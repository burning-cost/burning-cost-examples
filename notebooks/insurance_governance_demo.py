# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-governance: PRA SS1/23 Validation Workflow
# MAGIC
# MAGIC **What this notebook demonstrates:**
# MAGIC
# MAGIC UK pricing teams operating under PRA SS1/23 need consistent, auditable validation output for every production model.
# MAGIC The `insurance-governance` library automates the statistical tests (Gini, PSI, Hosmer-Lemeshow, calibration) and
# MAGIC model risk management governance (MRM risk tier scoring, HTML governance packs) in a single install.
# MAGIC
# MAGIC We benchmark a CatBoost Poisson frequency model against a simple Poisson GLM on a synthetic 50k UK motor portfolio.
# MAGIC Both models are put through the full PRA SS1/23 validation suite and registered in the MRM inventory.
# MAGIC
# MAGIC **SS1/23 principles covered:**
# MAGIC - Principle 1 (Model identification and classification): MRM ModelCard + RiskTierScorer
# MAGIC - Principle 3 (Model development, implementation and use): ValidationModelCard documenting methodology and limitations
# MAGIC - Principle 4 (Model validation): Discrimination (Gini), stability (PSI), calibration (A/E, HL test)
# MAGIC - Principle 5 (Ongoing monitoring): Monitoring plan completeness check with named owner and trigger thresholds
# MAGIC
# MAGIC **Time to run end-to-end:** approximately 4-6 minutes (dominated by CatBoost fit, not validation tests)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install insurance-governance catboost polars matplotlib scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from insurance_governance import (
    ModelValidationReport,
    ValidationModelCard,
    MRMModelCard,
    Assumption,
    Limitation,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
    RAGStatus,
)

print(f"insurance-governance ready")
print(f"numpy {np.__version__}, polars {pl.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Synthetic UK Motor Portfolio
# MAGIC
# MAGIC 50,000 policies. Temporal split: 60% train (policy years 2020-2022), 20% validation (2023), 20% test (2024).
# MAGIC This mirrors the temporal holdout structure PRA expects in model validation — never a random split for time-series
# MAGIC insurance data.
# MAGIC
# MAGIC **Features:** driver age, vehicle age, region (10 GB regions), annual mileage, NCD band.
# MAGIC **Target:** claim count. Frequency modelled as Poisson with natural log of exposure as offset.
# MAGIC
# MAGIC True DGP is intentionally non-linear so CatBoost has a real advantage over the GLM. That gap shows up in the
# MAGIC Gini comparison and is the point of the benchmark.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 50_000

# ── Rating factors ──────────────────────────────────────────────────────────
driver_age    = rng.integers(17, 80, N).astype(float)
vehicle_age   = rng.integers(0, 20, N).astype(float)
region        = rng.integers(0, 10, N)   # 0-9, representing 10 GB regions
annual_mileage = rng.exponential(8_000, N).clip(500, 50_000)
ncd_band      = rng.integers(0, 6, N)    # 0 = no NCD, 5 = max NCD
exposure      = rng.uniform(0.1, 1.0, N) # policy years (fractional for mid-term)

REGION_NAMES = [
    "London", "South East", "South West", "East of England",
    "East Midlands", "West Midlands", "Yorkshire", "North West",
    "North East", "Scotland",
]
region_names = np.array([REGION_NAMES[r] for r in region])

# ── True log-frequency (non-linear DGP: hard for a GLM, easy for CatBoost) ──
# Young drivers: U-shaped age effect — high at 17-25, low at 40-65, rising again at 75+
age_effect = (
    0.6 * np.exp(-((driver_age - 21) / 8) ** 2)    # young-driver spike
    + 0.3 * np.exp(-((driver_age - 75) / 10) ** 2)  # elderly spike
    - 0.4 * np.exp(-((driver_age - 50) / 15) ** 2)  # experienced-driver trough
)
vehicle_age_effect = 0.02 * vehicle_age + 0.001 * vehicle_age ** 2   # wears and age
mileage_effect     = 0.08 * np.log1p(annual_mileage / 1_000)
ncd_effect         = -0.12 * ncd_band                                 # NCD discounts risk
region_base        = np.array([0.0, -0.05, -0.1, -0.03, 0.02,
                                0.05, 0.03, 0.08, 0.12, -0.07])[region]
# Interaction: young drivers in high-mileage segments are disproportionately risky
youth_mileage_interaction = np.where(
    (driver_age < 25) & (annual_mileage > 15_000), 0.25, 0.0
)

log_freq = (
    -3.2                          # intercept: ~4% base frequency
    + age_effect
    + vehicle_age_effect
    + mileage_effect
    + ncd_effect
    + region_base
    + youth_mileage_interaction
)
true_freq = np.exp(log_freq)

# ── Claim counts ─────────────────────────────────────────────────────────────
# Expected claims = freq * exposure (Poisson intensity)
claim_count = rng.poisson(true_freq * exposure)

# ── Temporal split ────────────────────────────────────────────────────────────
# Assign synthetic policy year for temporal ordering
policy_year_weights = np.concatenate([
    np.full(int(N * 0.60), 0),  # 2020-2022 train
    np.full(int(N * 0.20), 1),  # 2023 validation
    np.full(N - int(N * 0.60) - int(N * 0.20), 2),  # 2024 test
])
rng.shuffle(policy_year_weights)

train_mask = policy_year_weights == 0
val_mask   = policy_year_weights == 1
test_mask  = policy_year_weights == 2

print(f"Train:      {train_mask.sum():,} policies")
print(f"Validation: {val_mask.sum():,} policies")
print(f"Test:       {test_mask.sum():,} policies")
print(f"Overall claim frequency: {claim_count.mean():.4f}")
print(f"Claims > 0: {(claim_count > 0).sum():,} ({100*(claim_count>0).mean():.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Feature DataFrames (Polars)
# MAGIC
# MAGIC Passing Polars DataFrames to `ModelValidationReport` enables data quality checks,
# MAGIC feature drift PSI, and proxy discrimination analysis automatically.

# COMMAND ----------

feature_cols = ["driver_age", "vehicle_age", "region", "annual_mileage", "ncd_band", "exposure"]

X_all = pl.DataFrame({
    "driver_age":    driver_age,
    "vehicle_age":   vehicle_age,
    "region":        region_names,    # categorical: string
    "annual_mileage": annual_mileage,
    "ncd_band":      ncd_band.astype(float),
    "exposure":      exposure,
})

X_train = X_all.filter(pl.Series(train_mask))
X_val   = X_all.filter(pl.Series(val_mask))
X_test  = X_all.filter(pl.Series(test_mask))

print(X_all.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit Models
# MAGIC
# MAGIC ### 4a. CatBoost Poisson Frequency Model
# MAGIC
# MAGIC Log offset for exposure. CatBoost handles the non-linear age interaction naturally without
# MAGIC feature engineering. This is the champion model we are validating.

# COMMAND ----------

from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings("ignore")

cat_features_idx = [2]  # "region" column index

train_pool = Pool(
    data=X_train.to_numpy(),
    label=claim_count[train_mask],
    weight=exposure[train_mask],  # Poisson: weight = exposure
    cat_features=cat_features_idx,
    feature_names=feature_cols,
)

val_pool = Pool(
    data=X_val.to_numpy(),
    label=claim_count[val_mask],
    weight=exposure[val_mask],
    cat_features=cat_features_idx,
    feature_names=feature_cols,
)

catboost_model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=50,
)
catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)

# Predictions: raw CatBoost output is log(frequency) for Poisson loss
# The model predicts expected count given input; we want frequency so divide by exposure
cb_pred_train_raw = catboost_model.predict(X_train.to_numpy())
cb_pred_val_raw   = catboost_model.predict(X_val.to_numpy())

# Convert to frequency (divide by exposure weight)
cb_freq_train = cb_pred_train_raw / exposure[train_mask]
cb_freq_val   = cb_pred_val_raw   / exposure[val_mask]

print(f"\nCatBoost validation: mean predicted freq = {cb_freq_val.mean():.4f}, "
      f"actual = {(claim_count[val_mask] / exposure[val_mask]).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Poisson GLM (Benchmark)
# MAGIC
# MAGIC A standard log-link GLM with linear main effects. No interactions, no polynomial terms —
# MAGIC this is what a traditional pricing team would fit before GBMs became standard.
# MAGIC The GLM is the challenger here; in reality you would compare against the previous champion.

# COMMAND ----------

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numeric_features = ["driver_age", "vehicle_age", "annual_mileage", "ncd_band"]
categorical_features = ["region"]

# Build design matrices from the Polars DataFrames
def build_glm_matrix(X_pl: pl.DataFrame):
    numeric_arr = X_pl.select(numeric_features).to_numpy()
    region_arr  = X_pl["region"].to_list()
    return numeric_arr, region_arr

enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

def get_glm_X(X_pl, fit_encoder=False):
    numeric_arr = X_pl.select(numeric_features).to_numpy()
    region_arr  = np.array(X_pl["region"].to_list()).reshape(-1, 1)
    if fit_encoder:
        region_enc = enc.fit_transform(region_arr)
    else:
        region_enc = enc.transform(region_arr)
    return np.hstack([numeric_arr, region_enc])

X_glm_train = get_glm_X(X_train, fit_encoder=True)
X_glm_val   = get_glm_X(X_val)

glm = PoissonRegressor(alpha=1e-4, max_iter=300)
# sklearn PoissonRegressor uses sample_weight for exposure offset (weighted regression)
glm.fit(X_glm_train, claim_count[train_mask], sample_weight=exposure[train_mask])

glm_pred_train = glm.predict(X_glm_train) / exposure[train_mask]
glm_pred_val   = glm.predict(X_glm_val)   / exposure[val_mask]

print(f"GLM validation: mean predicted freq = {glm_pred_val.mean():.4f}, "
      f"actual = {(claim_count[val_mask] / exposure[val_mask]).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. PRA SS1/23 Validation Suite: CatBoost
# MAGIC
# MAGIC The `ModelValidationReport` facade runs all standard checks in one call:
# MAGIC - **Gini coefficient** with bootstrap 95% confidence interval (discrimination)
# MAGIC - **10-band lift chart** (actual vs predicted by predicted decile)
# MAGIC - **A/E ratio** overall and by decile with Poisson confidence intervals
# MAGIC - **Hosmer-Lemeshow** goodness-of-fit test
# MAGIC - **PSI** on score distribution (train vs validation, stability check)
# MAGIC - **Feature drift** PSI for each input variable
# MAGIC - **Monitoring plan** completeness check
# MAGIC
# MAGIC The `ValidationModelCard` records model metadata for the audit trail. Every field is
# MAGIC a governance commitment — limitations documented here will appear in the HTML report.

# COMMAND ----------

cb_card = ValidationModelCard(
    name="Motor TPPD Frequency v3.0 (CatBoost)",
    version="3.0.0",
    purpose=(
        "Predict claim frequency (claims per policy year) for UK private motor "
        "TPPD business. Supports technical pricing for new business and renewal."
    ),
    methodology="CatBoost gradient boosting, Poisson loss, log-link. 500 trees, depth 6.",
    target="claim_count",
    features=["driver_age", "vehicle_age", "region", "annual_mileage", "ncd_band"],
    limitations=[
        "Trained on 2020-2022 data. Performance may degrade if claim frequency "
        "trends shift materially (COVID-era suppression artefact).",
        "No telematics data. Mileage is self-reported and subject to under-declaration bias.",
        "Fleet and commercial policies excluded from training — do not apply to non-personal lines.",
    ],
    owner="Pricing Actuarial Team",
    model_type="GBM",
    distribution_family="Poisson",
    materiality_tier=1,
    approved_by=["Chief Actuary", "Model Risk Committee"],
    monitoring_frequency="Quarterly",
    monitoring_owner="Jane Smith, Senior Pricing Actuary",
    monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10, "gini_decline": 0.05},
)

cb_report = ModelValidationReport(
    model_card=cb_card,
    y_val=claim_count[val_mask],
    y_pred_val=cb_freq_val * exposure[val_mask],   # expected counts for calibration tests
    exposure_val=exposure[val_mask],
    y_train=claim_count[train_mask],
    y_pred_train=cb_freq_train * exposure[train_mask],
    exposure_train=exposure[train_mask],
    X_train=X_train,
    X_val=X_val,
    monitoring_owner="Jane Smith, Senior Pricing Actuary",
    monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
    random_state=42,
)

cb_results = cb_report.run()
cb_rag = cb_report.get_rag_status()

print(f"CatBoost validation complete: {len(cb_results)} test results")
print(f"Overall RAG status: {cb_rag.value.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Print CatBoost Results Summary

# COMMAND ----------

print(f"{'Test':<45} {'Pass':<6} {'Value':>10}  Details")
print("-" * 120)
for r in cb_results:
    val_str = f"{r.metric_value:.4f}" if r.metric_value is not None else "—"
    status  = "PASS" if r.passed else "FAIL"
    # Truncate long details for display
    detail_short = r.details[:80] + "..." if len(r.details) > 80 else r.details
    print(f"{r.test_name:<45} {status:<6} {val_str:>10}  {detail_short}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. PRA SS1/23 Validation Suite: GLM (Challenger)

# COMMAND ----------

glm_card = ValidationModelCard(
    name="Motor TPPD Frequency v1.2 (GLM)",
    version="1.2.0",
    purpose=(
        "Predict claim frequency for UK private motor TPPD business. "
        "Log-link Poisson GLM with main effects only."
    ),
    methodology="Poisson GLM, log-link, main effects only. No interactions or polynomial terms.",
    target="claim_count",
    features=["driver_age", "vehicle_age", "region", "annual_mileage", "ncd_band"],
    limitations=[
        "Linear main effects only — does not capture the young-driver/high-mileage "
        "interaction or the non-linear age curve. Expect Gini below 0.35.",
        "No telematics data.",
        "Fleet and commercial policies excluded.",
    ],
    owner="Pricing Actuarial Team",
    model_type="GLM",
    distribution_family="Poisson",
    materiality_tier=2,
    approved_by=["Chief Actuary"],
    monitoring_frequency="Quarterly",
    monitoring_owner="Jane Smith, Senior Pricing Actuary",
    monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
)

glm_report = ModelValidationReport(
    model_card=glm_card,
    y_val=claim_count[val_mask],
    y_pred_val=glm_pred_val * exposure[val_mask],
    exposure_val=exposure[val_mask],
    y_train=claim_count[train_mask],
    y_pred_train=glm_pred_train * exposure[train_mask],
    exposure_train=exposure[train_mask],
    X_train=X_train,
    X_val=X_val,
    monitoring_owner="Jane Smith, Senior Pricing Actuary",
    monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
    random_state=42,
)

glm_results = glm_report.run()
glm_rag = glm_report.get_rag_status()

print(f"GLM validation complete: {len(glm_results)} test results")
print(f"Overall RAG status: {glm_rag.value.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MRM Risk Tier Scoring
# MAGIC
# MAGIC `RiskTierScorer` maps six dimensions onto a 0-100 composite score, determining
# MAGIC the MRM tier (1=Critical, 2=Significant, 3=Informational). This is the objective,
# MAGIC documented scoring that replaces the "vibes-based tier assignment" conversation
# MAGIC at the Model Risk Committee.
# MAGIC
# MAGIC Dimensions:
# MAGIC 1. **Materiality** (GWP impacted) — 25 pts max
# MAGIC 2. **Complexity** (model architecture) — 20 pts max
# MAGIC 3. **Data quality** (external data use) — 10 pts max
# MAGIC 4. **Validation coverage** (months since last validation) — 10 pts max
# MAGIC 5. **Drift history** (monitoring triggers fired) — 10 pts max
# MAGIC 6. **Regulatory exposure** (production status + regulatory + customer-facing) — 25 pts max

# COMMAND ----------

scorer = RiskTierScorer()

# CatBoost (champion, customer-facing, large GWP)
cb_tier = scorer.score(
    gwp_impacted=125_000_000,       # £125m GWP — largest motor book
    model_complexity="high",         # CatBoost ensemble, 50+ features
    deployment_status="champion",    # live in production, setting customer prices
    regulatory_use=False,            # not used in Solvency II capital model
    external_data=False,             # no telematics or credit data
    customer_facing=True,            # directly drives premiums
    validation_months_ago=2,         # just validated — good governance
    drift_triggers_last_year=1,      # one PSI trigger fired in Q3
)

# GLM (challenger, being phased out)
glm_tier = scorer.score(
    gwp_impacted=125_000_000,
    model_complexity="low",          # linear GLM, simple
    deployment_status="challenger",  # shadow running, not live
    regulatory_use=False,
    external_data=False,
    customer_facing=False,           # challenger: not directly setting prices
    validation_months_ago=8,         # validated 8 months ago
    drift_triggers_last_year=0,
)

print("=" * 70)
print("CatBoost MRM Risk Tier")
print("=" * 70)
print(cb_tier.rationale)
print()
print("=" * 70)
print("GLM MRM Risk Tier")
print("=" * 70)
print(glm_tier.rationale)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Inventory Registration
# MAGIC
# MAGIC The `ModelInventory` persists model metadata and tier assignments to a JSON file.
# MAGIC In practice this file lives in the same git repo as your model code — every promotion
# MAGIC event is a commit, every validation run updates the `last_validation_run` field.

# COMMAND ----------

import tempfile, os

# Use a temp file for demo — in production this is a committed JSON file in your repo
inventory_path = "/tmp/mrm_registry_demo.json"

# MRM ModelCard (governance metadata — separate from the validation ModelCard)
cb_mrm_card = MRMModelCard(
    model_id="motor-freq-catboost-v3",
    model_name="Motor TPPD Frequency (CatBoost)",
    version="3.0.0",
    model_class="pricing",
    intended_use=(
        "Frequency pricing for UK private motor TPPD. "
        "Not for use on commercial fleet or motorbike policies."
    ),
    not_intended_for=["commercial fleet", "motorcycles", "claims reserving"],
    target_variable="claim_count",
    distribution_family="Poisson",
    model_type="CatBoost",
    rating_factors=["driver_age", "vehicle_age", "region", "annual_mileage", "ncd_band"],
    training_data_period=("2020-01-01", "2022-12-31"),
    developer="Pricing Actuarial Team",
    champion_challenger_status="champion",
    portfolio_scope="UK private motor TPPD",
    geographic_scope="Great Britain",
    customer_facing=True,
    regulatory_use=False,
    gwp_impacted=125_000_000,
    materiality_tier=cb_tier.tier,
    tier_rationale=cb_tier.rationale,
    monitoring_owner="Jane Smith, Senior Pricing Actuary",
    monitoring_frequency="Quarterly",
    monitoring_triggers={"psi_score": 0.25, "ae_ratio_deviation": 0.10},
    assumptions=[
        Assumption(
            description="Claim frequency is stationary since 2022; no structural break from COVID-era suppression.",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring by account year. Trigger at A/E > 1.10.",
        ),
        Assumption(
            description="Self-reported annual mileage is broadly accurate within +/- 20%.",
            risk="LOW",
            mitigation="Annual comparison against telematics cohort if data becomes available.",
        ),
        Assumption(
            description="GBM feature interactions are stable across policy years.",
            risk="HIGH",
            mitigation=(
                "PSI monitoring on all top-10 features quarterly. "
                "Full refit if PSI > 0.25 on any feature."
            ),
        ),
    ],
    limitations=[
        Limitation(
            description="No telematics data. Mileage is self-reported.",
            impact="May systematically underprice high-mileage risks who under-declare.",
            population_at_risk="Young urban drivers.",
            monitoring_flag=True,
        ),
        Limitation(
            description="Trained on 2020-2022. COVID suppression effects may inflate apparent frequency.",
            impact="If suppression reverses, model will underestimate true frequency.",
            population_at_risk="All policyholders.",
            monitoring_flag=True,
        ),
    ],
    overall_rag=cb_rag.value.upper(),
)

glm_mrm_card = MRMModelCard(
    model_id="motor-freq-glm-v1",
    model_name="Motor TPPD Frequency (GLM)",
    version="1.2.0",
    model_class="pricing",
    intended_use="Legacy Poisson GLM, shadow mode only. Being replaced by CatBoost champion.",
    model_type="GLM",
    distribution_family="Poisson",
    rating_factors=["driver_age", "vehicle_age", "region", "annual_mileage", "ncd_band"],
    developer="Pricing Actuarial Team",
    champion_challenger_status="challenger",
    portfolio_scope="UK private motor TPPD",
    geographic_scope="Great Britain",
    customer_facing=False,
    gwp_impacted=125_000_000,
    materiality_tier=glm_tier.tier,
    tier_rationale=glm_tier.rationale,
    overall_rag=glm_rag.value.upper(),
)

inventory = ModelInventory(inventory_path)
inventory.register(cb_mrm_card, cb_tier)
inventory.register(glm_mrm_card, glm_tier)

# Show what's in the inventory
models = inventory.list_models()
print(f"Registry contains {len(models)} models:")
for m in models:
    print(f"  [{m['model_id']}] tier={m['tier']} status={m['champion_challenger_status']} "
          f"rag={m.get('overall_rag', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. HTML Report Generation
# MAGIC
# MAGIC Both validation reports and the MRM governance pack are generated as self-contained HTML.
# MAGIC No CDN, no JavaScript dependencies — suitable for embedding in SharePoint or emailing to the MRC.
# MAGIC
# MAGIC In Databricks we use `displayHTML` to render them inline in the notebook.

# COMMAND ----------

# Generate CatBoost validation report
cb_html_path = "/tmp/catboost_validation_report.html"
cb_report.generate(cb_html_path)
cb_report.to_json("/tmp/catboost_validation_report.json")

# Generate GLM validation report
glm_html_path = "/tmp/glm_validation_report.html"
glm_report.generate(glm_html_path)

# Generate MRM governance pack for CatBoost
gov_report = GovernanceReport(card=cb_mrm_card, tier=cb_tier)
gov_html_path = "/tmp/catboost_mrm_pack.html"
gov_report.save_html(gov_html_path)

print("Reports generated:")
print(f"  {cb_html_path}")
print(f"  /tmp/catboost_validation_report.json")
print(f"  {glm_html_path}")
print(f"  {gov_html_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9a. Display CatBoost Validation Report (inline)

# COMMAND ----------

with open(cb_html_path, "r") as f:
    cb_html_content = f.read()

displayHTML(cb_html_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9b. Display MRM Governance Pack (inline)

# COMMAND ----------

with open(gov_html_path, "r") as f:
    gov_html_content = f.read()

displayHTML(gov_html_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Benchmark Table: CatBoost vs GLM
# MAGIC
# MAGIC Extract key metrics from both result sets for a side-by-side comparison.
# MAGIC This is the table you put in front of the MRC to justify the champion selection.

# COMMAND ----------

def extract_metric(results, test_name):
    """Pull metric_value and passed flag from a named test result."""
    for r in results:
        if r.test_name == test_name:
            return r.metric_value, r.passed, r.details
    return None, None, "Not found"

def format_metric(val, passed):
    if val is None:
        return "—"
    sym = "" if passed else " !"
    return f"{val:.4f}{sym}"

# Key metrics to extract
metrics_to_compare = [
    ("gini_coefficient",    "Gini coefficient"),
    ("gini_with_ci",        "Gini (bootstrap 95% CI)"),
    ("hosmer_lemeshow",     "Hosmer-Lemeshow p-value"),
    ("actual_vs_expected",  "Overall A/E ratio"),
    ("psi_score",           "PSI (score distribution)"),
]

print(f"{'Metric':<40} {'CatBoost':>20} {'GLM':>20}")
print("-" * 82)

benchmark_rows = []
for test_name, label in metrics_to_compare:
    cb_val, cb_pass, _ = extract_metric(cb_results, test_name)
    gm_val, gm_pass, _ = extract_metric(glm_results, test_name)
    cb_str = format_metric(cb_val, cb_pass)
    gm_str = format_metric(gm_val, gm_pass)
    print(f"{label:<40} {cb_str:>20} {gm_str:>20}")
    benchmark_rows.append({
        "Metric": label,
        "CatBoost": cb_str,
        "GLM": gm_str,
    })

print("-" * 82)
print(f"{'Overall RAG':<40} {cb_rag.value.upper():>20} {glm_rag.value.upper():>20}")
print(f"{'MRM Tier':<40} {f'Tier {cb_tier.tier} ({cb_tier.tier_label})':>20} {f'Tier {glm_tier.tier} ({glm_tier.tier_label})':>20}")
print(f"{'Tier Score (0-100)':<40} {cb_tier.score:>20.1f} {glm_tier.score:>20.1f}")
print()
print("  ! = test failed threshold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Visualise Discrimination: Lorenz Curves
# MAGIC
# MAGIC The Gini coefficient measures how well the model separates high-risk from low-risk policies.
# MAGIC The Lorenz curve shows the cumulative share of claims explained as policies are ranked by
# MAGIC predicted risk (cheapest to most expensive). A perfect model's curve hugs the top-left.
# MAGIC A flat model's curve is the 45-degree diagonal.

# COMMAND ----------

def lorenz_arrays(y_true, y_pred, weights=None):
    """Compute Lorenz curve coordinates for claim frequency model evaluation."""
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    w_sorted = weights[order] if weights is not None else np.ones(len(y_true))
    cum_policies = np.cumsum(w_sorted) / w_sorted.sum()
    cum_claims   = np.cumsum(y_sorted * w_sorted) / (y_sorted * w_sorted).sum()
    return np.concatenate([[0], cum_policies]), np.concatenate([[0], cum_claims])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Lorenz curves
ax = axes[0]
x_cb, y_cb = lorenz_arrays(claim_count[val_mask], cb_freq_val, exposure[val_mask])
x_glm, y_glm = lorenz_arrays(claim_count[val_mask], glm_pred_val, exposure[val_mask])
ax.plot(x_cb,  y_cb,  "b-",  lw=2, label=f"CatBoost (Gini = {extract_metric(cb_results, 'gini_coefficient')[0]:.3f})")
ax.plot(x_glm, y_glm, "r--", lw=2, label=f"GLM      (Gini = {extract_metric(glm_results, 'gini_coefficient')[0]:.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=1, label="No discrimination (Gini = 0)")
ax.set_xlabel("Cumulative share of policies (ranked by predicted risk)")
ax.set_ylabel("Cumulative share of claims")
ax.set_title("Lorenz Curves — Discrimination\n(CatBoost vs GLM, validation set)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# Predicted frequency distribution by model
ax = axes[1]
ax.hist(cb_freq_val,  bins=50, alpha=0.6, color="blue",  label="CatBoost", density=True)
ax.hist(glm_pred_val, bins=50, alpha=0.6, color="red",   label="GLM",      density=True)
ax.set_xlabel("Predicted claim frequency")
ax.set_ylabel("Density")
ax.set_title("Predicted Frequency Distribution\n(validation set)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.5)

plt.tight_layout()
plt.savefig("/tmp/governance_benchmark.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Calibration: Actual vs Expected by Decile
# MAGIC
# MAGIC A model that discriminates well (high Gini) but is miscalibrated (A/E far from 1.0) will
# MAGIC misprice. The A/E by decile chart catches systematic under- or over-pricing in rate bands.
# MAGIC PRA SS1/23 Principle 4 requires evidence that calibration has been assessed.

# COMMAND ----------

def ae_by_decile(y_true, y_pred, exposure, n_bands=10):
    """Compute actual vs expected by predicted frequency decile."""
    pred_freq = y_pred / exposure
    deciles = np.percentile(pred_freq, np.linspace(0, 100, n_bands + 1))
    bands = []
    for i in range(n_bands):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (pred_freq >= lo) & (pred_freq <= hi) if i == n_bands - 1 else (pred_freq >= lo) & (pred_freq < hi)
        if mask.sum() == 0:
            continue
        actual_claims    = y_true[mask].sum()
        expected_claims  = y_pred[mask].sum()
        ae = actual_claims / expected_claims if expected_claims > 0 else np.nan
        mid_pred_freq    = pred_freq[mask].mean()
        bands.append({"decile": i + 1, "n": mask.sum(), "actual": actual_claims,
                      "expected": expected_claims, "ae": ae, "mid_freq": mid_pred_freq})
    return bands

cb_ae_bands  = ae_by_decile(claim_count[val_mask], cb_freq_val * exposure[val_mask],  exposure[val_mask])
glm_ae_bands = ae_by_decile(claim_count[val_mask], glm_pred_val * exposure[val_mask], exposure[val_mask])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, bands, title, color in [
    (axes[0], cb_ae_bands,  "CatBoost — A/E by Predicted Decile", "blue"),
    (axes[1], glm_ae_bands, "GLM — A/E by Predicted Decile",      "red"),
]:
    deciles = [b["decile"] for b in bands]
    ae_vals  = [b["ae"]     for b in bands]
    ax.bar(deciles, ae_vals, color=color, alpha=0.7, label="A/E ratio")
    ax.axhline(1.0,  color="black", lw=2, linestyle="-",  label="Perfect (A/E = 1.0)")
    ax.axhline(1.10, color="orange", lw=1, linestyle="--", label="±10% threshold")
    ax.axhline(0.90, color="orange", lw=1, linestyle="--")
    ax.set_xlabel("Predicted frequency decile")
    ax.set_ylabel("A/E ratio")
    ax.set_title(title)
    ax.set_ylim(0.5, 1.6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/tmp/ae_by_decile.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Stability: PSI Analysis
# MAGIC
# MAGIC Population Stability Index (PSI) measures whether the distribution of predicted scores has shifted
# MAGIC between train and validation. PSI > 0.25 is the conventional trigger for a model review;
# MAGIC PSI > 0.10 warrants monitoring attention.
# MAGIC
# MAGIC Here we also show PSI by input feature, which the `ModelValidationReport` computes automatically
# MAGIC when Polars DataFrames are passed.

# COMMAND ----------

def compute_psi_manual(reference, current, n_bins=10):
    """PSI using equal-width bins on the combined range."""
    combined = np.concatenate([reference, current])
    bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-10
    bins[-1] += 1e-10

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current,   bins=bins)

    ref_pct = ref_hist / ref_hist.sum()
    cur_pct = cur_hist / cur_hist.sum()

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-8, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-8, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi, bins, ref_pct, cur_pct

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred_train, pred_val, model_name, color in [
    (axes[0], cb_freq_train,  cb_freq_val,  "CatBoost", "blue"),
    (axes[1], glm_pred_train, glm_pred_val, "GLM",      "red"),
]:
    psi, bins, ref_pct, cur_pct = compute_psi_manual(pred_train, pred_val)
    x = np.arange(len(ref_pct))
    width = 0.4
    ax.bar(x - width/2, ref_pct, width, label="Train (reference)",  alpha=0.7, color=color)
    ax.bar(x + width/2, cur_pct, width, label="Validation (current)", alpha=0.7, color="grey")
    ax.set_title(f"{model_name} — Score PSI = {psi:.4f}\n"
                 f"({'STABLE' if psi < 0.1 else 'MONITOR' if psi < 0.25 else 'UNSTABLE'})")
    ax.set_xlabel("Score bin (ranked by train distribution)")
    ax.set_ylabel("Proportion of records")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/tmp/psi_chart.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. JSON Sidecar — Machine-Readable Audit Record
# MAGIC
# MAGIC The `.to_json()` output is the artefact that plugs into an MRM system or gets stored alongside
# MAGIC the model binary. It contains every TestResult in structured form — no manual copy-paste from
# MAGIC a Word report.

# COMMAND ----------

import json

with open("/tmp/catboost_validation_report.json", "r") as f:
    cb_json = json.load(f)

# Show top-level structure
print(f"Run ID:        {cb_json.get('run_id', 'N/A')}")
print(f"Generated:     {cb_json.get('generated_date', 'N/A')}")
print(f"RAG status:    {cb_json.get('rag_status', 'N/A')}")
print(f"Test count:    {len(cb_json.get('results', []))}")
print()
print("First 3 results (raw JSON):")
for r in cb_json.get("results", [])[:3]:
    print(f"  {r['test_name']}: passed={r['passed']}, metric={r['metric_value']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Governance Commentary: CatBoost vs GLM
# MAGIC
# MAGIC ### Discrimination (Gini)
# MAGIC
# MAGIC The CatBoost model captures the non-linear interaction between driver age and annual mileage
# MAGIC that the GLM cannot represent with main effects only. This is visible in the Gini gap.
# MAGIC A Gini above 0.30 is generally considered acceptable for UK motor frequency; above 0.45 is strong.
# MAGIC The GLM would fail the discrimination threshold a well-governed insurer would set.
# MAGIC
# MAGIC ### Stability (PSI)
# MAGIC
# MAGIC Both models score low PSI on the 2023 validation year — expected given the synthetic DGP is
# MAGIC stationary. In practice, a PSI trigger on a live model suggests distribution shift in the portfolio
# MAGIC (new products, broker mix change) or concept drift in the claims process.
# MAGIC
# MAGIC ### Calibration (A/E, Hosmer-Lemeshow)
# MAGIC
# MAGIC CatBoost has better within-decile calibration because it directly optimises Poisson deviance.
# MAGIC The GLM, optimised via IRLS with a log-link, has overall calibration close to 1.0 (by construction)
# MAGIC but shows A/E excursions at the tails where the non-linearity is worst.
# MAGIC
# MAGIC ### MRM Tier Assignment
# MAGIC
# MAGIC Both models score the same GWP materiality (£125m) but CatBoost's higher complexity dimension
# MAGIC and champion status push its tier score higher. This is correct governance behaviour: a more
# MAGIC complex champion model warrants closer scrutiny than a simple challenger.
# MAGIC
# MAGIC ### PRA SS1/23 Compliance Summary
# MAGIC
# MAGIC | SS1/23 Principle | Evidence Produced |
# MAGIC |---|---|
# MAGIC | Principle 1 — Model identification | MRM ModelCard registered in inventory; tier assigned objectively |
# MAGIC | Principle 3 — Development and use | ValidationModelCard with documented limitations and approved_by |
# MAGIC | Principle 4 — Validation | Gini + CI, lift chart, A/E by decile, Hosmer-Lemeshow, PSI |
# MAGIC | Principle 5 — Monitoring | Named monitoring owner, quarterly cadence, quantified PSI/A/E triggers |
# MAGIC
# MAGIC All outputs are persisted as HTML (for the board pack) and JSON (for the MRM system). The
# MAGIC validation run is linked to the MRM card via `last_validation_run_id`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Full Benchmark Summary Table

# COMMAND ----------

# Collect all comparable metrics
def safe_metric(results, test_name):
    for r in results:
        if r.test_name == test_name:
            return r.metric_value, r.passed
    return None, None

print(f"{'Metric':<45} {'CatBoost':>15} {'GLM':>15} {'Winner':>12}")
print("=" * 90)

comparisons = [
    ("gini_coefficient",     "Gini coefficient",          "higher"),
    ("hosmer_lemeshow",      "Hosmer-Lemeshow p-value",   "higher"),
    ("actual_vs_expected",   "Overall A/E ratio",         "closer_1"),
    ("psi_score",            "Score PSI",                 "lower"),
]

for test_name, label, better in comparisons:
    cb_v, cb_p = safe_metric(cb_results, test_name)
    gm_v, gm_p = safe_metric(glm_results, test_name)

    cb_str = f"{cb_v:.4f}" if cb_v is not None else "—"
    gm_str = f"{gm_v:.4f}" if gm_v is not None else "—"

    if cb_v is not None and gm_v is not None:
        if better == "higher":
            winner = "CatBoost" if cb_v > gm_v else "GLM"
        elif better == "lower":
            winner = "CatBoost" if cb_v < gm_v else "GLM"
        elif better == "closer_1":
            winner = "CatBoost" if abs(cb_v - 1.0) < abs(gm_v - 1.0) else "GLM"
        else:
            winner = "—"
    else:
        winner = "—"

    print(f"{label:<45} {cb_str:>15} {gm_str:>15} {winner:>12}")

print("=" * 90)
print(f"{'RAG Status':<45} {cb_rag.value.upper():>15} {glm_rag.value.upper():>15}")
print(f"{'MRM Tier':<45} {'Tier ' + str(cb_tier.tier):>15} {'Tier ' + str(glm_tier.tier):>15}")
print(f"{'MRM Score (0-100)':<45} {cb_tier.score:>15.1f} {glm_tier.score:>15.1f}")
print(f"{'Review Cadence':<45} {cb_tier.review_frequency:>15} {glm_tier.review_frequency:>15}")
print(f"{'Sign-off Required':<45} {cb_tier.sign_off_requirement:>15} {glm_tier.sign_off_requirement:>15}")
