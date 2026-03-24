# Databricks notebook source

# MAGIC %md
# MAGIC # End-to-End UK Motor Pricing Workflow
# MAGIC
# MAGIC A complete pricing model development and deployment pipeline using the Burning Cost library stack:
# MAGIC
# MAGIC | Library | Role |
# MAGIC |---|---|
# MAGIC | `insurance-synthetic` | Generate a realistic synthetic UK motor portfolio |
# MAGIC | `catboost` | Train a Poisson GBM frequency model |
# MAGIC | `shap-relativities` | Extract multiplicative rating factor relativities |
# MAGIC | `insurance-governance` | PRA SS1/23-compliant model validation report |
# MAGIC | `insurance-deploy` | Register models and run champion/challenger in shadow mode |
# MAGIC
# MAGIC The data is entirely synthetic. In production, replace Section 1 with your actual policy and claims extract. Everything from Section 2 onwards works identically on real data.

# COMMAND ----------

# MAGIC %pip install insurance-synthetic shap-relativities insurance-deploy insurance-governance catboost shap scipy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Generate a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC We use `insurance-synthetic` to produce 20,000 synthetic policies. The library fits a vine copula to reproduce the dependency structure between rating factors — tail dependence between young driver age, high vehicle group, and zero NCD is the key actuarial feature that generic data generators miss.
# MAGIC
# MAGIC In production: replace this cell with your actual portfolio extract. The column names below match the `uk_motor_schema()` definition, so you can rename your columns to match and the rest of the notebook runs unchanged.

# COMMAND ----------

from insurance_synthetic import InsuranceSynthesizer, uk_motor_schema

schema = uk_motor_schema()

schema_constraints = schema["constraints"]
schema_cols = {c.name: c for c in schema["columns"]}
categorical_cols = [
    c.name for c in schema["columns"] if c.dtype == "categorical"
]

rng = np.random.default_rng(42)
n_seed = 5_000

seed_driver_age = rng.integers(17, 80, size=n_seed)
seed_ncd_years = np.clip(
    rng.integers(0, 25, size=n_seed),
    0,
    seed_driver_age - 17,  # NCD can't exceed years eligible to drive
).astype(int)

seed_df = pl.DataFrame({
    "driver_age": seed_driver_age.tolist(),
    "vehicle_age": rng.integers(0, 20, size=n_seed).tolist(),
    "vehicle_group": rng.integers(1, 50, size=n_seed).tolist(),
    "region": rng.choice(schema_cols["region"].categories, size=n_seed).tolist(),
    "ncd_years": seed_ncd_years.tolist(),
    "cover_type": rng.choice(
        schema_cols["cover_type"].categories, size=n_seed,
        p=[0.75, 0.15, 0.10],
    ).tolist(),
    "payment_method": rng.choice(
        schema_cols["payment_method"].categories, size=n_seed,
        p=[0.55, 0.30, 0.15],
    ).tolist(),
    "annual_mileage": rng.integers(3_000, 25_000, size=n_seed).tolist(),
    "exposure": np.clip(rng.beta(5, 2, size=n_seed), 0.01, 1.0).tolist(),
    "claim_count": rng.choice([0, 1, 2], size=n_seed, p=[0.93, 0.065, 0.005]).tolist(),
    "claim_amount": [
        float(rng.gamma(shape=2.0, scale=2_500.0)) if c > 0 else 0.0
        for c in rng.choice([0, 1, 2], size=n_seed, p=[0.93, 0.065, 0.005])
    ],
})

print(f"Seed portfolio: {len(seed_df):,} policies")
print(f"Overall claim frequency: {seed_df['claim_count'].mean():.3f} claims/policy")

# COMMAND ----------

# Fit vine copula and generate a larger synthetic portfolio.
# trunc_lvl=3 captures the most important dependencies without O(d^2) cost.
synth = InsuranceSynthesizer(method="vine", trunc_lvl=3, random_state=42)

synth.fit(
    seed_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
    categorical_cols=categorical_cols,
)

N_SYNTHETIC = 20_000
portfolio = synth.generate(N_SYNTHETIC, constraints=schema_constraints)

print(f"Synthetic portfolio: {len(portfolio):,} policies")
print(f"Claim frequency: {portfolio['claim_count'].mean():.3f} claims/policy")
print(f"Mean exposure: {portfolio['exposure'].mean():.3f} policy years")
print(f"Cover type mix:")
print(portfolio['cover_type'].value_counts().sort('count', descending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Train a CatBoost Frequency Model
# MAGIC
# MAGIC We model claim frequency as Poisson with a log-link. CatBoost handles the categorical rating factors (region, cover_type, payment_method) natively without one-hot encoding — this is the main practical advantage over XGBoost for insurance pricing work.
# MAGIC
# MAGIC The offset `log(exposure)` is passed via `baseline`. This is equivalent to the GLM offset — it tells the model that a policy with 0.5 years of exposure should expect half as many claims as an identical policy with 1.0 year.

# COMMAND ----------

from catboost import CatBoostRegressor, Pool

# Split into train (70%) and holdout (30%)
n_train = int(0.7 * len(portfolio))
train_df = portfolio[:n_train]
holdout_df = portfolio[n_train:]

feature_cols = [
    "driver_age", "vehicle_age", "vehicle_group", "region",
    "ncd_years", "cover_type", "payment_method", "annual_mileage",
]

cat_feature_indices = [
    i for i, col in enumerate(feature_cols) if col in categorical_cols
]

train_pd = train_df.select(feature_cols + ["claim_count", "exposure"]).to_pandas()
holdout_pd = holdout_df.select(feature_cols + ["claim_count", "exposure"]).to_pandas()

train_log_exposure = np.log(train_pd["exposure"].clip(lower=1e-9)).values
holdout_log_exposure = np.log(holdout_pd["exposure"].clip(lower=1e-9)).values

train_pool = Pool(
    data=train_pd[feature_cols], label=train_pd["claim_count"],
    cat_features=cat_feature_indices, baseline=train_log_exposure,
)
holdout_pool = Pool(
    data=holdout_pd[feature_cols], label=holdout_pd["claim_count"],
    cat_features=cat_feature_indices, baseline=holdout_log_exposure,
)

# Challenger: more trees, deeper
model_v2 = CatBoostRegressor(
    loss_function="Poisson", iterations=500, depth=4,
    learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, verbose=0,
)
model_v2.fit(train_pool, eval_set=holdout_pool)

# Champion: fewer trees, shallower
model_v1 = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=3,
    learning_rate=0.08, l2_leaf_reg=5.0, random_seed=42, verbose=0,
)
model_v1.fit(train_pool)

v2_preds = model_v2.predict(holdout_pd[feature_cols])
v1_preds = model_v1.predict(holdout_pd[feature_cols])
actuals = holdout_pd["claim_count"].values

v2_rmse = float(np.sqrt(np.mean((v2_preds - actuals) ** 2)))
v1_rmse = float(np.sqrt(np.mean((v1_preds - actuals) ** 2)))

print(f"Champion (v1) holdout RMSE: {v1_rmse:.4f}")
print(f"Challenger (v2) holdout RMSE: {v2_rmse:.4f}")
print(f"Challenger lift: {(v1_rmse - v2_rmse) / v1_rmse:.1%} RMSE improvement")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: Extract SHAP Relativities
# MAGIC
# MAGIC We convert the challenger model's SHAP values into multiplicative rating factor relativities. The output is directly comparable to GLM `exp(beta)` relativities — a table of (feature, level, relativity) triples where the base level has relativity = 1.0.
# MAGIC
# MAGIC We use the holdout set for SHAP computation. Using training data would give overly optimistic relativities due to overfitting.

# COMMAND ----------

from shap_relativities import SHAPRelativities

holdout_features = holdout_df.select(feature_cols)
holdout_exposure = holdout_df["exposure"]

sr = SHAPRelativities(
    model=model_v2,
    X=holdout_features,
    exposure=holdout_exposure,
    categorical_features=["region", "cover_type", "payment_method"],
)
sr.fit()

base_levels = {
    "region": "London",
    "cover_type": "Comprehensive",
    "payment_method": "Annual",
}

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels=base_levels,
    ci_method="clt",
)

print("Rating factor relativities (challenger model, Poisson GBM):\n")
for feature in feature_cols:
    feature_rels = relativities.filter(pl.col("feature") == feature)
    if feature_rels.is_empty():
        continue
    print(f"  {feature}:")
    for row in feature_rels.iter_rows(named=True):
        level_str = str(row["level"])[:20].ljust(22)
        rel = row["relativity"]
        if rel is not None and not np.isnan(rel):
            ci_lo = row.get("lower_ci", float("nan"))
            ci_hi = row.get("upper_ci", float("nan"))
            if ci_lo is not None and not np.isnan(ci_lo):
                print(f"    {level_str} {rel:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
            else:
                print(f"    {level_str} {rel:.3f}")
    print()

baseline = sr.baseline()
print(f"Annualised base claim rate: {baseline:.4f} claims per policy-year")

print("\nSHAP diagnostic checks:")
checks = sr.validate()
for check_name, check_result in checks.items():
    status = "PASS" if check_result.passed else "WARN"
    print(f"  [{status}] {check_name}: {check_result.message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Model Validation Report
# MAGIC
# MAGIC `insurance-governance` provides PRA SS1/23-compliant model validation. `ModelCard` captures what the model is and who owns it. `ModelValidationReport` runs all standard tests (Gini with CI, A/E ratio, Hosmer-Lemeshow, lift chart) and writes an HTML report suitable for actuarial function review.

# COMMAND ----------

try:
    from insurance_governance.validation import (
        ModelCard, ModelValidationReport, PerformanceReport,
    )
    _governance_available = True
except ImportError:
    _governance_available = False
    print("insurance-governance not installed. Running manual validation checks.")

if _governance_available:
    _tmpdir_val = tempfile.mkdtemp(prefix="bc_examples_val_")
    html_path = Path(_tmpdir_val) / "validation_report.html"

    card = ModelCard(
        name="Motor Frequency Challenger v2",
        version="2.0.0",
        purpose="Predict claim frequency for UK private motor portfolio.",
        methodology="CatBoost Poisson GBM, 500 trees, depth 4, L2 reg 3.0.",
        target="claim_count",
        features=feature_cols,
        limitations=[
            "No telematics data — mileage is self-declared annual_mileage only",
            "Trained on synthetic data — must be retrained on actual portfolio",
        ],
        owner="Pricing Team",
    )

    train_pd_copy = train_pd.copy()
    train_pd_copy["predicted_rate"] = model_v2.predict(train_pd_copy[feature_cols])

    report = ModelValidationReport(
        model_card=card,
        y_val=actuals,
        y_pred_val=v2_preds,
        exposure_val=holdout_pd["exposure"].values,
        y_train=train_pd_copy["claim_count"].values,
        y_pred_train=train_pd_copy["predicted_rate"].values,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.generate(str(html_path))

    perf = PerformanceReport(actuals, v2_preds, exposure=holdout_pd["exposure"].values)
    gini_result = perf.gini_with_ci()
    ae_result = perf.ae_with_poisson_ci()

    print(f"Key validation metrics (holdout set, n={len(holdout_pd):,} policies):")
    print(f"  Gini: {gini_result.metric_value:.4f}  (95% CI: [{gini_result.extra['ci_lower']:.4f}, {gini_result.extra['ci_upper']:.4f}])")
    print(f"  A/E:  {ae_result.metric_value:.4f}  (95% CI: [{ae_result.extra['ci_lower']:.4f}, {ae_result.extra['ci_upper']:.4f}])")
    print(f"HTML report: {html_path}")
else:
    # Manual calibration check
    v2_predicted_total = float(np.sum(v2_preds * holdout_pd["exposure"].values))
    v2_observed_total = float(holdout_pd["claim_count"].sum())
    calibration_ratio = v2_predicted_total / v2_observed_total if v2_observed_total > 0 else float("nan")
    print(f"Calibration ratio (predicted / observed): {calibration_ratio:.4f}")

    mean_rate = float(np.mean(actuals))
    naive_preds = np.full_like(actuals, fill_value=mean_rate, dtype=float)
    naive_rmse = float(np.sqrt(np.mean((naive_preds - actuals) ** 2)))
    lift_vs_naive = (naive_rmse - v2_rmse) / naive_rmse
    print(f"RMSE lift vs naive: {lift_vs_naive:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5: Champion/Challenger Deployment (Shadow Mode)
# MAGIC
# MAGIC We register both models in a `ModelRegistry`, set up a shadow-mode experiment, and simulate quoting 1,000 renewal policies. Shadow mode means the champion always prices the live quote; the challenger scores in parallel but its price is never shown to the customer. This is the correct starting point — it gives us model quality data with zero FCA Consumer Duty risk.

# COMMAND ----------

from insurance_deploy import (
    ModelRegistry, Experiment, QuoteLogger, KPITracker, ModelComparison, ENBPAuditReport,
)

_tmpdir = tempfile.mkdtemp(prefix="bc_examples_")
registry_path = Path(_tmpdir) / "registry"
log_path = Path(_tmpdir) / "quotes.db"

registry = ModelRegistry(str(registry_path))

mv1 = registry.register(
    model_v1, name="motor_frequency", version="1.0",
    metadata={
        "training_date": "2025-01-01", "features": feature_cols,
        "objective": "Poisson", "iterations": 200, "depth": 3,
        "holdout_rmse": round(v1_rmse, 6),
    },
)
mv2 = registry.register(
    model_v2, name="motor_frequency", version="2.0",
    metadata={
        "training_date": "2025-07-01", "features": feature_cols,
        "objective": "Poisson", "iterations": 500, "depth": 4,
        "holdout_rmse": round(v2_rmse, 6),
    },
)

registry.set_champion("motor_frequency", "1.0")
print(f"Champion: {registry.champion('motor_frequency')}")

experiment = Experiment(
    name="motor_v2_vs_v1_shadow",
    champion=mv1, challenger=mv2,
    challenger_pct=0.10, mode="shadow",
)
print(f"Experiment: {experiment}")

logger = QuoteLogger(str(log_path))

# COMMAND ----------

# Simulate quoting 1,000 renewal policies
rng_quotes = np.random.default_rng(99)
n_quotes = 1_000
start_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
bound_count = 0
bound_policy_ids = []

for i in range(n_quotes):
    policy_id = f"POL-{i:05d}"
    arm = experiment.route(policy_id)
    live_mv = experiment.live_model(policy_id)

    row_idx = i % len(holdout_pd)
    risk_row = holdout_pd[feature_cols].iloc[[row_idx]]
    exposure = float(holdout_pd["exposure"].iloc[row_idx])

    champion_rate = float(model_v1.predict(risk_row)[0])
    challenger_rate = float(model_v2.predict(risk_row)[0])

    avg_severity = 4_500.0
    champion_premium = max(150.0, champion_rate * exposure * avg_severity)
    challenger_premium = max(150.0, challenger_rate * exposure * avg_severity)
    enbp = max(150.0, challenger_rate * exposure * avg_severity * 1.02)

    quote_ts = start_ts + timedelta(days=float(rng_quotes.uniform(0, 180)))

    logger.log_quote(
        policy_id=policy_id, experiment_name=experiment.name,
        arm=arm, model_version=live_mv.version_id,
        quoted_price=champion_premium, enbp=enbp,
        renewal_flag=True, exposure=exposure, timestamp=quote_ts,
    )

    if rng_quotes.random() < 0.35:
        bound_price = champion_premium * rng_quotes.uniform(0.97, 1.03)
        logger.log_bind(policy_id, bound_price=bound_price)
        bound_policy_ids.append(policy_id)
        bound_count += 1

        if rng_quotes.random() < 0.08:
            claim_amount = float(rng_quotes.gamma(shape=2.0, scale=2_500.0))
            logger.log_claim(
                policy_id=policy_id, claim_date=date(2025, 3, 15),
                claim_amount=claim_amount, development_month=12,
            )

print(f"Quoted: {n_quotes:,} | Bound: {bound_count:,} ({bound_count/n_quotes:.1%} hit rate)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6: KPI Tracking and ENBP Audit Report
# MAGIC
# MAGIC The FCA's ICOBS 6B multi-firm review (2023) found 83% of firms could not demonstrate which model priced each renewal. This framework is the answer — every quote is logged with model version, arm, price, and ENBP.

# COMMAND ----------

tracker = KPITracker(logger)

vol = tracker.quote_volume(experiment.name)
print("Quote volume by arm:")
for arm, stats in vol.items():
    print(f"  {arm.capitalize()}: {stats['n']:,} quotes, mean price £{stats['mean_price']:.2f}")

hr = tracker.hit_rate(experiment.name)
print("\nHit rate by arm:")
for arm, stats in hr.items():
    print(f"  {arm.capitalize()}: {stats['bound']:,} / {stats['quoted']:,} = {stats['hit_rate']:.1%}")

enbp_stats = tracker.enbp_compliance(experiment.name)
print("\nENBP compliance (renewal quotes):")
for arm, stats in enbp_stats.items():
    rate = stats['compliance_rate']
    rate_str = f"{rate:.1%}" if not np.isnan(rate) else "N/A"
    print(f"  {arm.capitalize()}: {stats['compliant']:,} compliant, {stats['breaches']:,} breaches — {rate_str}")

reporter = ENBPAuditReport(logger)
audit_md = reporter.generate(
    experiment_name=experiment.name,
    period_start="2025-01-01", period_end="2025-06-30",
    firm_name="Example Insurer Ltd", smf_holder="Jane Smith, Chief Actuary",
)
print("\n--- ENBP Audit Report (first 20 lines) ---")
for line in audit_md.split("\n")[:20]:
    print(line)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7: Statistical Comparison
# MAGIC
# MAGIC `ModelComparison.bootstrap_lr_test()` requires 12+ months of developed claims. With 6 months of simulated data, it will correctly return INSUFFICIENT_EVIDENCE. This is the expected outcome at this stage — not a failure. Re-run after 12-18 months with full claims development.

# COMMAND ----------

comparison = ModelComparison(tracker)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr_result = comparison.bootstrap_lr_test(
        experiment_name=experiment.name,
        development_months=12,
        n_bootstrap=1_000,  # use 10,000 in production
        seed=42,
    )

print("Loss ratio comparison (bootstrap test):")
print(lr_result.summary())
print("\nNote: INSUFFICIENT_EVIDENCE is the correct result at 6 months.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Libraries used:**
# MAGIC - `insurance-synthetic` — synthetic UK motor portfolio (20,000 policies)
# MAGIC - `catboost` — Poisson GBM frequency model (champion v1, challenger v2)
# MAGIC - `shap-relativities` — multiplicative rating factor relativities
# MAGIC - `insurance-governance` — PRA SS1/23 model validation
# MAGIC - `insurance-deploy` — model registry, shadow experiment, ENBP audit
# MAGIC
# MAGIC **Next steps in a real deployment:**
# MAGIC 1. Replace synthetic data with your actual portfolio extract
# MAGIC 2. Tune hyperparameters and feature set on your book
# MAGIC 3. Run the experiment for 12-18 months in shadow mode
# MAGIC 4. Re-run the bootstrap LR test once claims are developed
# MAGIC 5. Call `registry.set_champion("motor_frequency", "2.0")` once approved
