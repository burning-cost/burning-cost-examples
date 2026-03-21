# Databricks notebook source

# MAGIC %md
# MAGIC # FCA Model Governance Pipeline
# MAGIC ## CatBoost Frequency Model → Fairness Audit → Conformal Intervals → Drift Monitoring
# MAGIC
# MAGIC This notebook demonstrates the end-to-end governance workflow that a UK pricing team
# MAGIC needs to run under current FCA requirements. The four stages map directly to the
# MAGIC regulatory obligations:
# MAGIC
# MAGIC **Stage 1 — Model training** sets the baseline. A CatBoost Poisson frequency model
# MAGIC on UK motor data.
# MAGIC
# MAGIC **Stage 2 — Fairness audit** (FCA Consumer Duty PRIN 2A, FCA TR24/2, EP25/2).
# MAGIC Consumer Duty requires firms to *demonstrate* that model outcomes do not disadvantage
# MAGIC customers with protected characteristics. "We don't use gender as a rating factor" is
# MAGIC not sufficient — the FCA's concern is proxy discrimination, where correlated rating
# MAGIC factors transmit a protected characteristic's effect into the price. This stage uses
# MAGIC `insurance-fairness` to run a full proxy detection and bias metric audit.
# MAGIC
# MAGIC **Stage 3 — Conformal prediction intervals** (PRA SS1/23 model uncertainty).
# MAGIC SS1/23 requires quantification of model uncertainty and clear escalation criteria when
# MAGIC uncertainty is high. Distribution-free conformal intervals from `insurance-conformal`
# MAGIC provide coverage guarantees without parametric assumptions — something bootstrap
# MAGIC intervals cannot offer when the residual distribution is non-standard (as it always is
# MAGIC for Tweedie data with heavy-tailed high-risk segments).
# MAGIC
# MAGIC **Stage 4 — Drift monitoring** (PRA SS1/23 ongoing monitoring requirements).
# MAGIC SS1/23 requires documented monitoring at defined frequencies, with clear criteria
# MAGIC for escalation (recalibrate vs refit vs no action). `insurance-monitoring` implements
# MAGIC the decision framework from arXiv 2510.04556: PSI/CSI for feature drift, A/E ratio
# MAGIC for calibration drift, Gini z-test for discrimination power drift.
# MAGIC
# MAGIC ---
# MAGIC **Regulatory references:**
# MAGIC - FCA Consumer Duty Finalised Guidance FG22/5 (2023) and PRIN 2A
# MAGIC - FCA Thematic Review TR24/2 (2024): Pricing Practices and Data Analytics
# MAGIC - FCA Evaluation Paper EP25/2 (2025): Consumer Duty — Fair Value Outcomes
# MAGIC - PRA Supervisory Statement SS1/23: Model Risk Management Principles
# MAGIC - Equality Act 2010, Section 19: Indirect Discrimination

# COMMAND ----------

# MAGIC %pip install "insurance-fairness>=0.5.0" "insurance-conformal>=0.5.1" "insurance-monitoring>=0.6.0" catboost polars numpy scipy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: CatBoost Frequency Model
# MAGIC
# MAGIC We train a Poisson frequency model on synthetic UK motor data. The data-generating
# MAGIC process is calibrated to UK private motor loss ratios:
# MAGIC
# MAGIC - Base frequency ~0.12 claims per vehicle-year (broadly consistent with Thatcham data)
# MAGIC - Young drivers (<25) and high-powered vehicles have materially higher frequency
# MAGIC - NCD years reduce frequency — this is real in the data, not just a pricing convention
# MAGIC - Gender is **not** in the model (excluded per FCA guidance on protected attributes)
# MAGIC   but gender correlates with vehicle group and annual mileage — which IS in the model.
# MAGIC   This is the proxy risk the fairness audit will surface.
# MAGIC
# MAGIC We generate ~50,000 policies to give the model enough data to learn meaningful
# MAGIC relativities. The train/cal/test split is temporal by convention:
# MAGIC - 60% train (years 1-3 of a hypothetical 5-year window)
# MAGIC - 20% calibrate (year 4 — held out from model fitting, used for conformal calibration)
# MAGIC - 20% test  (year 5 — unseen, used for evaluation and monitoring baseline)

# COMMAND ----------

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

# Compatibility patch: np.trapezoid added in NumPy 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ── Data-generating process ───────────────────────────────────────────────────

rng = np.random.default_rng(seed=2024)

N = 50_000
TRAIN_FRAC = 0.60
CAL_FRAC   = 0.20
TEST_FRAC  = 0.20

# Features
driver_age      = rng.integers(18, 75, size=N).astype(float)
ncd_years       = rng.integers(0, 10, size=N).astype(float)    # no-claims discount years
vehicle_group   = rng.integers(1, 16, size=N).astype(float)    # ABI vehicle group 1-15
region          = rng.choice([1, 2, 3, 4, 5], size=N,          # 1=London, 2=SE, 3=Midlands,
                              p=[0.18, 0.20, 0.22, 0.20, 0.20]) # 4=North, 5=Scotland
annual_mileage  = rng.lognormal(mean=9.2, sigma=0.5, size=N)   # log-normal, typical UK distribution
vehicle_age     = rng.integers(0, 15, size=N).astype(float)
exposure        = rng.uniform(0.3, 1.0, size=N)                 # earned vehicle-years

# Gender: NOT in the model, but correlated with vehicle_group and annual_mileage
# This creates the proxy discrimination risk the fairness audit will detect.
# Male drivers (1) skew toward higher vehicle groups and higher mileage.
male_prob = np.clip(0.5 + 0.03 * (vehicle_group - 8) + 0.15 * (annual_mileage > 14000), 0.2, 0.85)
gender    = rng.binomial(1, male_prob).astype(float)  # 1=male, 0=female

# True frequency model (Poisson rate per vehicle-year)
age_factor      = np.where(driver_age < 25,  2.1,
                   np.where(driver_age < 30,  1.4,
                   np.where(driver_age < 60,  1.0,
                   np.where(driver_age < 70,  1.1, 1.4))))
ncd_factor      = np.exp(-0.14 * ncd_years)
vg_factor       = 1.0 + 0.06 * (vehicle_group - 8)
region_factor   = np.array([1.35, 1.15, 1.00, 0.90, 0.85])[region - 1]
mileage_factor  = (annual_mileage / 10_000) ** 0.30
base_freq       = 0.10

true_freq = base_freq * age_factor * ncd_factor * vg_factor * region_factor * mileage_factor

# Observed claim count (Poisson, scaled by exposure)
claim_count = rng.poisson(true_freq * exposure)

# Assemble feature matrix (gender excluded from model features)
feature_cols = ["driver_age", "ncd_years", "vehicle_group", "region",
                "annual_mileage", "vehicle_age"]

X_all = np.column_stack([
    driver_age, ncd_years, vehicle_group, region.astype(float),
    annual_mileage, vehicle_age,
])

# Temporal split: indices are ordered, split by position
n_train = int(N * TRAIN_FRAC)
n_cal   = int(N * CAL_FRAC)
n_test  = N - n_train - n_cal

idx_train = slice(0, n_train)
idx_cal   = slice(n_train, n_train + n_cal)
idx_test  = slice(n_train + n_cal, N)

X_train, y_train, exp_train = X_all[idx_train], claim_count[idx_train], exposure[idx_train]
X_cal,   y_cal,   exp_cal   = X_all[idx_cal],   claim_count[idx_cal],   exposure[idx_cal]
X_test,  y_test,  exp_test  = X_all[idx_test],  claim_count[idx_test],  exposure[idx_test]

print(f"Train: {n_train:,} policies | Cal: {n_cal:,} | Test: {n_test:,}")
print(f"Train frequency: {y_train.sum() / exp_train.sum():.4f} claims/vehicle-year")
print(f"Cal   frequency: {y_cal.sum()   / exp_cal.sum():.4f} claims/vehicle-year")
print(f"Test  frequency: {y_test.sum()  / exp_test.sum():.4f} claims/vehicle-year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit CatBoost Poisson Model
# MAGIC
# MAGIC Poisson regression with a log link — the industry standard for frequency modelling.
# MAGIC The exposure offset (log(exposure)) is passed as a baseline to ensure the model
# MAGIC predicts *rates* (claims per vehicle-year), not raw counts. This matters for the
# MAGIC A/E monitoring later: if you forget the exposure offset, the A/E ratio will fluctuate
# MAGIC with the average exposure in each monitoring window, which is noise.

# COMMAND ----------

from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
)

# Baseline = log(exposure): tells CatBoost to model log(E[y]) = log(exposure) + f(X)
model.fit(
    X_train, y_train,
    baseline=np.log(exp_train),
    eval_set=(X_cal, y_cal),
    eval_baseline=np.log(exp_cal),
)

# Point predictions on each split (raw claim count, not rate)
pred_train = model.predict(X_train, ntree_end=0)  # use all trees
pred_cal   = model.predict(X_cal)
pred_test  = model.predict(X_test)

# Rates (divide by exposure)
rate_train = pred_train / exp_train
rate_test  = pred_test  / exp_test

print(f"Model fitted. Trees: {model.tree_count_}")
print(f"Train A/E: {y_train.sum() / pred_train.sum():.4f}")
print(f"Cal   A/E: {y_cal.sum()   / pred_cal.sum():.4f}")
print(f"Test  A/E: {y_test.sum()  / pred_test.sum():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Fairness Audit (FCA Consumer Duty / TR24/2)
# MAGIC
# MAGIC The FCA's concern under Consumer Duty is differential outcomes by protected
# MAGIC characteristic. The audit has two parts:
# MAGIC
# MAGIC **Proxy detection**: does any rating factor in the model act as a proxy for gender?
# MAGIC We test this by fitting a secondary model that predicts gender from the rating
# MAGIC factors alone (proxy R-squared) and computing the mutual information between
# MAGIC each factor and gender. Factors above the amber threshold (proxy R² > 0.05) need
# MAGIC to be flagged in the governance record and considered for adjustment.
# MAGIC
# MAGIC **Bias metrics**: given the model's predictions, are there differential outcomes
# MAGIC by gender group? The FCA framework focuses on:
# MAGIC - *Demographic parity*: is the average predicted price materially different across groups?
# MAGIC - *Calibration by group*: within each prediction decile, is the A/E ratio similar
# MAGIC   across groups? A model can have the right overall A/E but systematically over- or
# MAGIC   under-charge one group at high or low risk levels.
# MAGIC - *Disparate impact ratio*: the ratio of average predictions across groups — the "80%
# MAGIC   rule" threshold from equal opportunity law.
# MAGIC
# MAGIC We build the audit dataset with gender included as a column (it's a protected attribute
# MAGIC we're auditing against, not a rating factor). The model is excluded from the feature
# MAGIC columns — the audit operates on the model's outputs, not its inputs.

# COMMAND ----------

from insurance_fairness import FairnessAudit

# Build audit dataset with predictions already added
# FairnessAudit operates on policy-level data; it needs predictions already computed.
# We audit on the test set — the held-out period that represents the live book.

audit_df = pl.DataFrame({
    "driver_age":     driver_age[idx_test],
    "ncd_years":      ncd_years[idx_test],
    "vehicle_group":  vehicle_group[idx_test],
    "region":         region[idx_test].astype(float),
    "annual_mileage": annual_mileage[idx_test],
    "vehicle_age":    vehicle_age[idx_test],
    "gender":         gender[idx_test],          # protected attribute
    "predicted_freq": pred_test / exp_test,      # model output: claim rate
    "claim_count":    y_test.astype(float),      # observed outcome
    "exposure":       exp_test,
})

audit = FairnessAudit(
    model=model,                       # pass model for proxy R² computation
    data=audit_df,
    protected_cols=["gender"],
    prediction_col="predicted_freq",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=["driver_age", "ncd_years", "vehicle_group",
                 "region", "annual_mileage", "vehicle_age"],
    model_name="Motor Frequency v1 — FCA Governance Audit",
    run_proxy_detection=True,
    run_counterfactual=False,          # counterfactual requires gender in model features
    proxy_catboost_iterations=150,
    n_bootstrap=0,                     # set >0 for CIs on parity metrics (slower)
)

print("Running fairness audit — proxy detection + bias metrics...")
report = audit.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Audit Results
# MAGIC
# MAGIC The overall RAG status and the flagged proxy factors are the governance deliverable.
# MAGIC An AMBER status means the factor warrants monitoring and explanation; RED means
# MAGIC the factor must be reviewed with compliance before the next pricing cycle.

# COMMAND ----------

report.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proxy Detection Detail
# MAGIC
# MAGIC For each rating factor, proxy R-squared measures how much of the variance in gender
# MAGIC can be explained by that factor alone. Mutual information captures non-linear
# MAGIC relationships. Both are exposure-weighted.
# MAGIC
# MAGIC The regulatorily material question is not "is there any correlation" — some correlation
# MAGIC is inevitable in any realistic rating plan — but "is the correlation material enough
# MAGIC that the factor is acting primarily as a gender proxy rather than a genuine risk predictor?"
# MAGIC The amber threshold at proxy R² > 0.05 is conservative but defensible for a TR24/2
# MAGIC submission. Factors above it need documented justification in the pricing governance pack.

# COMMAND ----------

pc_report = report.results["gender"]

if pc_report.proxy_detection is not None:
    pd_result = pc_report.proxy_detection
    rows = []
    for score in pd_result.scores:
        rows.append({
            "factor":              score.factor,
            "proxy_r2":            round(score.proxy_r2, 4),
            "mutual_information":  round(score.mutual_information, 4),
            "rag":                 score.rag,
        })
    proxy_df = pl.DataFrame(rows).sort("proxy_r2", descending=True)
    print("Proxy detection results (sorted by proxy R²):")
    print(proxy_df)
    print(f"\nFlagged factors: {pd_result.flagged_factors}")
else:
    print("Proxy detection not available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bias Metrics Summary
# MAGIC
# MAGIC Disparate impact ratio below 0.8 (or above 1.25 in either direction) is the standard
# MAGIC threshold from equal opportunity frameworks. The FCA has not published an explicit
# MAGIC numeric threshold for pricing models — TR24/2 requires firms to explain their
# MAGIC methodology and thresholds, not to hit a specific number. Document the choice.

# COMMAND ----------

if pc_report.demographic_parity is not None:
    dp = pc_report.demographic_parity
    print(f"Demographic parity log-ratio: {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f}) [{dp.rag.upper()}]")
    print(f"  Group means (0=female, 1=male): {dp.group_means}")

if pc_report.disparate_impact is not None:
    di = pc_report.disparate_impact
    print(f"Disparate impact ratio:         {di.ratio:.4f} [{di.rag.upper()}]")
    print(f"  Group means: {di.group_means}")

if pc_report.calibration is not None:
    cal = pc_report.calibration
    print(f"Max calibration disparity:      {cal.max_disparity:.4f} [{cal.rag.upper()}]")

print(f"\nOverall audit status: {report.overall_rag.upper()}")
if report.flagged_factors:
    print(f"Factors requiring governance review: {report.flagged_factors}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Conformal Prediction Intervals (PRA SS1/23)
# MAGIC
# MAGIC SS1/23 requires firms to quantify model uncertainty and demonstrate that escalation
# MAGIC criteria are triggered when uncertainty is material. Point predictions alone do not
# MAGIC satisfy this — a predicted frequency of 0.08 claims/year could be anywhere from
# MAGIC 0.04 to 0.18 for a young driver segment where the model has limited data.
# MAGIC
# MAGIC The correct approach for Poisson/Tweedie data is the `pearson_weighted`
# MAGIC non-conformity score: `|y - ŷ| / ŷ^(p/2)`. This accounts for the variance-mean
# MAGIC relationship in Poisson/Tweedie distributions — the variance grows with the mean,
# MAGIC so a residual of 0.5 at a high-risk prediction is less extreme than the same residual
# MAGIC at a low-risk prediction. Using the raw residual (or even the unweighted Pearson
# MAGIC residual) gives intervals that are systematically too narrow for high-risk policies.
# MAGIC
# MAGIC The calibration set here is the year-4 hold-out. The guarantee is marginal:
# MAGIC P(y in [lower, upper]) >= 1 - alpha, holding over the test-set population.
# MAGIC It is not a per-policy guarantee, and it is not conditional (see coverage_by_decile
# MAGIC output below — conditional coverage requires additional tools).

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor, CoverageDiagnostics

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",   # correct score for Poisson/Tweedie data
    distribution="tweedie",
    tweedie_power=1.0,                  # Poisson: p=1
)

# Calibrate on the year-4 hold-out (independent of both training and test sets)
cp.calibrate(X_cal, y_cal, exposure=exp_cal)

print(repr(cp))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 90% Prediction Intervals on the Test Set
# MAGIC
# MAGIC We produce intervals at alpha=0.10 (90% coverage target). The coverage check
# MAGIC by decile is the key governance diagnostic — if coverage is substantially below
# MAGIC target in the high-risk deciles, the model is under-representing uncertainty
# MAGIC precisely where it matters most for reserving and pricing.

# COMMAND ----------

ALPHA = 0.10

# Get intervals for the test set
intervals = cp.predict_interval(X_test, alpha=ALPHA)

# Empirical coverage
y_arr  = y_test.astype(float)
lower  = intervals["lower"].to_numpy()
upper  = intervals["upper"].to_numpy()
covered = (y_arr >= lower) & (y_arr <= upper)

print(f"Target coverage: {1 - ALPHA:.0%}")
print(f"Marginal coverage (test set): {covered.mean():.3%}")
print(f"Mean interval width: {(upper - lower).mean():.4f} claims/policy")
print(f"Median interval width: {float(pl.Series(upper - lower).median()):.4f}")
print()

# Show first few intervals
sample = intervals.head(10).with_columns([
    pl.Series("observed", y_test[:10].astype(float)),
    pl.Series("covered", covered[:10]),
])
print("Sample intervals (first 10 test policies):")
print(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage by Decile
# MAGIC
# MAGIC This is the governance diagnostic that matters. A model with good marginal coverage
# MAGIC but poor decile-level coverage is not fit for SS1/23 purposes — it is systematically
# MAGIC mis-sizing uncertainty for specific risk segments. The `pearson_weighted` score
# MAGIC should give substantially more uniform decile coverage than the raw residual score.

# COMMAND ----------

decile_cov = cp.coverage_by_decile(X_test, y_test, alpha=ALPHA, n_bins=10)

print(f"Coverage by risk decile (target: {1 - ALPHA:.0%}):")
print(f"{'Decile':>7} {'Mean Pred':>10} {'N Obs':>7} {'Coverage':>10} {'Flag':>5}")
print("-" * 42)
for row in decile_cov.iter_rows(named=True):
    deviation = abs(row["coverage"] - (1 - ALPHA))
    flag      = "  ***" if deviation > 0.05 else ""
    print(
        f"{row['decile']:>7} {row['mean_predicted']:>10.4f} "
        f"{row['n_obs']:>7} {row['coverage']:>10.1%}{flag}"
    )

decile_arr = decile_cov["coverage"].to_numpy()
print(f"\nMax decile deviation from target: {abs(decile_arr - (1 - ALPHA)).max():.1%}")
print("(*** = more than 5pp from target — review non-conformity score choice)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Portfolio Uncertainty Summary
# MAGIC
# MAGIC The conformal intervals can be aggregated to give the pricing committee a sense of
# MAGIC model-level uncertainty, not just policy-level uncertainty. The total claim count
# MAGIC upper bound (sum of per-policy upper bounds) provides a conservative estimate of
# MAGIC portfolio loss consistent with the 90% coverage level.

# COMMAND ----------

total_predicted = float(intervals["point"].sum())
total_lower     = float(intervals["lower"].sum())
total_upper     = float(intervals["upper"].sum())
total_observed  = float(y_test.sum())

print("Portfolio-level uncertainty summary (test period):")
print(f"  Observed claims:              {total_observed:,.0f}")
print(f"  Predicted claims (point):     {total_predicted:,.1f}")
print(f"  90% interval lower:           {total_lower:,.1f}")
print(f"  90% interval upper:           {total_upper:,.1f}")
print(f"  Interval width as % of point: {(total_upper - total_lower) / total_predicted:.1%}")
print()
print("SS1/23 interpretation:")
in_interval = total_lower <= total_observed <= total_upper
print(f"  Observed within 90% portfolio interval: {in_interval}")
print("  This interval is not a capital requirement — it is a model uncertainty")
print("  disclosure for the internal model validation report.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Drift Monitoring (PRA SS1/23 Ongoing Monitoring)
# MAGIC
# MAGIC SS1/23 requires documented monitoring at defined intervals (quarterly at minimum for
# MAGIC material models). The monitoring framework from arXiv 2510.04556 implements a
# MAGIC three-stage decision tree:
# MAGIC
# MAGIC 1. **Gini stable, A/E stable** — no action required
# MAGIC 2. **A/E drifted, Gini stable** — recalibrate the intercept (cheap, low risk)
# MAGIC 3. **Gini degraded** — refit the model (material change in risk ranking)
# MAGIC
# MAGIC The Murphy decomposition (optional) sharpens this: it decomposes the loss function
# MAGIC into discrimination (DSC) and miscalibration (MCB) components. If MCB is high but
# MAGIC DSC is stable, the evidence points to RECALIBRATE even before the Gini z-test
# MAGIC crosses red. This is the statistically correct approach — Gini z-tests at n=10,000
# MAGIC have limited power against small degradations.
# MAGIC
# MAGIC For this demonstration, we simulate a monitoring period where frequency has increased
# MAGIC by 12% (e.g. economic conditions, new risk types entering the book). The model was
# MAGIC trained on the earlier distribution and will show A/E drift.

# COMMAND ----------

from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi, csi

# Simulate a monitoring period: 6 months of live data after deployment.
# The "true" frequency has increased by 12% — consistent with claims inflation
# or a change in the book mix not captured in training data.
N_MON = 8_000
FREQ_UPLIFT = 1.12   # 12% frequency increase in the live period

rng_mon = np.random.default_rng(seed=2025)

driver_age_mon     = rng_mon.integers(18, 75, size=N_MON).astype(float)
ncd_years_mon      = rng_mon.integers(0, 10, size=N_MON).astype(float)
vehicle_group_mon  = rng_mon.integers(1, 16, size=N_MON).astype(float)
region_mon         = rng_mon.choice([1, 2, 3, 4, 5], size=N_MON,
                                     p=[0.20, 0.22, 0.22, 0.18, 0.18])  # slight mix shift: London up
annual_mileage_mon = rng_mon.lognormal(mean=9.25, sigma=0.52, size=N_MON)  # slight mileage increase
vehicle_age_mon    = rng_mon.integers(0, 15, size=N_MON).astype(float)
exposure_mon       = rng_mon.uniform(0.3, 1.0, size=N_MON)

age_factor_mon     = np.where(driver_age_mon < 25,  2.1,
                      np.where(driver_age_mon < 30,  1.4,
                      np.where(driver_age_mon < 60,  1.0,
                      np.where(driver_age_mon < 70,  1.1, 1.4))))
ncd_factor_mon     = np.exp(-0.14 * ncd_years_mon)
vg_factor_mon      = 1.0 + 0.06 * (vehicle_group_mon - 8)
region_factor_mon  = np.array([1.35, 1.15, 1.00, 0.90, 0.85])[region_mon - 1]
mileage_factor_mon = (annual_mileage_mon / 10_000) ** 0.30

true_freq_mon = (base_freq * FREQ_UPLIFT * age_factor_mon * ncd_factor_mon *
                 vg_factor_mon * region_factor_mon * mileage_factor_mon)

claim_count_mon = rng_mon.poisson(true_freq_mon * exposure_mon)

X_mon = np.column_stack([
    driver_age_mon, ncd_years_mon, vehicle_group_mon, region_mon.astype(float),
    annual_mileage_mon, vehicle_age_mon,
])
pred_mon = model.predict(X_mon)  # stale model — trained before the uplift

print(f"Monitoring period: {N_MON:,} policies")
print(f"Reference A/E (training): {y_train.sum() / pred_train.sum():.4f}")
print(f"Monitor  A/E (live):      {claim_count_mon.sum() / pred_mon.sum():.4f}")
print(f"Monitor frequency: {claim_count_mon.sum() / exposure_mon.sum():.4f} "
      f"(injected uplift: {FREQ_UPLIFT:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Drift: PSI and CSI
# MAGIC
# MAGIC PSI (Population Stability Index) measures overall distribution shift in the model
# MAGIC score. CSI (Characteristic Stability Index) does the same per input feature — it
# MAGIC tells you *which* factor is driving the drift.
# MAGIC
# MAGIC Thresholds: PSI < 0.10 = green, 0.10–0.25 = amber, > 0.25 = red. These are
# MAGIC industry standard heuristics from credit risk monitoring practice.
# MAGIC
# MAGIC When CSI is high on a specific feature, the next step is to check whether the
# MAGIC model's relative sensitivity to that feature has changed — a job for DriftAttributor
# MAGIC (TRIPODD). We omit that here for brevity but it's available in insurance-monitoring.

# COMMAND ----------

# Score PSI: monitor distribution of log-rate vs reference distribution
log_rate_ref = np.log(pred_train / exp_train + 1e-8)  # reference: training predictions
log_rate_mon = np.log(pred_mon / exposure_mon + 1e-8)  # monitor: live predictions

score_psi = psi(log_rate_ref, log_rate_mon, n_bins=10)
psi_band  = "green" if score_psi < 0.10 else ("amber" if score_psi < 0.25 else "red")
print(f"Score PSI (log-rate distribution): {score_psi:.4f} [{psi_band.upper()}]")
print()

# CSI per feature
ref_features_df = pl.DataFrame({
    col: X_train[:, i] for i, col in enumerate(feature_cols)
})
mon_features_df = pl.DataFrame({
    col: X_mon[:, i] for i, col in enumerate(feature_cols)
})

csi_result = csi(ref_features_df, mon_features_df, feature_cols)
print("Feature CSI (Characteristic Stability Index):")
print(csi_result.sort("csi", descending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### MonitoringReport: Full Traffic-Light Summary
# MAGIC
# MAGIC The MonitoringReport combines A/E ratio, Gini drift test, score PSI, and CSI into
# MAGIC a single governance output. The recommendation is the action item for the model
# MAGIC risk committee: NO_ACTION, MONITOR_CLOSELY, RECALIBRATE, REFIT, or INVESTIGATE.
# MAGIC
# MAGIC With Murphy decomposition enabled, the recommendation engine looks at whether the
# MAGIC drift is in the discrimination component (DSC — model ranking has degraded, REFIT)
# MAGIC or the miscalibration component (MCB — model mean is wrong but ranking is intact,
# MAGIC RECALIBRATE). This is the correct statistical distinction for SS1/23 materiality
# MAGIC assessment.

# COMMAND ----------

monitoring = MonitoringReport(
    reference_actual=y_train.astype(float),
    reference_predicted=pred_train,
    current_actual=claim_count_mon.astype(float),
    current_predicted=pred_mon,
    exposure=exposure_mon,
    reference_exposure=exp_train,
    feature_df_reference=ref_features_df,
    feature_df_current=mon_features_df,
    features=feature_cols,
    score_reference=log_rate_ref,
    score_current=log_rate_mon,
    murphy_distribution="poisson",   # Murphy decomposition: RECALIBRATE vs REFIT
    gini_bootstrap=False,            # set True for CI bands on Gini (slower)
    n_bootstrap=200,
)

print(f"Monitoring recommendation: {monitoring.recommendation}")
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring Results Table
# MAGIC
# MAGIC The to_polars() output is what gets written to a Delta monitoring table in
# MAGIC production. One row per metric, one column for value, one for traffic-light band.
# MAGIC Set up a Delta Live Table or Databricks workflow to run this every quarter and
# MAGIC alert the model risk committee when any metric hits amber.

# COMMAND ----------

monitoring_df = monitoring.to_polars()
print("Full monitoring report:")
print(monitoring_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring Narrative
# MAGIC
# MAGIC Translate the traffic-light output into the governance narrative the model risk
# MAGIC committee needs to record in the model log.

# COMMAND ----------

results = monitoring.results_

ae_val  = results["ae_ratio"]["value"]
ae_band = results["ae_ratio"]["band"]

gini_ref = results["gini"]["reference"]
gini_cur = results["gini"]["current"]
gini_chg = results["gini"]["change"]
gini_p   = results["gini"]["p_value"]
gini_band= results["gini"]["band"]

print("=" * 60)
print("MODEL MONITORING REPORT — GOVERNANCE NARRATIVE")
print("=" * 60)
print()
print(f"A/E Ratio: {ae_val:.4f} [{ae_band.upper()}]")
print(f"  95% CI: [{results['ae_ratio']['lower_ci']:.4f}, {results['ae_ratio']['upper_ci']:.4f}]")
print(f"  Observed claims: {results['ae_ratio']['n_claims']:.1f}, "
      f"Expected: {results['ae_ratio']['n_expected']:.1f}")
print()
print(f"Gini Coefficient: {gini_cur:.4f} (reference: {gini_ref:.4f}) [{gini_band.upper()}]")
print(f"  Change: {gini_chg:+.4f}, p-value: {gini_p:.4f}")
print()

if monitoring.murphy_available:
    m = results["murphy"]
    print(f"Murphy Decomposition [{m['verdict']}]:")
    print(f"  Discrimination (DSC):  {m['discrimination']:.4f} ({m['discrimination_pct']:.1%} of total deviance)")
    print(f"  Miscalibration (MCB):  {m['miscalibration']:.4f} ({m['miscalibration_pct']:.1%} of total deviance)")
    print(f"  Global MCB:            {m['global_mcb']:.4f}")
    print(f"  Local MCB:             {m['local_mcb']:.4f}")
    print()

if "max_csi" in results:
    mc = results["max_csi"]
    print(f"Worst feature CSI: {mc['value']:.4f} on '{mc['worst_feature']}' [{mc['band'].upper()}]")
    print()

if "score_psi" in results:
    print(f"Score PSI: {results['score_psi']['value']:.4f} [{results['score_psi']['band'].upper()}]")
    print()

print(f"RECOMMENDATION: {monitoring.recommendation}")
print()
print("Action required:")
rec = monitoring.recommendation
if rec == "NO_ACTION":
    print("  Model performance is within tolerance. Continue monitoring at scheduled frequency.")
elif rec == "RECALIBRATE":
    print("  A/E ratio has drifted but Gini is stable. The model is ranking risks correctly")
    print("  but the overall level is wrong. Recalibrate the intercept/base rate.")
    print("  Do not retrain — this is a pricing level adjustment, not a model rebuild.")
elif rec == "REFIT":
    print("  Gini has degraded. The model's ability to rank risks has deteriorated.")
    print("  Escalate to model owner. Schedule a refit on recent data within 60 days.")
    print("  Consider whether to recalibrate the intercept as a temporary measure.")
elif rec == "INVESTIGATE":
    print("  Multiple conflicting signals. Convene a model risk committee review.")
    print("  Do not recalibrate or refit until the root cause is understood.")
elif rec == "MONITOR_CLOSELY":
    print("  Amber signals present but no red thresholds crossed. Increase monitoring frequency.")
    print("  Schedule an off-cycle review in 30 days.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Four-Library Governance Pipeline
# MAGIC
# MAGIC This notebook has demonstrated a complete model governance pipeline aligned with UK
# MAGIC regulatory requirements:
# MAGIC
# MAGIC | Stage | Library | Regulatory Requirement | Output |
# MAGIC |-------|---------|------------------------|--------|
# MAGIC | Model training | CatBoost | Baseline model documentation | Fitted Poisson frequency model |
# MAGIC | Fairness audit | insurance-fairness | FCA Consumer Duty PRIN 2A, TR24/2, EP25/2 | RAG status, proxy detection, bias metrics |
# MAGIC | Conformal intervals | insurance-conformal | PRA SS1/23 model uncertainty quantification | 90% coverage-guaranteed prediction intervals |
# MAGIC | Drift monitoring | insurance-monitoring | PRA SS1/23 ongoing monitoring | A/E, Gini, PSI/CSI, Murphy decomposition, governance recommendation |
# MAGIC
# MAGIC ### What to operationalise
# MAGIC
# MAGIC In a production environment, this pipeline should run as a Databricks workflow:
# MAGIC
# MAGIC - **Monthly**: MonitoringReport on the live book (quick, uses cached model predictions)
# MAGIC - **Quarterly**: Full FairnessAudit + conformal interval review + MonitoringReport
# MAGIC - **On amber/red trigger**: Convene model risk committee within 10 business days
# MAGIC - **Annually at minimum**: Full model documentation refresh for the model inventory
# MAGIC
# MAGIC The outputs from each run should be written to a Delta governance table with
# MAGIC `model_version`, `run_date`, `metric`, `value`, `band` columns. This gives you
# MAGIC a time series of model health that satisfies the SS1/23 audit trail requirement.
