# Databricks notebook source

# MAGIC %md
# MAGIC # Burning Cost in 30 Minutes
# MAGIC
# MAGIC This notebook runs a complete mini-workflow using three libraries from the Burning Cost ecosystem:
# MAGIC
# MAGIC | Library | What it does |
# MAGIC |---|---|
# MAGIC | `insurance-causal` | Double Machine Learning to estimate causal effects on observational data |
# MAGIC | `insurance-conformal` | Distribution-free prediction intervals with coverage guarantees |
# MAGIC | `insurance-monitoring` | PSI/CSI drift detection, A/E monitoring, Gini discrimination tracking |
# MAGIC
# MAGIC **No data required.** Everything runs on a 5,000-row synthetic UK motor dataset generated below.
# MAGIC
# MAGIC ### The scenario
# MAGIC
# MAGIC You are a pricing actuary at a UK motor insurer. You have:
# MAGIC
# MAGIC - A fitted CatBoost frequency model (Poisson)
# MAGIC - A question: *does opting into telematics actually reduce claim frequency, or do safer drivers self-select into telematics?*
# MAGIC - A need for uncertainty quantification around frequency predictions
# MAGIC - A monitoring pack due at quarter-end
# MAGIC
# MAGIC All three problems are addressed below.

# COMMAND ----------

# MAGIC %pip install insurance-causal insurance-conformal insurance-monitoring catboost --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate Synthetic UK Motor Data
# MAGIC
# MAGIC 5,000 policies with realistic UK motor features. The data generating process has a deliberate confounding problem: drivers who opt into telematics also tend to be younger and lower-mileage. Naive regression will underestimate the true causal effect of telematics because those risk factors are correlated with both the treatment (opt-in) and the outcome (claim frequency).
# MAGIC
# MAGIC We split into a **training period** (first 3,500 rows) and a **monitoring period** (last 1,500 rows, with simulated distribution shift).

# COMMAND ----------

rng = np.random.default_rng(42)
N = 5_000
N_TRAIN = 3_500
TRUE_CAUSAL_EFFECT = -0.15  # telematics reduces log-frequency by 0.15 (~14% fewer claims)

# Rating factors
driver_age = rng.integers(18, 75, size=N).astype(float)
vehicle_group = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], size=N,
                            p=[0.08, 0.12, 0.18, 0.20, 0.18, 0.12, 0.07, 0.05])
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=N,
                        p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
postcode_area = rng.choice(
    ["London", "SE England", "Midlands", "North", "Scotland", "Wales"],
    size=N,
    p=[0.18, 0.16, 0.20, 0.22, 0.14, 0.10],
)
annual_mileage = rng.lognormal(mean=9.2, sigma=0.5, size=N).clip(1_000, 50_000)
exposure = rng.uniform(0.5, 1.0, size=N)

postcode_effect = {
    "London": 0.25, "SE England": 0.10, "Midlands": 0.05,
    "North": 0.00, "Scotland": -0.10, "Wales": -0.05,
}
log_rate = (
    -2.5
    + 0.012 * np.maximum(0, 25 - driver_age)
    - 0.008 * np.maximum(0, driver_age - 55)
    + 0.08 * vehicle_group
    - 0.07 * ncd_years
    + np.array([postcode_effect[p] for p in postcode_area])
    + 0.00015 * annual_mileage
)

# Telematics opt-in (confounded by risk factors)
logit_telematics = (
    1.2
    - 0.04 * (driver_age - 35)
    - 0.00008 * annual_mileage
    + 0.05 * (5 - ncd_years)
)
p_telematics = 1 / (1 + np.exp(-logit_telematics))
telematics_optin = rng.binomial(1, p_telematics)

log_rate_with_treatment = log_rate + TRUE_CAUSAL_EFFECT * telematics_optin
mu = np.exp(log_rate_with_treatment) * exposure
claim_count = rng.poisson(mu)
claim_amount = np.where(claim_count > 0, rng.lognormal(7.5, 0.8, size=N) * claim_count, 0.0)

df = pd.DataFrame({
    "driver_age": driver_age,
    "vehicle_group": vehicle_group.astype(float),
    "ncd_years": ncd_years.astype(float),
    "postcode_area": postcode_area,
    "annual_mileage": annual_mileage,
    "exposure": exposure,
    "telematics_optin": telematics_optin.astype(float),
    "claim_count": claim_count,
    "claim_amount": claim_amount,
})

df_train = df.iloc[:N_TRAIN].copy()
df_monitor = df.iloc[N_TRAIN:].copy()
df_monitor["driver_age"] = (df_monitor["driver_age"] + rng.normal(2, 1, size=len(df_monitor))).clip(18, 80)
df_monitor["annual_mileage"] = (df_monitor["annual_mileage"] * rng.lognormal(0.05, 0.05, size=len(df_monitor))).clip(1_000, 50_000)

print(f"Training period: {len(df_train):,} policies")
print(f"Monitoring period: {len(df_monitor):,} policies")
print(f"Overall claim frequency: {claim_count.sum() / exposure.sum():.4f} claims per car-year")
print(f"Telematics opt-in rate: {telematics_optin.mean():.1%}")
print(f"True causal effect of telematics: {TRUE_CAUSAL_EFFECT:.3f} (log-frequency)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit a CatBoost Frequency Model
# MAGIC
# MAGIC Standard Poisson GBM on training data. We hold out 20% of the training period for conformal calibration. The model learns log-rate from all features including telematics — but this rate estimate is confounded. The causal step below isolates the pure treatment effect.

# COMMAND ----------

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

FEATURES = ["driver_age", "vehicle_group", "ncd_years", "annual_mileage"]

X_train_full = df_train[FEATURES].values
y_train_full = df_train["claim_count"].values
w_train_full = df_train["exposure"].values

idx = np.arange(len(X_train_full))
idx_fit, idx_cal = train_test_split(idx, test_size=0.20, random_state=42)

X_fit, y_fit, w_fit = X_train_full[idx_fit], y_train_full[idx_fit], w_train_full[idx_fit]
X_cal, y_cal, w_cal = X_train_full[idx_cal], y_train_full[idx_cal], w_train_full[idx_cal]

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
)
model.fit(X_fit, y_fit, sample_weight=w_fit)

preds_cal = model.predict(X_cal)
ae_train = y_cal.sum() / (preds_cal * w_cal).sum()

print(f"CatBoost Poisson GBM fitted on {len(X_fit):,} policies")
print(f"Calibration set A/E ratio: {ae_train:.3f}  (1.00 = perfect balance)")
print(f"Model features: {FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Causal Inference — Does Telematics Reduce Claims?
# MAGIC
# MAGIC **The problem with naive regression:** telematics opt-in is correlated with driver age and mileage. Younger drivers opt in more. Younger drivers also have higher claim frequencies for unrelated reasons. A naive comparison mixes the *causal effect of telematics* with the *selection effect*.
# MAGIC
# MAGIC **Double Machine Learning** (Chernozhukov et al. 2018) fixes this by residualising both the outcome and the treatment on confounders, then regressing the outcome residual on the treatment residual. The result is free of confounding and has valid confidence intervals.
# MAGIC
# MAGIC The true causal effect in our DGP is **-0.15**. The naive estimate will be biased toward a larger reduction because low-risk drivers self-select into telematics.

# COMMAND ----------

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment

CONFOUNDERS = ["driver_age", "vehicle_group", "ncd_years", "annual_mileage"]

causal_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=BinaryTreatment(
        column="telematics_optin",
        positive_label="opted in",
        negative_label="not opted in",
    ),
    confounders=CONFOUNDERS,
    exposure_col="exposure",
    cv_folds=3,
    random_state=42,
)

print("Fitting DML model (cross-fitting nuisance models)...")
causal_model.fit(df_train)

ate = causal_model.average_treatment_effect()
print(ate)

freq_train = df_train["claim_count"] / df_train["exposure"]
naive_diff = (
    freq_train[df_train["telematics_optin"] == 1].mean()
    - freq_train[df_train["telematics_optin"] == 0].mean()
)

print(f"\nComparison:")
print(f"  True causal effect (DGP):  {TRUE_CAUSAL_EFFECT:.4f}")
print(f"  DML estimate:              {ate.estimate:.4f}  (95% CI: [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}])")
print(f"  Naive difference in means: {naive_diff:.4f}")
print()
print(f"The naive estimate is {abs(naive_diff - TRUE_CAUSAL_EFFECT) / abs(TRUE_CAUSAL_EFFECT):.0%} off the true effect.")
print(f"DML recovers the true causal effect within its confidence interval: ", end="")
print(ate.ci_lower <= TRUE_CAUSAL_EFFECT <= ate.ci_upper)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Conformal Prediction Intervals
# MAGIC
# MAGIC Point predictions from the CatBoost model tell you the expected claim count. **Split conformal prediction** adds distribution-free prediction intervals with a finite-sample marginal coverage guarantee: `P(Y_new in [l(X), u(X)]) >= 1 - alpha`.
# MAGIC
# MAGIC For Poisson/count data, the right non-conformity score is the **Pearson-weighted residual**: `(y - y_hat) / y_hat^(p/2)`. This handles the heteroscedasticity of count data — giving narrower intervals for low-risk policies and appropriately wide intervals for high-risk ones. We use `alpha=0.10` for 90% coverage.

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="poisson",
    tweedie_power=1.0,
)

cp.calibrate(X_cal, y_cal)
print(f"Calibrated on {cp.n_calibration_:,} observations")

X_monitor = df_monitor[FEATURES].values
y_monitor = df_monitor["claim_count"].values

intervals = cp.predict_interval(X_monitor, alpha=0.10)

covered = (
    (y_monitor >= intervals["lower"].to_numpy()) &
    (y_monitor <= intervals["upper"].to_numpy())
)

print(f"\nPrediction interval summary (monitoring period, alpha=0.10):")
print(f"  Target coverage:        90.0%")
print(f"  Achieved coverage:      {covered.mean():.1%}")
print(f"  Mean interval width:    {(intervals['upper'] - intervals['lower']).mean():.4f} claims")

# Coverage by decile — key diagnostic
decile_coverage = cp.coverage_by_decile(X_monitor, y_monitor, alpha=0.10)
print("\nCoverage by predicted risk decile:")
print(decile_coverage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Drift Monitoring
# MAGIC
# MAGIC The monitoring period has a mild distribution shift: drivers are slightly older and have higher mileage. The `MonitoringReport` runs three checks in one call:
# MAGIC
# MAGIC | Check | What it detects |
# MAGIC |---|---|
# MAGIC | **A/E ratio** | Calibration drift — model systematically over or underpredicts |
# MAGIC | **Gini coefficient** | Discrimination drift — model's rank-ordering has degraded |
# MAGIC | **CSI** | Feature drift — which rating factors have shifted |
# MAGIC
# MAGIC Traffic lights use the standard insurance monitoring thresholds (green / amber / red). The recommendation follows the three-stage decision tree: NO_ACTION, RECALIBRATE, or REFIT.

# COMMAND ----------

from insurance_monitoring import MonitoringReport

preds_train = model.predict(df_train[FEATURES].values)
preds_monitor = model.predict(X_monitor)

numeric_features = ["driver_age", "vehicle_group", "ncd_years", "annual_mileage"]
feat_train = pl.from_pandas(df_train[numeric_features])
feat_monitor = pl.from_pandas(df_monitor[numeric_features])

report = MonitoringReport(
    reference_actual=df_train["claim_count"].values,
    reference_predicted=preds_train * df_train["exposure"].values,
    current_actual=df_monitor["claim_count"].values,
    current_predicted=preds_monitor * df_monitor["exposure"].values,
    exposure=df_monitor["exposure"].values,
    reference_exposure=df_train["exposure"].values,
    feature_df_reference=feat_train,
    feature_df_current=feat_monitor,
    features=numeric_features,
    murphy_distribution="poisson",
    n_bootstrap=100,
)

print(f"Monitoring recommendation: {report.recommendation}")

ae = report.results_["ae_ratio"]
print(f"\nA/E ratio: {ae['value']:.3f}  [{ae['band'].upper()}]")
print(f"  95% CI: [{ae['lower_ci']:.3f}, {ae['upper_ci']:.3f}]")

gini = report.results_["gini"]
print(f"\nGini (reference): {gini['reference']:.3f}")
print(f"Gini (current):   {gini['current']:.3f}  [{gini['band'].upper()}]")
print(f"  Change: {gini['change']:+.3f}   p-value: {gini['p_value']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Feature Stability (CSI) and Murphy Decomposition

# COMMAND ----------

# CSI per feature
csi_df = pl.DataFrame(report.results_["csi"])
print("Feature stability (CSI):")
print("  PSI < 0.10: green (no shift)")
print("  PSI 0.10-0.25: amber (investigate)")
print("  PSI > 0.25: red (significant shift)")
print()
print(csi_df)

# Murphy decomposition: distinguishes RECALIBRATE vs REFIT
if report.murphy_available:
    m = report.results_["murphy"]
    print(f"\nMurphy decomposition ({m['verdict']}):")
    print(f"  Discrimination loss:  {m['discrimination']:.4f}  ({m['discrimination_pct']:.1f}% of total)")
    print(f"  Miscalibration loss:  {m['miscalibration']:.4f}  ({m['miscalibration_pct']:.1f}% of total)")

# Full results as a flat DataFrame — ready for Delta or MLflow
print(f"\nFull monitoring results:")
print(report.to_polars())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Standalone PSI — Quick Feature Drift Check
# MAGIC
# MAGIC `MonitoringReport` is the right tool for a full quarterly pack. For a quick per-feature check, the `psi()` function gives you that in one line.

# COMMAND ----------

from insurance_monitoring.drift import psi, wasserstein_distance

features_to_check = {
    "driver_age": (df_train["driver_age"].values, df_monitor["driver_age"].values),
    "annual_mileage": (df_train["annual_mileage"].values, df_monitor["annual_mileage"].values),
    "vehicle_group": (df_train["vehicle_group"].values, df_monitor["vehicle_group"].values),
    "ncd_years": (df_train["ncd_years"].values, df_monitor["ncd_years"].values),
}

print(f"{'Feature':<20} {'PSI':>8} {'Band':>8} {'Wasserstein':>14}")
print("-" * 55)
for name, (ref, cur) in features_to_check.items():
    p = psi(ref, cur, n_bins=10)
    w = wasserstein_distance(ref, cur)
    band = "green" if p < 0.10 else ("amber" if p < 0.25 else "red")
    print(f"{name:<20} {p:>8.4f} {band:>8} {w:>12.3f}")

w_age = wasserstein_distance(df_train["driver_age"].values, df_monitor["driver_age"].values)
print(f"\ndriver_age has shifted by {w_age:.1f} years on average (Wasserstein)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was demonstrated:**
# MAGIC
# MAGIC 1. **CatBoost Poisson frequency model** on 3,500 synthetic UK motor policies
# MAGIC
# MAGIC 2. **Causal effect of telematics opt-in** using Double Machine Learning:
# MAGIC    - The naive estimate was biased due to selection (safer drivers opting in)
# MAGIC    - DML recovered the true causal effect (-0.15 log-rate reduction)
# MAGIC    - Wrong causal estimates lead to wrong telematics pricing
# MAGIC
# MAGIC 3. **90% prediction intervals** via split conformal prediction:
# MAGIC    - Distribution-free, finite-sample coverage guarantee
# MAGIC    - Pearson-weighted score correctly handles count data heteroscedasticity
# MAGIC
# MAGIC 4. **Quarterly monitoring check** against a monitoring period with mild distribution shift:
# MAGIC    - A/E ratio, Gini drift test, and per-feature CSI in one call
# MAGIC    - Murphy decomposition distinguishes RECALIBRATE vs REFIT decisions
# MAGIC    - Output is a flat DataFrame ready to write to Delta or MLflow
# MAGIC
# MAGIC **Next steps — go deeper:**
# MAGIC
# MAGIC - `insurance-causal`: heterogeneous CATE by segment, GATES/BLP/CLAN inference, FCA-compliant renewal pricing optimisation
# MAGIC - `insurance-conformal`: locally-weighted intervals, joint frequency/severity conformal, Solvency II SCR bounds
# MAGIC - `insurance-monitoring`: TRIPODD drift attribution, anytime-valid A/B testing (mSPRT), PIT-based sequential calibration monitoring
