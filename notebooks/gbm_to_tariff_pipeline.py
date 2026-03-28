# Databricks notebook source

# MAGIC %md
# MAGIC # From GBM to Production Tariff: Distributional Modelling, SHAP Relativities, and Smoothed Rating Tables
# MAGIC
# MAGIC ## What this notebook solves
# MAGIC
# MAGIC Building a GBM that outperforms the production GLM is the easy part. Turning that GBM
# MAGIC into an actual production tariff — one that pricing committees can read, actuarial
# MAGIC function can sign off, and rating engines can ingest — is where most teams get stuck.
# MAGIC
# MAGIC This pipeline chains four libraries to close that gap:
# MAGIC
# MAGIC | Step | Library | Output |
# MAGIC |------|---------|--------|
# MAGIC | 1. Generate synthetic portfolio | numpy | 25,000-policy UK motor dataset |
# MAGIC | 2. Fit distributional GBM | `insurance-distributional` | Per-risk mean AND volatility |
# MAGIC | 3. Extract rating factor relativities | `shap-relativities` | GLM-format factor table with CIs |
# MAGIC | 4. Smooth age and NCD curves | `insurance-whittaker` | Actuarially defensible rating tables |
# MAGIC | 5. Build conformal prediction intervals | `insurance-conformal` | Distribution-free premium intervals |
# MAGIC
# MAGIC ## Why these four libraries together
# MAGIC
# MAGIC **`insurance-distributional`** gives you more than E[Y|X]. It models both the conditional
# MAGIC mean and the conditional dispersion, so you get a volatility score per risk. Two policies
# MAGIC with the same predicted pure premium can have very different risk profiles — a young driver
# MAGIC in a sports car has higher variance than an experienced driver in a standard vehicle, even
# MAGIC at the same expected cost. That distinction matters for safety loading, reinsurance, and
# MAGIC capital allocation.
# MAGIC
# MAGIC **`shap-relativities`** translates the GBM into a format actuaries and regulators recognise:
# MAGIC a multiplicative factor table with one row per rating factor level, directly comparable to
# MAGIC exp(beta) from a GLM. Without this step, a GBM result is not auditable by a pricing
# MAGIC committee.
# MAGIC
# MAGIC **`insurance-whittaker`** addresses the actuarial production problem: raw SHAP relativities
# MAGIC for driver age are noisy at the tails (ages 17–20 and 75+) because few policies sit there.
# MAGIC Whittaker-Henderson smoothing with automatic REML lambda selection produces a smooth,
# MAGIC defensible curve without arbitrary manual intervention.
# MAGIC
# MAGIC **`insurance-conformal`** provides prediction intervals with a finite-sample coverage
# MAGIC guarantee. Unlike parametric bootstrap intervals, conformal intervals hold even when
# MAGIC the residual distribution is non-standard — which it always is for compound Poisson-Gamma
# MAGIC pure premium data. These intervals directly support PRA SS1/23 model uncertainty requirements.
# MAGIC
# MAGIC ---
# MAGIC **Regulatory references:**
# MAGIC - PRA Supervisory Statement SS1/23: Model Risk Management Principles (uncertainty quantification)
# MAGIC - FCA Consumer Duty FG22/5: Explainability requirements for pricing models
# MAGIC - IFRS 17: Per-risk variance quantification for reserving
# MAGIC - Biessy (2026), ASTIN Bulletin: Whittaker-Henderson smoothing — the methodology behind Step 4

# COMMAND ----------

# MAGIC %pip install insurance-distributional shap-relativities insurance-whittaker insurance-conformal catboost shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

print("Imports ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC We generate 25,000 synthetic UK motor policies from a known data-generating process.
# MAGIC The DGP has three key actuarial features:
# MAGIC
# MAGIC - **Young driver loading:** drivers under 25 have a 90% frequency loading (exp(0.64) = 1.90)
# MAGIC - **NCD discount:** each NCD year reduces frequency by ~11% (exp(-0.115) = 0.89)
# MAGIC - **Heterogeneous dispersion:** older vehicles have higher variance relative to mean
# MAGIC   (phi increases with vehicle_age). This is what `insurance-distributional` is designed
# MAGIC   to detect — two identical expected costs with different risk profiles.
# MAGIC
# MAGIC The pure premium target is incurred cost per earned year: Tweedie distributed with
# MAGIC a point mass at zero (policies with no claims) and a heavy right tail. p=1.5 is the
# MAGIC standard starting point for UK motor.
# MAGIC
# MAGIC In production: replace this cell with your policy and claims extract. The column
# MAGIC names below match standard UK motor schemas.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 25_000

# Rating factors
driver_age = rng.integers(17, 80, size=n).astype(float)
vehicle_age = rng.integers(0, 16, size=n).astype(float)
ncd_years = np.minimum(rng.integers(0, 10, size=n), (driver_age - 17)).clip(0).astype(float)
vehicle_group = rng.integers(1, 51, size=n).astype(float)
annual_mileage = rng.integers(3_000, 25_001, size=n).astype(float)
area_codes = rng.integers(0, 6, size=n)  # 0=A (rural) to 5=F (inner city)
exposure = np.clip(rng.beta(5, 2, size=n), 0.05, 1.0)

# True log-linear predictor for mean frequency
# Intercept gives ~0.10 base rate (annualised)
log_mu = (
    -2.30                                                   # base: ~0.10 freq/year
    + 0.64 * (driver_age < 25).astype(float)               # young driver loading
    + 0.35 * (driver_age >= 70).astype(float)              # elderly driver loading
    - 0.115 * ncd_years                                     # NCD discount ~11%/year
    + 0.018 * vehicle_group                                 # vehicle group loading
    + 0.05 * area_codes                                     # urban loading per band
    + 0.12 * (annual_mileage > 15_000).astype(float)        # high-mileage loading
)
mu_true = np.exp(log_mu) * exposure  # expected claims (not rate)

# Heterogeneous dispersion: older vehicles have higher variance relative to mean
# This is the key feature that standard GLMs miss
phi_true = 0.5 + 0.04 * vehicle_age  # phi in [0.5, 1.1]

# Simulate compound Poisson-Gamma (Tweedie p=1.5) correctly
# At p=1.5: alpha_sev = 1.0, so individual severity ~ Exponential
p_tweedie = 1.5
alpha_sev = (2 - p_tweedie) / (p_tweedie - 1)  # = 1.0
lam = mu_true ** (2 - p_tweedie) / (phi_true * (2 - p_tweedie))
counts = rng.poisson(lam)
beta_sev = mu_true / (lam * alpha_sev + 1e-12)
y_incurred = np.array([
    rng.gamma(alpha_sev, beta_sev[i], size=c).sum() if c > 0 else 0.0
    for i, c in enumerate(counts)
])
pure_premium = y_incurred / exposure  # incurred per earned year (the Tweedie target)

# Claim count separately for the SHAP / frequency model
claim_count = counts.astype(float)

# Build feature matrix for modelling (numpy — no pandas dependency)
area_labels = np.array(["A", "B", "C", "D", "E", "F"])[area_codes]

X_np = np.column_stack([
    driver_age, vehicle_age, ncd_years, vehicle_group,
    annual_mileage, area_codes, exposure,
])
feature_names = [
    "driver_age", "vehicle_age", "ncd_years", "vehicle_group",
    "annual_mileage", "area_code", "exposure",
]

# 60% train / 20% calibration / 20% test split (temporal ordering by policy index)
n_train = int(0.60 * n)
n_cal   = int(0.20 * n)

X_train = X_np[:n_train, :6]            # exclude exposure from feature matrix
X_cal   = X_np[n_train:n_train+n_cal, :6]
X_test  = X_np[n_train+n_cal:, :6]

y_train = pure_premium[:n_train]
y_cal   = pure_premium[n_train:n_train+n_cal]
y_test  = pure_premium[n_train+n_cal:]

exp_train = exposure[:n_train]
exp_cal   = exposure[n_train:n_train+n_cal]
exp_test  = exposure[n_train+n_cal:]

feat_names_no_exposure = feature_names[:6]

print(f"Portfolio: {n:,} policies, {exposure.sum():.0f} earned years")
print(f"Claim frequency: {counts.sum() / exposure.sum():.4f} per earned year")
print(f"Claim rate (young drivers, age<25): "
      f"{counts[driver_age<25].sum() / exposure[driver_age<25].sum():.4f}")
print(f"Claim rate (experienced, age 30-55): "
      f"{counts[(driver_age>=30) & (driver_age<=55)].sum() / exposure[(driver_age>=30) & (driver_age<=55)].sum():.4f}")
print(f"Zero-loss policies: {(y_incurred == 0).mean()*100:.1f}%")
print(f"\nSplit: {n_train:,} train | {n_cal:,} calibration | {n - n_train - n_cal:,} test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit TweedieGBM — Mean and Volatility per Risk
# MAGIC
# MAGIC Standard GBMs model E[Y|X]. `TweedieGBM` from `insurance-distributional` jointly models
# MAGIC both E[Y|X] (the pure premium) and Var[Y|X] / mu^p (the dispersion phi), using a
# MAGIC double-GLM approach: CatBoost Tweedie loss for the mean, Gamma regression on out-of-fold
# MAGIC Pearson residuals for the dispersion.
# MAGIC
# MAGIC The key output is `pred.volatility_score()` — the coefficient of variation (CV = SD/mean)
# MAGIC per risk. Two policies priced at £350 can have CV 0.8 (stable risk) and CV 2.1 (volatile
# MAGIC risk). The appropriate safety loading, reinsurance attachment, and capital treatment differ.
# MAGIC
# MAGIC The DGP has phi increasing with vehicle_age, so we expect TweedieGBM to recover higher
# MAGIC dispersion for older vehicles. Standard CatBoost would miss this entirely.

# COMMAND ----------

from insurance_distributional import TweedieGBM, tweedie_deviance

print("Fitting TweedieGBM(power=1.5)...")
print("This takes ~2-5 minutes (CatBoost + 3-fold OOF for phi model).\n")

tweedie_model = TweedieGBM(power=1.5, phi_cv_folds=3, random_state=42)
tweedie_model.fit(X_train, y_train, exposure=exp_train)

print("Fitting complete.")
print(tweedie_model)

# COMMAND ----------

# Predictions on test set
pred_test = tweedie_model.predict(X_test, exposure=exp_test)
vol_test = pred_test.volatility_score()

# Validate: dispersion should increase with vehicle age in our DGP
vehicle_age_test = X_test[:, 1]  # column index 1 = vehicle_age
young_vehicle_mask = vehicle_age_test <= 3
old_vehicle_mask   = vehicle_age_test >= 10

mean_vol_young = vol_test[young_vehicle_mask].mean()
mean_vol_old   = vol_test[old_vehicle_mask].mean()

print("Distributional prediction summary (test set):")
print(f"  Mean predicted pure premium:  £{pred_test.mean.mean():,.0f}")
print(f"  Mean observed pure premium:   £{y_test.mean():,.0f}")
print(f"  Mean volatility score (CV):   {vol_test.mean():.4f}")
print()
print("Heterogeneous dispersion recovery — the key test:")
print(f"  New vehicles (age 0-3):   mean CV = {mean_vol_young:.4f}")
print(f"  Old vehicles (age 10+):   mean CV = {mean_vol_old:.4f}")
print(f"  Ratio:                    {mean_vol_old / mean_vol_young:.2f}x  (true ratio from DGP: ~{(0.5+0.04*12)/(0.5+0.04*2):.2f}x)")

dev_test = tweedie_deviance(y_test, pred_test.mean, power=1.5)
null_mean = np.full(len(y_test), y_train.mean())
dev_null = tweedie_deviance(y_test, null_mean, power=1.5)
d_squared = 1 - dev_test / dev_null
print(f"\nTweedie D² (pseudo-R²): {d_squared:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Extract SHAP Relativities — GLM-Format Factor Table
# MAGIC
# MAGIC The TweedieGBM is a black box to any pricing committee. `shap-relativities` converts
# MAGIC the GBM into a format actuaries recognise: a table of (feature, level, relativity)
# MAGIC triples, where relativity = exp(SHAP value), exposure-weighted across the portfolio.
# MAGIC
# MAGIC For continuous features (driver_age, ncd_years), the output is one relativity per
# MAGIC observed value — a full non-linear curve. This is what the GLM cannot produce without
# MAGIC manual binning. For categorical features (area_code), it is one relativity per level.
# MAGIC
# MAGIC We compute SHAP on the **calibration set** (not training), which avoids inflated
# MAGIC confidence intervals from in-sample overfitting. Using the training set would produce
# MAGIC relativities that look tighter than they are.
# MAGIC
# MAGIC Note: we use a CatBoostRegressor fitted with Tweedie loss here (rather than the
# MAGIC TweedieGBM directly) because `shap-relativities` integrates with CatBoost's
# MAGIC TreeExplainer. The TweedieGBM's mean model is a standard CatBoost Tweedie model.

# COMMAND ----------

import polars as pl
from catboost import CatBoostRegressor, Pool

# Refit a plain CatBoost Tweedie model for SHAP extraction.
# TweedieGBM internally uses CatBoost — we expose the equivalent here so
# shap-relativities can call its TreeExplainer directly.
log_exposure_train = np.log(exp_train.clip(1e-9))
log_exposure_cal   = np.log(exp_cal.clip(1e-9))

catboost_model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
catboost_model.fit(
    Pool(X_train, y_train, baseline=log_exposure_train)
)

# COMMAND ----------

from shap_relativities import SHAPRelativities

# Use calibration set for SHAP computation
cal_df = pl.DataFrame({
    name: X_cal[:, i].tolist()
    for i, name in enumerate(feat_names_no_exposure)
})

sr = SHAPRelativities(
    model=catboost_model,
    X=cal_df,
    exposure=pl.Series("exposure", exp_cal.tolist()),
    categorical_features=["area_code"],
)
sr.fit()

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0},  # area A (rural) = base
    ci_method="clt",
)

baseline_rate = sr.baseline()
print(f"Baseline annualised claim rate: {baseline_rate:.4f} per policy year")
print()

# Show area and vehicle group relativities (discrete factors)
print("Area relativities (0=A rural → 5=F inner city):")
area_rels = relativities.filter(pl.col("feature") == "area_code").sort("level")
for row in area_rels.iter_rows(named=True):
    level_map = {0: "A (rural)", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F (inner city)"}
    rel = row["relativity"]
    ci_lo = row.get("lower_ci", float("nan"))
    ci_hi = row.get("upper_ci", float("nan"))
    if rel is not None and not np.isnan(rel):
        print(f"  {level_map.get(int(row['level']), str(row['level'])):20s}  {rel:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

print()

# Show SHAP validation checks before trusting any numbers
print("SHAP diagnostic checks:")
checks = sr.validate()
for check_name, result in checks.items():
    status = "PASS" if result.passed else "WARN"
    print(f"  [{status}] {check_name}: {result.message}")

# COMMAND ----------

# Extract continuous age and NCD curves from SHAP
# These are the curves we will smooth in Step 4.
age_rels = relativities.filter(pl.col("feature") == "driver_age").sort("level")
ncd_rels = relativities.filter(pl.col("feature") == "ncd_years").sort("level")

print(f"\nRaw SHAP driver age relativities: {len(age_rels)} bands (one per observed age)")
print(age_rels.head(10))

print(f"\nRaw SHAP NCD relativities: {len(ncd_rels)} levels")
print(ncd_rels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Smooth Rating Curves with Whittaker-Henderson
# MAGIC
# MAGIC Raw SHAP relativities for driver_age are noisy at the tails — ages 17–20 and 75–79
# MAGIC have sparse exposure in the calibration set, so the SHAP values there are driven by
# MAGIC individual policy effects, not stable underlying rates. A pricing committee will
# MAGIC immediately challenge a 40-year-old relativity that is higher than a 38-year-old
# MAGIC when the true risk curve is monotonically declining in the 30–60 age range.
# MAGIC
# MAGIC Whittaker-Henderson smoothing (Biessy 2026, ASTIN Bulletin) solves this with:
# MAGIC 1. **Penalised least squares:** fitting a smooth curve through the raw relativity points
# MAGIC 2. **REML lambda selection:** automatic smoothing level, no manual parameter choice
# MAGIC 3. **Bayesian credible intervals:** wider where exposure is thin (ages 17–20, 75+)
# MAGIC
# MAGIC We smooth on the log-relativity scale (REML on log values, then exponentiate) to
# MAGIC ensure all smoothed relativities remain positive. This is the actuarial convention
# MAGIC for multiplicative tariff tables.
# MAGIC
# MAGIC The NCD scale is short (0–9 years), so the smoother is less critical there — but
# MAGIC it still removes the occasional non-monotone anomaly that causes actuarial review
# MAGIC challenges.

# COMMAND ----------

from insurance_whittaker import WhittakerHenderson1D

# ---- Driver age curve smoothing ----

# Extract age values and raw log-relativities from SHAP output
age_vals = age_rels["level"].cast(pl.Float64).to_numpy()
raw_age_rels = age_rels["relativity"].to_numpy()

# Exposure weighting: SHAP exposure weighted (proxy: count of policies per age in cal set)
# Use the calibration set exposures as weights — thin ages get less influence
unique_ages, age_counts = np.unique(X_cal[:, 0].astype(int), return_counts=True)
age_exposure_map = dict(zip(unique_ages, age_counts.astype(float)))

# Match exposure to the SHAP age levels (some ages may not appear in calibration)
age_weights = np.array([age_exposure_map.get(int(a), 1.0) for a in age_vals])

# Work on log-relativity scale (avoids negative smoothed values)
log_age_rels = np.log(np.clip(raw_age_rels, 1e-4, None))

wh_age = WhittakerHenderson1D(order=2, lambda_method="reml")
smooth_age = wh_age.fit(age_vals, log_age_rels, weights=age_weights)

smoothed_age_rels = np.exp(smooth_age.fitted)

print(f"Driver age curve smoothing:")
print(f"  REML lambda: {smooth_age.lambda_:.1f}  (higher = smoother)")
print(f"  Effective degrees of freedom: {smooth_age.edf:.2f}")
print()
print(f"  Age 17 (young driver):   raw={np.exp(log_age_rels[age_vals==17][0]):.3f}  "
      f"smoothed={smoothed_age_rels[age_vals==17][0]:.3f}" if 17 in age_vals else "  Age 17: not in calibration set")
print(f"  Age 40 (experienced):    raw={np.exp(log_age_rels[age_vals==40][0]):.3f}  "
      f"smoothed={smoothed_age_rels[age_vals==40][0]:.3f}" if 40 in age_vals else "  Age 40: not in calibration set")

# ---- NCD curve smoothing ----

ncd_vals = ncd_rels["level"].cast(pl.Float64).to_numpy()
raw_ncd_rels = ncd_rels["relativity"].to_numpy()

unique_ncds, ncd_counts = np.unique(X_cal[:, 2].astype(int), return_counts=True)
ncd_exposure_map = dict(zip(unique_ncds, ncd_counts.astype(float)))
ncd_weights = np.array([ncd_exposure_map.get(int(v), 1.0) for v in ncd_vals])

log_ncd_rels = np.log(np.clip(raw_ncd_rels, 1e-4, None))

wh_ncd = WhittakerHenderson1D(order=2, lambda_method="reml")
smooth_ncd = wh_ncd.fit(ncd_vals, log_ncd_rels, weights=ncd_weights)

smoothed_ncd_rels = np.exp(smooth_ncd.fitted)

print(f"\nNCD curve smoothing:")
print(f"  REML lambda: {smooth_ncd.lambda_:.1f}")
print(f"  Effective degrees of freedom: {smooth_ncd.edf:.2f}")

# True expected NCD discount per year from DGP: exp(-0.115) = 0.891
# The smoother should recover a monotonically declining curve
print(f"\nSmoothed NCD relativities (true discount per year: exp(-0.115) = {np.exp(-0.115):.3f}):")
for i, (ncd, rel) in enumerate(zip(ncd_vals[:6], smoothed_ncd_rels[:6])):
    print(f"  NCD {int(ncd):1d}:  {rel:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Assemble the Production Rating Table
# MAGIC
# MAGIC The output of Steps 3 and 4 is a complete multiplicative rating table.
# MAGIC For continuous factors (age, NCD), we use the smoothed Whittaker-Henderson values.
# MAGIC For discrete factors (area), we use the SHAP relativities directly — no smoothing
# MAGIC needed for a 6-level categorical.
# MAGIC
# MAGIC This is the artefact that goes to the rating engine team. It is in the same format
# MAGIC as an Emblem or Radar output: (feature, level, relativity). An actuary can read it,
# MAGIC a pricing committee can challenge individual entries, and a compliance team can trace
# MAGIC each relativity back to the underlying model evidence.
# MAGIC
# MAGIC The Whittaker credible intervals propagate through: for thin-data ages (17–20, 75+),
# MAGIC the interval is wide, correctly signalling that the relativity is less certain. The
# MAGIC pricing committee may choose to load these bands conservatively within the CI bounds.

# COMMAND ----------

# Assemble rating table for the full continuous age and NCD scales
age_df = pl.from_dict({
    "feature": ["driver_age"] * len(age_vals),
    "level": age_vals.tolist(),
    "relativity_raw": np.exp(log_age_rels).tolist(),
    "relativity": smoothed_age_rels.tolist(),
    "ci_lower": np.exp(smooth_age.fitted - 1.96 * smooth_age.std_fitted).tolist(),
    "ci_upper": np.exp(smooth_age.fitted + 1.96 * smooth_age.std_fitted).tolist(),
    "weight": age_weights.tolist(),
    "smoothed": [True] * len(age_vals),
})

ncd_df = pl.from_dict({
    "feature": ["ncd_years"] * len(ncd_vals),
    "level": ncd_vals.tolist(),
    "relativity_raw": np.exp(log_ncd_rels).tolist(),
    "relativity": smoothed_ncd_rels.tolist(),
    "ci_lower": np.exp(smooth_ncd.fitted - 1.96 * smooth_ncd.std_fitted).tolist(),
    "ci_upper": np.exp(smooth_ncd.fitted + 1.96 * smooth_ncd.std_fitted).tolist(),
    "weight": ncd_weights.tolist(),
    "smoothed": [True] * len(ncd_vals),
})

area_df = area_rels.with_columns([
    pl.col("relativity").alias("relativity_raw"),
    pl.lit(True).alias("smoothed"),
    pl.col("lower_ci").alias("ci_lower"),
    pl.col("upper_ci").alias("ci_upper"),
    pl.lit(1.0).alias("weight"),
]).select(["feature", "level", "relativity_raw", "relativity", "ci_lower", "ci_upper", "weight", "smoothed"])

rating_table = pl.concat([age_df, ncd_df, area_df])

print("Rating table assembled:")
print(f"  Rows: {len(rating_table):,}")
print(f"  Features: {rating_table['feature'].unique().sort().to_list()}")
print()
print("Sample — driver age (young driver range, ages 17–25):")
print(
    rating_table
    .filter(pl.col("feature") == "driver_age")
    .filter(pl.col("level").cast(pl.Int64) <= 25)
    .select(["level", "relativity_raw", "relativity", "ci_lower", "ci_upper"])
    .with_columns(pl.all().round(4))
)

print("\nSample — NCD scale:")
print(
    rating_table
    .filter(pl.col("feature") == "ncd_years")
    .select(["level", "relativity_raw", "relativity", "ci_lower", "ci_upper"])
    .with_columns(pl.all().round(4))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Conformal Prediction Intervals — Premium Uncertainty Quantification
# MAGIC
# MAGIC The rating table gives expected relativities. But how uncertain are the individual
# MAGIC policy predictions? This matters in three contexts:
# MAGIC
# MAGIC 1. **PRA SS1/23 compliance:** Requires quantification of model uncertainty and documented
# MAGIC    escalation criteria when uncertainty is high. Parametric intervals assume variance
# MAGIC    scales as mu^p across the whole book — an assumption that breaks for non-standard
# MAGIC    residual distributions.
# MAGIC
# MAGIC 2. **Premium sufficiency review:** A policy predicted at £420 — is the true expected
# MAGIC    cost plausibly £600? If the 90% upper bound is £800, that is a meaningful risk.
# MAGIC
# MAGIC 3. **Reinsurance attachment:** The 99.5th percentile per risk is an input to SCR
# MAGIC    calculations under Solvency II internal models.
# MAGIC
# MAGIC `insurance-conformal` provides distribution-free coverage: P(y in [lower, upper]) >= 1-alpha
# MAGIC holds without any parametric assumption. The only requirement is exchangeability of
# MAGIC calibration and test data.
# MAGIC
# MAGIC We use the `pearson_weighted` non-conformity score, which normalises residuals by
# MAGIC the Tweedie standard deviation (mu^(p/2)). This is the correct choice for pure premium
# MAGIC data — it produces exchangeable scores across risk levels rather than treating a £1
# MAGIC error on a £100 policy the same as a £1 error on a £10,000 policy.

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor

# Conformal predictor wraps the CatBoost model
cp = InsuranceConformalPredictor(
    model=catboost_model,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)

# Calibrate on the held-out calibration set (must not overlap training)
log_exp_cal_flat = log_exposure_cal.reshape(-1, 1) if log_exposure_cal.ndim == 1 else log_exposure_cal
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals on the test set
intervals_90 = cp.predict_interval(X_test, alpha=0.10)

# 99% intervals for SCR use (Solvency II 99.5th percentile input)
intervals_99 = cp.predict_interval(X_test, alpha=0.01)

print("Conformal prediction intervals (test set, n={:,} policies):".format(len(X_test)))
print()

lower_90 = intervals_90["lower"].to_numpy()
upper_90 = intervals_90["upper"].to_numpy()
point    = intervals_90["point"].to_numpy()

width_90 = upper_90 - lower_90
covered_90 = float(((y_test >= lower_90) & (y_test <= upper_90)).mean())

print(f"  90% intervals:")
print(f"    Coverage on test: {covered_90:.3f} (target: >= 0.90)")
print(f"    Mean width:       £{width_90.mean():,.0f}")
print(f"    Median width:     £{np.median(width_90):,.0f}")

# High-risk vs low-risk coverage check
# Young drivers should have wider absolute intervals but similar coverage
young_mask = X_test[:, 0] < 25
exp_mask   = (X_test[:, 0] >= 30) & (X_test[:, 0] < 60)

cov_young = float(((y_test[young_mask] >= lower_90[young_mask]) & (y_test[young_mask] <= upper_90[young_mask])).mean())
cov_exp   = float(((y_test[exp_mask]   >= lower_90[exp_mask])   & (y_test[exp_mask]   <= upper_90[exp_mask])).mean())

print()
print(f"  Coverage by risk segment (90% intervals):")
print(f"    Young drivers (age < 25):    {cov_young:.3f}  (n={young_mask.sum():,})")
print(f"    Experienced (age 30–59):     {cov_exp:.3f}  (n={exp_mask.sum():,})")

print()
print(f"  99% intervals (SCR / reinsurance attachment use):")
lower_99 = intervals_99["lower"].to_numpy()
upper_99 = intervals_99["upper"].to_numpy()
covered_99 = float(((y_test >= lower_99) & (y_test <= upper_99)).mean())
print(f"    Coverage on test: {covered_99:.3f} (target: >= 0.99)")
print(f"    Mean width:       £{(upper_99 - lower_99).mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Premium Sufficiency Analysis
# MAGIC
# MAGIC The conformal intervals feed directly into a premium sufficiency control:
# MAGIC for each policy, does the quoted premium cover the upper bound of the confidence
# MAGIC interval for the expected loss? If not, the policy may be structurally underpriced.
# MAGIC
# MAGIC This is not a new idea — actuaries have always applied safety loadings. What changes
# MAGIC is that the loading is now calibrated per-risk using the model's own uncertainty estimate,
# MAGIC rather than a flat scalar applied to all risks equally. A high-uncertainty risk (wide
# MAGIC interval) receives a larger loading than a low-uncertainty risk.

# COMMAND ----------

# Pure premium predictions on test set
pred_pure = catboost_model.predict(X_test, ntree_start=0, ntree_end=0)
# CatBoost Tweedie predicts in response space — already in £/year terms

# Construct policy analysis dataframe
n_test = len(X_test)
driver_age_test   = X_test[:, 0]
vehicle_age_test  = X_test[:, 1]
ncd_test          = X_test[:, 2]
area_code_test    = X_test[:, 5]
area_label_test   = np.array(["A", "B", "C", "D", "E", "F"])[area_code_test.astype(int).clip(0, 5)]

policy_df = pl.DataFrame({
    "driver_age":      driver_age_test.tolist(),
    "vehicle_age":     vehicle_age_test.tolist(),
    "ncd_years":       ncd_test.tolist(),
    "area":            area_label_test.tolist(),
    "pure_premium_pred": pred_pure.tolist(),
    "pure_premium_obs":  y_test.tolist(),
    "interval_90_lower": lower_90.tolist(),
    "interval_90_upper": upper_90.tolist(),
    "interval_99_upper": upper_99.tolist(),
    "volatility_score":  vol_test[n_train + n_cal:].tolist() if len(vol_test) > n_cal else pred_test.volatility_score().tolist(),
}).with_columns([
    pl.col("pure_premium_pred").alias("technical_price"),
    (pl.col("interval_90_upper") / pl.col("pure_premium_pred").clip(lower_bound=1e-6)).alias("loading_factor_90"),
    (pl.col("interval_99_upper") / pl.col("pure_premium_pred").clip(lower_bound=1e-6)).alias("loading_factor_99"),
])

print("Premium sufficiency analysis:")
print()
print("Mean loading factors needed to cover upper CI bounds:")
print(f"  90% upper bound loading: {policy_df['loading_factor_90'].mean():.3f}x")
print(f"  99% upper bound loading: {policy_df['loading_factor_99'].mean():.3f}x")
print()
print("Loading factor by driver age band:")
print(
    policy_df.with_columns(
        pl.when(pl.col("driver_age") < 25).then(pl.lit("17-24"))
          .when(pl.col("driver_age") < 35).then(pl.lit("25-34"))
          .when(pl.col("driver_age") < 50).then(pl.lit("35-49"))
          .when(pl.col("driver_age") < 65).then(pl.lit("50-64"))
          .otherwise(pl.lit("65+"))
          .alias("age_band")
    )
    .group_by("age_band")
    .agg([
        pl.col("loading_factor_90").mean().round(3).alias("mean_loading_90"),
        pl.col("volatility_score").mean().round(4).alias("mean_volatility"),
        pl.len().alias("n_policies"),
    ])
    .sort("age_band")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Four libraries. One coherent workflow. Each step adds something no prior step could provide:
# MAGIC
# MAGIC | Step | Library | What it adds |
# MAGIC |------|---------|--------------|
# MAGIC | 2 | `insurance-distributional` | Per-risk volatility score (not just expected cost) |
# MAGIC | 3 | `shap-relativities` | Auditable factor table from a GBM — the "GBM goes to production" step |
# MAGIC | 4 | `insurance-whittaker` | Actuarially defensible rating curves, automatic smoothing, credible intervals |
# MAGIC | 5–6 | `insurance-conformal` | Distribution-free premium intervals, coverage-guaranteed |
# MAGIC
# MAGIC **The DGP validation results you should see:**
# MAGIC - TweedieGBM correctly recovers higher dispersion for older vehicles (true DGP: phi
# MAGIC   increases ~0.04 per year of vehicle age)
# MAGIC - SHAP relativities show young driver loading > 1.5x (true DGP: exp(0.64) = 1.90)
# MAGIC - NCD smoothed curve is monotonically declining (true discount: exp(-0.115) = 0.89/year)
# MAGIC - Conformal 90% intervals achieve >= 90% empirical coverage on test data
# MAGIC
# MAGIC **Next steps in a real deployment:**
# MAGIC 1. Replace the synthetic DGP with your actual policy and claims extract
# MAGIC 2. Tune TweedieGBM hyperparameters on your book (phi_cv_folds=5 for production)
# MAGIC 3. Export the rating table to your rating engine format (Emblem, Radar, or flat CSV)
# MAGIC 4. Set escalation thresholds on the conformal intervals for underwriter referral
# MAGIC 5. Re-calibrate the conformal predictor quarterly as the book evolves

# COMMAND ----------

print("Pipeline complete.")
print()
print("Artefacts produced:")
print(f"  1. TweedieGBM: mean + volatility per risk")
print(f"  2. SHAP relativities: {len(relativities):,} (feature, level, relativity) triples")
print(f"  3. Smoothed rating table: {len(rating_table):,} rows, {rating_table['feature'].n_unique()} features")
print(f"  4. Conformal intervals: 90% and 99% coverage, n={n_test:,} test policies")
print()
print("Key results:")
print(f"  Tweedie D²:                  {d_squared:.4f}")
print(f"  Phi ratio (old/new vehicle): {mean_vol_old/mean_vol_young:.2f}x  (DGP: ~{(0.5+0.04*12)/(0.5+0.04*2):.2f}x)")
print(f"  Conformal 90% coverage:      {covered_90:.3f}")
print(f"  Conformal 99% coverage:      {covered_99:.3f}")
print(f"  Mean loading factor (90%):   {policy_df['loading_factor_90'].mean():.3f}x")
