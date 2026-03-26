# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-distributional: Distributional GBM for Pure Premium Pricing
# MAGIC ## TweedieGBM and GammaGBM — Modelling the Full Conditional Distribution
# MAGIC
# MAGIC A standard GLM predicts one thing: E[Y|X]. Two risks with the same expected
# MAGIC pure premium but very different volatility profiles are priced identically.
# MAGIC That is a problem the moment you need to:
# MAGIC
# MAGIC - Add a safety loading that varies by risk volatility (not just expected loss)
# MAGIC - Calibrate prediction intervals for reinsurance attachment points
# MAGIC - Set capital reserves that reflect per-risk uncertainty under IFRS 17
# MAGIC - Decide which risks to refer to underwriters based on intrinsic uncertainty
# MAGIC
# MAGIC **Distributional GBM** — based on the ASTIN 2024 Best Paper (So & Valdez 2024)
# MAGIC — fits the full conditional distribution of losses, not just the mean. It models
# MAGIC both mu(x) = E[Y|X] and phi(x) = Var[Y|X] / V(mu) as functions of risk
# MAGIC characteristics. The result is a per-risk uncertainty score alongside the
# MAGIC usual point prediction.
# MAGIC
# MAGIC **This notebook:**
# MAGIC
# MAGIC 1. Loads a synthetic UK motor portfolio (50,000 policies) via `insurance_datasets`
# MAGIC 2. Fits **TweedieGBM** (compound Poisson-Gamma, the standard for pure premium)
# MAGIC 3. Uses **DistributionalPrediction** to extract per-risk uncertainty scores
# MAGIC 4. Shows **GammaGBM** for a severity-only model on claims-only data
# MAGIC 5. Evaluates with **tweedie_deviance**, **coverage**, and **pit_values**
# MAGIC 6. Interprets the volatility score in commercial pricing terms

# COMMAND ----------

# MAGIC %pip install catboost insurance-distributional insurance-datasets polars pyarrow matplotlib scipy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load the Motor Portfolio
# MAGIC
# MAGIC We use `load_motor()` from `insurance_datasets`: 50,000 synthetic UK motor
# MAGIC policies with a realistic confounding structure. The dataset is a pandas
# MAGIC DataFrame with one row per policy and 18 columns.
# MAGIC
# MAGIC The target for TweedieGBM is the **pure premium**: total incurred cost per
# MAGIC earned year. For policies with no claims this is zero, which is exactly
# MAGIC what the Tweedie compound Poisson-Gamma handles — a point mass at zero
# MAGIC plus a continuous positive distribution for claim-generating risks.
# MAGIC
# MAGIC We train on the most recent three accident years (2021-2023) and validate
# MAGIC on the holdout year (2023) using an inception-year split.

# COMMAND ----------

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_datasets import load_motor
import insurance_distributional
print(f"insurance-distributional version: {insurance_distributional.__version__}")

# ─── Load data ────────────────────────────────────────────────────────────────
df = load_motor(n_policies=50_000, seed=42)

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClaim frequency:   {df['claim_count'].sum() / df['exposure'].sum():.4f} per earned year")
print(f"Mean pure premium: £{(df['incurred'] / df['exposure']).mean():,.0f} per earned year")
print(f"Policies with claims: {(df['claim_count'] > 0).sum():,} ({(df['claim_count'] > 0).mean()*100:.1f}%)")
print(f"Zero-loss policies:   {(df['incurred'] == 0).sum():,} ({(df['incurred'] == 0).mean()*100:.1f}%)")

# Pure premium: incurred per earned year (Tweedie target)
df["pure_premium"] = df["incurred"] / df["exposure"]

print(f"\nPure premium summary:")
print(f"  Mean: £{df['pure_premium'].mean():,.0f}")
print(f"  Std:  £{df['pure_premium'].std():,.0f}")
print(f"  Max:  £{df['pure_premium'].max():,.0f}")
print(f"  % zeros: {(df['pure_premium'] == 0).mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Feature Engineering and Train/Test Split
# MAGIC
# MAGIC Rating factors: vehicle age, vehicle group (ABI 1-50), driver age, driver
# MAGIC experience, NCD years, conviction points, annual mileage, area band,
# MAGIC occupation class, policy type, and NCD protected status.
# MAGIC
# MAGIC We use an inception-year split: train on policies with inception year ≤ 2022,
# MAGIC test on 2023. This mimics the real workflow where you fit on historical years
# MAGIC and predict on the new year.

# COMMAND ----------

# Feature columns to use
FEATURE_COLS = [
    "vehicle_age",
    "vehicle_group",
    "driver_age",
    "driver_experience",
    "ncd_years",
    "conviction_points",
    "annual_mileage",
    "occupation_class",
    "ncd_protected",
]
CATEGORICAL_COLS = ["area", "policy_type"]

# Encode categoricals as integers for CatBoost
area_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
type_map = {"Comp": 0, "TPFT": 1}
df["area_code"]   = df["area"].map(area_map).fillna(0).astype(int)
df["policy_type_code"] = df["policy_type"].map(type_map).fillna(0).astype(int)

ALL_FEATURES = FEATURE_COLS + ["area_code", "policy_type_code"]

# Train/test split by inception year
train_mask = df["inception_year"] <= 2022
test_mask  = df["inception_year"] == 2023

df_train = df[train_mask].reset_index(drop=True)
df_test  = df[test_mask].reset_index(drop=True)

print(f"Training set: {len(df_train):,} policies (inception year ≤ 2022)")
print(f"Test set:     {len(df_test):,} policies (inception year 2023)")
print(f"\nTrain claim frequency: {df_train['claim_count'].sum() / df_train['exposure'].sum():.4f}")
print(f"Test  claim frequency: {df_test['claim_count'].sum()  / df_test['exposure'].sum():.4f}")

X_train = df_train[ALL_FEATURES].values
X_test  = df_test[ALL_FEATURES].values
y_train = df_train["pure_premium"].values
y_test  = df_test["pure_premium"].values
exp_train = df_train["exposure"].values
exp_test  = df_test["exposure"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fit TweedieGBM (Pure Premium — Compound Poisson-Gamma)
# MAGIC
# MAGIC `TweedieGBM(power=1.5)` is the standard for UK motor pure premium:
# MAGIC - Compound Poisson-Gamma handles the mixed discrete/continuous distribution
# MAGIC   of losses (point mass at zero from nil-claim policies + positive losses)
# MAGIC - Power p=1.5 is typical; p can be estimated from data but 1.5 is a
# MAGIC   good starting point for most UK motor books
# MAGIC - The model jointly estimates mu(x) and phi(x) using coordinate descent:
# MAGIC   CatBoost Tweedie loss for the mean, Gamma regression on OOF Pearson
# MAGIC   residuals for the dispersion (Smyth-Jørgensen double GLM)
# MAGIC
# MAGIC The `phi_cv_folds=3` setting uses 3-fold cross-fitting to get unbiased OOF
# MAGIC residuals for the dispersion model. This prevents in-sample overfitting from
# MAGIC making phi estimates unrealistically small.

# COMMAND ----------

from insurance_distributional import TweedieGBM

print("Fitting TweedieGBM(power=1.5) on training data...")
print("This takes approximately 3-8 minutes (CatBoost + 3-fold OOF for phi).\n")

tweedie_model = TweedieGBM(
    power=1.5,
    phi_cv_folds=3,
    random_state=42,
)
tweedie_model.fit(X_train, y_train, exposure=exp_train)
print("Fitting complete.")
print(tweedie_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: DistributionalPrediction — Per-Risk Mean and Uncertainty
# MAGIC
# MAGIC `model.predict()` returns a `DistributionalPrediction` object with:
# MAGIC - `pred.mean` — E[Y|X], the standard pure premium estimate
# MAGIC - `pred.variance` — Var[Y|X] = phi * mu^p, the conditional variance
# MAGIC - `pred.volatility_score()` — CoV = sqrt(Var)/mean, a dimensionless
# MAGIC   risk uncertainty ranking
# MAGIC
# MAGIC The volatility score is the key output. Two risks with the same predicted
# MAGIC mean but different CoV have different risk profiles. The high-CoV risk
# MAGIC warrants a larger safety loading and more careful reinsurance structuring.

# COMMAND ----------

from insurance_distributional import DistributionalPrediction

# Predictions on test set
pred_test = tweedie_model.predict(X_test, exposure=exp_test)
print(pred_test)
print()

# ─── Summary statistics ───────────────────────────────────────────────────────
print("Test set prediction summary:")
print(f"  Mean predicted pure premium:    £{pred_test.mean.mean():,.0f}")
print(f"  Mean observed pure premium:     £{y_test.mean():,.0f}")
print(f"  Mean predicted phi (dispersion):{pred_test.phi.mean():.4f}")
print(f"  Phi range:                      [{pred_test.phi.min():.4f}, {pred_test.phi.max():.4f}]")
print(f"  Mean volatility score (CoV):    {pred_test.volatility_score().mean():.4f}")
print(f"  CoV range:                      [{pred_test.volatility_score().min():.4f}, "
      f"{pred_test.volatility_score().max():.4f}]")
print()

# ─── Volatility spread across key rating factors ──────────────────────────────
df_pred = df_test[["driver_age", "ncd_years", "vehicle_group", "area", "conviction_points"]].copy()
df_pred["predicted_mean"]      = pred_test.mean
df_pred["predicted_phi"]       = pred_test.phi
df_pred["volatility_score"]    = pred_test.volatility_score()

# NCD quintile
df_pred["ncd_band"] = df_pred["ncd_years"].apply(
    lambda x: "0" if x == 0 else ("1-2" if x <= 2 else ("3-4" if x <= 4 else "5"))
)

print("Mean volatility score by NCD years (0 = most risky, 5 = safest):")
print(
    df_pred.groupby("ncd_band")["volatility_score"]
    .agg(mean="mean", count="count")
    .sort_index()
    .round(4)
)
print()

# Driver age band
df_pred["age_band"] = pd.cut(
    df_pred["driver_age"], bins=[16, 25, 35, 50, 65, 90],
    labels=["17-25", "26-35", "36-50", "51-65", "66+"]
)
print("Mean volatility score by driver age band:")
print(
    df_pred.groupby("age_band", observed=True)["volatility_score"]
    .agg(mean="mean", count="count")
    .round(4)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Scoring — tweedie_deviance, coverage, pit_values
# MAGIC
# MAGIC Three complementary diagnostics:
# MAGIC
# MAGIC 1. **Tweedie deviance** — proper scoring rule for mean fit quality. Lower
# MAGIC    is better. Compares to a null model (constant mean) to get a D² statistic.
# MAGIC 2. **Coverage** — what fraction of actuals fall within the predicted
# MAGIC    prediction interval? Should equal the nominal confidence level if
# MAGIC    the distributional model is well-calibrated.
# MAGIC 3. **PIT values** (probability integral transform) — if the fitted
# MAGIC    distribution is correct, PIT values should be Uniform(0, 1). A
# MAGIC    histogram of PITs reveals shape misspecification.

# COMMAND ----------

from insurance_distributional import tweedie_deviance, coverage, pit_values

# ─── Tweedie deviance ─────────────────────────────────────────────────────────
dev_test = tweedie_deviance(y_test, pred_test.mean, power=1.5)
print(f"Tweedie deviance (test):   {dev_test:.6f}")

# Null model deviance (constant mean = training mean)
null_mean = np.full(len(y_test), y_train.mean())
dev_null = tweedie_deviance(y_test, null_mean, power=1.5)
print(f"Null model deviance:       {dev_null:.6f}")
print(f"D² (pseudo-R²):            {1 - dev_test / dev_null:.4f}")
print()

# ─── Log score ───────────────────────────────────────────────────────────────
log_score_test = tweedie_model.log_score(X_test, y_test, exposure=exp_test)
print(f"Log score (mean NLL, test): {log_score_test:.6f}")
print()

# ─── Coverage at 80%, 90%, 95% ────────────────────────────────────────────────
print("Coverage (Monte Carlo prediction intervals from fitted distribution):")
print("Target   Empirical   Delta")
print("-" * 35)
for target in [0.80, 0.90, 0.95]:
    n_samples_mc = 5_000
    samples = pred_test._sample(n_samples=n_samples_mc, rng=np.random.default_rng(42))
    lower_q = (1 - target) / 2
    upper_q = 1 - lower_q
    lo = np.quantile(samples, lower_q, axis=1)
    hi = np.quantile(samples, upper_q, axis=1)
    cov = coverage(y_test, lo, hi)
    print(f"  {target:.0%}     {cov:.3f}       {cov - target:+.3f}")
print()

# ─── PIT values ──────────────────────────────────────────────────────────────
# PIT for continuous positive observations only (skip zeros)
pos_mask = y_test > 0
if pos_mask.sum() > 100:
    samples_pos = pred_test._sample(n_samples=5_000, rng=np.random.default_rng(42))[pos_mask]
    y_pos = y_test[pos_mask]
    pit_vals = pit_values(y_pos, samples_pos)
    print(f"PIT values (claims-only, n={pos_mask.sum():,}):")
    print(f"  Mean:   {pit_vals.mean():.4f}  (ideal: 0.500)")
    print(f"  Std:    {pit_vals.std():.4f}  (ideal: {1/np.sqrt(12):.4f} for Uniform)")
    ks_stat = float(np.max(np.abs(
        np.sort(pit_vals) - np.linspace(0, 1, len(pit_vals))
    )))
    print(f"  KS:     {ks_stat:.4f}  (lower = better calibration)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: GammaGBM — Severity Model on Claims-Only Data
# MAGIC
# MAGIC `GammaGBM` models the Gamma distribution: appropriate for strictly positive
# MAGIC continuous losses (claim amounts), where the compound-Poisson structure
# MAGIC of Tweedie is not needed. We fit it on claims-only records.
# MAGIC
# MAGIC Use case: separate frequency-severity pricing where you want a distributional
# MAGIC severity model (not just the mean) for reinsurance layer pricing or per-risk
# MAGIC capital assessment.

# COMMAND ----------

from insurance_distributional import GammaGBM

# Claims-only data
train_claims = df_train[df_train["claim_count"] > 0].reset_index(drop=True)
test_claims  = df_test[df_test["claim_count"] > 0].reset_index(drop=True)

# Severity: average cost per claim
train_claims["avg_severity"] = train_claims["incurred"] / train_claims["claim_count"]
test_claims["avg_severity"]  = test_claims["incurred"]  / test_claims["claim_count"]

X_train_sev = train_claims[ALL_FEATURES].values
X_test_sev  = test_claims[ALL_FEATURES].values
y_train_sev = train_claims["avg_severity"].values
y_test_sev  = test_claims["avg_severity"].values

print(f"Claims-only training set: {len(train_claims):,} records")
print(f"Claims-only test set:     {len(test_claims):,} records")
print(f"Mean severity (train): £{y_train_sev.mean():,.0f}")
print(f"Mean severity (test):  £{y_test_sev.mean():,.0f}")
print()

print("Fitting GammaGBM on claims-only severity data...")
gamma_model = GammaGBM(
    phi_cv_folds=3,
    random_state=42,
)
gamma_model.fit(X_train_sev, y_train_sev)
print("Fitting complete.")
print(gamma_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: GammaGBM Predictions — Per-Risk Severity Uncertainty

# COMMAND ----------

from insurance_distributional import gamma_deviance

pred_sev = gamma_model.predict(X_test_sev)
print(pred_sev)
print()

# Severity deviance
dev_sev = gamma_deviance(y_test_sev, pred_sev.mean)
null_sev = gamma_deviance(y_test_sev, np.full(len(y_test_sev), y_train_sev.mean()))
d2_sev = 1 - dev_sev / null_sev

print(f"Gamma deviance (test):   {dev_sev:.6f}")
print(f"Null model deviance:     {null_sev:.6f}")
print(f"D² (pseudo-R²):          {d2_sev:.4f}")
print()

print(f"Severity prediction summary:")
print(f"  Mean predicted severity:  £{pred_sev.mean.mean():,.0f}")
print(f"  Mean observed severity:   £{y_test_sev.mean():,.0f}")
print(f"  Phi range:                [{pred_sev.phi.min():.4f}, {pred_sev.phi.max():.4f}]")
print(f"  Volatility (CoV) range:   [{pred_sev.volatility_score().min():.3f}, "
      f"{pred_sev.volatility_score().max():.3f}]")
print()

# Spread of CoV by vehicle group decile
df_sev_pred = test_claims[["vehicle_group", "driver_age"]].copy()
df_sev_pred["pred_severity"]   = pred_sev.mean
df_sev_pred["volatility_cov"]  = pred_sev.volatility_score()
df_sev_pred["vg_decile"] = pd.qcut(df_sev_pred["vehicle_group"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

print("Severity volatility by vehicle group quintile (Q1=cheapest, Q5=most expensive):")
print(
    df_sev_pred.groupby("vg_decile", observed=True)[["pred_severity", "volatility_cov"]]
    .mean()
    .round({"pred_severity": 0, "volatility_cov": 4})
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Visualisations

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "insurance-distributional: TweedieGBM and GammaGBM — Motor Portfolio\n"
    "50,000 synthetic UK motor policies | Per-risk mean and uncertainty modelling",
    fontsize=13, fontweight="bold",
)

# ─── 1. Predicted mean vs observed (Tweedie test set) ─────────────────────────
ax = axes[0, 0]
sample_idx = np.random.default_rng(77).integers(0, len(y_test), 2000)
ax.scatter(
    pred_test.mean[sample_idx], y_test[sample_idx],
    alpha=0.15, s=6, color="#4CAF50"
)
max_val = max(float(pred_test.mean.max()), float(y_test.max()))
ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, alpha=0.7, label="Perfect prediction")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Predicted pure premium (log scale)")
ax.set_ylabel("Observed pure premium (log scale)")
ax.set_title("1. TweedieGBM: Predicted vs Observed\n(positive observations only)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ─── 2. Phi distribution across test portfolio ────────────────────────────────
ax = axes[0, 1]
ax.hist(pred_test.phi, bins=60, color="#4CAF50", alpha=0.75, edgecolor="none", density=True)
ax.axvline(pred_test.phi.mean(), color="red", lw=2, linestyle="--",
           label=f"Mean phi = {pred_test.phi.mean():.3f}")
ax.axvline(np.median(pred_test.phi), color="orange", lw=1.5, linestyle=":",
           label=f"Median phi = {np.median(pred_test.phi):.3f}")
ax.set_xlabel("phi (dispersion parameter)")
ax.set_ylabel("Density")
ax.set_title("2. TweedieGBM: Phi Distribution\n(per-risk dispersion)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ─── 3. Volatility score by driver age band ───────────────────────────────────
ax = axes[0, 2]
age_bands = ["17-25", "26-35", "36-50", "51-65", "66+"]
age_cov = []
for band in age_bands:
    mask = df_pred["age_band"] == band
    if mask.sum() > 0:
        age_cov.append(df_pred.loc[mask, "volatility_score"].mean())
    else:
        age_cov.append(0)

colours_age = ["#F44336", "#FF9800", "#FFC107", "#4CAF50", "#2196F3"]
bars = ax.bar(age_bands, age_cov, color=colours_age, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, age_cov):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Driver age band")
ax.set_ylabel("Mean volatility score (CoV)")
ax.set_title("3. TweedieGBM: Volatility by Age Band\nHigher = more uncertain", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# ─── 4. PIT histogram (claims only) ──────────────────────────────────────────
ax = axes[1, 0]
if pos_mask.sum() > 100:
    ax.hist(pit_vals, bins=25, color="#4CAF50", alpha=0.75, edgecolor="none", density=True)
    ax.axhline(1.0, color="red", lw=2, linestyle="--", label="Uniform(0,1)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"4. PIT Histogram (claims only, n={pos_mask.sum():,})\nFlat = well-calibrated distribution", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "Insufficient\nclaims for PIT", ha="center", va="center", fontsize=12)
    ax.set_title("4. PIT Histogram", fontsize=10)

# ─── 5. GammaGBM: severity CoV by vehicle group quintile ─────────────────────
ax = axes[1, 1]
vg_groups = ["Q1", "Q2", "Q3", "Q4", "Q5"]
vg_mean_sev = df_sev_pred.groupby("vg_decile", observed=True)["pred_severity"].mean().reindex(vg_groups)
vg_cov      = df_sev_pred.groupby("vg_decile", observed=True)["volatility_cov"].mean().reindex(vg_groups)

ax2_twin = ax.twinx()
bars2 = ax.bar(vg_groups, vg_mean_sev.values, color="#2196F3", alpha=0.5, width=0.4, label="Mean severity (£)")
line2, = ax2_twin.plot(vg_groups, vg_cov.values, "ro-", lw=2, markersize=8, label="Volatility CoV")
ax.set_xlabel("Vehicle group quintile (Q1=cheapest)")
ax.set_ylabel("Mean predicted severity (£)", color="#2196F3")
ax2_twin.set_ylabel("Volatility score (CoV)", color="red")
ax.set_title("5. GammaGBM: Severity and Volatility\nby Vehicle Group Quintile", fontsize=10)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax.grid(alpha=0.3, axis="y")

# ─── 6. Coverage calibration curve (Tweedie MC) ───────────────────────────────
ax = axes[1, 2]
target_levels = np.linspace(0.50, 0.97, 20)
empirical_cov = []
mc_samples_cov = pred_test._sample(n_samples=5_000, rng=np.random.default_rng(99))
for tgt in target_levels:
    lo_q = (1 - tgt) / 2
    hi_q = 1 - lo_q
    lo_v = np.quantile(mc_samples_cov, lo_q, axis=1)
    hi_v = np.quantile(mc_samples_cov, hi_q, axis=1)
    empirical_cov.append(coverage(y_test, lo_v, hi_v))

ax.plot(target_levels, target_levels, "k--", lw=1.5, label="Perfect calibration")
ax.plot(target_levels, empirical_cov, color="#4CAF50", lw=2, label="TweedieGBM coverage")
ax.fill_between(target_levels, target_levels, empirical_cov, alpha=0.15, color="#4CAF50")
ax.set_xlabel("Nominal coverage level")
ax.set_ylabel("Empirical coverage")
ax.set_title("6. Coverage Calibration Curve\n(Monte Carlo prediction intervals)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
display(fig)
plt.close()
print("Plots rendered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Safety Loading — Practical Use of the Volatility Score
# MAGIC
# MAGIC A simple but commercially useful application: compute a technical price that
# MAGIC includes a safety loading proportional to the per-risk volatility score (CoV).
# MAGIC
# MAGIC ```
# MAGIC Technical price = E[Y|X] * (1 + lambda * CoV(X))
# MAGIC ```
# MAGIC
# MAGIC where lambda is a loading parameter set by the pricing team (e.g. 0.10 = 10%
# MAGIC of the volatility score). This means:
# MAGIC - High-CoV risks pay more relative to their expected loss
# MAGIC - Low-CoV risks (stable, predictable) pay closer to the expected loss
# MAGIC - The loading is risk-specific, not a flat percentage uplift
# MAGIC
# MAGIC This is a minimal illustration. In practice, lambda would be calibrated to
# MAGIC the target return on capital and the portfolio loss distribution.

# COMMAND ----------

# Compute safety-loaded prices for test set
lambda_loading = 0.10   # 10% per unit of CoV — a conservative illustration

cov_scores   = pred_test.volatility_score()
base_premium = pred_test.mean
loaded_price = base_premium * (1 + lambda_loading * cov_scores)

print("Safety Loading Analysis (lambda = 10% per CoV unit)")
print("=" * 60)
print(f"\n  Mean base pure premium:    £{base_premium.mean():,.0f}")
print(f"  Mean CoV:                  {cov_scores.mean():.4f}")
print(f"  Mean safety loading:       {(loaded_price / base_premium - 1).mean()*100:.2f}%")
print(f"  Mean loaded premium:       £{loaded_price.mean():,.0f}")
print()

# Distribution of loading percentage
loading_pct = (loaded_price / base_premium - 1) * 100
print(f"  Loading distribution:")
print(f"    5th percentile:    {np.percentile(loading_pct, 5):.2f}%")
print(f"    25th percentile:   {np.percentile(loading_pct, 25):.2f}%")
print(f"    Median:            {np.median(loading_pct):.2f}%")
print(f"    75th percentile:   {np.percentile(loading_pct, 75):.2f}%")
print(f"    95th percentile:   {np.percentile(loading_pct, 95):.2f}%")
print()

# How does loading vary by age band?
df_loading = df_pred[["age_band"]].copy()
df_loading["loading_pct"] = loading_pct
print("Mean safety loading by driver age band:")
print(
    df_loading.groupby("age_band", observed=True)["loading_pct"]
    .mean()
    .round(2)
    .rename("loading (%)")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we built:**
# MAGIC
# MAGIC - `TweedieGBM(power=1.5)` fitted on 50k synthetic UK motor policies,
# MAGIC   jointly modelling the conditional mean AND dispersion of pure premiums.
# MAGIC - `GammaGBM` on claims-only records for the severity distribution.
# MAGIC - Diagnostic evaluation with tweedie_deviance, coverage, and PIT values.
# MAGIC - A simple volatility-loaded technical price based on per-risk CoV.
# MAGIC
# MAGIC **Why this matters:**
# MAGIC
# MAGIC | Question | Standard GLM | Distributional GBM |
# MAGIC |---|---|---|
# MAGIC | E[Y\|X] pure premium | Yes | Yes |
# MAGIC | Per-risk Var[Y\|X] | No | Yes |
# MAGIC | Calibrated prediction intervals | No | Yes |
# MAGIC | Volatility-based safety loading | No | Yes |
# MAGIC | Solvency II risk adjustment | Coarse | Risk-specific |
# MAGIC
# MAGIC **When to use TweedieGBM vs GammaGBM:**
# MAGIC - **TweedieGBM** for pure premium models where the data includes policies
# MAGIC   with zero claims. The compound structure handles the zero mass correctly.
# MAGIC - **GammaGBM** when you have already filtered to claims-only and want to
# MAGIC   model the severity distribution (e.g. separate frequency-severity).
# MAGIC
# MAGIC **Compute note:**
# MAGIC - Fitting TweedieGBM with 3-fold OOF cross-fitting takes 3-8 minutes on
# MAGIC   a standard Databricks cluster. Use `phi_cv_folds=1` to skip cross-fitting
# MAGIC   if speed is critical (accuracy of phi estimates will be lower).

# COMMAND ----------

print("=" * 65)
print("Workflow complete — insurance-distributional demo")
print("=" * 65)
print(f"""
TweedieGBM (pure premium, power=1.5):
  Training set:          {len(df_train):,} policies
  Test set:              {len(df_test):,} policies
  Tweedie deviance:      {dev_test:.6f}
  Null deviance:         {dev_null:.6f}
  D² (pseudo-R²):        {1 - dev_test / dev_null:.4f}
  Log score (NLL):       {log_score_test:.6f}
  Mean phi (test):       {pred_test.phi.mean():.4f}
  Mean CoV (test):       {pred_test.volatility_score().mean():.4f}

GammaGBM (severity, claims only):
  Training set:          {len(train_claims):,} claim records
  Test set:              {len(test_claims):,} claim records
  Gamma deviance:        {dev_sev:.6f}
  D² (pseudo-R²):        {d2_sev:.4f}
  Mean CoV (test):       {pred_sev.volatility_score().mean():.4f}

Key takeaway:
  The volatility score (CoV) varies by risk profile — young drivers
  and high-vehicle-group risks have higher intrinsic uncertainty.
  A standard GLM assigns the same uncertainty to all risks.
  The distributional model enables risk-specific safety loading
  and calibrated prediction intervals at each observation.
""")
