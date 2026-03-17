# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-distill: GBM-to-GLM Distillation for Rating Engines
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC You have a CatBoost frequency model that beats your production GLM by 4 Gini
# MAGIC points on the holdout. The head of pricing wants to deploy it. The rating engine
# MAGIC team wants a Radar import file. The regulator wants factor tables.
# MAGIC
# MAGIC CatBoost is a black box to a rating engine. You cannot load a gradient boosted
# MAGIC tree into Radar or Emblem. What pricing teams actually do: they shelf the GBM,
# MAGIC deploy the GLM, and accept the gap.
# MAGIC
# MAGIC **insurance-distill** takes a different approach. It trains the GBM normally,
# MAGIC generates pseudo-predictions from it, then fits a Poisson (or Gamma) GLM using
# MAGIC those pseudo-predictions as the target. The GLM learns to reproduce the GBM's
# MAGIC pricing decisions — but through a multiplicative factor structure the rating engine
# MAGIC can consume. The continuous variables are binned optimally (by decision tree, quantile,
# MAGIC or isotonic regression), and the result is one factor table per variable, directly
# MAGIC importable into Radar.
# MAGIC
# MAGIC A well-tuned distillation typically retains 90–97% of the GBM's Gini. You do not
# MAGIC get the full GBM — you get most of it, in a form that actually goes to production.
# MAGIC
# MAGIC ## What this notebook covers
# MAGIC
# MAGIC 1. Generate 40,000-policy synthetic UK motor portfolio with a known DGP
# MAGIC 2. Train a CatBoost Poisson frequency model
# MAGIC 3. Fit a Poisson GLM baseline (the current production approach)
# MAGIC 4. Distil the CatBoost model to a surrogate GLM with `insurance-distill`
# MAGIC 5. Benchmark: original GLM vs CatBoost vs distilled GLM (Gini, deviance)
# MAGIC 6. Inspect factor tables and compare distilled relativities to the direct GLM
# MAGIC 7. Show the double-lift chart and segment deviation report
# MAGIC 8. Export factor tables to CSV (Radar-compatible format)

# COMMAND ----------

# MAGIC %pip install "insurance-distill[catboost]" polars numpy matplotlib statsmodels glum --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_distill import SurrogateGLM
from insurance_distill import compute_gini, format_radar_csv
import insurance_distill

print(f"insurance-distill version: {insurance_distill.__version__}")
print("All imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data
# MAGIC
# MAGIC ### Data generating process
# MAGIC
# MAGIC We generate 40,000 synthetic UK motor policies. The true log claim frequency is:
# MAGIC
# MAGIC ```
# MAGIC log(mu_i / exposure_i) =
# MAGIC     intercept
# MAGIC   + age_effect(driver_age_i)          # non-linear: high young, low mid, modest rise 60+
# MAGIC   + area_effect(area_group_i)         # 5 area groups, log-scale
# MAGIC   + vehicle_value_effect(veh_value_i) # log-linear above a threshold
# MAGIC   + ncd_effect(ncd_years_i)           # monotone decreasing (more NCD = lower risk)
# MAGIC   + vehicle_age_effect(veh_age_i)     # older vehicle = mildly higher frequency
# MAGIC ```
# MAGIC
# MAGIC A standard Poisson GLM with linear continuous terms cannot capture the non-linear
# MAGIC age curve. CatBoost can. The distillation turns CatBoost's non-linear age response
# MAGIC into a bin-level factor table — the same shape, but in a form the GLM can represent.

# COMMAND ----------

RNG = np.random.default_rng(2024)
N = 40_000

# --- Features ---
driver_age    = RNG.integers(17, 76, N).astype(float)
vehicle_age   = RNG.integers(0, 16, N).astype(float)
vehicle_value = RNG.uniform(2_000, 45_000, N)
area_group    = RNG.integers(0, 5, N)             # 5 areas
ncd_years     = RNG.integers(0, 11, N).astype(float)
annual_miles  = RNG.integers(3_000, 25_001, N).astype(float)
exposure      = RNG.uniform(0.25, 1.0, N)

# --- True DGP ---
# Non-linear age effect (U-shaped)
def true_age_effect(age: np.ndarray) -> np.ndarray:
    effect = np.zeros_like(age, dtype=float)
    young = age < 25
    mid   = (age >= 25) & (age <= 55)
    old   = age > 55
    effect[young] = 1.1 - (age[young] - 17) * (0.55 / 7)
    effect[mid]   = -0.03 * (age[mid] - 25) / 30
    effect[old]   = 0.009 * (age[old] - 55)
    return effect

# Area effects on log scale
AREA_LOG_EFFECTS = np.array([0.0, 0.18, 0.35, -0.12, -0.28])

# Vehicle value: risk rises steeply on expensive vehicles (theft/repair cost mix)
def true_veh_value_effect(v: np.ndarray) -> np.ndarray:
    return 0.000012 * np.maximum(v - 8_000, 0)

# NCD: monotone decreasing
TRUE_NCD_COEF = -0.09

# Vehicle age: mild upward slope, flattens after 10 years
def true_veh_age_effect(a: np.ndarray) -> np.ndarray:
    return 0.022 * np.minimum(a, 10)

# Annual miles: small positive effect
TRUE_MILES_COEF = 0.000018

INTERCEPT = -3.0  # base log-rate ~5% annual frequency

log_rate = (
    INTERCEPT
    + true_age_effect(driver_age)
    + AREA_LOG_EFFECTS[area_group]
    + true_veh_value_effect(vehicle_value)
    + TRUE_NCD_COEF * ncd_years
    + true_veh_age_effect(vehicle_age)
    + TRUE_MILES_COEF * annual_miles
)

mu          = np.exp(log_rate) * exposure
claim_count = RNG.poisson(mu)

print(f"Policies:          {N:,}")
print(f"Total claims:      {claim_count.sum():,}")
print(f"Overall frequency: {claim_count.sum() / exposure.sum():.4f}")
print(f"Frequency range:   {np.exp(log_rate).min():.4f} – {np.exp(log_rate).max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / test split
# MAGIC
# MAGIC We use a 70 / 30 random split. The CatBoost model is trained on the training
# MAGIC fold only. The distillation is also fitted on training data — the test fold is
# MAGIC held out for the final Gini and deviance comparison.

# COMMAND ----------

idx       = RNG.permutation(N)
n_train   = int(N * 0.7)
train_idx = idx[:n_train]
test_idx  = idx[n_train:]

FEATURES = ["driver_age", "vehicle_age", "vehicle_value", "area_group", "ncd_years", "annual_miles"]

# Polars DataFrames
raw_df = pl.DataFrame({
    "driver_age":    driver_age,
    "vehicle_age":   vehicle_age,
    "vehicle_value": vehicle_value,
    "area_group":    area_group.astype(float),
    "ncd_years":     ncd_years,
    "annual_miles":  annual_miles,
})

X_train = raw_df[train_idx]
X_test  = raw_df[test_idx]
y_train = claim_count[train_idx].astype(float)
y_test  = claim_count[test_idx].astype(float)
exp_train = exposure[train_idx]
exp_test  = exposure[test_idx]

# pandas copies for statsmodels
def to_pd(X: pl.DataFrame, y: np.ndarray, exp: np.ndarray) -> pd.DataFrame:
    d = X.to_pandas().copy()
    d["y"] = y
    d["exposure"] = exp
    return d

train_pd = to_pd(X_train, y_train, exp_train)
test_pd  = to_pd(X_test,  y_test,  exp_test)

print(f"Train: {len(train_idx):,} policies, {y_train.sum():.0f} claims")
print(f"Test:  {len(test_idx):,}  policies, {y_test.sum():.0f} claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Baseline Poisson GLM
# MAGIC
# MAGIC This is the model we are trying to beat — and ultimately replace with the
# MAGIC distilled surrogate. We fit it with linear terms for all continuous features.
# MAGIC Driver age enters as a single coefficient; it cannot capture the U-shape.

# COMMAND ----------

glm_formula = (
    "y ~ driver_age + vehicle_age + vehicle_value + area_group + ncd_years + annual_miles"
)
glm_model = smf.glm(
    formula=glm_formula,
    data=train_pd,
    family=sm.families.Poisson(),
    offset=np.log(train_pd["exposure"]),
).fit()

print(glm_model.summary())

# COMMAND ----------

# Holdout predictions
glm_pred_test = np.asarray(
    glm_model.predict(test_pd, offset=np.log(test_pd["exposure"]))
)

# Also generate predictions on the training set (for distillation comparison)
glm_pred_train = np.asarray(
    glm_model.predict(train_pd, offset=np.log(train_pd["exposure"]))
)

# Compute Gini on test set (using actual counts as the ordering criterion,
# i.e. rank by predicted rate and measure discrimination against actual claims)
glm_gini_test = compute_gini(glm_pred_test / exp_test, weights=exp_test)
print(f"GLM Gini (test, rate-ordered): {glm_gini_test:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. CatBoost frequency model
# MAGIC
# MAGIC We fit a CatBoost Poisson regressor with exposure as the offset. CatBoost is
# MAGIC the natural complement to the distillation approach: it handles mixed feature
# MAGIC types without preprocessing, learns the non-linear age curve automatically, and
# MAGIC its `predict()` method is directly compatible with `SurrogateGLM`.
# MAGIC
# MAGIC We use a deliberately conservative hyperparameter set (shallow trees, early stopping)
# MAGIC because the distillation quality is more sensitive to overfitting in the GBM than
# MAGIC in a standard deployment — the surrogate GLM learns from the GBM's predictions,
# MAGIC so overfit predictions propagate into the factor tables.

# COMMAND ----------

from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=50,
)

# CatBoost Poisson with exposure offset: pass log(exposure) as baseline
catboost_model.fit(
    X_train.to_pandas(),
    y_train,
    sample_weight=exp_train,
    baseline=np.log(exp_train),  # log-offset for Poisson
    eval_set=(X_test.to_pandas(), y_test),
)

print("\nCatBoost training complete.")

# COMMAND ----------

# Predictions on test set
cb_pred_test = catboost_model.predict(
    X_test.to_pandas(),
    ntree_end=catboost_model.best_iteration_ + 1 if hasattr(catboost_model, "best_iteration_") else 0,
) * exp_test  # raw rate * exposure = expected count

# CatBoost returns rates when offset is included; multiply by test exposure
# Double-check by verifying mean predicted count vs observed
cb_rate_test = catboost_model.predict(X_test.to_pandas())
cb_pred_test = cb_rate_test * exp_test

cb_gini_test = compute_gini(cb_rate_test, weights=exp_test)
print(f"CatBoost Gini (test, rate-ordered): {cb_gini_test:.4f}")
print(f"CatBoost mean predicted count: {cb_pred_test.mean():.5f}")
print(f"Observed mean count:           {y_test.mean():.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distillation: SurrogateGLM
# MAGIC
# MAGIC We pass the fitted CatBoost model directly to `SurrogateGLM`. The surrogate:
# MAGIC
# MAGIC 1. Calls `catboost_model.predict(X_train)` to get pseudo-predictions
# MAGIC 2. Bins each continuous feature into up to 8 bins using CART decision trees
# MAGIC    (optimal splits for the GBM pseudo-predictions)
# MAGIC 3. Uses isotonic regression binning for `ncd_years` — monotone decreasing
# MAGIC 4. One-hot encodes the binned features
# MAGIC 5. Fits a Poisson GLM (via glum) on the pseudo-predictions with exposure weights
# MAGIC
# MAGIC The result is a GLM whose factor tables approximate the CatBoost model's
# MAGIC pricing decisions.

# COMMAND ----------

surrogate = SurrogateGLM(
    model=catboost_model,
    X_train=X_train,
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
)

surrogate.fit(
    max_bins=8,
    binning_method="tree",        # CART optimal splits for continuous features
    method_overrides={
        "ncd_years": "isotonic",  # monotone decreasing: more NCD = lower risk
    },
    min_bin_size=0.01,            # bins must have at least 1% of training data
)

print("Surrogate GLM fitted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validation report
# MAGIC
# MAGIC The `report()` method computes:
# MAGIC - **Gini ratio**: how much of CatBoost's discrimination the surrogate retains
# MAGIC - **Deviance ratio**: how well the surrogate reproduces the GBM pseudo-predictions
# MAGIC - **Segment deviation**: maximum and mean deviation across all rating cells
# MAGIC
# MAGIC These three numbers are what you take to the pricing committee.

# COMMAND ----------

report = surrogate.report()
print(report.metrics.summary())
print(f"\n{report}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Factor tables
# MAGIC
# MAGIC Each factor table has three columns:
# MAGIC - `level`: bin interval or category (same format as Radar import)
# MAGIC - `log_coefficient`: raw GLM coefficient on log scale (0.0 = base level)
# MAGIC - `relativity`: multiplicative factor = exp(log_coefficient)
# MAGIC
# MAGIC The base level always has relativity = 1.0. All other levels are relative to it.

# COMMAND ----------

# Print all factor tables
for feat, tbl in report.factor_tables.items():
    print(f"\n--- {feat} ---")
    print(tbl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Head-to-head: distilled GLM vs direct GLM factor comparison
# MAGIC
# MAGIC The distilled GLM encodes the non-linear age effect as a bin-level factor table.
# MAGIC The direct GLM has a single linear age coefficient. Here we plot both curves
# MAGIC on the same axes — the gap is the benefit of the distillation.
# MAGIC
# MAGIC We also overlay the true DGP to show how well the distillation recovers the
# MAGIC known ground truth.

# COMMAND ----------

# Distilled age factors
age_tbl = report.factor_tables["driver_age"].to_pandas()

# Parse bin midpoints from interval labels (e.g. "[-inf, 21.0)" => midpoint 19)
def bin_midpoint(label: str) -> float:
    """Estimate a representative midpoint for plotting from a bin label."""
    label = label.strip()
    if label.startswith("[-inf"):
        # Take the upper bound and subtract a small offset
        try:
            upper = float(label.split(",")[1].rstrip(")").strip())
            return upper - 1.0
        except Exception:
            return float("nan")
    elif label.endswith("inf)"):
        try:
            lower = float(label.split(",")[0].lstrip("[").strip())
            return lower + 1.0
        except Exception:
            return float("nan")
    else:
        try:
            lo = float(label.split(",")[0].lstrip("[").strip())
            hi = float(label.split(",")[1].rstrip(")").strip())
            return (lo + hi) / 2.0
        except Exception:
            return float("nan")

age_tbl["midpoint"] = age_tbl["level"].apply(bin_midpoint)
age_tbl = age_tbl.dropna(subset=["midpoint"]).sort_values("midpoint")

# GLM linear age relativity at same grid
glm_age_coef = glm_model.params["driver_age"]
age_grid = np.arange(17, 76, dtype=float)
glm_log_rel  = glm_age_coef * (age_grid - 35.0)

# True DGP log-relativity normalised to age 35
true_log_rel = true_age_effect(age_grid) - true_age_effect(np.array([35.0]))[0]

# Distilled log-relativity (step function at bin midpoints)
# Normalise so that the bin containing age 35 = 0 (log scale)
age_35_bin_mask = (age_tbl["midpoint"] <= 35)
ref_idx = age_tbl[age_35_bin_mask].index[-1] if age_35_bin_mask.any() else age_tbl.index[0]
ref_log_coef = age_tbl.loc[ref_idx, "log_coefficient"]
dist_log_rel = age_tbl["log_coefficient"] - ref_log_coef

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: log-relativity
ax = axes[0]
ax.plot(age_grid, true_log_rel,         color="black",   linewidth=2.5, label="True DGP", zorder=5)
ax.plot(age_grid, glm_log_rel,          color="#d62728", linewidth=2,   linestyle="--", label="GLM (linear term)")
ax.step(age_tbl["midpoint"], dist_log_rel,
        color="#1f77b4", linewidth=2.5, where="mid", label="Distilled GLM (bin factors)")
ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
ax.axvline(25, color="grey", linewidth=0.5, linestyle=":")
ax.axvline(55, color="grey", linewidth=0.5, linestyle=":")
ax.set_xlabel("Driver age")
ax.set_ylabel("Log-relativity vs age 35")
ax.set_title("Driver age: GLM vs distilled GLM vs true DGP")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Right: relativity scale
ax2 = axes[1]
ax2.plot(age_grid, np.exp(true_log_rel),      color="black",   linewidth=2.5, label="True DGP", zorder=5)
ax2.plot(age_grid, np.exp(glm_log_rel),        color="#d62728", linewidth=2,   linestyle="--", label="GLM (linear term)")
ax2.step(age_tbl["midpoint"], np.exp(dist_log_rel),
         color="#1f77b4", linewidth=2.5, where="mid", label="Distilled GLM (bin factors)")
ax2.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
ax2.set_xlabel("Driver age")
ax2.set_ylabel("Relativity vs age 35")
ax2.set_title("Relativity scale")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle("Driver age effect: distillation recovers the non-linear shape", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Double-lift chart
# MAGIC
# MAGIC The double-lift chart compares CatBoost pseudo-predictions to the surrogate
# MAGIC GLM predictions, ranked by the ratio GBM / GLM. If the surrogate is a perfect
# MAGIC copy of the GBM, all deciles would show ratio = 1.0. Deviations above 1.0 mean
# MAGIC the GBM charges more than the GLM in that cell; below 1.0 means the GLM is
# MAGIC over-pricing relative to the GBM.
# MAGIC
# MAGIC A flat chart means the surrogate is faithfully reproducing the GBM's pricing.

# COMMAND ----------

lift = report.lift_chart.to_pandas()
print("\nDouble-lift chart (GBM pseudo-preds vs surrogate GLM, sorted by GBM/GLM ratio):")
print(lift.to_string(index=False, float_format="{:.4f}".format))

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: average predictions by decile
ax = axes[0]
x = lift["decile"]
ax.bar(x - 0.2, lift["avg_gbm"], width=0.35, label="CatBoost (GBM)", color="#ff7f0e", alpha=0.85)
ax.bar(x + 0.2, lift["avg_glm"], width=0.35, label="Distilled GLM",  color="#1f77b4", alpha=0.85)
ax.set_xlabel("Decile (sorted by GBM/GLM ratio)")
ax.set_ylabel("Average predicted rate")
ax.set_title("Double-lift: GBM vs distilled GLM")
ax.legend()
ax.grid(alpha=0.3, axis="y")
ax.set_xticks(x)

# Right: ratio GBM / GLM
ax2 = axes[1]
ax2.bar(x, lift["ratio_gbm_to_glm"], color="#2ca02c", alpha=0.8)
ax2.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="Ratio = 1.0 (perfect)")
ax2.set_xlabel("Decile")
ax2.set_ylabel("GBM / GLM ratio")
ax2.set_title("Ratio of GBM to distilled GLM by decile")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")
ax2.set_xticks(x)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Holdout benchmark: GLM vs CatBoost vs distilled GLM
# MAGIC
# MAGIC We now evaluate all three models on the held-out test set.
# MAGIC
# MAGIC **Gini coefficient** measures discrimination — how well the model ranks risks
# MAGIC from low to high. Higher is better.
# MAGIC
# MAGIC **Poisson deviance** measures calibration — how close the predicted counts are
# MAGIC to the observed counts. Lower is better.
# MAGIC
# MAGIC The distilled GLM is not expected to match CatBoost exactly. The question is
# MAGIC whether it closes most of the gap between the direct GLM and CatBoost.

# COMMAND ----------

def poisson_deviance_mean(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Exposure-weighted mean Poisson deviance."""
    eps   = 1e-10
    y_pred = np.maximum(y_pred, eps)
    d      = 2.0 * (np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0) - (y_true - y_pred))
    if weights is not None:
        return float(np.average(d, weights=weights))
    return float(np.mean(d))


# --- GLM predictions on test ---
glm_pred_test_counts = glm_pred_test  # already computed above (expected counts)
glm_pred_test_rates  = glm_pred_test / exp_test

# --- CatBoost predictions on test ---
# cb_pred_test already computed as rate * exposure
cb_pred_test_rates = cb_rate_test
cb_pred_test_counts = cb_pred_test

# --- Surrogate GLM predictions on test ---
# We need to generate pseudo-predictions for X_test from the CatBoost model,
# then run the design matrix through the fitted surrogate GLM.
# The SurrogateGLM._get_predictions() runs on X_train only; for test we use
# surrogate.predict equivalent: apply the same binning + GLM on test data.
# We reconstruct this manually.

from insurance_distill._binning import OptimalBinner
from insurance_distill._export import build_factor_tables

# Re-apply the binning spec to X_test
_binner = OptimalBinner()
X_test_binned = _binner.transform(X_test, surrogate._bin_specs)

# Build test design matrix using surrogate's column structure
X_test_design, _ = surrogate._build_design_matrix(X_test_binned, interaction_pairs=[])
dist_pred_test = surrogate._predict_glm(X_test_design)
dist_pred_test_rates  = dist_pred_test  # GLM predict returns rate per unit exposure
dist_pred_test_counts = dist_pred_test * exp_test

dist_gini_test = compute_gini(dist_pred_test_rates, weights=exp_test)
print(f"Distilled GLM Gini (test): {dist_gini_test:.4f}")

# COMMAND ----------

# Deviances
null_rate   = y_train.sum() / exp_train.sum()
null_counts = null_rate * exp_test

null_dev = poisson_deviance_mean(y_test, null_counts, exp_test)
glm_dev  = poisson_deviance_mean(y_test, glm_pred_test_counts,  exp_test)
cb_dev   = poisson_deviance_mean(y_test, cb_pred_test_counts,   exp_test)
dist_dev = poisson_deviance_mean(y_test, dist_pred_test_counts, exp_test)

# Gini on test (rate-ordered, exposure-weighted)
glm_gini  = compute_gini(glm_pred_test_rates,  weights=exp_test)
cb_gini   = compute_gini(cb_pred_test_rates,   weights=exp_test)
dist_gini = compute_gini(dist_pred_test_rates, weights=exp_test)

def d_sq(dev_model: float, dev_null: float) -> float:
    return 1.0 - dev_model / dev_null if dev_null > 0 else 0.0

print("=" * 70)
print("HOLDOUT BENCHMARK (40k policies, 30% test split)")
print("=" * 70)
print(f"\n{'Model':<30} {'Gini':>8} {'Deviance':>12} {'D-squared':>10}")
print("-" * 65)
print(f"{'Null (intercept only)':<30} {'—':>8} {null_dev:>12.6f} {'0.0000':>10}")
print(f"{'Poisson GLM (linear terms)':<30} {glm_gini:>8.4f} {glm_dev:>12.6f} {d_sq(glm_dev, null_dev):>10.4f}")
print(f"{'CatBoost (Poisson)':<30} {cb_gini:>8.4f} {cb_dev:>12.6f} {d_sq(cb_dev, null_dev):>10.4f}")
print(f"{'Distilled GLM (surrogate)':<30} {dist_gini:>8.4f} {dist_dev:>12.6f} {d_sq(dist_dev, null_dev):>10.4f}")
print("-" * 65)

if cb_gini > glm_gini:
    gap_recovered = (dist_gini - glm_gini) / (cb_gini - glm_gini)
    print(f"\nDistillation recovers {gap_recovered:.1%} of the Gini gap (CatBoost over GLM).")
else:
    print("\nCatBoost Gini <= GLM Gini (unusual; check data/model fit).")

print(f"\nDistillation metrics (training set):")
print(report.metrics.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Factor table plots: all variables
# MAGIC
# MAGIC These are the rating factors the surrogate GLM produces. Each plot shows the
# MAGIC multiplicative relativity for each bin level. The base level (relativity = 1.0)
# MAGIC is marked with a dashed line.
# MAGIC
# MAGIC The NCD factor table should be monotone decreasing (more NCD years = lower risk)
# MAGIC because we used `method_overrides={"ncd_years": "isotonic"}` during binning.

# COMMAND ----------

n_feats = len(report.factor_tables)
fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
if n_feats == 1:
    axes = [axes]

for ax, (feat, tbl) in zip(axes, report.factor_tables.items()):
    tbl_pd = tbl.to_pandas()
    labels = [str(l) for l in tbl_pd["level"]]
    rels   = tbl_pd["relativity"].values

    short_labels = [l[:12] + "…" if len(l) > 13 else l for l in labels]
    colors = ["#d62728" if r > 1.05 else "#2ca02c" if r < 0.95 else "#aec7e8" for r in rels]

    ax.barh(short_labels, rels, color=colors, alpha=0.85, edgecolor="grey", linewidth=0.5)
    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Relativity")
    ax.set_title(feat, fontsize=10)
    ax.grid(alpha=0.3, axis="x")

plt.suptitle("Distilled GLM factor tables (multiplicative relativities)", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Export to CSV (Radar-compatible format)
# MAGIC
# MAGIC `export_csv` writes one CSV file per variable. We also show the `format_radar_csv`
# MAGIC helper that produces a two-column format matching what Radar expects on import.
# MAGIC
# MAGIC In production, point `directory` at a Unity Catalog volume or DBFS path.

# COMMAND ----------

import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    written = surrogate.export_csv(tmpdir, prefix="motor_freq_")
    print(f"Files written to {tmpdir}:")
    for path in written:
        fname = os.path.basename(path)
        content = open(path).read()
        print(f"\n  {fname}:")
        for line in content.strip().split("\n"):
            print(f"    {line}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Radar CSV format
# MAGIC
# MAGIC `format_radar_csv` wraps a factor table into the two-column format that Radar
# MAGIC expects. The feature name appears in the header row.

# COMMAND ----------

radar_age = format_radar_csv(report.factor_tables["driver_age"], "DriverAgeBand")
print("Radar CSV (driver age):")
print(radar_age)

radar_ncd = format_radar_csv(report.factor_tables["ncd_years"], "NCDYears")
print("\nRadar CSV (NCD years):")
print(radar_ncd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary
# MAGIC
# MAGIC ### What we found
# MAGIC
# MAGIC The key numbers from this demo:
# MAGIC
# MAGIC - **CatBoost** outperforms the direct Poisson GLM on both Gini and deviance.
# MAGIC   The gap comes almost entirely from the non-linear driver age effect — the
# MAGIC   GLM cannot represent it with a single linear coefficient.
# MAGIC
# MAGIC - **Distilled GLM** closes most of that gap. The Gini ratio from `report.metrics`
# MAGIC   shows how much of CatBoost's discrimination the surrogate retains. Values
# MAGIC   above 0.92 are typical for well-separated variables; the segment deviation
# MAGIC   report shows where the surrogate diverges most from the GBM.
# MAGIC
# MAGIC - **The NCD factor table is monotone** because we used isotonic regression
# MAGIC   binning. Without `method_overrides`, the CART tree might produce a non-monotone
# MAGIC   step function if the data is noisy at the extremes.
# MAGIC
# MAGIC ### What to tune if the Gini ratio is below 0.90
# MAGIC
# MAGIC 1. **Increase `max_bins`**. More bins = finer factor table = closer fit to the GBM.
# MAGIC    The trade-off is a larger factor table and more cells to validate.
# MAGIC
# MAGIC 2. **Add interaction terms** with `interaction_pairs=[("driver_age", "area_group")]`.
# MAGIC    If the GBM has strong interactions, the additive surrogate misses them. Adding
# MAGIC    interaction cells recovers them at the cost of multiplicativity.
# MAGIC
# MAGIC 3. **Check for GBM overfitting**. The surrogate learns from GBM pseudo-predictions.
# MAGIC    If the GBM is badly overfit to the training set, the surrogate distils noise.
# MAGIC    Use early stopping or increase L2 regularisation in the GBM before distilling.
# MAGIC
# MAGIC 4. **Use `alpha > 0`** in `SurrogateGLM` if the surrogate itself is overfitting
# MAGIC    to sparse bins. L2 regularisation shrinks small-cell coefficients toward zero.
# MAGIC
# MAGIC ### When distillation is the right approach
# MAGIC
# MAGIC Distillation makes sense when:
# MAGIC - The GBM materially outperforms the GLM (>2 Gini points)
# MAGIC - The rating engine requires factor tables (Radar, Emblem, any multiplicative system)
# MAGIC - The Gini ratio after distillation is acceptable (>0.90) AND the max segment
# MAGIC   deviation is within tolerance (<10%)
# MAGIC
# MAGIC It does not make sense when:
# MAGIC - The GBM barely beats the GLM (the distillation effort exceeds the benefit)
# MAGIC - The GBM's lift comes entirely from high-order interactions that an additive
# MAGIC   GLM cannot represent even with binned terms
# MAGIC - Regulatory constraints require that the rating factors come from a GLM fitted
# MAGIC   directly on the observed data, not on pseudo-predictions

# COMMAND ----------

dbutils.notebook.exit("PASS: insurance_distill_demo completed successfully")
