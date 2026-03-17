# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-distill: GBM-to-GLM Distillation for Rating Engines
# MAGIC
# MAGIC **The problem:** Your CatBoost frequency model outperforms the production GLM by 4 Gini points. The head of pricing wants to see factor tables. The rating engine team needs a Radar import. You cannot load a gradient boosted tree into Emblem.
# MAGIC
# MAGIC The GBM lives in a notebook. Factor tables go to production.
# MAGIC
# MAGIC **What this notebook shows:** `insurance-distill` bridges the gap by fitting a surrogate Poisson GLM on the GBM's pseudo-predictions — the GBM's output is used as the target, not the actual claims. The resulting factor tables are multiplicative (log-link by construction), compatible with any rating engine that uses a base rate × relativities structure, and can be exported as CSV files for direct import.
# MAGIC
# MAGIC **The honest accounting:** a well-tuned distillation retains 90–97% of the GBM's Gini. You give up some discrimination; you get interpretability, rating engine compatibility, and a factor table that a pricing committee can interrogate.
# MAGIC
# MAGIC **What you will cover:**
# MAGIC 1. Generate a 30,000-policy synthetic UK motor portfolio with a known frequency DGP
# MAGIC 2. Train a CatBoost Poisson model — the model that sits in a notebook
# MAGIC 3. Distil the GBM into a surrogate GLM using `insurance-distill`
# MAGIC 4. Inspect the validation metrics: Gini ratio, deviance ratio, segment deviation
# MAGIC 5. Compare GBM vs GLM predictions on a double-lift chart
# MAGIC 6. Print factor tables for each rating variable
# MAGIC 7. Show R² of distilled GLM predictions vs original GBM predictions

# COMMAND ----------

# MAGIC %pip install "insurance-distill[catboost]" polars

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor portfolio
# MAGIC
# MAGIC We build a 30,000-policy portfolio with a known log-linear frequency DGP. The true parameters are:
# MAGIC
# MAGIC ```
# MAGIC log(freq) = log(exposure)
# MAGIC           + intercept (-2.8)
# MAGIC           + 0.50 * young_driver   (age < 25)
# MAGIC           + 0.28 * senior_driver  (age >= 70)
# MAGIC           - 0.10 * ncd_years      (per year, up to 5)
# MAGIC           + area_effect           (A=0, B=0.12, C=0.24, D=0.38, E=0.55)
# MAGIC           + 0.40 * has_convictions
# MAGIC           + 0.03 * vehicle_group  (groups 1–10)
# MAGIC ```
# MAGIC
# MAGIC These are the coefficients the distilled GLM should approximately recover.

# COMMAND ----------

from __future__ import annotations

import numpy as np
import polars as pl

SEED = 42
N_POLICIES = 30_000

rng = np.random.default_rng(SEED)

# --- Features ---
driver_age = rng.integers(17, 85, N_POLICIES)
ncd_years = rng.integers(0, 6, N_POLICIES)          # 0-5 years no-claims discount
area = rng.choice(["A", "B", "C", "D", "E"], N_POLICIES, p=[0.30, 0.25, 0.20, 0.15, 0.10])
vehicle_group = rng.integers(1, 11, N_POLICIES)     # 1-10
has_convictions = rng.binomial(1, 0.08, N_POLICIES)
exposure = rng.uniform(0.3, 1.0, N_POLICIES)        # fraction of year earned

# --- True DGP ---
TRUE_PARAMS = {
    "intercept": -2.8,
    "young_driver (age<25)": 0.50,
    "senior_driver (age>=70)": 0.28,
    "ncd_years (per year)": -0.10,
    "area_B": 0.12,
    "area_C": 0.24,
    "area_D": 0.38,
    "area_E": 0.55,
    "has_convictions": 0.40,
    "vehicle_group (per group)": 0.03,
}

area_effect = np.select(
    [area == "A", area == "B", area == "C", area == "D", area == "E"],
    [0.0, 0.12, 0.24, 0.38, 0.55],
)

log_mu = (
    TRUE_PARAMS["intercept"]
    + np.log(exposure)
    + TRUE_PARAMS["young_driver (age<25)"] * (driver_age < 25).astype(float)
    + TRUE_PARAMS["senior_driver (age>=70)"] * (driver_age >= 70).astype(float)
    + TRUE_PARAMS["ncd_years (per year)"] * ncd_years
    + area_effect
    + TRUE_PARAMS["has_convictions"] * has_convictions
    + TRUE_PARAMS["vehicle_group (per group)"] * vehicle_group
)

claim_counts = rng.poisson(np.exp(log_mu))

# --- Assemble Polars DataFrame ---
df = pl.DataFrame({
    "driver_age": driver_age,
    "ncd_years": ncd_years,
    "area": area,
    "vehicle_group": vehicle_group,
    "has_convictions": has_convictions,
    "exposure": exposure,
    "claim_count": claim_counts,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Total claims: {df['claim_count'].sum():,}")
print(f"Overall frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
print(f"\nColumns: {df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC Print the ground-truth DGP parameters for reference. These are what we expect the distilled GLM to approximately recover.

# COMMAND ----------

print("True frequency DGP parameters:")
print("-" * 52)
for k, v in TRUE_PARAMS.items():
    relativity = np.exp(v)
    if k == "intercept":
        print(f"  {k:40s}  {v:+.3f}  (base)")
    else:
        print(f"  {k:40s}  {v:+.3f}  => relativity {relativity:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train CatBoost Poisson model
# MAGIC
# MAGIC CatBoost handles the `area` string feature natively — no label encoding needed. We use `loss_function="Poisson"` which is the standard choice for insurance frequency modelling.
# MAGIC
# MAGIC Note that the CatBoost model sees raw continuous features (driver_age, ncd_years, vehicle_group) with no manual binning. That is deliberate: the GBM will find its own split points; it is the distillation step that will bin them for the GLM surrogate.

# COMMAND ----------

from catboost import CatBoostRegressor, Pool

FEATURE_COLS = ["driver_age", "ncd_years", "area", "vehicle_group", "has_convictions"]
CAT_FEATURES = ["area"]

X_train = df.select(FEATURE_COLS)
y_train = df["claim_count"].to_numpy().astype(float)
exposure_arr = df["exposure"].to_numpy()

pool = Pool(
    data=X_train.to_pandas(),
    label=y_train,
    weight=exposure_arr,
    cat_features=CAT_FEATURES,
)

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=5,
    loss_function="Poisson",
    eval_metric="Poisson",
    random_seed=SEED,
    verbose=0,
)
model.fit(pool)

gbm_preds = model.predict(X_train.to_pandas())
print(f"CatBoost model trained. Feature importances:")
for feat, imp in sorted(
    zip(FEATURE_COLS, model.get_feature_importance()),
    key=lambda x: -x[1]
):
    print(f"  {feat:25s}  {imp:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Distil the GBM into a surrogate GLM
# MAGIC
# MAGIC `SurrogateGLM` takes the fitted CatBoost model, the training data, the actual targets, and the exposure. It:
# MAGIC 1. Calls `model.predict()` to get pseudo-predictions
# MAGIC 2. Bins continuous features using CART trees fit on the pseudo-predictions
# MAGIC 3. One-hot encodes the binned features (reference-coded)
# MAGIC 4. Fits a Poisson GLM with log link on the pseudo-predictions
# MAGIC
# MAGIC We specify `categorical_features=["area", "has_convictions"]` to pass those through without binning. Everything else gets tree-based binning with a maximum of 8 bins.

# COMMAND ----------

from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=model,
    X_train=X_train,
    y_train=y_train,
    exposure=exposure_arr,
    family="poisson",
)

surrogate.fit(
    features=["driver_age", "ncd_years", "vehicle_group"],
    categorical_features=["area", "has_convictions"],
    max_bins=8,
    binning_method="tree",
)

print("SurrogateGLM fitted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validation metrics
# MAGIC
# MAGIC `surrogate.report()` returns a `DistillationReport` with all scalar metrics, factor tables, and the double-lift chart.
# MAGIC
# MAGIC The key numbers to watch:
# MAGIC - **Gini ratio**: how much of the GBM's discrimination the GLM retains. Target >= 0.90.
# MAGIC - **Deviance ratio**: analogous to R² for GLMs, measured against the GBM's predictions.
# MAGIC - **Max segment deviation**: the worst-case relative error between GBM and GLM across all cells of the factor table. This is the most operationally relevant check — if it exceeds 10%, individual cells will misprice materially against the GBM.

# COMMAND ----------

report = surrogate.report()

print("=" * 52)
print("Distillation validation report")
print("=" * 52)
print(report.metrics.summary())
print()
print(f"Factor tables available for: {list(report.factor_tables.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Double-lift chart: GBM vs GLM
# MAGIC
# MAGIC The lift chart ranks policies by the GBM's predicted frequency (deciles), then shows the mean GBM prediction and mean GLM prediction in each decile. If the distillation is faithful, the two curves should track closely. Divergence at the extremes (decile 1 and decile 10) is the most common failure mode — the GLM may not fully capture the tail behaviour of the GBM.

# COMMAND ----------

print("Double-lift chart (GBM pseudo-predictions vs GLM surrogate):")
print()
print(report.lift_chart)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Factor tables
# MAGIC
# MAGIC Each factor table has three columns:
# MAGIC - `level`: the bin label (e.g. `[25.0, 40.0)`) or category value
# MAGIC - `log_coefficient`: the raw GLM coefficient on the log scale (0.0 for the base level)
# MAGIC - `relativity`: `exp(log_coefficient)` — the multiplicative factor
# MAGIC
# MAGIC The base level always has `relativity = 1.0`. All other levels are expressed relative to it.
# MAGIC
# MAGIC This is the same format as `exp(beta)` output from Emblem or Radar.

# COMMAND ----------

for feature_name, ft in report.factor_tables.items():
    print(f"\nFactor table: {feature_name}")
    print("-" * 55)
    print(ft)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. R² of distilled GLM vs original GBM predictions
# MAGIC
# MAGIC R² here measures how well the GLM surrogate reproduces the GBM's predicted values — not the actual claims. If the GBM's predictions explain variance in the actual claims, and the GLM captures most of that variance, then the factor tables are a faithful representation of the GBM.
# MAGIC
# MAGIC A high R² alongside a high Gini ratio gives confidence that the distillation has worked: the GLM is producing numbers that are close to what the GBM would produce, not just tracking the central tendency while missing the extremes.

# COMMAND ----------

from sklearn.metrics import r2_score

glm_preds = surrogate._glm_predictions
r2 = r2_score(gbm_preds, glm_preds)

print(f"R² (distilled GLM vs GBM predictions): {r2:.4f}")
print()

# Sanity check: mean predictions should be close
print(f"Mean GBM prediction:  {gbm_preds.mean():.6f}")
print(f"Mean GLM prediction:  {glm_preds.mean():.6f}")
print(f"Ratio (GLM/GBM mean): {glm_preds.mean() / gbm_preds.mean():.4f}")

# Percentile comparison
pcts = [5, 25, 50, 75, 95]
print(f"\nPrediction distribution comparison:")
print(f"  {'Pct':>5s}  {'GBM':>10s}  {'GLM':>10s}  {'Diff%':>8s}")
for p in pcts:
    g = float(np.percentile(gbm_preds, p))
    l = float(np.percentile(glm_preds, p))
    diff_pct = (l - g) / g * 100
    print(f"  {p:>5d}  {g:>10.6f}  {l:>10.6f}  {diff_pct:>+7.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare distilled relativities to true DGP
# MAGIC
# MAGIC The NCD factor table should show a monotone downward pattern — more NCD years should mean lower predicted frequency. The area factor table should show A < B < C < D < E. These are sanity checks that the distilled GLM has learned the right direction of effects, even if the exact magnitudes differ from the DGP (the GBM was fit on noisy realisations of claims, not the true DGP directly).

# COMMAND ----------

print("NCD years factor table (expect monotone decrease):")
print(report.factor_tables.get("ncd_years", "not found"))

print()
print("Area factor table (expect A < B < C < D < E):")
print(report.factor_tables.get("area", "not found"))

print()
print("Convictions factor table (expect >1.0 for convicted):")
print(report.factor_tables.get("has_convictions", "not found"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC What we showed:
# MAGIC
# MAGIC - **CatBoost Poisson model** trained on 30,000-policy synthetic portfolio with known DGP
# MAGIC - **`SurrogateGLM`** distilled the GBM into a multiplicative Poisson GLM using pseudo-predictions
# MAGIC - **Validation metrics**: Gini ratio, deviance ratio, segment deviation — the three numbers a pricing actuary needs before loading factor tables into a rating engine
# MAGIC - **Double-lift chart**: visual check that the GLM tracks the GBM across the predicted frequency distribution
# MAGIC - **Factor tables**: one DataFrame per variable, in the same `level / log_coefficient / relativity` format as Emblem or Radar output
# MAGIC - **R²**: measures how faithfully the distilled GLM reproduces the GBM's predictions
# MAGIC
# MAGIC **Production checklist before loading factor tables into a rating engine:**
# MAGIC 1. Gini ratio >= 0.90 (above 0.95 is excellent)
# MAGIC 2. Max segment deviation < 10% (single cells should not deviate materially)
# MAGIC 3. Direction of effects is consistent with domain knowledge (monotone NCD, higher area hazard in urban areas)
# MAGIC 4. Double-lift chart tracks reasonably across all deciles — inspect the tails specifically
# MAGIC 5. Mean prediction ratio is close to 1.0 (no systematic over/underpricing from the distillation)
