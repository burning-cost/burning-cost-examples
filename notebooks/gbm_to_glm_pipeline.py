# Databricks notebook source

# MAGIC %md
# MAGIC # GBM-to-GLM Pipeline: Build, Interpret, and Monitor a Motor Frequency Model
# MAGIC
# MAGIC The standard tension in insurance pricing is between model performance and interpretability. A CatBoost GBM outperforms a GLM on lift, but pricing committees, underwriters, and regulators expect to see multiplicative rating factor tables — the format GLMs produce naturally.
# MAGIC
# MAGIC This notebook closes that gap: train a GBM with proper temporal cross-validation, extract SHAP relativities into a GLM-equivalent tariff table, then set up monitoring so you know when the live model starts to drift.
# MAGIC
# MAGIC **Libraries:**
# MAGIC - `insurance-datasets` — synthetic UK motor data with known DGP
# MAGIC - `insurance-cv` — temporal CV splits that respect IBNR structure
# MAGIC - `shap-relativities` — SHAP values to multiplicative rating factor table
# MAGIC - `insurance-monitoring` — PSI/CSI/A&E/Gini monitoring

# COMMAND ----------

# MAGIC %pip install insurance-datasets insurance-cv shap-relativities insurance-monitoring catboost shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

import numpy as _np_compat
if not hasattr(_np_compat, 'trapezoid'):
    _np_compat.trapezoid = _np_compat.trapz

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Synthetic UK Motor Portfolio
# MAGIC
# MAGIC `insurance-datasets` generates a fully synthetic portfolio from a known data generating process. The true parameters are exported so you can check that your model is recovering the right structure — useful for methodology validation. In production, replace this with your own policy extract.

# COMMAND ----------

from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=15_000, seed=42)

print(f"Portfolio: {len(df):,} policies")
print(f"Accident years: {sorted(df['accident_year'].unique())}")
print(f"Overall claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f} claims/year")
print(f"\nTrue DGP parameters:")
for k, v in MOTOR_TRUE_FREQ_PARAMS.items():
    print(f"  {k:<25} {v:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Temporal Walk-Forward Cross-Validation
# MAGIC
# MAGIC Standard random k-fold is wrong for insurance. A policy written in 2023 can end up in the training set while a policy from 2019 ends up in test — that is not how a live model is deployed. `insurance-cv`'s `walk_forward_split` forces training data to precede test data in calendar time, with a 3-month IBNR buffer.

# COMMAND ----------

from insurance_cv import walk_forward_split

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)

print(f"Generated {len(splits)} temporal folds:")
for s in splits:
    train_idx, test_idx = s.get_indices(df)
    print(f"  {s.label[:70]} | train n={len(train_idx):,}, test n={len(test_idx):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train CatBoost Poisson Frequency Model
# MAGIC
# MAGIC Train on 2019-2022, hold out 2023 for final evaluation. The `log(exposure)` offset is the correct way to handle variable earned years in a Poisson model.

# COMMAND ----------

from catboost import CatBoostRegressor, Pool

FEATURE_COLS = ["vehicle_age", "vehicle_group", "driver_age", "ncd_years", "area", "policy_type"]
CAT_FEATURE_NAMES = ["area", "policy_type"]
cat_indices = [FEATURE_COLS.index(c) for c in CAT_FEATURE_NAMES]

train_pd = df[df["accident_year"] <= 2022][FEATURE_COLS + ["claim_count", "exposure"]]
holdout_pd = df[df["accident_year"] == 2023][FEATURE_COLS + ["claim_count", "exposure"]]

print(f"Training set: {len(train_pd):,} policies (2019-2022)")
print(f"Holdout set:  {len(holdout_pd):,} policies (2023)")

train_log_exp = np.log(train_pd["exposure"].clip(lower=1e-9)).values
holdout_log_exp = np.log(holdout_pd["exposure"].clip(lower=1e-9)).values

train_pool = Pool(
    data=train_pd[FEATURE_COLS], label=train_pd["claim_count"],
    cat_features=cat_indices, baseline=train_log_exp,
)
holdout_pool = Pool(
    data=holdout_pd[FEATURE_COLS], label=holdout_pd["claim_count"],
    cat_features=cat_indices, baseline=holdout_log_exp,
)

model = CatBoostRegressor(
    loss_function="Poisson", iterations=400, depth=4,
    learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, verbose=0,
)
model.fit(train_pool, eval_set=holdout_pool)

holdout_preds = model.predict(holdout_pd[FEATURE_COLS])
actuals = holdout_pd["claim_count"].values
holdout_exp = holdout_pd["exposure"].values

rmse = float(np.sqrt(np.mean((holdout_preds - actuals) ** 2)))
mean_rate = actuals.sum() / holdout_exp.sum()
naive_preds = mean_rate * holdout_exp
naive_rmse = float(np.sqrt(np.mean((naive_preds - actuals) ** 2)))
calibration = (holdout_preds * holdout_exp).sum() / actuals.sum()

print(f"\nHoldout (2023): RMSE={rmse:.5f}, lift over naive={((naive_rmse - rmse) / naive_rmse):.1%}, cal={calibration:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Extract SHAP Relativities
# MAGIC
# MAGIC `SHAPRelativities` converts the model's SHAP values into the multiplicative format actuaries review: a (feature, level, relativity) table where the base level for each factor has relativity = 1.0.

# COMMAND ----------

from shap_relativities import SHAPRelativities

holdout_features = pl.from_pandas(holdout_pd[FEATURE_COLS])
holdout_exposure = pl.Series("exposure", holdout_pd["exposure"].values)

sr = SHAPRelativities(
    model=model, X=holdout_features, exposure=holdout_exposure,
    categorical_features=CAT_FEATURE_NAMES,
)
sr.fit()

base_levels = {"area": "C", "policy_type": "Comp"}
relativities = sr.extract_relativities(
    normalise_to="base_level", base_levels=base_levels, ci_method="clt",
)

print("Rating factor relativities (Poisson GBM, holdout data):\n")
for feature in FEATURE_COLS:
    feature_rels = relativities.filter(pl.col("feature") == feature)
    if feature_rels.is_empty():
        continue
    print(f"  {feature}:")
    for row in feature_rels.sort("level").iter_rows(named=True):
        rel = row["relativity"]
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            continue
        ci_lo = row.get("lower_ci")
        ci_hi = row.get("upper_ci")
        level_str = str(row["level"])[:18].ljust(20)
        if ci_lo is not None and not (isinstance(ci_lo, float) and np.isnan(ci_lo)):
            print(f"    {level_str}  {rel:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
        else:
            print(f"    {level_str}  {rel:.3f}")
    print()

baseline_rate = sr.baseline()
print(f"Annualised base claim rate: {baseline_rate:.4f} claims/policy-year")

checks = sr.validate()
print("SHAP diagnostic checks:")
for check_name, result in checks.items():
    print(f"  [{'PASS' if result.passed else 'WARN'}] {check_name}: {result.message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Multiplicative Tariff vs GBM Predictions
# MAGIC
# MAGIC A multiplicative tariff is the standard output for pricing committees. The pure premium for any risk is: `premium = base_rate * rel_vehicle_group * rel_driver_age * ... * exposure`. Here we compute that premium for each holdout policy and compare to the GBM predictions.

# COMMAND ----------

# Build lookup: {feature: {level: relativity}}
tariff: dict = {}
for row in relativities.iter_rows(named=True):
    feat, rel = row["feature"], row["relativity"]
    if rel is None or (isinstance(rel, float) and np.isnan(rel)):
        continue
    if feat not in tariff:
        tariff[feat] = {}
    tariff[feat][row["level"]] = rel

def nearest_key(d: dict, val) -> float:
    keys = [k for k in d.keys() if isinstance(k, (int, float))]
    if not keys:
        return d.get(val, 1.0)
    return d[min(keys, key=lambda k: abs(k - val))]

tariff_preds = np.zeros(len(holdout_pd))
for i, (_, row) in enumerate(holdout_pd.iterrows()):
    rate = baseline_rate
    for feat in FEATURE_COLS:
        if feat in tariff:
            val = row[feat]
            rate *= tariff[feat].get(val, 1.0) if isinstance(val, str) else nearest_key(tariff[feat], val)
    tariff_preds[i] = rate

tariff_vs_gbm_corr = float(np.corrcoef(tariff_preds, holdout_preds)[0, 1])
tariff_rmse = float(np.sqrt(np.mean((tariff_preds - actuals) ** 2)))

print(f"GBM RMSE:    {rmse:.5f}")
print(f"Tariff RMSE: {tariff_rmse:.5f}")
print(f"Tariff vs GBM correlation: {tariff_vs_gbm_corr:.4f}")
print(f"Tariff explains {tariff_vs_gbm_corr**2:.1%} of the GBM's variance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Establish Monitoring Baselines
# MAGIC
# MAGIC Before the model goes live, capture baseline statistics against which to measure future drift: PSI/CSI (has the risk mix changed?), A/E ratio (is the model still calibrated?), Gini (is the model still discriminating well?).

# COMMAND ----------

from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import csi

continuous_features = ["vehicle_age", "vehicle_group", "driver_age", "ncd_years"]
train_features_pl = pl.from_pandas(train_pd[continuous_features])
holdout_features_pl = pl.from_pandas(holdout_pd[continuous_features])

train_preds = model.predict(train_pd[FEATURE_COLS])
train_actuals = train_pd["claim_count"].values
train_exp = train_pd["exposure"].values

monitoring = MonitoringReport(
    reference_actual=train_actuals, reference_predicted=train_preds,
    current_actual=actuals, current_predicted=holdout_preds,
    exposure=holdout_exp, reference_exposure=train_exp,
    feature_df_reference=train_features_pl, feature_df_current=holdout_features_pl,
    features=continuous_features,
    score_reference=np.log(np.clip(train_preds, 1e-9, None)),
    score_current=np.log(np.clip(holdout_preds, 1e-9, None)),
    n_bootstrap=200,
)

print("Baseline monitoring (training vs holdout):")
report_df = monitoring.to_polars()
for row in report_df.filter(
    pl.col("metric").is_in(["ae_ratio", "gini_current", "gini_change", "score_psi"])
).iter_rows(named=True):
    if not np.isnan(row["value"]):
        print(f"  {row['metric']:<25} {row['value']:.4f}  [{row['band'].upper()}]")
print(f"\nRecommendation: {monitoring.recommendation}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Simulate Portfolio Shift and Detect Drift
# MAGIC
# MAGIC We simulate what happens 18 months after deployment: younger drivers, higher vehicle groups, more urban areas — the classic post-aggregator composition change.

# COMMAND ----------

rng = np.random.default_rng(123)
n_new = 5_000

new_data = pd.DataFrame({
    "vehicle_age": rng.integers(0, 10, n_new),
    "vehicle_group": rng.integers(25, 50, n_new),
    "driver_age": rng.integers(17, 35, n_new),
    "ncd_years": rng.integers(0, 3, n_new),
    "area": rng.choice(["D", "E", "F"], n_new, p=[0.40, 0.35, 0.25]),
    "policy_type": rng.choice(["Comp", "TPFT"], n_new, p=[0.85, 0.15]),
    "exposure": np.clip(rng.beta(5, 2, n_new), 0.1, 1.0),
})

new_log_mu = (
    MOTOR_TRUE_FREQ_PARAMS["intercept"]
    + 0.025 * new_data["vehicle_group"]
    + 0.55
    - 0.12 * new_data["ncd_years"]
    + new_data["area"].map({"A": 0, "B": 0.1, "C": 0.2, "D": 0.35, "E": 0.5, "F": 0.65})
    + np.log(new_data["exposure"])
)
new_claims = rng.poisson(np.exp(new_log_mu))
new_preds = model.predict(new_data[FEATURE_COLS])
new_features_pl = pl.from_pandas(new_data[continuous_features])

monitoring_live = MonitoringReport(
    reference_actual=train_actuals, reference_predicted=train_preds,
    current_actual=new_claims.astype(float), current_predicted=new_preds,
    exposure=new_data["exposure"].values, reference_exposure=train_exp,
    feature_df_reference=train_features_pl, feature_df_current=new_features_pl,
    features=continuous_features,
    score_reference=np.log(np.clip(train_preds, 1e-9, None)),
    score_current=np.log(np.clip(new_preds, 1e-9, None)),
    n_bootstrap=200,
)

print("Monitoring after portfolio shift:")
live_df = monitoring_live.to_polars()
for row in live_df.filter(
    pl.col("metric").is_in(["ae_ratio", "gini_current", "gini_change", "score_psi"])
).iter_rows(named=True):
    if not np.isnan(row["value"]):
        print(f"  {row['metric']:<25} {row['value']:.4f}  [{row['band'].upper()}]")

print(f"\nRecommendation: {monitoring_live.recommendation}")

drift_csi = csi(train_features_pl, new_features_pl, continuous_features)
print("\nCSI per factor after portfolio shift:")
for row in drift_csi.iter_rows(named=True):
    print(f"  {row['feature']:<20} CSI = {row['csi']:.4f}  [{row['band'].upper()}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was demonstrated:**
# MAGIC 1. `insurance-datasets` — 15,000 synthetic UK motor policies with known DGP
# MAGIC 2. `insurance-cv` — walk-forward CV with IBNR buffer (correct temporal split)
# MAGIC 3. CatBoost Poisson GBM — lift measurement vs naive baseline
# MAGIC 4. `shap-relativities` — SHAP to multiplicative tariff table for pricing committee
# MAGIC 5. Tariff vs GBM comparison — quantifies information loss from linearisation
# MAGIC 6. `insurance-monitoring` — baseline A/E, Gini, CSI at go-live
# MAGIC 7. Drift detection — portfolio shift correctly flagged for action
# MAGIC
# MAGIC The SHAP tariff table from Step 4 is the deliverable for your pricing committee. The monitoring pack from Step 6 is what your quarterly model review should show.
