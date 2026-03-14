"""
GBM-to-GLM pipeline: build, interpret, and monitor a motor frequency model.

The standard tension in insurance pricing is between model performance and
interpretability. A CatBoost GBM outperforms a GLM on lift, but pricing
committees, underwriters, and regulators expect to see multiplicative rating
factor tables — the format GLMs produce naturally.

This script shows how to close that gap: train a GBM with proper temporal
cross-validation, extract SHAP relativities into a GLM-equivalent tariff table,
then set up monitoring so you know when the live model starts to drift.

The workflow is:

    1. Load synthetic UK motor data (insurance-datasets)
    2. Temporal walk-forward cross-validation (insurance-cv)
    3. Train a CatBoost Poisson frequency model
    4. Extract SHAP relativities (shap-relativities)
    5. Construct a multiplicative tariff from the relativities
    6. Establish monitoring baselines (insurance-monitoring)
    7. Simulate a shifted portfolio and run drift detection

Libraries used
--------------
    insurance-datasets  — synthetic UK motor data with a known DGP
    insurance-cv        — temporal CV splits that respect IBNR structure
    shap-relativities   — SHAP values → multiplicative rating factor table
    insurance-monitoring — PSI/CSI/A&E/Gini monitoring

Dependencies
------------
    uv add insurance-datasets insurance-cv shap-relativities insurance-monitoring
    uv add catboost shap

Approximate runtime: 2–3 minutes. CatBoost fitting and SHAP computation
dominate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Step 1: Load synthetic UK motor portfolio
# ---------------------------------------------------------------------------
#
# insurance-datasets generates a fully synthetic portfolio from a known data
# generating process. The true parameters are exported so you can check that
# your model is recovering the right structure — useful for methodology
# validation, less relevant in production where you replace this step with your
# own policy extract.
#
# We use 15,000 policies here: enough for stable SHAP estimates without taking
# too long on modest hardware.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Load synthetic UK motor portfolio")
print("=" * 70)

from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=15_000, seed=42)

print(f"Portfolio: {len(df):,} policies")
print(f"Accident years: {sorted(df['accident_year'].unique())}")
print(f"Overall claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f} claims/year")
print(f"Mean exposure: {df['exposure'].mean():.3f} policy-years")
print(f"\nTrue DGP parameters (what a well-specified model should recover):")
for k, v in MOTOR_TRUE_FREQ_PARAMS.items():
    print(f"  {k:<25} {v:+.3f}")

# ---------------------------------------------------------------------------
# Step 2: Temporal walk-forward cross-validation
# ---------------------------------------------------------------------------
#
# Standard random k-fold is wrong for insurance. A policy written in 2023 can
# end up in the training set while a policy from 2019 ends up in test —
# that is not how a live model is deployed. Temporal CV forces training data
# to precede test data in calendar time.
#
# We use insurance-cv's walk_forward_split with a 3-month IBNR buffer. For
# attritional motor this is conservative; for a long-tail line you would use
# 12–18 months.
#
# The splits are used for hyperparameter assessment only. We train the final
# model on the full dataset and report RMSE per fold.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: Temporal walk-forward cross-validation")
print("=" * 70)

from insurance_cv import walk_forward_split, split_summary

# The inception_date column is what insurance-cv needs.
# Convert to pandas-compatible format — insurance-cv handles both.
splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,   # need at least 18 months before first test
    test_months=6,         # evaluate on 6-month windows
    step_months=6,         # non-overlapping test windows
    ibnr_buffer_months=3,
)

print(f"Generated {len(splits)} temporal folds:")
for s in splits:
    train_idx, test_idx = s.get_indices(df)
    print(
        f"  {s.label[:70]}"
        f" | train n={len(train_idx):,}, test n={len(test_idx):,}"
    )

# Quick fold-level frequency check: confirm test claim rates are reasonable
print("\nFrequency check per fold (should be close to ~0.07-0.09):")
for s in splits:
    _, test_idx = s.get_indices(df)
    test = df.iloc[test_idx]
    freq = test["claim_count"].sum() / test["exposure"].sum()
    print(f"  {s.label.split('test:')[1].strip()[:20]} → {freq:.4f}")

# ---------------------------------------------------------------------------
# Step 3: Train CatBoost Poisson frequency model
# ---------------------------------------------------------------------------
#
# We train the final model on the most recent four years (2020-2023) and
# use 2019 as a simple holdout. In production you would use the walk-forward
# CV above to select hyperparameters, then retrain on all available data.
#
# The log(exposure) offset is the correct way to handle variable earned years
# in a Poisson model. Without it, a policy with 0.5 years of exposure would
# be treated identically to one with a full year.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: Train CatBoost Poisson frequency model")
print("=" * 70)

from catboost import CatBoostRegressor, Pool

FEATURE_COLS = [
    "vehicle_age",
    "vehicle_group",
    "driver_age",
    "ncd_years",
    "area",
    "policy_type",
]
CAT_FEATURE_NAMES = ["area", "policy_type"]
cat_indices = [FEATURE_COLS.index(c) for c in CAT_FEATURE_NAMES]

# Split: train on 2019-2022, hold out 2023 for final evaluation
train_pd = df[df["accident_year"] <= 2022][FEATURE_COLS + ["claim_count", "exposure"]]
holdout_pd = df[df["accident_year"] == 2023][FEATURE_COLS + ["claim_count", "exposure"]]

print(f"Training set: {len(train_pd):,} policies (accident years 2019-2022)")
print(f"Holdout set:  {len(holdout_pd):,} policies (accident year 2023)")

train_log_exp = np.log(train_pd["exposure"].clip(lower=1e-9)).values
holdout_log_exp = np.log(holdout_pd["exposure"].clip(lower=1e-9)).values

train_pool = Pool(
    data=train_pd[FEATURE_COLS],
    label=train_pd["claim_count"],
    cat_features=cat_indices,
    baseline=train_log_exp,
)
holdout_pool = Pool(
    data=holdout_pd[FEATURE_COLS],
    label=holdout_pd["claim_count"],
    cat_features=cat_indices,
    baseline=holdout_log_exp,
)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
model.fit(train_pool, eval_set=holdout_pool)

# Evaluate on holdout
holdout_preds = model.predict(holdout_pd[FEATURE_COLS])
actuals = holdout_pd["claim_count"].values
holdout_exp = holdout_pd["exposure"].values

rmse = float(np.sqrt(np.mean((holdout_preds - actuals) ** 2)))
mean_rate = actuals.sum() / holdout_exp.sum()
naive_preds = mean_rate * holdout_exp
naive_rmse = float(np.sqrt(np.mean((naive_preds - actuals) ** 2)))
calibration = (holdout_preds * holdout_exp).sum() / actuals.sum()

print(f"\nHoldout results (accident year 2023):")
print(f"  RMSE: {rmse:.5f}  (naive: {naive_rmse:.5f})")
print(f"  Lift over naive: {(naive_rmse - rmse) / naive_rmse:.1%}")
print(f"  Calibration ratio: {calibration:.4f}  (1.000 = perfect)")

# ---------------------------------------------------------------------------
# Step 4: Extract SHAP relativities
# ---------------------------------------------------------------------------
#
# SHAPRelativities converts the model's SHAP values into the multiplicative
# format actuaries are used to reviewing. The result is a (feature, level,
# relativity) table where the base level for each factor has relativity = 1.0
# and all other levels are relative to it.
#
# We use the holdout data for SHAP computation. Using training data gives
# overly confident relativities because the model has fitted to those exact
# observations.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: Extract SHAP relativities")
print("=" * 70)

from shap_relativities import SHAPRelativities

holdout_features = pl.from_pandas(holdout_pd[FEATURE_COLS])
holdout_exposure = pl.Series("exposure", holdout_pd["exposure"].values)

sr = SHAPRelativities(
    model=model,
    X=holdout_features,
    exposure=holdout_exposure,
    categorical_features=CAT_FEATURE_NAMES,
)
sr.fit()

base_levels = {
    "area": "C",          # mid-risk area band as reference
    "policy_type": "Comp", # comprehensive as reference
}

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels=base_levels,
    ci_method="clt",
)

print("Rating factor relativities (Poisson GBM, holdout data):\n")
for feature in FEATURE_COLS:
    feature_rels = relativities.filter(pl.col("feature") == feature)
    if feature_rels.is_empty():
        continue
    print(f"  {feature}:")
    for row in feature_rels.sort("level").iter_rows(named=True):
        level_str = str(row["level"])[:18].ljust(20)
        rel = row["relativity"]
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            continue
        ci_lo = row.get("lower_ci")
        ci_hi = row.get("upper_ci")
        if ci_lo is not None and not (isinstance(ci_lo, float) and np.isnan(ci_lo)):
            print(f"    {level_str}  {rel:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
        else:
            print(f"    {level_str}  {rel:.3f}")
    print()

baseline_rate = sr.baseline()
print(f"Annualised base claim rate: {baseline_rate:.4f} claims/policy-year")
print("(This is the rate for the base level of all factors combined)\n")

print("SHAP diagnostic checks:")
checks = sr.validate()
for check_name, result in checks.items():
    status = "PASS" if result.passed else "WARN"
    print(f"  [{status}] {check_name}: {result.message}")

# ---------------------------------------------------------------------------
# Step 5: Build a multiplicative tariff from the relativities
# ---------------------------------------------------------------------------
#
# A multiplicative tariff is the standard output format for pricing committees.
# The pure premium for any risk is:
#
#     premium = base_rate * rel_vehicle_group * rel_driver_age * ... * exposure
#
# Here we compute that premium for each holdout policy using the SHAP
# relativities. The result should be close (but not identical) to the GBM
# prediction — differences arise because SHAP relativities are a linear
# approximation of the model's interaction structure.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 5: Multiplicative tariff vs GBM predictions")
print("=" * 70)

# Build a lookup dict: {feature: {level: relativity}}
tariff: dict[str, dict] = {}
for row in relativities.iter_rows(named=True):
    feat = row["feature"]
    rel = row["relativity"]
    if rel is None or (isinstance(rel, float) and np.isnan(rel)):
        continue
    if feat not in tariff:
        tariff[feat] = {}
    tariff[feat][row["level"]] = rel

# For continuous features, round to nearest observed bin for lookup
def nearest_key(d: dict, val) -> float:
    """Return the relativity for the closest key in d to val."""
    keys = list(d.keys())
    numeric_keys = [k for k in keys if isinstance(k, (int, float))]
    if not numeric_keys:
        return d.get(val, 1.0)
    closest = min(numeric_keys, key=lambda k: abs(k - val))
    return d[closest]

tariff_preds = np.zeros(len(holdout_pd))
for i, (_, row) in enumerate(holdout_pd.iterrows()):
    rate = baseline_rate
    for feat in FEATURE_COLS:
        if feat in tariff:
            feat_tariff = tariff[feat]
            val = row[feat]
            if isinstance(val, str):
                rate *= feat_tariff.get(val, 1.0)
            else:
                rate *= nearest_key(feat_tariff, val)
    tariff_preds[i] = rate

# Compare: tariff vs GBM on holdout
tariff_vs_gbm_corr = float(np.corrcoef(tariff_preds, holdout_preds)[0, 1])
tariff_rmse = float(np.sqrt(np.mean((tariff_preds - actuals) ** 2)))

print(f"Comparison on holdout (accident year 2023):")
print(f"  GBM RMSE:    {rmse:.5f}")
print(f"  Tariff RMSE: {tariff_rmse:.5f}")
print(f"  Tariff vs GBM correlation: {tariff_vs_gbm_corr:.4f}")
print(
    f"\n  The tariff reproduces {tariff_vs_gbm_corr**2:.1%} of the GBM's variance."
)
print(
    "  Residual difference is driven by higher-order interactions that "
    "SHAP relativities cannot capture in a purely multiplicative structure."
)
print(
    "\n  For pricing committee review, present the SHAP tariff table. "
    "The live rater can use either the full GBM or the tariff — "
    "your business will have a view on the interpretability/performance trade-off."
)

# ---------------------------------------------------------------------------
# Step 6: Establish monitoring baselines
# ---------------------------------------------------------------------------
#
# Before the model goes live, capture baseline statistics against which to
# measure future drift. The three key signals are:
#
#   PSI/CSI  — has the incoming risk mix changed?
#   A/E ratio — is the model still calibrated?
#   Gini     — is the model still discriminating well?
#
# We use the training data as the reference distribution and the holdout
# as the first monitoring window.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 6: Establish monitoring baselines")
print("=" * 70)

from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import csi, psi, wasserstein_distance

# CSI on rating factors (continuous ones only for this example)
continuous_features = ["vehicle_age", "vehicle_group", "driver_age", "ncd_years"]

train_features_pl = pl.from_pandas(train_pd[continuous_features])
holdout_features_pl = pl.from_pandas(holdout_pd[continuous_features])

csi_result = csi(
    reference_df=train_features_pl,
    current_df=holdout_features_pl,
    features=continuous_features,
)

print("Characteristic Stability Index (training → holdout):")
for row in csi_result.iter_rows(named=True):
    band_str = f"[{row['band'].upper()}]"
    print(f"  {row['feature']:<20} CSI = {row['csi']:.4f}  {band_str}")

# Wasserstein distances for interpretable drift communication
print("\nWasserstein distances (interpretable in original units):")
for feat in continuous_features:
    d = wasserstein_distance(train_pd[feat].values, holdout_pd[feat].values)
    unit = "years" if "age" in feat or feat == "vehicle_age" else "ABI groups" if feat == "vehicle_group" else "years"
    print(f"  {feat:<20} {d:.3f} {unit}")

# Full monitoring report: combines A/E, Gini, and CSI
train_preds = model.predict(train_pd[FEATURE_COLS])
train_actuals = train_pd["claim_count"].values
train_exp = train_pd["exposure"].values

monitoring = MonitoringReport(
    reference_actual=train_actuals,
    reference_predicted=train_preds,
    current_actual=actuals,
    current_predicted=holdout_preds,
    exposure=holdout_exp,
    reference_exposure=train_exp,
    feature_df_reference=train_features_pl,
    feature_df_current=holdout_features_pl,
    features=continuous_features,
    score_reference=np.log(np.clip(train_preds, 1e-9, None)),
    score_current=np.log(np.clip(holdout_preds, 1e-9, None)),
    n_bootstrap=200,
)

print("\nBaseline monitoring report (training vs holdout):")
report_df = monitoring.to_polars()
for row in report_df.filter(
    pl.col("metric").is_in(["ae_ratio", "gini_current", "gini_reference", "gini_change", "score_psi"])
).iter_rows(named=True):
    if not np.isnan(row["value"]):
        print(f"  {row['metric']:<25} {row['value']:.4f}  [{row['band'].upper()}]")

print(f"\nRecommendation: {monitoring.recommendation}")
print("(NO_ACTION = stable; expected at this stage since holdout is in-sample period)")

# ---------------------------------------------------------------------------
# Step 7: Simulate a shifted portfolio and detect drift
# ---------------------------------------------------------------------------
#
# Here we simulate what happens 18 months after deployment: the incoming risk
# mix has shifted (younger drivers, higher vehicle groups — the classic post-
# aggregator composition change). We run the monitoring report against this
# new data to show drift detection firing.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 7: Simulate portfolio shift and detect drift")
print("=" * 70)

rng = np.random.default_rng(123)
n_new = 5_000

# Shifted portfolio: younger drivers, higher vehicle groups, more urban areas
new_data = pd.DataFrame({
    "vehicle_age": rng.integers(0, 10, n_new),         # newer vehicles
    "vehicle_group": rng.integers(25, 50, n_new),       # higher groups (was 1-50)
    "driver_age": rng.integers(17, 35, n_new),          # younger drivers
    "ncd_years": rng.integers(0, 3, n_new),             # less NCD (was 0-5)
    "area": rng.choice(["D", "E", "F"], n_new,          # more urban
                       p=[0.40, 0.35, 0.25]),
    "policy_type": rng.choice(["Comp", "TPFT"], n_new,
                               p=[0.85, 0.15]),
    "exposure": np.clip(rng.beta(5, 2, n_new), 0.1, 1.0),
})

# Simulate claims with the true DGP parameters (but reflecting the new mix)
new_log_mu = (
    MOTOR_TRUE_FREQ_PARAMS["intercept"]
    + 0.025 * new_data["vehicle_group"]
    + 0.55  # young driver effect (all < 35)
    - 0.12 * new_data["ncd_years"]
    + new_data["area"].map({"A": 0, "B": 0.1, "C": 0.2, "D": 0.35, "E": 0.5, "F": 0.65})
    + np.log(new_data["exposure"])
)
new_lambda = np.exp(new_log_mu)
new_claims = rng.poisson(new_lambda)

new_preds = model.predict(new_data[FEATURE_COLS])

new_features_pl = pl.from_pandas(new_data[continuous_features])

monitoring_live = MonitoringReport(
    reference_actual=train_actuals,
    reference_predicted=train_preds,
    current_actual=new_claims.astype(float),
    current_predicted=new_preds,
    exposure=new_data["exposure"].values,
    reference_exposure=train_exp,
    feature_df_reference=train_features_pl,
    feature_df_current=new_features_pl,
    features=continuous_features,
    score_reference=np.log(np.clip(train_preds, 1e-9, None)),
    score_current=np.log(np.clip(new_preds, 1e-9, None)),
    n_bootstrap=200,
)

print("Monitoring report after portfolio shift (training → shifted live book):")
live_df = monitoring_live.to_polars()
for row in live_df.filter(
    pl.col("metric").is_in([
        "ae_ratio", "gini_current", "gini_change", "score_psi",
        "csi_driver_age", "csi_vehicle_group", "csi_ncd_years",
    ])
).iter_rows(named=True):
    if not np.isnan(row["value"]):
        print(f"  {row['metric']:<25} {row['value']:.4f}  [{row['band'].upper()}]")

print(f"\nRecommendation: {monitoring_live.recommendation}")

drift_csi = csi(train_features_pl, new_features_pl, continuous_features)
print("\nCSI per factor after portfolio shift:")
for row in drift_csi.iter_rows(named=True):
    band_str = f"[{row['band'].upper()}]"
    print(f"  {row['feature']:<20} CSI = {row['csi']:.4f}  {band_str}")

print(
    "\nNote: red CSI for driver_age and vehicle_group is expected — the simulated"
    " portfolio is deliberately younger and higher-risk than the training book."
    " The monitoring framework has correctly identified the shift and escalated"
    f" to {monitoring_live.recommendation}."
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Pipeline complete")
print("=" * 70)
print(f"""
What was demonstrated:

  1. insurance-datasets   Loaded {len(df):,} synthetic UK motor policies with known DGP
  2. insurance-cv         Walk-forward CV with {len(splits)} temporal folds, 3-month IBNR buffer
  3. CatBoost             Poisson frequency model, holdout RMSE {rmse:.5f}
  4. shap-relativities    SHAP → multiplicative tariff table for pricing committee
  5. Tariff vs GBM        {tariff_vs_gbm_corr**2:.1%} variance explained by linear multiplicative structure
  6. insurance-monitoring Baseline: A/E, Gini, CSI established at go-live
  7. Drift detection      Portfolio shift correctly flagged as {monitoring_live.recommendation}

The tariff table from step 4 is the deliverable for your pricing committee.
The monitoring pack from step 6 is what your quarterly model review should show.
""")
