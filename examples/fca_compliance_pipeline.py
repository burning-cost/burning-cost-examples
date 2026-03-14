"""
FCA compliance pipeline: fairness audit and model validation for a pricing model.

UK pricing actuaries operate under two overlapping regulatory frameworks that
require documented evidence of compliance:

  FCA Consumer Duty (PRIN 2A): firms must demonstrate good outcomes for all
  customer groups, including monitoring for differential outcomes by
  characteristics such as gender or socioeconomic group.

  PRA SS1/23: model risk management requirements that mandate formal validation
  of all material risk models, including sensitivity analysis and out-of-time
  performance tests.

This script runs a pricing model through both frameworks using the
insurance-fairness and insurance-validation libraries. The output is a
structured audit record — the kind of thing you attach to a pricing committee
paper or model validation submission.

The workflow is:

    1. Load synthetic motor data and train a quick CatBoost model
    2. Proxy discrimination audit (insurance-fairness)
    3. PRA SS1/23 model validation report (insurance-validation)
    4. Interpret results and identify any action items

Libraries used
--------------
    insurance-datasets    — synthetic UK motor data
    insurance-fairness    — FCA Consumer Duty proxy discrimination audit
    insurance-validation  — PRA SS1/23 model validation report

Dependencies
------------
    uv add insurance-datasets insurance-fairness insurance-validation catboost shap

Approximate runtime: 3–5 minutes. The proxy detection step (which fits small
CatBoost models to test each rating factor as a proxy) takes the most time.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Step 1: Load data and train the model under audit
# ---------------------------------------------------------------------------
#
# In a real compliance exercise the model already exists — you are auditing
# something that is live or about to go live. Here we train a CatBoost
# frequency model so the rest of the script has something to audit.
#
# We add synthetic demographic proxies to the dataset. Real motor portfolios
# do not contain gender or ethnicity columns (they are prohibited rating
# factors in the UK post-2013 ECJ ruling), but you might have:
#   - aggregator channel flag (correlates with customer age profile)
#   - occupation class (correlates with socioeconomic group)
#   - area-level ONS deprivation index (correlates with ethnicity)
#
# We use occupation_class and a synthetic socioeconomic_proxy as the
# "protected-adjacent" columns to audit.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Load data and train CatBoost frequency model")
print("=" * 70)

from insurance_datasets import load_motor

df_raw = load_motor(n_policies=20_000, seed=42)

# Add a binary disadvantaged_area proxy: area bands E and F are
# disproportionately inner-city and correlate with lower socioeconomic status.
# This is a realistic proxy concern — the FCA has flagged postcode-based
# pricing as a proxy risk.
rng = np.random.default_rng(42)
disadvantaged = (df_raw["area"].isin(["E", "F"])).astype(int)
# Add a small amount of noise so the proxy is imperfect (as in real data)
noise = rng.binomial(1, 0.08, len(df_raw))  # 8% misclassification
df_raw = df_raw.assign(
    disadvantaged_area=((disadvantaged + noise) % 2).astype(int)
)

print(f"Portfolio: {len(df_raw):,} policies")
print(f"Disadvantaged area proportion: {df_raw['disadvantaged_area'].mean():.2%}")
print(f"Occupation class distribution:\n{df_raw['occupation_class'].value_counts().sort_index()}")

FEATURE_COLS = [
    "vehicle_age",
    "vehicle_group",
    "driver_age",
    "ncd_years",
    "area",
    "policy_type",
    "occupation_class",
]
CAT_COLS = ["area", "policy_type"]
cat_indices = [FEATURE_COLS.index(c) for c in CAT_COLS]

from catboost import CatBoostRegressor, Pool

n_train = int(0.75 * len(df_raw))
train_pd = df_raw.iloc[:n_train].copy()
holdout_pd = df_raw.iloc[n_train:].copy()

train_log_exp = np.log(train_pd["exposure"].clip(lower=1e-9).values)
holdout_log_exp = np.log(holdout_pd["exposure"].clip(lower=1e-9).values)

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
    iterations=300,
    depth=4,
    learning_rate=0.06,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
model.fit(train_pool, eval_set=holdout_pool)

holdout_pd = holdout_pd.copy()
holdout_pd["predicted_rate"] = model.predict(holdout_pd[FEATURE_COLS])

holdout_actuals = holdout_pd["claim_count"].values
holdout_preds = holdout_pd["predicted_rate"].values
holdout_exp = holdout_pd["exposure"].values

holdout_rmse = float(np.sqrt(np.mean((holdout_preds - holdout_actuals) ** 2)))
calibration = float((holdout_preds * holdout_exp).sum() / max(holdout_actuals.sum(), 1e-9))
print(f"\nModel trained. Holdout RMSE: {holdout_rmse:.5f}, calibration: {calibration:.4f}")

# ---------------------------------------------------------------------------
# Step 2: FCA Consumer Duty proxy discrimination audit
# ---------------------------------------------------------------------------
#
# The FCA does not require pricing to produce equal outcomes across demographic
# groups — it requires you to demonstrate that:
#   (a) protected characteristics are not rating factors (direct discrimination)
#   (b) rating factors are not functioning as proxies for protected characteristics
#       in a way that produces unjustifiable differential outcomes (indirect discrimination)
#
# insurance-fairness runs three types of test:
#
#   Proxy detection   — R-squared and mutual information between each rating
#                        factor and the protected/proxied characteristic
#   Demographic parity — exposure-weighted mean prediction ratio across groups
#   Calibration by group — does the model's accuracy differ between groups?
#   Disparate impact ratio — 4/5ths rule (now used in UK financial services)
#
# We audit against two proxied characteristics:
#   occupation_class   — proxy for socioeconomic group
#   disadvantaged_area — binary proxy for area deprivation/ethnicity
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: FCA Consumer Duty proxy discrimination audit")
print("=" * 70)

from insurance_fairness import FairnessAudit

# Convert to Polars — insurance-fairness prefers Polars but accepts pandas
holdout_pl = pl.from_pandas(holdout_pd)

audit = FairnessAudit(
    model=model,
    data=holdout_pl,
    protected_cols=["disadvantaged_area", "occupation_class"],
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=FEATURE_COLS,
    model_name="Motor Frequency Model v1.0",
    run_proxy_detection=True,
    run_counterfactual=False,   # would need protected col as model input
    n_bootstrap=0,              # skip bootstrapped CIs for demo speed
    proxy_catboost_iterations=100,
)

print("Running audit... (proxy detection fits CatBoost models per factor)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fairness_report = audit.run()

fairness_report.summary()

# Drill into proxy detection scores
print("\nDetailed proxy scores by factor:")
for pc, pc_result in fairness_report.results.items():
    if pc_result.proxy_detection is None:
        continue
    print(f"\n  Protected characteristic: {pc}")
    for score in sorted(pc_result.proxy_detection.scores,
                        key=lambda s: s.proxy_r2, reverse=True)[:5]:
        print(
            f"    {score.factor:<25}"
            f"  proxy_R2={score.proxy_r2:.4f}"
            f"  MI={score.mutual_information:.4f}"
            f"  [{score.rag.upper()}]"
        )

if fairness_report.flagged_factors:
    print(f"\nFactors requiring investigation: {', '.join(fairness_report.flagged_factors)}")
    print(
        "These factors have statistically meaningful associations with the"
        " proxied characteristics. You need to determine whether these"
        " associations are actuarially justified (risk-based) or not."
    )
else:
    print("\nNo factors flagged as proxies at the default thresholds.")

# Save the markdown audit report
tmpdir = Path(tempfile.mkdtemp(prefix="bc_compliance_"))
audit_md_path = tmpdir / "fairness_audit.md"
fairness_report.to_markdown(str(audit_md_path))
print(f"\nMarkdown audit report saved to: {audit_md_path}")

# The to_dict() output is what you would log to your model risk system
audit_dict = fairness_report.to_dict()
print(f"\nAudit summary for model risk log:")
print(f"  Overall RAG: {audit_dict['overall_rag'].upper()}")
print(f"  Policies audited: {audit_dict['n_policies']:,}")
print(f"  Total exposure: {audit_dict['total_exposure']:,.1f} years")
print(f"  Protected characteristics audited: {', '.join(audit_dict['protected_cols'])}")
print(f"  Rating factors tested: {len(audit_dict['factor_cols'])}")

# ---------------------------------------------------------------------------
# Step 3: PRA SS1/23 model validation report
# ---------------------------------------------------------------------------
#
# PRA SS1/23 requires insurers to produce formal model validation documentation
# for all material models. The minimum requirements are:
#
#   - Data quality assessment
#   - Performance metrics (RMSE, Gini, A/E, lift chart)
#   - Stability analysis (out-of-time and out-of-segment)
#   - Discrimination metrics (Gini with confidence interval)
#   - Hosmer-Lemeshow calibration test
#
# insurance-validation generates an HTML report covering these requirements.
# The ModelCard captures the model metadata that goes in section 1 of the
# validation document. The ModelValidationReport runs all the tests.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: PRA SS1/23 model validation report")
print("=" * 70)

from insurance_validation import ModelCard, ModelValidationReport, PerformanceReport

# ModelCard: the metadata that goes in the validation document header
card = ModelCard(
    name="Motor Frequency Model",
    version="1.0.0",
    purpose=(
        "Predict claim frequency for UK private motor portfolio. "
        "Output is used directly as the frequency component of the pure premium."
    ),
    methodology=(
        "CatBoost gradient boosting with Poisson objective and log(exposure) offset. "
        "300 trees, depth 4, L2 regularisation 3.0."
    ),
    target="claim_count",
    features=FEATURE_COLS,
    limitations=[
        "No telematics data — mileage is self-declared annual_mileage only",
        "No claims history beyond NCD years (no prior frequency data)",
        "Trained on synthetic data — must be retrained on actual portfolio before deployment",
        "Area bands are coarser than postcode-district level",
    ],
    owner="Pricing Team",
)

# Training set predictions for comparison
train_pd_copy = train_pd.copy()
train_pd_copy["predicted_rate"] = model.predict(train_pd_copy[FEATURE_COLS])
train_preds = train_pd_copy["predicted_rate"].values
train_actuals = train_pd_copy["claim_count"].values
train_exp = train_pd_copy["exposure"].values

print("Running validation tests...")
report = ModelValidationReport(
    model_card=card,
    y_val=holdout_actuals,
    y_pred_val=holdout_preds,
    exposure_val=holdout_exp,
    y_train=train_actuals,
    y_pred_train=train_preds,
)

# Write the HTML report
html_path = tmpdir / "validation_report.html"
json_path = tmpdir / "validation_report.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report.generate(str(html_path))
    report.to_json(str(json_path))

print(f"HTML validation report: {html_path}")
print(f"JSON validation report: {json_path}")

# Run individual metrics and print them.
# PerformanceReport returns TestResult objects: check .passed, .metric_value,
# .details, and .extra for per-band data.
perf = PerformanceReport(holdout_actuals, holdout_preds, exposure=holdout_exp)

gini_result = perf.gini_with_ci()
ae_result = perf.ae_with_poisson_ci()
hl_result = perf.hosmer_lemeshow_test()

# Extract values from TestResult
gini_val = gini_result.metric_value
gini_ci_lo = gini_result.extra.get("ci_lower")
gini_ci_hi = gini_result.extra.get("ci_upper")

ae_val = ae_result.metric_value
ae_ci_lo = ae_result.extra.get("ci_lower")
ae_ci_hi = ae_result.extra.get("ci_upper")

hl_stat = hl_result.extra.get("hl_statistic")
hl_pval = hl_result.extra.get("p_value")

print(f"\nKey validation metrics (holdout set, n={len(holdout_pd):,} policies):")
print(f"  Gini coefficient:  {gini_val:.4f}  (95% CI: [{gini_ci_lo:.4f}, {gini_ci_hi:.4f}])")
print(f"  A/E ratio:         {ae_val:.4f}  (95% CI: [{ae_ci_lo:.4f}, {ae_ci_hi:.4f}])")
print(f"  Hosmer-Lemeshow:   chi2 = {hl_stat:.2f}, p = {hl_pval:.4f}  ({'PASS' if hl_result.passed else 'REVIEW'})")
print(f"  Calibration ratio: {calibration:.4f}")

# Lift chart — returns a list[TestResult], per-band data is in extra["bands"]
print("\nLift chart (10 bands, ordered by predicted rate):")
lift_results = perf.lift_chart(n_bands=10)
lift_bands = lift_results[0].extra.get("bands", [])
print(f"  {'Band':<8} {'Pred':>10} {'Actual':>10} {'A/E':>8}")
for b in lift_bands:
    ae = b["actual_rate"] / b["predicted_rate"] if b["predicted_rate"] > 0 else float("nan")
    print(
        f"  {b['band']:<8} {b['predicted_rate']:>10.4f} {b['actual_rate']:>10.4f} {ae:>8.4f}"
    )

# Out-of-segment validation: Gini by area band
print("\nOut-of-segment validation (Gini by area band):")
from insurance_fairness import gini_by_group

holdout_pl_with_preds = holdout_pl.with_columns(
    pl.Series("predicted_rate", holdout_preds)
)
gini_by_area = gini_by_group(
    df=holdout_pl_with_preds,
    protected_col="area",
    prediction_col="predicted_rate",
    exposure_col="exposure",
)
# GiniResult stores results in .group_ginis dict (group_str -> gini_float)
for group_val, gini in sorted(gini_by_area.group_ginis.items()):
    n = int((holdout_pd["area"] == group_val).sum())
    print(f"  Area {group_val}: Gini = {gini:.4f}  (n={n:,})")
print(f"  Overall Gini: {gini_by_area.overall_gini:.4f}")
print(f"  Max disparity across areas: {gini_by_area.max_disparity:.4f}")

# ---------------------------------------------------------------------------
# Step 4: Interpret results and action items
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: Summary and action items")
print("=" * 70)

overall_rag = fairness_report.overall_rag.upper()

print(f"""
Compliance summary
------------------
Fairness audit:    {overall_rag}  (Consumer Duty / Equality Act 2010)
Validation Gini:   {gini_val:.4f}  (95% CI [{gini_ci_lo:.4f}, {gini_ci_hi:.4f}])
A/E ratio:         {ae_val:.4f}  (95% CI [{ae_ci_lo:.4f}, {ae_ci_hi:.4f}])
H-L p-value:       {hl_pval:.4f}  ({'calibration OK' if hl_result.passed else 'calibration concerns'})

Required before deployment
--------------------------
1. If any factors were flagged by the fairness audit, document the actuarial
   justification for including them (or remove them from the tariff). The FCA
   will ask for this documentation on a Consumer Duty review.

2. The model validation report (saved to {html_path}) must be
   reviewed and signed off by an independent validator per PRA SS1/23.

3. Hosmer-Lemeshow p = {hl_pval:.4f}: {'acceptable - model is calibrated.' if hl_result.passed else 'requires investigation - recalibrate or review feature set.'}

4. Gini = {gini_val:.4f}: the model {'has meaningful discriminatory power' if (gini_val is not None and gini_val > 0.15) else 'has limited discriminatory power - consider feature engineering'}.

5. Record this audit in your model risk register with date {fairness_report.audit_date}
   and overall RAG status {overall_rag}.

Ongoing monitoring
------------------
Re-run this script quarterly with the live portfolio. The FCA's Consumer Duty
multi-firm review (2024) found that many firms do not monitor differential
outcomes continuously - regulators are now looking for evidence of ongoing
rather than point-in-time compliance.
""")
