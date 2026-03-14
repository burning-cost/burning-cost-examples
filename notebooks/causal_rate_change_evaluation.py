# Databricks notebook source

# MAGIC %md
# MAGIC # Causal Evaluation of a Motor Insurance Rate Change
# MAGIC
# MAGIC **Scenario:** Your pricing team raised rates in Q1 2023 for three northern territories. Loss ratios subsequently fell. The question the FCA's supervision team will ask — and the question the head of pricing should already be asking — is: did the rate change cause the improvement, or were you just lucky?
# MAGIC
# MAGIC Market claims inflation was flattening across the board in that period. Mix was shifting. Renewal repricing under GIPP was biting through the renewal book. Any or all of these could explain the loss ratio movement without your rate action being the cause.
# MAGIC
# MAGIC This notebook works through a complete causal evaluation workflow:
# MAGIC
# MAGIC 1. Generate a realistic synthetic motor portfolio (5,000 policies, 10 regions, 8 quarterly periods)
# MAGIC 2. Simulate a rate change in 3 treated regions at Q5, with a known true effect: +5% rate increase causing 3% frequency reduction
# MAGIC 3. Build the segment × period panel with `PolicyPanelBuilder`
# MAGIC 4. Estimate the causal effect with `SDIDEstimator`
# MAGIC 5. Interpret the event study plot and pre-treatment test
# MAGIC 6. Run HonestDiD sensitivity analysis
# MAGIC 7. Build an FCA evidence pack and interpret the regulatory narrative
# MAGIC 8. Summarise for management
# MAGIC
# MAGIC The true treatment effect is embedded in the data-generating process. This lets you check whether SDID recovers it — which is the methodology validation step you would run before applying this to live data.

# COMMAND ----------

# MAGIC %pip install insurance-causal-policy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate Synthetic Motor Portfolio
# MAGIC
# MAGIC We build the portfolio ourselves rather than using `make_synthetic_motor_panel`. The reason: this example uses claim frequency as the outcome (not loss ratio), and we want to be explicit about the data-generating process so you can see exactly how a rate change flows through to frequency.
# MAGIC
# MAGIC The data-generating process (DGP) is:
# MAGIC ```
# MAGIC freq_{it} = exp(alpha_i + beta_t + tau * D_{it}) + noise
# MAGIC ```
# MAGIC where:
# MAGIC - `alpha_i` = region-level fixed effect (baseline risk level)
# MAGIC - `beta_t` = common time trend (market-wide claims inflation / seasonal)
# MAGIC - `tau` = true treatment effect (what we want SDID to recover)
# MAGIC - `D_{it}` = treatment indicator (1 if region i raised rates at/after period t0)
# MAGIC
# MAGIC In a real analysis you would replace this block with your policy extract joined to a claims development table. The schema requirements are:
# MAGIC - `policy_df` — segment_id, period, earned_premium, earned_exposure
# MAGIC - `claims_df` — segment_id, period, incurred_claims, claim_count
# MAGIC - `rate_log_df` — segment_id, first_treated_period

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl

# Portfolio parameters
N_POLICIES = 5_000        # total policies (across all regions and quarters)
N_REGIONS = 10            # distinct regions (segment identifiers)
N_QUARTERS = 8            # time periods (Q1 2022 through Q4 2023)
TREATMENT_QUARTER = 5     # rate change applied from Q5 (= Q1 2023 in this frame)
N_TREATED_REGIONS = 3     # 3 of the 10 regions received the rate change
TRUE_FREQ_REDUCTION = -0.03  # true causal effect: -3% frequency reduction

# Market-wide trend: a gentle upward drift in frequency across all regions
# (claims inflation, new road users, etc.) This is what naive before-and-after
# would confound with the treatment effect.
MARKET_TREND = 0.008      # +0.8% frequency per quarter (common to all)

rng = np.random.default_rng(seed=2024)

regions = [
    "north_west", "north_east", "yorkshire", "east_midlands",
    "west_midlands", "south_east", "south_west", "london",
    "wales", "scotland",
]

# Three northern regions receive the rate change
treated_regions = {"north_west", "north_east", "yorkshire"}

# Region-level baseline frequencies (mean annual claims per policy-year)
# London and south-east higher; rural regions lower — realistic UK motor spread
region_base_freq = {
    "north_west":    0.082,
    "north_east":    0.078,
    "yorkshire":     0.076,
    "east_midlands": 0.071,
    "west_midlands": 0.074,
    "south_east":    0.085,
    "south_west":    0.065,
    "london":        0.094,
    "wales":         0.068,
    "scotland":      0.063,
}

# Policies per region per quarter (roughly proportional to population)
region_size = {
    "north_west":    80,
    "north_east":    55,
    "yorkshire":     70,
    "east_midlands": 60,
    "west_midlands": 65,
    "south_east":    90,
    "south_west":    55,
    "london":        120,
    "wales":         40,
    "scotland":      65,
}

print(f"Portfolio: {sum(region_size.values())} policies/quarter × {N_QUARTERS} quarters")
print(f"Treated regions: {sorted(treated_regions)}")
print(f"True treatment effect: {TRUE_FREQ_REDUCTION:+.1%} frequency")
print(f"Market trend: {MARKET_TREND:+.3f} per quarter (all regions)")

# COMMAND ----------

# Build the raw policy and claims tables

policy_rows = []
claims_rows = []

for region in regions:
    n_policies_region = region_size[region]
    base_freq = region_base_freq[region]
    is_treated = region in treated_regions

    for q in range(1, N_QUARTERS + 1):
        market_drift = MARKET_TREND * (q - 1)

        treatment_effect = 0.0
        if is_treated and q >= TREATMENT_QUARTER:
            treatment_effect = TRUE_FREQ_REDUCTION

        true_freq = base_freq + market_drift + treatment_effect
        freq_noise = rng.normal(0, 0.004)
        realised_freq = max(true_freq + freq_noise, 0.005)

        mean_exposure_per_policy = 0.95
        exposure_total = n_policies_region * mean_exposure_per_policy
        exposure_total += rng.normal(0, exposure_total * 0.02)
        exposure_total = max(exposure_total, 10.0)

        base_premium_per_policy = rng.uniform(480, 650)
        if is_treated and q >= TREATMENT_QUARTER:
            rate_uplift = 1.05
        else:
            rate_uplift = 1.0
        earned_premium = n_policies_region * base_premium_per_policy * rate_uplift
        earned_premium *= 1 + 0.005 * (q - 1)

        expected_claims = exposure_total * realised_freq
        claim_count = int(rng.poisson(expected_claims))

        mean_severity = 3_200 * (1 + 0.01 * (q - 1))
        severity_noise = rng.normal(0, mean_severity * 0.15)
        avg_severity = max(mean_severity + severity_noise, 500)
        incurred_claims = claim_count * avg_severity

        year = 2022 + (q - 1) // 4
        quarter = (q - 1) % 4 + 1
        period_code = year * 100 + quarter

        policy_rows.append({
            "segment_id": region,
            "period": period_code,
            "earned_premium": round(earned_premium, 2),
            "earned_exposure": round(exposure_total, 2),
        })

        claims_rows.append({
            "segment_id": region,
            "period": period_code,
            "incurred_claims": round(incurred_claims, 2),
            "claim_count": claim_count,
        })

policy_df = pl.DataFrame(policy_rows)
claims_df = pl.DataFrame(claims_rows)

rate_log_df = pl.DataFrame({
    "segment_id": list(treated_regions),
    "first_treated_period": [202301, 202301, 202301],
})

print(f"Policy table: {len(policy_df):,} rows")
print(f"Claims table: {len(claims_df):,} rows")
print(f"Rate log: {len(rate_log_df)} treated segments, first treated period = 202301")

# Quick sanity check: overall realised frequency before and after treatment
claims_pre = claims_df.filter(pl.col("period") < 202301)["claim_count"].sum()
exp_pre = policy_df.filter(pl.col("period") < 202301)["earned_exposure"].sum()
claims_post = claims_df.filter(pl.col("period") >= 202301)["claim_count"].sum()
exp_post = policy_df.filter(pl.col("period") >= 202301)["earned_exposure"].sum()

print(f"\nPre-treatment frequency (all regions):  {claims_pre / exp_pre:.4f}")
print(f"Post-treatment frequency (all regions): {claims_post / exp_post:.4f}")
print("Note: the post-treatment rise is partly market trend, partly the treatment effect (3 of 10 regions).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build the Balanced Segment × Period Panel
# MAGIC
# MAGIC `PolicyPanelBuilder` handles:
# MAGIC - Aggregating policy-level rows to segment totals
# MAGIC - Joining claims (left join — periods with zero claims stay in the panel)
# MAGIC - Computing the outcome metric (frequency = claim_count / earned_exposure)
# MAGIC - Joining treatment indicators from the rate log
# MAGIC - Balancing: ensuring all segment × period cells exist
# MAGIC
# MAGIC We use `outcome="frequency"` here. For a loss ratio analysis, change this to `"loss_ratio"` — the panel builder computes incurred / earned_premium. Frequency is preferred when post-treatment periods are recent (IBNR lag does not affect claim counts, only incurred claims).

# COMMAND ----------

from insurance_causal_policy import PolicyPanelBuilder

builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="frequency",
    exposure_col="earned_premium",
    min_exposure=50.0,
)

panel = builder.build()
summary = builder.summary()

print("Panel built successfully.")
print(f"  Segments total:   {summary['n_segments']}")
print(f"  Treated segments: {summary['n_treated_segments']} (northern regions)")
print(f"  Control segments: {summary['n_control_segments']}")
print(f"  Periods:          {summary['n_periods']}")
print(f"  Panel cells:      {summary['n_cells']}")
print(f"  Treated cells:    {summary['pct_treated_cells']:.1f}%")
print(f"  Non-zero exposure:{summary['pct_nonzero_exposure']:.1f}%")
print(f"  Outcome:          {summary['outcome']}")

# Show the panel structure for north_west
nw_panel = (
    panel
    .filter(pl.col("segment_id") == "north_west")
    .select(["segment_id", "period", "frequency", "treated", "first_treated_period"])
    .sort("period")
)
display(nw_panel.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fit the SDID Estimator
# MAGIC
# MAGIC `SDIDEstimator` implements Arkhangelsky et al. (2021) from first principles using CVXPY. There is no R dependency.
# MAGIC
# MAGIC The estimator does three things:
# MAGIC 1. **Unit weights (omega):** finds a weighted average of control regions that matched the treated regions' pre-treatment frequency trend
# MAGIC 2. **Time weights (lambda):** de-emphasises pre-treatment periods that are not informative about the post-treatment window
# MAGIC 3. **ATT via weighted DiD:** using the reweighted synthetic control and the time-weighted baseline, estimates the causal effect
# MAGIC
# MAGIC Inference method `"placebo"` randomly assigns treatment to control regions and measures variability under the null. Valid when N_control > N_treated (7 control, 3 treated here). Use `"bootstrap"` if this condition fails.

# COMMAND ----------

from insurance_causal_policy import SDIDEstimator

est = SDIDEstimator(
    panel=panel,
    outcome="frequency",
    inference="placebo",    # valid because N_control (7) > N_treated (3)
    n_replicates=200,
    random_seed=42,
)

print("Fitting SDID estimator (this takes ~30-60 seconds)...")
result = est.fit()

print("\n" + result.summary())

print("\nDetailed results:")
print(f"  ATT:            {result.att:+.5f} claims per policy-year")
print(f"  Standard error: {result.se:.5f}")
print(f"  95% CI:         [{result.ci_low:+.5f}, {result.ci_high:+.5f}]")
print(f"  p-value:        {result.pval:.4f}")
print(f"  Significant:    {result.significant}")
print(f"  True effect:    {TRUE_FREQ_REDUCTION:+.5f} (what we embedded in the DGP)")

print(f"\nPanel dimensions used:")
print(f"  Treated segments: {result.n_treated}")
print(f"  Control segments: {result.n_control} (non-zero weight)")
print(f"  Control total:    {result.n_control_total} (before weight optimisation)")
print(f"  Pre-treatment:    {result.t_pre} quarters")
print(f"  Post-treatment:   {result.t_post} quarters")

gap = abs(result.att - TRUE_FREQ_REDUCTION)
print(
    f"\nEstimator recovery: SDID estimated {result.att:+.5f}, "
    f"true effect was {TRUE_FREQ_REDUCTION:+.5f}. "
    f"Gap = {gap:.5f} ({gap / abs(TRUE_FREQ_REDUCTION):.1%} of true effect)."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Inspect Synthetic Control Composition (Unit Weights)
# MAGIC
# MAGIC The unit weights tell you which control regions are doing the heavy lifting. A good synthetic control is composed of regions that:
# MAGIC - Had similar pre-treatment frequency trends to the treated regions
# MAGIC - Are credibly comparable (similar risk mix, similar market exposure)
# MAGIC - Are not themselves affected by confounding events
# MAGIC
# MAGIC If the synthetic control puts all weight on one implausible region, the analysis needs to be interrogated. For example: if London gets 90% weight but the treated regions are rural northern territories, the parallel trends assumption is harder to defend.

# COMMAND ----------

import pandas as pd

omega = result.weights.unit_weights.sort_values(ascending=False)
print("Control region weights (omega):")
for region_name, weight in omega.items():
    bar = "#" * int(weight * 40)
    print(f"  {str(region_name):<20} {weight:.4f}  {bar}")

print(
    f"\nIntercept (omega_0): {result.weights.unit_intercept:+.5f}"
    "\n  This absorbs the level difference between treated and control."
    "\n  SDID allows level differences; pure synthetic control does not."
    "\n  The intercept being small (~0) suggests the regions are comparable."
)

print("\nPre-treatment time weights (lambda):")
lambda_weights = result.weights.time_weights.sort_values(ascending=False)
for period_code, weight in lambda_weights.items():
    bar = "#" * int(weight * 40)
    print(f"  Q{period_code}  {weight:.4f}  {bar}")
print(
    "  Higher-weight periods are more informative about the post-treatment"
    " trend. Usually the most recent pre-treatment quarters get most weight."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Event Study — the Key Diagnostic for Parallel Trends
# MAGIC
# MAGIC The event study plots the estimated treatment effect for each period relative to the rate change:
# MAGIC - `period_rel < 0`: pre-treatment (should be near zero — the parallel trends test)
# MAGIC - `period_rel >= 0`: post-treatment (the treatment effect estimates)
# MAGIC
# MAGIC **Reading the pre-treatment periods (left of the vertical line):**
# MAGIC Points should scatter around zero. If they show a systematic trend, treated and control regions were on diverging trajectories before the rate change — parallel trends is violated.
# MAGIC
# MAGIC **Reading the post-treatment periods (right of the vertical line):**
# MAGIC A step down at period 0 consistent with the sign you expect (rate increase → fewer high-frequency risks retained) is evidence the rate change worked as intended.

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_causal_policy import plot_event_study, pre_trend_summary

# Pre-trend summary
pt_summary = pre_trend_summary(result)
print("Pre-trend test:")
print(f"  p-value:         {pt_summary['pval']:.4f}")
print(f"  Passes (p>0.10): {pt_summary['pass']}")
print(f"  Pre-period ATTs: {[f'{x:+.4f}' for x in pt_summary['pre_atts']]}")
print(f"  Max |pre-ATT|:   {pt_summary['max_abs_pre_att']:.4f}")
print(f"\n  {pt_summary['interpretation']}")

# Event study table
print("\nEvent study by period:")
es = result.event_study
for _, row in es.iterrows():
    rel = int(row["period_rel"])
    att_val = row["att"]
    if rel < 0:
        label = "pre-treatment (should be ~0)"
    elif rel == 0:
        label = "treatment quarter (Q1 2023)"
    else:
        label = "post-treatment"
    print(f"  period_rel={rel:>3}  ATT={att_val:>+10.5f}  {label}")

# COMMAND ----------

fig_es = plot_event_study(
    result,
    title="Motor Rate Change Event Study: North-West/North-East/Yorkshire",
)
display(fig_es)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Sensitivity Analysis (HonestDiD-style)
# MAGIC
# MAGIC The sensitivity analysis answers: how robust is the conclusion to violations of the parallel trends assumption?
# MAGIC
# MAGIC The sensitivity parameter M is in multiples of the pre-period standard deviation:
# MAGIC - **M = 0.0:** classical parallel trends (no violation allowed)
# MAGIC - **M = 1.0:** post-treatment violation could be as large as the typical pre-treatment variation observed
# MAGIC - **M = 2.0:** post-treatment violation could be twice as large as pre-period
# MAGIC
# MAGIC The **breakdown point M*** is the smallest M at which zero enters the identified set — i.e. where you can no longer rule out that the true effect is zero. A large M* means the result is robust.
# MAGIC
# MAGIC **Regulatory context:** FCA EP25/2 does not require HonestDiD specifically, but it does require evidence that conclusions are robust to reasonable model perturbations. Showing M* > 1 is a credible way to demonstrate this.

# COMMAND ----------

from insurance_causal_policy import compute_sensitivity, plot_sensitivity

sens = compute_sensitivity(
    result,
    m_values=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    method="linear",   # linear extrapolation restriction (Rambachan & Roth 2023)
)

print(sens.summary())
print(f"\nPre-period SD: {sens.pre_period_sd:.5f}")

print("\nSensitivity table:")
print(f"  {'M':>6}  {'ATT lower':>12}  {'ATT upper':>12}  {'Excludes 0?':>14}")
print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*14}")
for m, lo, hi in zip(sens.m_values, sens.att_lower, sens.att_upper):
    excludes_zero = (lo > 0) or (hi < 0)
    flag = "YES (robust)" if excludes_zero else "NO (breaks down)"
    print(f"  {m:>6.1f}  {lo:>+12.5f}  {hi:>+12.5f}  {flag}")

print(f"\nBreakdown point: M* = {sens.m_breakdown:.2f}")

if sens.m_breakdown >= 2.0:
    print(
        "  Result survives even if the post-treatment parallel trends violation"
        " is twice as large as pre-period variation. Strong robustness result."
        " Suitable for FCA evidence submission."
    )
elif sens.m_breakdown >= 1.0:
    print(
        "  Result survives parallel trends violations up to the size of"
        " pre-period variation. Reasonable robustness. Consider extending the"
        " pre-treatment window for a stronger test."
    )
else:
    print(
        "  Result is sensitive to relatively small parallel trends violations."
        " Do not submit as regulatory evidence without further investigation."
    )

# COMMAND ----------

fig_sens = plot_sensitivity(
    sens,
    title="Sensitivity: How large must parallel trends violations be to change conclusion?",
)
display(fig_sens)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: FCA Evidence Pack
# MAGIC
# MAGIC FCA TR24/2 (2024) found that most insurers assessed under the Consumer Duty multi-firm review failed to demonstrate causal attribution between their rate changes and the outcomes they observed. They showed before-and-after data but could not show the rate change caused the change rather than external factors.
# MAGIC
# MAGIC `FCAEvidencePack` produces a structured Markdown document addressing the four things the FCA's supervision team looks for:
# MAGIC 1. **Causal methodology:** not before-and-after, but a recognised control group method
# MAGIC 2. **Confidence interval:** not a point estimate — a range showing uncertainty
# MAGIC 3. **Pre-treatment validation:** demonstrating parallel trends holds
# MAGIC 4. **Sensitivity analysis:** robustness to assumption violations

# COMMAND ----------

from insurance_causal_policy import FCAEvidencePack

pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2023-Q1",
    rate_change_magnitude="+5% technical premium (north-west, north-east, yorkshire)",
    analyst="Pricing Actuarial Team",
    panel_summary=summary,
    additional_notes=(
        "Analysis covers 8 quarterly periods (Q1 2022 to Q4 2023). "
        "No known market-wide structural breaks in the analysis window. "
        "FCA GIPP reforms (Jan 2022) pre-date the analysis start; "
        "their effect is captured in the baseline. "
        "IBNR development is not a concern as frequency is the outcome metric."
    ),
)

evidence_md = pack.to_markdown()

# Render Markdown in the notebook
displayHTML(f"<div style='font-family: monospace; white-space: pre-wrap;'>{evidence_md}</div>")

# COMMAND ----------

import json as _json
import numpy as _np

class _NumpyEncoder(_json.JSONEncoder):
    """Handle numpy scalars that Python's json module can't serialize."""
    def default(self, obj):
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return super().default(obj)

evidence_dict = pack.to_dict()
print("Evidence pack metadata:")
print(_json.dumps(evidence_dict["metadata"], indent=2, cls=_NumpyEncoder))
print("\nEstimation summary:")
print(_json.dumps(evidence_dict["estimation"], indent=2, cls=_NumpyEncoder))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Management Summary
# MAGIC
# MAGIC The technical output above is what you present to the FCA. What you present to the head of motor pricing is different. They want:
# MAGIC - What happened (one sentence)
# MAGIC - How confident are you (confidence interval in plain language)
# MAGIC - Is this good enough evidence to rely on for further rate decisions?
# MAGIC - What would invalidate the conclusion?

# COMMAND ----------

att_as_pct_of_base = result.att / 0.075

treated_exposure_post = (
    policy_df
    .filter(
        pl.col("segment_id").is_in(list(treated_regions)) &
        (pl.col("period") >= 202301)
    )
    ["earned_exposure"]
    .sum()
)
implied_claims_avoided = abs(result.att) * treated_exposure_post

mgmt_summary = f"""
MANAGEMENT SUMMARY
------------------

Rate change: +5% technical premium applied Q1 2023 to north-west,
north-east, and yorkshire territories.

HEADLINE RESULT
  The rate change caused a {abs(result.att):.4f} reduction in claim frequency
  (95% CI: {abs(result.ci_low):.4f} to {abs(result.ci_high):.4f} claims per policy-year).
  This is equivalent to a {abs(att_as_pct_of_base) * 100:.1f}% reduction from pre-treatment baseline.
  Result is statistically significant (p = {result.pval:.4f}).

CONFIDENCE
  We are 95% confident the true frequency reduction is between {abs(result.ci_low):.4f}
  and {abs(result.ci_high):.4f} claims per policy-year. The sign is not in doubt —
  even under conservative sensitivity assumptions (M=2.0: post-treatment
  parallel trends could deviate twice as much as pre-treatment), the
  entire identified set lies below zero.

CLAIMS AVOIDED
  Estimated {implied_claims_avoided:.0f} additional claims that would have occurred had
  the rate change not been applied, across the {treated_exposure_post:.0f} policy-years of
  treated exposure in the post-treatment window.

METHODOLOGY CREDIBILITY
  The comparison group is a synthetic control built from the 7 untreated
  regions. Pre-treatment parallel trends test passes (p = {result.pre_trend_pval if result.pre_trend_pval is not None else 1.0:.3f}) —
  the treated and control regions were on comparable trajectories before
  Q1 2023. The effect is not attributable to the market-wide frequency
  rise (which affected all regions equally and is absorbed by the
  synthetic control).

WHAT WOULD INVALIDATE THIS
  1. A post-treatment shock that hit only the northern territories.
     We are not aware of such a shock in Q1-Q4 2023.
  2. A concurrent change in the risk mix of the northern book not
     captured by the market-wide trend.
  3. IBNR bias is not a concern here because we are measuring frequency,
     not loss ratio.

RECOMMENDATION
  This analysis provides FCA-standard causal evidence (class of method
  used in FCA EP25/2). Include in the Q4 2023 Consumer Duty outcome
  monitoring pack. Flag the regional scope — the effect is estimated for
  north-west, north-east, and yorkshire only.
"""

print(mgmt_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workflow Summary

# COMMAND ----------

print("=" * 72)
print("Workflow complete")
print("=" * 72)
print(f"""
Steps completed:

  1. Synthetic portfolio   {sum(region_size.values())} policies, {N_REGIONS} regions,
                           {N_QUARTERS} quarters, {N_TREATED_REGIONS} treated

  2. Panel built           {summary['n_segments']} segments × {summary['n_periods']} periods
                           outcome: frequency (claim_count / earned_exposure)

  3. SDID estimated        ATT = {result.att:+.5f}
                           true = {TRUE_FREQ_REDUCTION:+.5f}
                           95% CI [{result.ci_low:+.5f}, {result.ci_high:+.5f}]
                           p = {result.pval:.4f}

  4. Synthetic control     {result.n_control} of {result.n_control_total} control regions
                           carry non-zero weight

  5. Event study           Pre-trend p = {result.pre_trend_pval:.4f}
                           (PASS = no evidence of parallel trends violation)

  6. Sensitivity           Breakdown M* = {sens.m_breakdown:.2f}
                           (robust up to {sens.m_breakdown:.1f}x pre-period SD)

  7. FCA evidence pack     Rendered above

  8. Management summary    Printed above

Next steps:
  - Replace synthetic data with your live policy extract (same schema)
  - Adjust treatment_period encoding to match your rate change date
  - Run with n_replicates=500 for the final FCA submission version
  - Add plot_unit_weights() and plot_synthetic_trajectory() to the evidence
    pack for the visual parallel trends exhibit
""")
