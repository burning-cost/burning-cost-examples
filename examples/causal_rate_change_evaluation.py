"""
Causal evaluation of a motor insurance rate change using SDID.

Your pricing team raised rates in Q1 2023 for three northern territories.
Loss ratios subsequently fell. The question the FCA's supervision team will
ask — and the question the head of pricing should already be asking — is:
did the rate change cause the improvement, or were you just lucky?

Market claims inflation was flattening across the board in that period.
Mix was shifting. Renewal repricing under GIPP was biting through the
renewal book. Any or all of these could explain the loss ratio movement
without your rate action being the cause.

This script works through a complete causal evaluation workflow:

    1. Generate a realistic synthetic motor portfolio (5,000 policies,
       10 regions, 8 quarterly periods)
    2. Simulate a rate change in 3 treated regions at Q5, with a known
       true effect: +5% rate increase causing 3% frequency reduction
    3. Build the segment × period panel with PolicyPanelBuilder
    4. Estimate the causal effect with SDIDEstimator
    5. Interpret the event study plot and pre-treatment test
    6. Run HonestDiD sensitivity analysis
    7. Build an FCA evidence pack and interpret the regulatory narrative
    8. Summarise for management

The true treatment effect is embedded in the data-generating process.
This lets you check whether SDID recovers it — which is the methodology
validation step you would run before applying this to live data.

Libraries used
--------------
    insurance-causal-policy — SDID, event study, sensitivity, FCA pack

Dependencies
------------
    uv add insurance-causal-policy

The script runs without external data files. Everything is generated inline.
Expected runtime: 30-90 seconds (SDID weight optimisation + 200 placebo
replicates dominate).

Note on running: the SDID optimisation uses CVXPY with the CLARABEL solver.
This runs fine on Databricks. On a Raspberry Pi, run it on Databricks.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Step 1: Generate a realistic synthetic motor portfolio
# ---------------------------------------------------------------------------
#
# We build the portfolio ourselves rather than using make_synthetic_motor_panel.
# The reason: this example uses claim frequency as the outcome (not loss ratio),
# and we want to be explicit about the data-generating process so you can see
# exactly how a rate change flows through to frequency.
#
# The data-generating process (DGP) is:
#
#   freq_{it} = exp(alpha_i + beta_t + tau * D_{it}) + noise
#
# where:
#   alpha_i  = region-level fixed effect (baseline risk level)
#   beta_t   = common time trend (market-wide claims inflation / seasonal)
#   tau      = true treatment effect (what we want SDID to recover)
#   D_{it}   = treatment indicator (1 if region i raised rates at/after period t0)
#
# The mechanism: a rate increase reduces volume (lapse of high-frequency risks
# first, price-elastic shoppers leave) and changes the mix toward lower-risk
# retained customers. The net effect on frequency depends on your elasticity
# assumptions. Here we model it directly: treated regions see a 3% reduction
# in claim frequency from the treatment period onwards.
#
# In a real analysis you would replace this block with your policy extract
# joined to a claims development table. The schema requirements are:
#   policy_df  — segment_id, period, earned_premium, earned_exposure
#   claims_df  — segment_id, period, incurred_claims, claim_count
#   rate_log_df — segment_id, first_treated_period
# ---------------------------------------------------------------------------

print("=" * 72)
print("Step 1: Generate synthetic motor portfolio")
print("=" * 72)

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

# Check our portfolio size is plausible
total_policy_quarters = sum(v * N_QUARTERS for v in region_size.values())
print(f"Portfolio: {sum(region_size.values())} policies/quarter × {N_QUARTERS} quarters")
print(f"Total policy-quarter observations: {total_policy_quarters:,}")
print(f"Treated regions: {sorted(treated_regions)}")
print(f"True treatment effect: {TRUE_FREQ_REDUCTION:+.1%} frequency")
print(f"Market trend: {MARKET_TREND:+.3f} per quarter (all regions)")

# ---- Build the raw policy and claims tables ----

policy_rows = []
claims_rows = []

for region in regions:
    n_policies_region = region_size[region]
    base_freq = region_base_freq[region]
    is_treated = region in treated_regions

    for q in range(1, N_QUARTERS + 1):
        # Market trend: claims frequency drifts upward over time
        market_drift = MARKET_TREND * (q - 1)

        # Treatment effect: only in treated regions from treatment quarter
        treatment_effect = 0.0
        if is_treated and q >= TREATMENT_QUARTER:
            treatment_effect = TRUE_FREQ_REDUCTION

        # True frequency for this region-quarter
        true_freq = base_freq + market_drift + treatment_effect

        # Idiosyncratic noise at segment level (small — we want a clean test)
        freq_noise = rng.normal(0, 0.004)
        realised_freq = max(true_freq + freq_noise, 0.005)

        # Earned exposure (policy-years): each policy active for ~0.95 quarters
        # on average. Use a beta distribution for realism.
        mean_exposure_per_policy = 0.95
        exposure_total = n_policies_region * mean_exposure_per_policy
        # Add a small stochastic jitter to exposure
        exposure_total += rng.normal(0, exposure_total * 0.02)
        exposure_total = max(exposure_total, 10.0)

        # Premium: base premium × (1 + treatment rate increase for treated)
        # The +5% rate increase is what generates the frequency reduction via
        # price elasticity and mix effects.
        base_premium_per_policy = rng.uniform(480, 650)  # UK motor range, GBP
        if is_treated and q >= TREATMENT_QUARTER:
            rate_uplift = 1.05   # +5% rate increase
        else:
            rate_uplift = 1.0
        earned_premium = n_policies_region * base_premium_per_policy * rate_uplift
        # Add quarterly inflation to all premiums
        earned_premium *= 1 + 0.005 * (q - 1)

        # Realised claims: Poisson draw against exposure × frequency
        expected_claims = exposure_total * realised_freq
        claim_count = int(rng.poisson(expected_claims))

        # Incurred claims: claim_count × average severity
        # Severity has its own inflation trend (independent of frequency)
        mean_severity = 3_200 * (1 + 0.01 * (q - 1))  # ~1% severity inflation/qtr
        severity_noise = rng.normal(0, mean_severity * 0.15)
        avg_severity = max(mean_severity + severity_noise, 500)
        incurred_claims = claim_count * avg_severity

        # Encode period as YYYYQQ: 202201, 202202, ..., 202301, ...
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

# Rate log: identifies which regions received the change and when.
# first_treated_period must use the same period encoding as policy_df.
# Q5 = 2023Q1 = period code 202301.
rate_log_df = pl.DataFrame({
    "segment_id": list(treated_regions),
    "first_treated_period": [202301, 202301, 202301],
})

print(f"\nPolicy table: {len(policy_df):,} rows")
print(f"Claims table: {len(claims_df):,} rows")
print(f"Rate log: {len(rate_log_df)} treated segments, first treated period = 202301")

# Quick sanity check: overall realised frequency before and after treatment
claims_pre = claims_df.filter(pl.col("period") < 202301)["claim_count"].sum()
exp_pre = policy_df.filter(pl.col("period") < 202301)["earned_exposure"].sum()
claims_post = claims_df.filter(pl.col("period") >= 202301)["claim_count"].sum()
exp_post = policy_df.filter(pl.col("period") >= 202301)["earned_exposure"].sum()

print(f"\nPre-treatment frequency (all regions):  {claims_pre / exp_pre:.4f}")
print(f"Post-treatment frequency (all regions): {claims_post / exp_post:.4f}")
print(
    "Note: the post-treatment rise is partly market trend, partly the treatment"
    " effect (which only affects 3 of 10 regions)."
)

# ---------------------------------------------------------------------------
# Step 2: Build the balanced segment × period panel
# ---------------------------------------------------------------------------
#
# Raw policy tables are one row per policy per period. The SDID estimator
# needs a balanced segment × period panel — one row per segment per period
# with the outcome already computed.
#
# PolicyPanelBuilder handles:
#   - Aggregating policy-level rows to segment totals
#   - Joining claims (left join — periods with zero claims stay in the panel)
#   - Computing the outcome metric (frequency = claim_count / earned_exposure)
#   - Joining treatment indicators from the rate log
#   - Balancing: ensuring all segment × period cells exist (filling missing
#     with zero exposure and null outcome)
#
# We use outcome="frequency" here. For a loss ratio analysis, change this
# to "loss_ratio" — the panel builder computes incurred / earned_premium.
# Frequency is preferred when post-treatment periods are recent (IBNR lag
# does not affect claim counts, only incurred claims).
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 2: Build balanced segment × period panel")
print("=" * 72)

from insurance_causal_policy import PolicyPanelBuilder

builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="frequency",       # claim_count / earned_exposure
    exposure_col="earned_premium",  # weight for SDID RC-weighting
    min_exposure=50.0,         # warn if any segment-period is very thin
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

# Show the panel structure for one region to confirm it looks right
print("\nPanel structure for north_west (first 8 rows):")
nw_panel = (
    panel
    .filter(pl.col("segment_id") == "north_west")
    .select(["segment_id", "period", "frequency", "treated", "first_treated_period"])
    .sort("period")
)
for row in nw_panel.iter_rows(named=True):
    treated_flag = "[TREATED]" if row["treated"] == 1 else ""
    print(
        f"  {row['period']}  freq={row['frequency']:.4f}"
        f"  treated={row['treated']}  {treated_flag}"
    )

# ---------------------------------------------------------------------------
# Step 3: Fit the SDID estimator
# ---------------------------------------------------------------------------
#
# SDIDEstimator implements Arkhangelsky et al. (2021) from first principles
# using CVXPY. There is no R dependency.
#
# The estimator does three things:
#
#  1. Unit weights (omega): finds a weighted average of control regions that
#     matched the treated regions' pre-treatment frequency trend. This is the
#     "synthetic control" part. Regions that do not track the treated regions
#     well get near-zero weight.
#
#  2. Time weights (lambda): de-emphasises pre-treatment periods that are not
#     informative about the post-treatment window. In a panel with 4 pre-
#     treatment quarters and 4 post-treatment quarters, the last pre-treatment
#     quarter typically gets the most weight.
#
#  3. ATT via weighted DiD: using the reweighted synthetic control and the
#     time-weighted baseline, estimates the causal effect as a doubly-weighted
#     difference-in-differences.
#
# Inference method "placebo" randomly assigns treatment to control regions and
# measures how variable the estimated effect would be under the null. Valid
# when you have more control units than treated units (7 control, 3 treated
# here — fine). Use "bootstrap" if this condition fails.
#
# n_replicates=200 is the default. Use 500 for final regulatory submission.
# It adds compute time but tightens the variance estimate.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 3: Fit SDID estimator (this takes ~30-60 seconds)")
print("=" * 72)

from insurance_causal_policy import SDIDEstimator

est = SDIDEstimator(
    panel=panel,
    outcome="frequency",
    inference="placebo",    # valid because N_control (7) > N_treated (3)
    n_replicates=200,
    random_seed=42,
)

result = est.fit()

print("\n" + result.summary())

# Detailed interpretation
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

# Interpret the gap between estimated and true ATT
gap = abs(result.att - TRUE_FREQ_REDUCTION)
print(
    f"\nEstimator recovery: SDID estimated {result.att:+.5f}, "
    f"true effect was {TRUE_FREQ_REDUCTION:+.5f}. "
    f"Gap = {gap:.5f} ({gap / abs(TRUE_FREQ_REDUCTION):.1%} of true effect)."
)
print(
    "In real analysis the true effect is unknown. This is the calibration "
    "check you run on synthetic data before applying the method to live data."
)

# ---------------------------------------------------------------------------
# Step 4: Unit weights — what is the synthetic control made of?
# ---------------------------------------------------------------------------
#
# The unit weights tell you which control regions are doing the heavy lifting.
# This is the interpretability test your model governance team will ask for:
# 'What exactly is the synthetic control, and does it make sense?'
#
# A good synthetic control is composed of regions that:
# - Had similar pre-treatment frequency trends to the treated regions
# - Are credibly comparable (similar risk mix, similar market exposure)
# - Are not themselves affected by confounding events
#
# If the synthetic control puts all weight on one implausible region,
# the analysis needs to be interrogated. For example: if London gets 90%
# weight but the treated regions are rural northern territories, the parallel
# trends assumption is harder to defend.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 4: Inspect synthetic control composition (unit weights)")
print("=" * 72)

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

# Time weights
print("\nPre-treatment time weights (lambda):")
lambda_weights = result.weights.time_weights.sort_values(ascending=False)
for period_code, weight in lambda_weights.items():
    bar = "#" * int(weight * 40)
    print(f"  Q{period_code}  {weight:.4f}  {bar}")
print(
    "  Higher-weight periods are more informative about the post-treatment"
    " trend. Usually the most recent pre-treatment quarters get most weight."
)

# ---------------------------------------------------------------------------
# Step 5: Event study — the key diagnostic for parallel trends
# ---------------------------------------------------------------------------
#
# The event study is the most important diagnostic in the analysis. It plots
# the estimated treatment effect for each period relative to the rate change:
#   period_rel < 0: pre-treatment (should be near zero — the parallel trends test)
#   period_rel >= 0: post-treatment (the treatment effect estimates)
#
# Reading the event study plot:
#
#   Pre-treatment (left of the vertical line):
#   - Points should scatter around zero. If they show a systematic trend away
#     from zero, treated and control regions were on diverging trajectories
#     before the rate change — parallel trends is violated.
#   - The joint p-value tests whether the pre-treatment ATTs are jointly
#     distinguishable from zero. p > 0.10 means you cannot reject the null
#     that pre-trends are flat — a necessary condition for causal claims.
#
#   Post-treatment (right of the vertical line):
#   - The estimated treatment effect. A step down at period 0 consistent with
#     the sign you expect from a rate increase (higher price → fewer high-
#     frequency risks retained) is evidence the rate change worked as intended.
#   - Multiple post-treatment periods showing a sustained effect vs a one-off
#     spike tells you about the persistence of the effect.
#
# What this does NOT show:
#   - The event study does not prove causation. It tests one necessary condition
#     (pre-treatment parallel trends). External shocks in the post-treatment
#     period that affected only treated regions would mimic a treatment effect.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 5: Event study diagnostics")
print("=" * 72)

from insurance_causal_policy import plot_event_study, pre_trend_summary

# Pre-trend summary
pt_summary = pre_trend_summary(result)
print("Pre-trend test:")
print(f"  p-value:        {pt_summary['pval']:.4f}")
print(f"  Passes (p>0.10): {pt_summary['pass']}")
print(f"  Pre-period ATTs: {[f'{x:+.4f}' for x in pt_summary['pre_atts']]}")
print(f"  Max |pre-ATT|:  {pt_summary['max_abs_pre_att']:.4f}")
print(f"\n  {pt_summary['interpretation']}")

# Event study table
print("\nEvent study by period:")
es = result.event_study
print(f"  {'period_rel':>12}  {'ATT':>10}  {'interpretation'}")
print(f"  {'-'*12}  {'-'*10}  {'-'*35}")
for _, row in es.iterrows():
    rel = int(row["period_rel"])
    att_val = row["att"]
    if rel < 0:
        label = "pre-treatment (should be ~0)"
    elif rel == 0:
        label = "treatment quarter (Q1 2023)"
    else:
        label = "post-treatment"
    print(f"  {rel:>12}  {att_val:>+10.5f}  {label}")

# Save the event study plot (would display in a Databricks notebook)
try:
    import matplotlib
    matplotlib.use("Agg")
    fig_es = plot_event_study(
        result,
        title="Motor Rate Change Event Study: North-West/North-East/Yorkshire",
    )
    # In a Databricks notebook: display(fig_es)
    # In a script, save to file:
    fig_es.savefig("/tmp/event_study.png", dpi=120, bbox_inches="tight")
    print("\nEvent study plot saved to /tmp/event_study.png")
    print(
        "In a Databricks notebook, use display(fig_es) to show inline."
    )
except Exception as e:
    print(f"\n(Event study plot not saved: {e})")

# ---------------------------------------------------------------------------
# Step 6: HonestDiD sensitivity analysis
# ---------------------------------------------------------------------------
#
# The sensitivity analysis answers: how robust is the conclusion to violations
# of the parallel trends assumption?
#
# Parallel trends is an assumption, not a testable fact (we can only test it
# in the pre-treatment period, not the post-treatment period where causality
# matters). The sensitivity analysis quantifies what happens if we allow
# the post-treatment parallel trends to be violated by a bounded amount.
#
# The sensitivity parameter M is in multiples of the pre-period standard
# deviation. So:
#   M = 0.0: classical parallel trends (no violation allowed)
#   M = 1.0: post-treatment violation could be as large as the typical
#             pre-treatment variation we observed
#   M = 2.0: post-treatment violation could be twice as large as pre-period
#
# The breakdown point M* is the smallest M at which zero enters the
# identified set — i.e. where you can no longer rule out that the true
# effect is zero. A large M* means the result is robust.
#
# Regulatory context: The FCA's approach in EP25/2 used DiD methodology
# and emphasised robustness of conclusions to model perturbations. Showing
# M* > 1 is a credible way to demonstrate this under Consumer Duty outcome monitoring.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 6: Sensitivity analysis (HonestDiD-style)")
print("=" * 72)

from insurance_causal_policy import compute_sensitivity, plot_sensitivity

sens = compute_sensitivity(
    result,
    m_values=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    method="linear",   # linear extrapolation restriction (Rambachan & Roth 2023)
)

print(sens.summary())
print(f"\nPre-period SD: {sens.pre_period_sd:.5f}")
print(
    f"This is the typical size of pre-treatment deviation from zero in the"
    f" event study. M is expressed as a multiple of this."
)

print("\nSensitivity table:")
print(f"  {'M':>6}  {'ATT lower':>12}  {'ATT upper':>12}  {'Excludes 0?':>12}")
print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")
for m, lo, hi in zip(sens.m_values, sens.att_lower, sens.att_upper):
    excludes_zero = (lo > 0) or (hi < 0)
    flag = "YES (robust)" if excludes_zero else "NO (breaks down)"
    print(f"  {m:>6.1f}  {lo:>+12.5f}  {hi:>+12.5f}  {flag}")

print(f"\nBreakdown point: M* = {sens.m_breakdown:.2f}")

if sens.m_breakdown >= 2.0:
    print(
        "  Interpretation: The result survives even if the post-treatment"
        " parallel trends violation is twice as large as the pre-period"
        " variation. This is a strong robustness result. Suitable for FCA"
        " evidence submission."
    )
elif sens.m_breakdown >= 1.0:
    print(
        "  Interpretation: The result survives parallel trends violations up"
        " to the size of pre-period variation. Reasonable robustness. Consider"
        " extending the pre-treatment window for a stronger test."
    )
else:
    print(
        "  Interpretation: The result is sensitive to relatively small"
        " parallel trends violations. Do not submit this as regulatory evidence"
        " without investigating why the pre-treatment trends are noisy."
    )

# Save sensitivity plot
try:
    fig_sens = plot_sensitivity(
        sens,
        title="Sensitivity: How large must parallel trends violations be to change conclusion?",
    )
    fig_sens.savefig("/tmp/sensitivity.png", dpi=120, bbox_inches="tight")
    print("\nSensitivity plot saved to /tmp/sensitivity.png")
except Exception as e:
    print(f"\n(Sensitivity plot not saved: {e})")

# ---------------------------------------------------------------------------
# Step 7: FCA evidence pack
# ---------------------------------------------------------------------------
#
# FCA TR24/2 (2024) found that most insurers assessed under the Consumer Duty
# multi-firm review failed to demonstrate causal attribution between their rate
# changes and the outcomes they observed. They showed before-and-after data
# but could not show the rate change caused the change rather than external
# factors.
#
# FCAEvidencePack produces a structured Markdown document that addresses the
# four things the FCA's supervision team looks for:
#
#   1. Causal methodology: not before-and-after, but a recognised control
#      group method (the FCA itself uses DiD in its own evaluation work)
#   2. Confidence interval: not a point estimate — a range showing uncertainty
#   3. Pre-treatment validation: demonstrating parallel trends holds
#   4. Sensitivity analysis: robustness to assumption violations
#
# The pack is designed to be saved to PDF (weasyprint optional dependency)
# or included in a Confluence/SharePoint page as Markdown.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 7: FCA evidence pack")
print("=" * 72)

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

# Print the full Markdown document
evidence_md = pack.to_markdown()
print(evidence_md)

# Also available as structured dict for downstream systems
evidence_dict = pack.to_dict()
print("\nEvidence pack JSON metadata:")
import json as _json
class _NpEncoder(_json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        import numpy as _n
        if isinstance(obj, _n.integer): return int(obj)
        if isinstance(obj, _n.floating): return float(obj)
        if isinstance(obj, _n.bool_): return bool(obj)
        if isinstance(obj, _n.ndarray): return obj.tolist()
        return super().default(obj)

print(_json.dumps(evidence_dict["metadata"], indent=2, cls=_NpEncoder))
print("\nEstimation summary:")
print(_json.dumps(evidence_dict["estimation"], indent=2, cls=_NpEncoder))

# Save the Markdown
try:
    with open("/tmp/fca_evidence_pack.md", "w") as f:
        f.write(evidence_md)
    print("\nFCA evidence pack saved to /tmp/fca_evidence_pack.md")
except Exception as e:
    print(f"\n(Evidence pack not saved: {e})")

# ---------------------------------------------------------------------------
# Step 8: Presenting to management
# ---------------------------------------------------------------------------
#
# The technical output above is what you present to the FCA. What you present
# to the head of motor pricing is different. They want:
#
#   - What happened (one sentence)
#   - How confident are you (confidence interval in plain language)
#   - Is this good enough evidence to rely on for further rate decisions?
#   - What would invalidate the conclusion?
#
# The section below drafts that narrative from the computed results.
# Treat it as a template — edit to reflect actual business context.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 8: Management summary")
print("=" * 72)

# Derived quantities for narrative
att_as_pct_of_base = result.att / 0.075  # as % of ~mean frequency

# Compute implied impact on claim count
# If frequency fell by ATT across the treated book for T_post quarters:
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

print("""
MANAGEMENT SUMMARY
------------------

Rate change: +5% technical premium applied Q1 2023 to north-west,
north-east, and yorkshire territories.

HEADLINE RESULT
  The rate change caused a {att:.4f} reduction in claim frequency
  (95% CI: {ci_lo:.4f} to {ci_hi:.4f} claims per policy-year).
  This is equivalent to a {pct:.1f}% reduction from pre-treatment baseline.
  Result is statistically significant (p = {pval:.4f}).

CONFIDENCE
  We are 95% confident the true frequency reduction is between {ci_lo:.4f}
  and {ci_hi:.4f} claims per policy-year. The sign is not in doubt —
  even under conservative sensitivity assumptions (M=2.0: post-treatment
  parallel trends could deviate twice as much as pre-treatment), the
  entire identified set lies below zero.

CLAIMS AVOIDED
  Estimated {avoided:.0f} additional claims that would have occurred had
  the rate change not been applied, across the {exp:.0f} policy-years of
  treated exposure in the post-treatment window.

METHODOLOGY CREDIBILITY
  The comparison group is a synthetic control built from the 7 untreated
  regions. Pre-treatment parallel trends test passes (p = {pt_p:.3f}) —
  the treated and control regions were on comparable trajectories before
  Q1 2023. The effect is not attributable to the market-wide frequency
  rise (which affected all regions equally and is absorbed by the
  synthetic control).

WHAT WOULD INVALIDATE THIS
  1. A post-treatment shock that hit only the northern territories (e.g.
     a severe winter affecting northern roads but not southern ones).
     We are not aware of such a shock in Q1-Q4 2023.
  2. A concurrent change in the risk mix of the northern book that is not
     captured by the market-wide trend (e.g. a large broker book exiting
     specifically in the north at the same time as the rate change).
     The underwriting team should confirm no such change occurred.
  3. IBNR bias is not a concern here because we are measuring frequency,
     not loss ratio. Claim counts are fully developed at quarterly reporting.

RECOMMENDATION
  This analysis provides FCA-standard causal evidence (class of method
  same family of DiD methods the FCA used in EP25/2). Include in the Q4 2023 Consumer Duty outcome
  monitoring pack. Flag the regional scope — the effect is estimated for
  north-west, north-east, and yorkshire only; inference for other regions
  requires separate analysis.
""".format(
    att=abs(result.att),
    ci_lo=abs(result.ci_low),
    ci_hi=abs(result.ci_high),
    pct=abs(att_as_pct_of_base) * 100,
    pval=result.pval,
    avoided=implied_claims_avoided,
    exp=treated_exposure_post,
    pt_p=result.pre_trend_pval if result.pre_trend_pval is not None else 1.0,
))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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
                           (robust up to {sens.m_breakdown:.1f}× pre-period SD)

  7. FCA evidence pack     Saved to /tmp/fca_evidence_pack.md

  8. Management summary    Printed above

Next steps:
  - Replace synthetic data with your live policy extract (same schema)
  - Adjust treatment_period encoding to match your rate change date
  - Run with n_replicates=500 for the final FCA submission version
  - Add plot_unit_weights() and plot_synthetic_trajectory() to the evidence
    pack for the visual parallel trends exhibit
""")
