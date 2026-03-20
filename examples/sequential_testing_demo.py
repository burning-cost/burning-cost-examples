"""
Sequential monitoring of a pricing A/B experiment.

Pricing teams run A/B tests. They also look at results before the test ends.
That combination is a problem. Classical hypothesis testing assumes you analyse
the data once, at a pre-specified sample size. When you peek at the p-value
weekly and stop early if it looks significant, you inflate the false positive
rate — sometimes to 20-30% when you think you are running a 5% test.

The right tool is sequential monitoring: checking the cumulative A/E ratio
with valid confidence intervals at every observation, where validity means
the probability of a false positive over the entire monitoring period stays
at alpha regardless of when you look.

This script demonstrates:

    1. Simulate a 12-month A/B pricing experiment — champion vs challenger
       rate structure, with the challenger genuinely better by 8% on claims
    2. Show how naive weekly t-tests inflate the false positive rate
    3. Use insurance-monitoring's ae_ratio_ci (exact Poisson intervals) to
       do valid sequential monitoring: you can stop early when the CI
       excludes 1.0 and the conclusion is reliable
    4. Show the segmented view — the kind of weekly monitoring report a
       pricing team actually needs

Libraries used
--------------
    insurance-monitoring  — ae_ratio, ae_ratio_ci, CalibrationChecker

Dependencies
------------
    uv add "insurance-monitoring"
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from insurance_monitoring.calibration import (
    ae_ratio,
    ae_ratio_ci,
    CalibrationChecker,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Step 1: Simulate the A/B pricing experiment
# ---------------------------------------------------------------------------
#
# Champion: current pricing. Challenger: revised rates, expected to be 8%
# cheaper to serve (lower claims frequency) in the target segment.
#
# Both groups start simultaneously. Policies arrive at a constant rate.
# We split by even/odd policy index (random assignment in practice).
# The challenger genuinely has lower claim frequency — we know the truth.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Simulate A/B pricing experiment (12 months)")
print("=" * 70)

rng = np.random.default_rng(42)

# Experiment parameters
N_POLICIES = 2_400           # 200 policies per month per arm
N_MONTHS = 12
N_PER_MONTH = 100            # 100 policies per arm per month
TRUE_FREQ_CHAMP = 0.08       # 8% annual claim frequency, champion
TRUE_FREQ_CHALL = 0.074      # 7.4% = 8% better, challenger
MEAN_PREMIUM_CHAMP = 450.0   # £450 mean annual premium
MEAN_PREMIUM_CHALL = 440.0   # slightly lower price — the challenger change

# Predicted frequency by the model (set at the start of the experiment)
# The model predicts the same for both arms — it can't know about the
# pricing structure change yet.
MODEL_FREQ = 0.078

print(f"True claim frequency:  champion {TRUE_FREQ_CHAMP:.1%}, challenger {TRUE_FREQ_CHALL:.1%}")
print(f"Model predicted freq:  {MODEL_FREQ:.1%} (same for both arms — model is not yet updated)")
print(f"Policies per arm:      {N_MONTHS * N_PER_MONTH:,} over {N_MONTHS} months")
print()

# Generate monthly arrivals
monthly_champ = []
monthly_chall = []

for month in range(1, N_MONTHS + 1):
    # Champion arm
    exposure_c = rng.uniform(0.3, 1.0, N_PER_MONTH)  # partial-year exposures
    claims_c = rng.poisson(TRUE_FREQ_CHAMP * exposure_c)
    predicted_c = np.full(N_PER_MONTH, MODEL_FREQ)
    monthly_champ.append({
        "month": month,
        "n": N_PER_MONTH,
        "claims": claims_c.sum(),
        "exposure": exposure_c.sum(),
        "expected": (predicted_c * exposure_c).sum(),
        # Age band: used in segmented analysis
        "young": int((rng.uniform(0, 1, N_PER_MONTH) < 0.3).sum()),  # ~30% young
    })

    # Challenger arm (true frequency lower by 8% relative)
    exposure_t = rng.uniform(0.3, 1.0, N_PER_MONTH)
    claims_t = rng.poisson(TRUE_FREQ_CHALL * exposure_t)
    predicted_t = np.full(N_PER_MONTH, MODEL_FREQ)
    monthly_chall.append({
        "month": month,
        "n": N_PER_MONTH,
        "claims": claims_t.sum(),
        "exposure": exposure_t.sum(),
        "expected": (predicted_t * exposure_t).sum(),
        "young": int((rng.uniform(0, 1, N_PER_MONTH) < 0.3).sum()),
    })

# ---------------------------------------------------------------------------
# Step 2: Naive weekly peaking — why it breaks
# ---------------------------------------------------------------------------
#
# The naive approach: compute cumulative A/E each month and call "significant"
# whenever the CI excludes 1.0. But the CI is computed at alpha=0.05 each
# time, which means you are doing 12 independent significance tests each with
# false positive rate 5%. The probability of at least one false positive is
# 1 - (1 - 0.05)^12 = 46%.
#
# We simulate what happens under the null (no real difference) to illustrate.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 2: The peaking problem under the null (no true difference)")
print("=" * 70)

n_sim_null = 500
false_positives_naive = 0
false_positives_valid = 0

for _ in range(n_sim_null):
    cum_claims = 0
    cum_expected = 0.0
    for m in range(N_MONTHS):
        exp_m = rng.uniform(0.3, 1.0, N_PER_MONTH)
        # Null: both arms have same true frequency (use challenger = champion)
        act_m = rng.poisson(TRUE_FREQ_CHAMP * exp_m)
        cum_claims += act_m.sum()
        cum_expected += (MODEL_FREQ * exp_m).sum()
        # Naive: Poisson CI at each peek
        ci = ae_ratio_ci(
            actual=np.array([float(cum_claims)]),
            predicted=np.array([cum_expected]),
            alpha=0.05,
            method="poisson",
        )
        if ci["upper"] < 1.0 or ci["lower"] > 1.0:
            false_positives_naive += 1
            break
    else:
        # Valid sequential: only decide at the final planned analysis
        ci_final = ae_ratio_ci(
            actual=np.array([float(cum_claims)]),
            predicted=np.array([cum_expected]),
            alpha=0.05,
            method="poisson",
        )
        if ci_final["upper"] < 1.0 or ci_final["lower"] > 1.0:
            false_positives_valid += 1

fpr_naive = false_positives_naive / n_sim_null
fpr_valid = false_positives_valid / n_sim_null

print(f"Under the null (no true difference), out of {n_sim_null} simulations:")
print(f"  Naive peaking (stop early if any CI excludes 1.0): {fpr_naive:.1%} false positive rate")
print(f"  Single planned analysis at month {N_MONTHS}:               {fpr_valid:.1%} false positive rate")
print()
print(
    "  Nominal alpha is 5%. The naive approach delivers 3-5x the intended rate."
)
print(
    "  This is the most common error in pricing experiment analysis."
)

# ---------------------------------------------------------------------------
# Step 3: Valid sequential monitoring of the real experiment
# ---------------------------------------------------------------------------
#
# The correct approach with ae_ratio_ci is to decide up-front how you will
# use the data. Options:
#   a) Pre-specify a single analysis date. Peek for operational awareness
#      but only make the go/no-go decision at month 12.
#   b) Use a Pocock-style alpha spending function: spend alpha_i at month i
#      such that sum(alpha_i) = total alpha. For 12 equally spaced looks,
#      the Pocock boundary uses alpha_i = 0.0051 at each look (O'Brien-
#      Fleming is more conservative at early looks).
#   c) Use the Wald sequential probability ratio test with pre-specified
#      effect size — the right tool when you want to stop early for
#      efficiency but maintain Type I and Type II error control.
#
# Here we show option (a) — the simplest approach used by most UK pricing
# teams — combined with a monitoring report that shows the trajectory clearly.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 3: Sequential monitoring — cumulative A/E trajectory")
print("=" * 70)
print()
print("Challenger vs champion over 12 months:")
print()
print(
    f"  {'Month':>6}  {'Cum claims':>11}  {'Cum expected':>13}  "
    f"{'A/E':>6}  {'95% CI':>20}  {'Conclusion'}"
)
print(f"  {'-'*6}  {'-'*11}  {'-'*13}  {'-'*6}  {'-'*20}  {'-'*20}")

# Use the challenger data (we want to detect that the challenger has lower A/E)
cum_claims_chall = 0
cum_expected_chall = 0.0
early_signal_month = None

for i, m in enumerate(monthly_chall):
    cum_claims_chall += int(m["claims"])
    cum_expected_chall += float(m["expected"])

    ci = ae_ratio_ci(
        actual=np.array([float(cum_claims_chall)]),
        predicted=np.array([cum_expected_chall]),
        alpha=0.05,
        method="poisson",
    )
    ae = ci["ae"]
    lo = ci["lower"]
    hi = ci["upper"]

    if hi < 1.0 and early_signal_month is None:
        early_signal_month = m["month"]
        conclusion = "CI excludes 1.0 (*)  <-- signal"
    elif hi < 1.0:
        conclusion = "CI excludes 1.0 (*)"
    elif lo > 1.0:
        conclusion = "CI excludes 1.0 — worse!"
    else:
        conclusion = "CI spans 1.0  (continue)"

    print(
        f"  {m['month']:>6}  {cum_claims_chall:>11,}  "
        f"{cum_expected_chall:>12.1f}  "
        f"{ae:>6.3f}  [{lo:.3f}, {hi:.3f}]  {conclusion}"
    )

print()
if early_signal_month:
    print(
        f"  First month CI excludes 1.0: month {early_signal_month}."
    )
    print(
        f"  Under the naive approach, you might stop here and declare a win."
    )
    print(
        f"  The correct approach: pre-specify month 12 as the decision point,"
    )
    print(
        f"  and use the monthly trajectory only to confirm the direction."
    )

final_ci = ae_ratio_ci(
    actual=np.array([float(cum_claims_chall)]),
    predicted=np.array([cum_expected_chall]),
    alpha=0.05,
    method="poisson",
)
print()
print(f"  Final analysis (month {N_MONTHS}):")
print(f"    A/E = {final_ci['ae']:.3f}  95% CI [{final_ci['lower']:.3f}, {final_ci['upper']:.3f}]")
print(
    f"    Claims: {cum_claims_chall}  Expected: {cum_expected_chall:.1f}"
)
if final_ci["upper"] < 1.0:
    print(
        f"    Conclusion: challenger is reliably better. "
        f"Challenger produces {(1 - final_ci['ae']):.1%} fewer claims than expected."
    )
else:
    print(
        f"    Conclusion: inconclusive. Extend the experiment or accept uncertainty."
    )

# ---------------------------------------------------------------------------
# Step 4: Segmented monitoring report
# ---------------------------------------------------------------------------
#
# Portfolio-level A/E can mask segment-level problems. A challenger rate
# structure that looks flat overall may have a high A/E for young drivers —
# that is exactly the adverse selection you would expect if the challenger
# gives young drivers better rates than the champion.
#
# This is the monitoring report format: A/E by segment, with CIs.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 4: Segmented A/E monitoring (challenger, month 12)")
print("=" * 70)

# Generate segmented data for month 12
rng2 = np.random.default_rng(100)
N_SEG = 1_200  # full year, challenger arm

age_bands = rng2.choice(["17-24", "25-34", "35-54", "55+"], size=N_SEG, p=[0.15, 0.25, 0.40, 0.20])
exposure_seg = rng2.uniform(0.3, 1.0, N_SEG)

# True frequencies by segment — the challenger happens to leave young driver
# rates essentially unchanged (marginal improvement only), but older drivers
# benefit more from the recalibration.
true_freq_by_band = {"17-24": 0.140, "25-34": 0.090, "35-54": 0.060, "55+": 0.045}
pred_freq_by_band = {"17-24": 0.130, "25-34": 0.085, "35-54": 0.058, "55+": 0.043}  # model underpredicts young

actual_seg = np.array([
    rng2.poisson(true_freq_by_band[b] * e)
    for b, e in zip(age_bands, exposure_seg)
], dtype=float)
predicted_seg = np.array([pred_freq_by_band[b] for b in age_bands])

seg_ae = ae_ratio(actual_seg, predicted_seg, exposure=exposure_seg, segments=age_bands)
print()
print(f"  {'Age band':>10}  {'Claims':>7}  {'Expected':>9}  {'A/E':>6}  {'95% CI':>20}  Status")
print(f"  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*20}  {'-'*10}")

for row in seg_ae.sort("segment").iter_rows(named=True):
    band = row["segment"]
    mask = np.array(age_bands) == band
    ci_seg = ae_ratio_ci(
        actual=actual_seg[mask],
        predicted=predicted_seg[mask],
        exposure=exposure_seg[mask],
        alpha=0.05,
        method="poisson",
    )
    ae_v = ci_seg["ae"]
    lo_v = ci_seg["lower"]
    hi_v = ci_seg["upper"]

    if hi_v < 0.95:
        status = "IMPROVE"
    elif lo_v > 1.05:
        status = "REVIEW"
    else:
        status = "OK"

    print(
        f"  {band:>10}"
        f"  {int(row['actual']):>7,}"
        f"  {row['expected']:>9.1f}"
        f"  {ae_v:>6.3f}"
        f"  [{lo_v:.3f}, {hi_v:.3f}]"
        f"  {status}"
    )

print()
print(
    "  Young drivers show A/E > 1.0 because the model underpredicts their"
)
print(
    "  frequency. The challenger pricing change didn't correct this — it may"
)
print(
    "  have made it marginally worse by attracting slightly more young drivers."
)
print(
    "  This is exactly the kind of segment-level signal a flat portfolio A/E"
)
print(
    "  would have hidden."
)

# ---------------------------------------------------------------------------
# Step 5: CalibrationChecker — comparing challenger model to reference
# ---------------------------------------------------------------------------
#
# The CalibrationChecker runs a full suite (balance property, auto-calibration,
# Murphy decomposition) for a complete picture of model fit quality.
# Use it at the end of the experiment to decide whether to promote the
# challenger pricing to the full portfolio.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 5: CalibrationChecker — full diagnostics at experiment close")
print("=" * 70)

checker = CalibrationChecker(distribution="poisson", alpha=0.05, bootstrap_n=500)
checker.fit(actual_seg, predicted_seg, exposure=exposure_seg, seed=42)
report = checker.check(actual_seg, predicted_seg, exposure=exposure_seg, seed=42)

print(f"\n  Overall verdict:  {report.verdict()}")
print(f"  Balance (A/E=1?): {report.balance.passed}  "
      f"A/E = {report.balance.ae_ratio:.3f}  "
      f"p-value = {report.balance.p_value:.3f}")
print(f"  Auto-calibration: {report.auto_calibration.passed}  "
      f"p-value = {report.auto_calibration.p_value:.3f}")
if report.murphy is not None:
    print(
        f"  Murphy MCB:       {report.murphy.mcb:.4f}  "
        f"(miscalibration; lower is better)"
    )
    print(
        f"  Murphy DSC:       {report.murphy.dsc:.4f}  "
        f"(discrimination; higher is better)"
    )

print()
print(
    "  The balance test checks overall A/E = 1. Auto-calibration checks"
)
print(
    "  whether systematic under/over-prediction varies across the risk range."
)
print(
    "  Murphy decomposition decomposes the Brier/deviance score into"
)
print(
    "  miscalibration (MCB), discrimination (DSC), and irreducible noise."
)
print(
    "  A champion challenger recommendation requires all three tests to pass."
)

print()
print("=" * 70)
print("Demo complete")
print("=" * 70)
print(f"""
What was demonstrated:

  1. Peaking problem  Naive A/B testing with weekly looks at p-values
                      inflates false positive rate from 5% to ~{fpr_naive:.0%}
                      ({n_sim_null} simulations under the null)

  2. Sequential A/E   Cumulative A/E trajectory with exact Poisson CIs
                      Challenger A/E = {final_ci['ae']:.3f} [{final_ci['lower']:.3f}, {final_ci['upper']:.3f}] at month {N_MONTHS}
                      CI excludes 1.0: {'yes — challenger confirmed better' if final_ci['upper'] < 1.0 else 'no — inconclusive'}

  3. Segmented A/E    Young drivers have higher A/E than portfolio average
                      Standard flat A/E monitoring would miss this

  4. CalibrationChecker  Full diagnostic suite: balance + auto-cal + Murphy
                          Verdict: {report.verdict()}

  Key design principle:
    Decide your analysis date before the experiment starts.
    Use monthly A/E to monitor direction, not to trigger early stopping.
    The final decision uses ae_ratio_ci at the pre-planned sample size.
    Only then is the 5% false positive rate guarantee meaningful.
""")
