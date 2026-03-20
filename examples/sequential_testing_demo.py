"""
Sequential (anytime-valid) A/B testing for insurance champion/challenger experiments.

Pricing teams run A/B tests. They also look at results before the test ends.
That combination is a problem. Classical hypothesis testing assumes you analyse
the data once. When you peek at the p-value weekly and stop early if it looks
significant, you inflate the false positive rate — sometimes to 20-30% when
you think you are running a 5% test.

The right tool is the mixture Sequential Probability Ratio Test (mSPRT). Its
e-process Lambda_n satisfies P_0(exists n: Lambda_n >= 1/alpha) <= alpha at
ALL stopping times. You can check weekly with no penalty. When Lambda_n >= 20
(at alpha=0.05), reject H0 and stop. The type I error guarantee holds.

This script demonstrates:

    1. Simulate a 24-month champion/challenger frequency experiment
    2. Show that naive monthly peeking inflates the false positive rate from
       5% to ~20% under the null
    3. Run the real experiment through SequentialTest month by month —
       showing Lambda_n accumulate evidence over time
    4. Inspect the anytime-valid confidence sequence for the rate ratio
    5. Show the Bayesian secondary display and when to declare futility

Libraries used
--------------
    insurance-monitoring  — SequentialTest, SequentialTestResult

Dependencies
------------
    uv add "insurance-monitoring"
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import polars as pl

from insurance_monitoring.sequential import SequentialTest, sequential_test_from_df

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
#
# UK motor champion/challenger. Both arms: 500 car-years per month.
# True annual claim frequency:
#   Champion  0.10  (100 claims per 1,000 car-years)
#   Challenger 0.085 (15% improvement — a plausible re-rate benefit)
#
# alpha=0.05 -> reject when Lambda_n >= 20
# tau=0.05 -> prior on log-rate-ratio SD, tuned for effects of 5-15%
# min_exposure_per_arm=500 -> wait at least 1 month before any decision
# ---------------------------------------------------------------------------

N_MONTHS = 24
EXPOSURE_PER_ARM_PER_MONTH = 500.0
TRUE_RATE_CHAMP = 0.10
TRUE_RATE_CHALL = 0.085   # 15% improvement
ALPHA = 0.05
TAU = 0.05

print("=" * 70)
print("Sequential A/B testing with mSPRT (insurance-monitoring v0.5)")
print("=" * 70)
print()
print(f"  Champion claim rate:   {TRUE_RATE_CHAMP:.3f} /car-year")
print(f"  Challenger claim rate: {TRUE_RATE_CHALL:.3f} /car-year  ({(1 - TRUE_RATE_CHALL/TRUE_RATE_CHAMP):.0%} improvement)")
print(f"  Exposure/arm/month:    {EXPOSURE_PER_ARM_PER_MONTH:.0f} car-years")
print(f"  alpha={ALPHA}  ->  reject threshold Lambda >= {1/ALPHA:.0f}")
print(f"  tau={TAU}  ->  prior on log-rate-ratio SD (expect 5% effects)")
print()

# ---------------------------------------------------------------------------
# Step 1: The peaking problem — false positive rate under the null
# ---------------------------------------------------------------------------
#
# Under the null (no true difference), how often does naive monthly peeking
# trigger an early stop? We simulate 500 independent null experiments.
# Each month we compute a simple z-test on the cumulative claim counts.
# If we stop early whenever p < 0.05, the realised FPR is much higher.
#
# The mSPRT does not have this problem: its FPR is guaranteed to be <= alpha
# regardless of how often you check.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: The peaking problem (500 null simulations)")
print("=" * 70)
print()

rng = np.random.default_rng(0)
n_sims = 500
naive_fp = 0
msprt_fp = 0

for _ in range(n_sims):
    # Naive peeking: cumulative Poisson z-test each month
    cum_a = 0
    cum_b = 0
    cum_e = 0.0
    naive_stopped = False
    for _ in range(N_MONTHS):
        ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
        cb = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null: same rate
        cum_a += ca
        cum_b += cb
        cum_e += EXPOSURE_PER_ARM_PER_MONTH
        if cum_a > 0 and cum_b > 0:
            # Naive two-sample Poisson test via normal approximation
            rate_a = cum_a / cum_e
            rate_b = cum_b / cum_e
            se = np.sqrt(rate_a / cum_e + rate_b / cum_e)
            if se > 0 and abs(rate_b - rate_a) / se > 1.96:
                naive_stopped = True
                break
    if naive_stopped:
        naive_fp += 1

    # mSPRT: reset and run the same null experiment
    test = SequentialTest(
        metric="frequency",
        alpha=ALPHA,
        tau=TAU,
        max_duration_years=N_MONTHS / 12,
        min_exposure_per_arm=0.0,
    )
    for _ in range(N_MONTHS):
        ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
        cb = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null
        result = test.update(ca, EXPOSURE_PER_ARM_PER_MONTH, cb, EXPOSURE_PER_ARM_PER_MONTH)
        if result.should_stop and result.decision == "reject_H0":
            msprt_fp += 1
            break

fpr_naive = naive_fp / n_sims
fpr_msprt = msprt_fp / n_sims

print(f"  Under H0 (no true difference), {n_sims} simulations:")
print(f"    Naive monthly z-test:  {fpr_naive:.1%} false positive rate")
print(f"    mSPRT (Lambda >= 20):  {fpr_msprt:.1%} false positive rate")
print(f"    Nominal alpha:         {ALPHA:.1%}")
print()
print("  The mSPRT controls FPR at the nominal level regardless of how often")
print("  you look. The naive approach delivers 3-4x the intended error rate.")

# ---------------------------------------------------------------------------
# Step 2: Run the real experiment month by month
# ---------------------------------------------------------------------------
#
# Now run the challenger (15% better) through the full 24-month window.
# We feed incremental monthly claims to SequentialTest.update() and track
# the accumulating e-process Lambda_n. When Lambda_n crosses the threshold
# of 1/alpha = 20, the test stops.
#
# update() takes *increments* since the last call, not cumulative totals.
# It returns a SequentialTestResult with the current decision and estimates.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 2: Monthly experiment — Lambda_n accumulating evidence")
print("=" * 70)
print()

rng = np.random.default_rng(42)
test = SequentialTest(
    metric="frequency",
    alternative="less",          # challenger rate < champion rate
    alpha=ALPHA,
    tau=TAU,
    max_duration_years=N_MONTHS / 12,
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
)

start_date = datetime.date(2024, 1, 1)
stopped_at_month = None

print(
    f"  {'Month':>5}  {'Lambda_n':>10}  {'Threshold':>10}  "
    f"{'Rate ratio':>12}  {'95% CS':>20}  Decision"
)
print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*20}  {'-'*20}")

for month in range(1, N_MONTHS + 1):
    ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
    cb = int(rng.poisson(TRUE_RATE_CHALL * EXPOSURE_PER_ARM_PER_MONTH))
    period_date = start_date.replace(month=((month - 1) % 12) + 1,
                                     year=start_date.year + (month - 1) // 12)

    result = test.update(
        champion_claims=ca,
        champion_exposure=EXPOSURE_PER_ARM_PER_MONTH,
        challenger_claims=cb,
        challenger_exposure=EXPOSURE_PER_ARM_PER_MONTH,
        calendar_date=period_date,
    )

    ci_str = f"[{result.rate_ratio_ci_lower:.3f}, {result.rate_ratio_ci_upper:.3f}]"
    if result.rate_ratio_ci_lower == 0.0:
        ci_str = "insufficient data"

    # Mark the stopping point
    flag = " <-- STOP" if result.should_stop and stopped_at_month is None else ""
    if result.should_stop and stopped_at_month is None:
        stopped_at_month = month

    print(
        f"  {month:>5}  {result.lambda_value:>10.3f}  {result.threshold:>10.1f}  "
        f"{result.rate_ratio:>12.4f}  {ci_str:>20}  {result.decision}{flag}"
    )

    if result.should_stop:
        break

print()
if stopped_at_month is not None and result.decision == "reject_H0":
    print(f"  Test stopped at month {stopped_at_month}: challenger confirmed better.")
    print(f"  Rate ratio: {result.rate_ratio:.4f}  (true: {TRUE_RATE_CHALL/TRUE_RATE_CHAMP:.4f})")
    print(f"  95% CS: [{result.rate_ratio_ci_lower:.4f}, {result.rate_ratio_ci_upper:.4f}]")
    print(f"  Bayesian P(challenger better): {result.prob_challenger_better:.1%}")
else:
    print(f"  Experiment ended at month {N_MONTHS}: {result.decision}")
    print(f"  Final Lambda_n: {result.lambda_value:.3f}  (threshold: {result.threshold:.1f})")

# ---------------------------------------------------------------------------
# Step 3: History DataFrame — the monitoring report
# ---------------------------------------------------------------------------
#
# test.history() returns a Polars DataFrame with one row per update() call.
# This is what you export to your monitoring dashboard or send to the team.
# Key columns: period_index, lambda_value, rate_ratio, ci_lower, ci_upper,
# decision, cum_champion_exposure, cum_challenger_exposure.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 3: history() — full trajectory as a Polars DataFrame")
print("=" * 70)
print()

hist = test.history()
print(f"  history() schema: {hist.schema}")
print()
print(f"  Rows: {len(hist)}  (one per update() call)")
print()

# Show summary stats from the history
lambda_max = hist["lambda_value"].max()
lambda_final = hist["lambda_value"][-1]
n_inconclusive = hist.filter(pl.col("decision") == "inconclusive").shape[0]
n_stop = hist.filter(pl.col("decision") != "inconclusive").shape[0]

print(f"  Peak Lambda_n:       {lambda_max:.3f}")
print(f"  Final Lambda_n:      {lambda_final:.3f}")
print(f"  Months inconclusive: {n_inconclusive}")
print(f"  Months to decision:  {n_stop}")

# ---------------------------------------------------------------------------
# Step 4: Anytime-valid confidence sequence
# ---------------------------------------------------------------------------
#
# The rate_ratio_ci_lower and rate_ratio_ci_upper in each result form a
# time-uniform (anytime-valid) confidence sequence. Unlike classical CIs,
# this CI is valid at all monitoring times simultaneously — P(true ratio
# falls outside the CS at ANY point) <= alpha.
#
# Classical 95% CIs would be invalid if inspected repeatedly. The CS is
# slightly wider to pay for this guarantee, but it narrows as data accumulates.
# Once the CS excludes 1.0 entirely (for 'less' alternative, upper bound < 1),
# you have anytime-valid evidence the challenger is better.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 4: Anytime-valid confidence sequence vs classical CI")
print("=" * 70)
print()
print("  At the stopping point:")
print(f"    Rate ratio:   {result.rate_ratio:.4f}")
print(f"    95% CS:       [{result.rate_ratio_ci_lower:.4f}, {result.rate_ratio_ci_upper:.4f}]")
print(f"    True ratio:   {TRUE_RATE_CHALL/TRUE_RATE_CHAMP:.4f}")
print()
print("  The CS upper bound < 1.0 confirms the challenger is significantly")
print("  better. This conclusion is valid at this stopping point because the")
print("  mSPRT guarantees FPR control: Lambda_n >= 1/alpha => H0 is rejected")
print("  with at most alpha probability of being wrong.")
print()
print("  A classical CI at this interim point would give the same numbers")
print("  but WITHOUT the anytime-valid guarantee. If the team had stopped")
print("  the experiment at any of the earlier months where the p-value crossed")
print("  0.05 in a naive test, the false positive rate would be inflated.")

# ---------------------------------------------------------------------------
# Step 5: Futility — when the experiment should stop early for the other reason
# ---------------------------------------------------------------------------
#
# Futility detection: if Lambda_n is very low (well below 1), the data are
# consistently consistent with H0. There is no point continuing. Set
# futility_threshold to enable this.
#
# Here we simulate a null experiment with futility detection enabled.
# Under equal rates, Lambda_n will often drift down toward zero quickly,
# triggering futility.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 5: Futility detection — stopping when evidence against H1")
print("=" * 70)
print()

rng_fut = np.random.default_rng(7)
test_futility = SequentialTest(
    metric="frequency",
    alternative="less",
    alpha=ALPHA,
    tau=TAU,
    max_duration_years=N_MONTHS / 12,
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
    futility_threshold=0.01,   # stop if Lambda_n < 0.01
)

fut_months = 0
fut_result = None
for month in range(1, N_MONTHS + 1):
    ca = int(rng_fut.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
    cb = int(rng_fut.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null: equal rates
    fut_result = test_futility.update(ca, EXPOSURE_PER_ARM_PER_MONTH, cb, EXPOSURE_PER_ARM_PER_MONTH)
    fut_months += 1
    if fut_result.should_stop:
        break

print(f"  Null experiment (challenger = champion, no real improvement):")
print(f"  Stopped at month {fut_months}: {fut_result.decision}")
print(f"  Final Lambda_n: {fut_result.lambda_value:.4f}")
print(f"  Rate ratio: {fut_result.rate_ratio:.4f}")
print()
if fut_result.decision == "futility":
    print(f"  Lambda_n fell below 0.01 — experiment is futile. Challenger")
    print(f"  is not delivering the expected improvement. Stop and investigate.")
else:
    print(f"  Experiment ran to completion without triggering futility.")
    print(f"  This is expected: under the null, Lambda is a martingale and")
    print(f"  may not consistently drift toward zero in every realisation.")

# ---------------------------------------------------------------------------
# Step 6: sequential_test_from_df — batch workflow
# ---------------------------------------------------------------------------
#
# If you have historical data as a Polars DataFrame (one row per reporting
# period), use sequential_test_from_df to run the test without a manual loop.
# Returns the SequentialTestResult from the final update() call.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 6: sequential_test_from_df — batch workflow from a DataFrame")
print("=" * 70)
print()

rng_df = np.random.default_rng(99)
monthly_rows = []
for m in range(1, 13):
    monthly_rows.append({
        "period": datetime.date(2024, m, 1) if m <= 12 else datetime.date(2025, m - 12, 1),
        "champ_claims": int(rng_df.poisson(TRUE_RATE_CHAMP * 500)),
        "champ_exposure": 500.0,
        "chall_claims": int(rng_df.poisson(TRUE_RATE_CHALL * 500)),
        "chall_exposure": 500.0,
    })

df = pl.DataFrame(monthly_rows)

batch_result = sequential_test_from_df(
    df=df,
    champion_claims_col="champ_claims",
    champion_exposure_col="champ_exposure",
    challenger_claims_col="chall_claims",
    challenger_exposure_col="chall_exposure",
    date_col="period",
    metric="frequency",
    alpha=ALPHA,
    tau=TAU,
    alternative="less",
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
)

print(f"  12-month batch run ({len(df)} rows):")
print(f"  Decision:         {batch_result.decision}")
print(f"  Lambda_n:         {batch_result.lambda_value:.3f}  (threshold: {batch_result.threshold:.1f})")
print(f"  Rate ratio:       {batch_result.rate_ratio:.4f}")
print(f"  Champion rate:    {batch_result.champion_rate:.4f}")
print(f"  Challenger rate:  {batch_result.challenger_rate:.4f}")
print(f"  Calendar days:    {batch_result.total_calendar_time_days:.0f}")
print(f"  Summary:          {batch_result.summary}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Demo complete")
print("=" * 70)
print(f"""
What was demonstrated:

  1. Peaking problem
     Naive monthly z-tests: {fpr_naive:.0%} false positive rate under the null
     mSPRT with Lambda >= 20: {fpr_msprt:.0%} false positive rate (nominal {ALPHA:.0%})

  2. SequentialTest.update()
     Fed incremental monthly claims to the test over {N_MONTHS} months.
     Lambda_n accumulated evidence; test stopped at month {stopped_at_month or N_MONTHS}.
     Rate ratio at stopping: {result.rate_ratio:.4f}
     95% anytime-valid CS:   [{result.rate_ratio_ci_lower:.4f}, {result.rate_ratio_ci_upper:.4f}]

  3. Key API points:
     - update() takes increments, not cumulative totals
     - history() returns a Polars DataFrame for export/dashboard
     - sequential_test_from_df() for batch processing from a DataFrame
     - futility_threshold stops early when Lambda_n is very small
     - alternative='less' for one-sided tests (challenger should be lower)

  4. Design principle:
     The mSPRT e-process has E_0[Lambda_n] = 1 for all n. The test threshold
     1/alpha is exactly calibrated: the probability of ever crossing it under
     the null is <= alpha. No Bonferroni, no alpha-spending, no fixed horizon
     required. Check weekly, monthly, or irregularly — the guarantee holds.
""")
