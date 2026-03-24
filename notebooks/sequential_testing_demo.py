# Databricks notebook source

# MAGIC %md
# MAGIC # Sequential (Anytime-Valid) A/B Testing for Champion/Challenger Experiments
# MAGIC
# MAGIC Pricing teams run A/B tests. They also look at results before the test ends. That combination is a problem. Classical hypothesis testing assumes you analyse the data once. When you peek at the p-value weekly and stop early if it looks significant, you inflate the false positive rate — sometimes to 20-30% when you think you're running a 5% test.
# MAGIC
# MAGIC The right tool is the mixture Sequential Probability Ratio Test (mSPRT). Its e-process Lambda_n satisfies `P_0(exists n: Lambda_n >= 1/alpha) <= alpha` at ALL stopping times. You can check weekly with no penalty.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Simulate the peaking problem — show naive FPR inflation
# MAGIC 2. Run a real experiment month-by-month showing Lambda_n accumulate evidence
# MAGIC 3. Inspect the anytime-valid confidence sequence
# MAGIC 4. Futility detection — stopping when there is no real improvement
# MAGIC 5. Batch workflow via `sequential_test_from_df`

# COMMAND ----------

# MAGIC %pip install "insurance-monitoring" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import datetime
import warnings

import numpy as np
import polars as pl

from insurance_monitoring.sequential import SequentialTest, sequential_test_from_df

warnings.filterwarnings("ignore")

# Experiment parameters
N_MONTHS = 24
EXPOSURE_PER_ARM_PER_MONTH = 500.0
TRUE_RATE_CHAMP = 0.10       # 100 claims per 1,000 car-years
TRUE_RATE_CHALL = 0.085      # 15% improvement — plausible re-rate benefit
ALPHA = 0.05
TAU = 0.05                   # prior on log-rate-ratio SD, tuned for 5-15% effects

print(f"Champion claim rate:   {TRUE_RATE_CHAMP:.3f} /car-year")
print(f"Challenger claim rate: {TRUE_RATE_CHALL:.3f} /car-year  ({(1 - TRUE_RATE_CHALL/TRUE_RATE_CHAMP):.0%} improvement)")
print(f"alpha={ALPHA}  ->  reject threshold Lambda >= {1/ALPHA:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: The Peaking Problem — False Positive Rate Under the Null
# MAGIC
# MAGIC Under the null (no true difference), how often does naive monthly peeking trigger an early stop? We simulate 500 independent null experiments. The mSPRT does not have this problem: its FPR is guaranteed to be <= alpha regardless of how often you check.

# COMMAND ----------

rng = np.random.default_rng(0)
n_sims = 500
naive_fp = 0
msprt_fp = 0

for _ in range(n_sims):
    # Naive peeking: cumulative Poisson z-test each month
    cum_a = cum_b = 0
    cum_e = 0.0
    naive_stopped = False
    for _ in range(N_MONTHS):
        ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
        cb = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null
        cum_a += ca; cum_b += cb; cum_e += EXPOSURE_PER_ARM_PER_MONTH
        if cum_a > 0 and cum_b > 0:
            rate_a = cum_a / cum_e; rate_b = cum_b / cum_e
            se = np.sqrt(rate_a / cum_e + rate_b / cum_e)
            if se > 0 and abs(rate_b - rate_a) / se > 1.96:
                naive_stopped = True; break
    if naive_stopped:
        naive_fp += 1

    # mSPRT: same null experiment
    test = SequentialTest(
        metric="frequency", alpha=ALPHA, tau=TAU,
        max_duration_years=N_MONTHS / 12, min_exposure_per_arm=0.0,
    )
    for _ in range(N_MONTHS):
        ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
        cb = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null
        result = test.update(ca, EXPOSURE_PER_ARM_PER_MONTH, cb, EXPOSURE_PER_ARM_PER_MONTH)
        if result.should_stop and result.decision == "reject_H0":
            msprt_fp += 1; break

fpr_naive = naive_fp / n_sims
fpr_msprt = msprt_fp / n_sims

print(f"Under H0 (no true difference), {n_sims} simulations:")
print(f"  Naive monthly z-test:  {fpr_naive:.1%} false positive rate")
print(f"  mSPRT (Lambda >= 20):  {fpr_msprt:.1%} false positive rate")
print(f"  Nominal alpha:         {ALPHA:.1%}")
print(f"\nThe mSPRT controls FPR at the nominal level regardless of how often you look.")
print(f"The naive approach delivers {fpr_naive/ALPHA:.0f}x the intended error rate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Monthly Experiment — Lambda_n Accumulating Evidence
# MAGIC
# MAGIC Now run the challenger (15% better) through the full 24-month window. We feed incremental monthly claims to `SequentialTest.update()` and track the accumulating e-process Lambda_n. When Lambda_n crosses the threshold of 1/alpha = 20, the test stops.
# MAGIC
# MAGIC `update()` takes *increments* since the last call, not cumulative totals.

# COMMAND ----------

rng = np.random.default_rng(42)
test = SequentialTest(
    metric="frequency",
    alternative="less",          # challenger rate < champion rate
    alpha=ALPHA, tau=TAU,
    max_duration_years=N_MONTHS / 12,
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
)

start_date = datetime.date(2024, 1, 1)
stopped_at_month = None

print(f"  {'Month':>5}  {'Lambda_n':>10}  {'Rate ratio':>12}  {'95% CS':>22}  Decision")
print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*22}  {'-'*20}")

for month in range(1, N_MONTHS + 1):
    ca = int(rng.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
    cb = int(rng.poisson(TRUE_RATE_CHALL * EXPOSURE_PER_ARM_PER_MONTH))
    period_date = start_date.replace(
        month=((month - 1) % 12) + 1,
        year=start_date.year + (month - 1) // 12,
    )

    result = test.update(
        champion_claims=ca, champion_exposure=EXPOSURE_PER_ARM_PER_MONTH,
        challenger_claims=cb, challenger_exposure=EXPOSURE_PER_ARM_PER_MONTH,
        calendar_date=period_date,
    )

    ci_str = f"[{result.rate_ratio_ci_lower:.3f}, {result.rate_ratio_ci_upper:.3f}]"
    if result.rate_ratio_ci_lower == 0.0:
        ci_str = "insufficient data"

    flag = " <-- STOP" if result.should_stop and stopped_at_month is None else ""
    if result.should_stop and stopped_at_month is None:
        stopped_at_month = month

    print(f"  {month:>5}  {result.lambda_value:>10.3f}  {result.rate_ratio:>12.4f}  {ci_str:>22}  {result.decision}{flag}")

    if result.should_stop:
        break

print()
if stopped_at_month is not None and result.decision == "reject_H0":
    print(f"Test stopped at month {stopped_at_month}: challenger confirmed better.")
    print(f"Rate ratio: {result.rate_ratio:.4f}  (true: {TRUE_RATE_CHALL/TRUE_RATE_CHAMP:.4f})")
    print(f"95% CS: [{result.rate_ratio_ci_lower:.4f}, {result.rate_ratio_ci_upper:.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: History DataFrame — the Monitoring Report
# MAGIC
# MAGIC `test.history()` returns a Polars DataFrame with one row per `update()` call. This is what you export to your monitoring dashboard. Key columns: `period_index`, `lambda_value`, `rate_ratio`, `ci_lower`, `ci_upper`, `decision`.

# COMMAND ----------

hist = test.history()
print(f"history() schema: {hist.schema}")
print(f"Rows: {len(hist)}  (one per update() call)")

lambda_max = hist["lambda_value"].max()
n_inconclusive = hist.filter(pl.col("decision") == "inconclusive").shape[0]

print(f"\nPeak Lambda_n:       {lambda_max:.3f}")
print(f"Final Lambda_n:      {hist['lambda_value'][-1]:.3f}")
print(f"Months inconclusive: {n_inconclusive}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Futility Detection
# MAGIC
# MAGIC If Lambda_n is very low (well below 1), the data are consistently consistent with H0. There is no point continuing. Set `futility_threshold` to enable this — under the null, Lambda_n will often drift toward zero quickly.

# COMMAND ----------

rng_fut = np.random.default_rng(7)
test_futility = SequentialTest(
    metric="frequency", alternative="less",
    alpha=ALPHA, tau=TAU,
    max_duration_years=N_MONTHS / 12,
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
    futility_threshold=0.01,  # stop if Lambda_n < 0.01
)

fut_months = 0
fut_result = None
for month in range(1, N_MONTHS + 1):
    ca = int(rng_fut.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))
    cb = int(rng_fut.poisson(TRUE_RATE_CHAMP * EXPOSURE_PER_ARM_PER_MONTH))  # null
    fut_result = test_futility.update(ca, EXPOSURE_PER_ARM_PER_MONTH, cb, EXPOSURE_PER_ARM_PER_MONTH)
    fut_months += 1
    if fut_result.should_stop:
        break

print(f"Null experiment (equal rates): stopped at month {fut_months}: {fut_result.decision}")
print(f"Final Lambda_n: {fut_result.lambda_value:.4f}")
if fut_result.decision == "futility":
    print("Lambda_n fell below 0.01 — experiment is futile. Challenger is not delivering improvement.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: sequential_test_from_df — Batch Workflow
# MAGIC
# MAGIC If you have historical data as a Polars DataFrame (one row per reporting period), use `sequential_test_from_df` to run the test without a manual loop.

# COMMAND ----------

rng_df = np.random.default_rng(99)
monthly_rows = []
for m in range(1, 13):
    monthly_rows.append({
        "period": datetime.date(2024, m, 1),
        "champ_claims": int(rng_df.poisson(TRUE_RATE_CHAMP * 500)),
        "champ_exposure": 500.0,
        "chall_claims": int(rng_df.poisson(TRUE_RATE_CHALL * 500)),
        "chall_exposure": 500.0,
    })

df = pl.DataFrame(monthly_rows)

batch_result = sequential_test_from_df(
    df=df,
    champion_claims_col="champ_claims", champion_exposure_col="champ_exposure",
    challenger_claims_col="chall_claims", challenger_exposure_col="chall_exposure",
    date_col="period",
    metric="frequency", alpha=ALPHA, tau=TAU, alternative="less",
    min_exposure_per_arm=EXPOSURE_PER_ARM_PER_MONTH,
)

print(f"12-month batch run ({len(df)} rows):")
print(f"  Decision:        {batch_result.decision}")
print(f"  Lambda_n:        {batch_result.lambda_value:.3f}  (threshold: {batch_result.threshold:.1f})")
print(f"  Rate ratio:      {batch_result.rate_ratio:.4f}")
print(f"  Champion rate:   {batch_result.champion_rate:.4f}")
print(f"  Challenger rate: {batch_result.challenger_rate:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Key design principle:** The mSPRT e-process has `E_0[Lambda_n] = 1` for all n. The test threshold `1/alpha` is exactly calibrated: the probability of ever crossing it under the null is <= alpha. No Bonferroni, no alpha-spending, no fixed horizon required. Check weekly, monthly, or irregularly — the guarantee holds.
# MAGIC
# MAGIC **Key API points:**
# MAGIC - `update()` takes increments, not cumulative totals
# MAGIC - `history()` returns a Polars DataFrame for export/dashboard
# MAGIC - `sequential_test_from_df()` for batch processing from a DataFrame
# MAGIC - `futility_threshold` stops early when Lambda_n is very small
# MAGIC - `alternative='less'` for one-sided tests (challenger should be lower)
