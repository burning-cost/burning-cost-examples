# Databricks notebook source

# MAGIC %md
# MAGIC # Thin Data Pipeline: Bühlmann-Straub Credibility for Sparse Portfolio Segments
# MAGIC
# MAGIC The most common pricing problem actuaries face is not "how do I build a GBM" — it is "what do I do with the segment that has 47 claims in five years?"
# MAGIC
# MAGIC Standard GLMs and GBMs are unreliable when fitted to a single thin segment. The raw loss ratio bounces around with small-number noise. But ignoring the segment's own experience entirely and applying the portfolio mean is also wrong: the segment may genuinely have different underlying risk.
# MAGIC
# MAGIC Bühlmann-Straub credibility is the actuarial solution. It is a rigorous, data-driven blend: the weight given to each segment's own experience is proportional to that segment's credibility factor Z_i, which depends on how noisy the data is relative to the between-segment signal.
# MAGIC
# MAGIC **Libraries:**
# MAGIC - `credibility` — Bühlmann-Straub and hierarchical credibility models
# MAGIC - `insurance-cv` — temporal walk-forward cross-validation
# MAGIC - `insurance-datasets` — synthetic UK motor data

# COMMAND ----------

# MAGIC %pip install credibility insurance-cv insurance-datasets --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Motor Data and Define Thin Segments
# MAGIC
# MAGIC We load 30,000 policies across 5 accident years. Young urban drivers (driver_age < 25, area D/E/F) are the classic thin segment in UK motor: high risk, high volatility, disproportionately important for pricing accuracy.

# COMMAND ----------

from insurance_datasets import load_motor

df_raw = load_motor(n_policies=30_000, seed=42)

df_raw = df_raw.assign(
    age_band=pd.cut(
        df_raw["driver_age"],
        bins=[0, 24, 34, 50, 999],
        labels=["17-24", "25-34", "35-50", "51+"],
    ).astype(str),
    urban=df_raw["area"].isin(["D", "E", "F"]).map({True: "urban", False: "rural"}),
)
df_raw["segment"] = df_raw["age_band"] + "_" + df_raw["urban"]

print(f"Portfolio: {len(df_raw):,} policies across {df_raw['accident_year'].nunique()} accident years")
print(f"Segments defined:\n{df_raw['segment'].value_counts().sort_index()}")

# Build panel: one row per (segment, accident_year)
panel = (
    df_raw.groupby(["segment", "accident_year"])
    .agg(
        total_claims=("claim_count", "sum"),
        total_exposure=("exposure", "sum"),
    )
    .assign(loss_rate=lambda x: x["total_claims"] / x["total_exposure"])
    .reset_index()
)

print(f"\nPanel: {panel['segment'].nunique()} segments x {panel['accident_year'].nunique()} years = {len(panel)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Noise in Raw Loss Ratios by Segment
# MAGIC
# MAGIC Raw loss ratios = total claims / total exposure. For well-populated segments (mature rural drivers) these are stable year-on-year. For young urban drivers, the annual count is too small to distinguish signal from noise. The coefficient of variation (CV) of the annual loss rate is the key diagnostic — high CV = noisy = low credibility.

# COMMAND ----------

segment_stats = (
    panel.groupby("segment")
    .agg(
        mean_loss_rate=("loss_rate", "mean"),
        std_loss_rate=("loss_rate", "std"),
        total_exposure=("total_exposure", "sum"),
        total_claims=("total_claims", "sum"),
    )
    .assign(cv=lambda x: x["std_loss_rate"] / x["mean_loss_rate"].clip(lower=1e-9))
    .sort_values("total_exposure", ascending=True)
)

print(f"{'Segment':<22} {'Exp':>8} {'Claims':>8} {'Mean LR':>9} {'CV':>7}  Note")
print("-" * 70)
for _, row in segment_stats.iterrows():
    thin_flag = " <- thin" if row["total_exposure"] < 200 else ""
    print(
        f"  {row.name:<20} {row['total_exposure']:>8.0f} {row['total_claims']:>8.0f}"
        f" {row['mean_loss_rate']:>9.4f} {row['cv']:>7.3f}{thin_flag}"
    )

# Show year-on-year variability for the thinnest segment
thinnest = segment_stats.index[0]
print(f"\nYear-on-year loss rates — thinnest segment ({thinnest}):")
thin_panel = panel[panel["segment"] == thinnest].sort_values("accident_year")
for _, row in thin_panel.iterrows():
    bar = "#" * int(row["loss_rate"] / 0.01)
    print(f"  {int(row['accident_year'])}: {row['loss_rate']:.4f}  n_claims={row['total_claims']:.0f}  {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Bühlmann-Straub Credibility Estimation
# MAGIC
# MAGIC `BuhlmannStraub.fit()` estimates three structural parameters:
# MAGIC - `mu` — grand weighted mean loss rate (the complement)
# MAGIC - `v` — EPV: expected process variance (within-segment noise)
# MAGIC - `a` — VHM: variance of hypothetical means (between-segment signal)
# MAGIC
# MAGIC From these it computes `k = v/a` (Bühlmann's k) and per-segment credibility factors `Z_i = w_i / (w_i + k)`. A segment needs `w_i = k` exposure-years to achieve `Z = 0.5`.

# COMMAND ----------

from credibility import BuhlmannStraub

bs = BuhlmannStraub()
bs.fit(
    panel,
    group_col="segment",
    period_col="accident_year",
    loss_col="loss_rate",
    weight_col="total_exposure",
)

results_table = bs.summary()
print(results_table)

print(f"\nKey parameter: k = {bs.k_:.1f} exposure-years needed for Z = 0.50")
print(f"Portfolio mean loss rate (complement): {bs.mu_hat_:.4f}")

print("\nCredibility factors by segment:")
print(f"  {'Segment':<22} {'Exposure':>10} {'Z':>7} {'Obs. Rate':>10} {'Cred. Rate':>11} {'Shift':>8}")
print("  " + "-" * 72)
for row in bs.premiums_.sort("exposure").iter_rows(named=True):
    shift = row["credibility_premium"] - row["observed_mean"]
    print(
        f"  {str(row['group']):<22} {row['exposure']:>10.0f} {row['Z']:>7.3f}"
        f" {row['observed_mean']:>10.4f} {row['credibility_premium']:>11.4f}"
        f" {shift:>+8.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Stability Comparison — Raw Rates vs Credibility Estimates
# MAGIC
# MAGIC The credibility estimate for each segment is a stable single value (using all years). Compare it to the within-segment standard deviation of the raw annual rates. For thin segments, credibility shrinks outlier years toward the mean, reducing volatility that would otherwise propagate into prices.

# COMMAND ----------

credibility_premiums = dict(zip(bs.premiums_["group"].to_list(), bs.premiums_["credibility_premium"].to_list()))
z_factors = dict(zip(bs.z_["group"].to_list(), bs.z_["Z"].to_list()))

print(f"{'Segment':<22} {'Raw SD':>8} {'Cred. Rate':>11} {'Z':>7}  Interpretation")
print("-" * 70)
for seg, seg_panel in panel.groupby("segment"):
    raw_sd = float(seg_panel["loss_rate"].std()) if len(seg_panel) > 1 else float("nan")
    cred_rate = credibility_premiums.get(seg, float("nan"))
    z = z_factors.get(seg, float("nan"))
    if np.isnan(raw_sd):
        interp = "single year"
    elif z < 0.3:
        interp = "mostly collective mean"
    elif z < 0.7:
        interp = "balanced blend"
    else:
        interp = "mostly own experience"
    print(f"  {str(seg):<20} {raw_sd:>8.4f} {cred_rate:>11.4f} {z:>7.3f}  {interp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Temporal CV Validation
# MAGIC
# MAGIC The strongest evidence for credibility is out-of-sample performance: do credibility-weighted estimates better predict the next year's loss rate than raw historical averages? We use `insurance-cv`'s `policy_year_split` with an expanding-window validation.

# COMMAND ----------

from insurance_cv import policy_year_split

panel_for_cv = panel.copy()
panel_for_cv["inception_date"] = pd.to_datetime(panel_for_cv["accident_year"].astype(str) + "-01-01")

splits = policy_year_split(
    panel_for_cv, date_col="inception_date",
    n_years_train=3, n_years_test=1, step_years=1,
)

print(f"Temporal CV: {len(splits)} folds\n")
print(f"{'Fold':<8} {'Test Year':>10} {'Cred MAE':>10} {'Naive MAE':>10} {'Improvement':>12}")
print("-" * 54)

fold_results = []
for i, split in enumerate(splits, 1):
    train_idx, test_idx = split.get_indices(panel_for_cv)
    train_panel = panel_for_cv.iloc[train_idx]
    test_panel = panel_for_cv.iloc[test_idx]

    bs_fold = BuhlmannStraub()
    try:
        bs_fold.fit(
            train_panel, group_col="segment", period_col="accident_year",
            loss_col="loss_rate", weight_col="total_exposure",
        )
        cred_premiums_fold = dict(zip(bs_fold.premiums_["group"].to_list(), bs_fold.premiums_["credibility_premium"].to_list()))
        portfolio_mean = bs_fold.mu_hat_
    except Exception as e:
        print(f"  Fold {i}: could not fit ({e}), skipping")
        continue

    naive_rates = (
        train_panel.groupby("segment")
        .apply(lambda g: (g["total_claims"].sum() / max(g["total_exposure"].sum(), 1e-9)))
        .to_dict()
    )

    maes_cred, maes_naive = [], []
    for _, test_row in test_panel.iterrows():
        seg = test_row["segment"]
        actual = test_row["loss_rate"]
        cred_pred = cred_premiums_fold.get(seg, portfolio_mean)
        naive_pred = naive_rates.get(seg, portfolio_mean)
        exp_weight = test_row["total_exposure"]
        maes_cred.append(abs(cred_pred - actual) * exp_weight)
        maes_naive.append(abs(naive_pred - actual) * exp_weight)

    total_exp = test_panel["total_exposure"].sum()
    mae_cred = sum(maes_cred) / max(total_exp, 1e-9)
    mae_naive = sum(maes_naive) / max(total_exp, 1e-9)
    improvement = (mae_naive - mae_cred) / max(mae_naive, 1e-9)

    test_year = int(test_panel["accident_year"].iloc[0])
    fold_results.append((i, test_year, mae_cred, mae_naive, improvement))
    print(f"  {i:<8} {test_year:>10} {mae_cred:>10.5f} {mae_naive:>10.5f} {improvement:>11.1%}")

if fold_results:
    mean_improvement = np.mean([r[4] for r in fold_results])
    print(f"\nMean improvement: {mean_improvement:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **When to use credibility in practice:**
# MAGIC - Scheme or cohort pricing where segments have < k exposure-years
# MAGIC - Regional pricing in thin geographic areas
# MAGIC - NCD pricing at the extremes (NCD=0 or NCD=max)
# MAGIC - Any situation where you cannot justify a standalone model per segment but also cannot apply the portfolio mean without adjustment
# MAGIC
# MAGIC **What to report to a pricing committee:**
# MAGIC - k (years needed for full credibility)
# MAGIC - The Z factors per segment (`bs.z_`)
# MAGIC - The credibility premiums (`bs.premiums_`)
# MAGIC - Out-of-sample MAE improvement over naive estimates
# MAGIC
# MAGIC For multi-level structures (regions > districts > sectors) see `HierarchicalBuhlmannStraub` in the `credibility` package.
