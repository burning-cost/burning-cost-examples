# Databricks notebook source

# MAGIC %md
# MAGIC # Heterogeneous Treatment Effects: Causal Forest Demo
# MAGIC
# MAGIC **What this notebook demonstrates:** estimating heterogeneous price elasticity across customer segments using CausalForestDML from the `insurance-causal` library.
# MAGIC
# MAGIC The key practical insight: individual-level CATEs from a causal forest are noisy. You cannot reliably rank individual customers by their price sensitivity from a single year of renewal data. What you *can* do reliably is estimate group average treatment effects (GATES) for well-defined segments. GATES across NCD bands are the right granularity for a rate review discussion.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Generate synthetic UK motor renewal data with known heterogeneous price elasticity
# MAGIC 2. Fit `HeterogeneousElasticityEstimator` and compute per-customer CATEs
# MAGIC 3. Formal heterogeneity test via BLP (Best Linear Predictor)
# MAGIC 4. GATES: group average effects by CATE quantile
# MAGIC 5. RATE/AUTOC: does the CATE ranking identify genuinely high-effect customers?
# MAGIC 6. Simple GATE by channel for the rate review meeting

# COMMAND ----------

# MAGIC %pip install "insurance-causal" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
    make_hte_renewal_data,
    true_cate_by_ncd,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic UK Motor Renewal Data
# MAGIC
# MAGIC `make_hte_renewal_data` generates a renewal book with known CATE structure:
# MAGIC - NCD=0: tau = -0.30 (most price-elastic — young, price-constrained)
# MAGIC - NCD=5: tau = -0.10 (least elastic — loyal, value their NCD protection)
# MAGIC
# MAGIC The treatment `log_price_change` is confounded: risk factors (NCD, age, channel) determine most of the re-rate. DML cross-fitting identifies the causal effect from the exogenous portion of price variation.

# COMMAND ----------

df = make_hte_renewal_data(n=8_000, seed=42)

print(f"  Records:           {len(df):,}")
print(f"  Renewal rate:      {df['renewed'].mean():.1%}")
print(f"  Price change SD:   {df['log_price_change'].std():.4f} log-scale")
print(f"  NCD distribution:")
ncd_dist = df.group_by("ncd_years").agg(pl.len().alias("n")).sort("ncd_years")
for row in ncd_dist.iter_rows(named=True):
    print(f"    NCD {row['ncd_years']}: {row['n']:,} policies")

print()
print("  True CATE by NCD band (the ground truth we aim to recover):")
truth = true_cate_by_ncd(df)
for row in truth.iter_rows(named=True):
    print(
        f"    NCD {row['ncd_years']}: tau = {row['true_cate_mean']:.2f}  "
        f"({row['n']:,} policies)"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit HeterogeneousElasticityEstimator
# MAGIC
# MAGIC `HeterogeneousElasticityEstimator` wraps CausalForestDML with insurance defaults:
# MAGIC - CatBoost nuisance models (handles categoricals natively)
# MAGIC - `honest=True` (Athey & Imbens 2016 sample splitting for valid inference)
# MAGIC - `min_samples_leaf=20` (sparse rating segments need larger leaves)
# MAGIC
# MAGIC `fit()` accepts polars DataFrames directly. The outcome is the binary renewal indicator; the treatment is `log_price_change` (the confounded but partially exogenous variable); the confounders are risk factors that jointly determine price AND outcome.

# COMMAND ----------

confounders = ["age", "ncd_years", "vehicle_group", "channel", "region"]

est = HeterogeneousElasticityEstimator(
    binary_outcome=True,
    n_estimators=200,
    n_folds=5,
    catboost_iterations=200,
    min_samples_leaf=20,
    random_state=42,
)

est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)

# ATE: average treatment effect across all customers
ate, ate_lb, ate_ub = est.ate()
print(f"  Average Treatment Effect (ATE):")
print(f"    Estimated: {ate:.4f}  95% CI [{ate_lb:.4f}, {ate_ub:.4f}]")
print(f"    True mean: {df['true_cate'].mean():.4f}")

cates = est.cate(df)
print(f"\n  Individual CATE estimates: mean={cates.mean():.4f}, std={cates.std():.4f}")
print("  Note: individual CATEs are noisy. Use GATES for segment-level inference.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: GATES by NCD Band
# MAGIC
# MAGIC `est.gate(df, by="ncd_years")` computes group average treatment effects for each level of a categorical variable. This is where the actionable insight lives: the difference in price elasticity between NCD=0 and NCD=5 directly informs how much retention discount is justified per band.
# MAGIC
# MAGIC Wide CIs on small segments are expected and honest — don't report them as if they were reliable.

# COMMAND ----------

gates_ncd = est.gate(df, by="ncd_years")

print(f"  {'NCD':>5}  {'GATE':>10}  {'95% CI':>28}  {'n':>7}  {'True CATE':>10}")
print(f"  {'-'*5}  {'-'*10}  {'-'*28}  {'-'*7}  {'-'*10}")

true_by_ncd = {
    row["ncd_years"]: row["true_cate_mean"]
    for row in truth.iter_rows(named=True)
}

for row in gates_ncd.iter_rows(named=True):
    ncd = row["ncd_years"]
    true_val = true_by_ncd.get(ncd, float("nan"))
    print(
        f"  {ncd:>5}  {row['cate']:>+10.4f}  "
        f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]  "
        f"{row['n']:>7,}  "
        f"{true_val:>+10.4f}"
    )

print()
print("  Practical implication: a 5% retention discount is worth offering to")
print("  NCD=0 customers but not to NCD=5 customers whose renewal decision")
print("  is much less price-sensitive.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Formal HTE Inference — BLP, GATES, CLAN
# MAGIC
# MAGIC `HeterogeneousInference` runs the Chernozhukov et al. (2020/2025) procedure:
# MAGIC
# MAGIC - **BLP (Best Linear Predictor):** beta_2 > 0 and p < 0.05 formally confirms that the CATE proxy explains real variation in treatment effects. This is the test you cite in governance documentation.
# MAGIC - **GATES:** Partitions customers into K=5 CATE quantile groups. Should increase monotonically.
# MAGIC - **CLAN:** Compares covariate means between the highest and lowest GATE groups — tells you *which* risk factors drive the heterogeneity.

# COMMAND ----------

inf = HeterogeneousInference(
    n_splits=50,    # 100 is standard; 50 for demo speed
    k_groups=5,
    alpha=0.05,
    random_state=42,
)

hte_result = inf.run(
    df=df,
    estimator=est,
    cate_proxy=cates,
    confounders=confounders,
)

print(hte_result.summary())

print()
if hte_result.blp.heterogeneity_detected:
    print("  BLP confirms heterogeneity: beta_2 > 0 and p < 0.05.")
    print("  The CATE proxy S(x) explains real variation in individual treatment effects.")
else:
    print("  BLP did not detect significant heterogeneity at p < 0.05.")

print()
print("  GATES monotonicity check:", hte_result.gates.gates_increasing)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: RATE/AUTOC — Does the CATE Ranking Have Targeting Value?
# MAGIC
# MAGIC RATE asks: if you offer a retention discount to the top-q fraction of customers ranked by predicted elasticity, do they actually have larger causal effects than average?
# MAGIC
# MAGIC - **AUTOC > 0 and p < 0.05:** the ranking has targeting value. Use individual CATE scores to prioritise retention offers.
# MAGIC - **AUTOC not significant:** the individual CATE ranking is noise. Use segment-level GATES instead.
# MAGIC
# MAGIC This is the honest test most practitioners skip. Running it and reporting the result — even when negative — is good practice.

# COMMAND ----------

evaluator = TargetingEvaluator(
    method="autoc",
    n_bootstrap=200,
    random_state=42,
)

rate_result = evaluator.fit(
    estimator=est,
    df=df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)

print(f"  AUTOC:   {rate_result.rate:.4f}")
print(f"  SE:      {rate_result.se:.4f}")
print(f"  p-value: {rate_result.p_value:.4f}  (one-sided H0: RATE <= 0)")
print()

if rate_result.p_value < 0.05:
    print("  AUTOC > 0 and significant: CATE ranking identifies real targeting value.")
else:
    print("  AUTOC not significant at p=0.05. Use segment GATES, not individual CATEs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Simple GATE by Channel — for the Rate Review Meeting
# MAGIC
# MAGIC The formal inference machinery is for governance and model validation. For day-to-day pricing decisions, a simple `gate()` call by a business segment is often more useful than the full BLP/CLAN output.
# MAGIC
# MAGIC PCW customers (comparison shoppers) should show the most negative GATE. Direct customers less negative. This table informs channel-specific renewal discount strategy directly.

# COMMAND ----------

gates_channel = est.gate(df, by="channel")
print(f"  {'Channel':>10}  {'GATE':>10}  {'95% CI':>28}  {'n':>7}")
print(f"  {'-'*10}  {'-'*10}  {'-'*28}  {'-'*7}")
for row in gates_channel.sort("cate").iter_rows(named=True):
    print(
        f"  {row['channel']:>10}  {row['cate']:>+10.4f}  "
        f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]  "
        f"{row['n']:>7,}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `causal_forest` subpackage does NOT try to give reliable individual CATE estimates for every customer. It gives reliable GROUP estimates (GATES) and formal tests for WHETHER heterogeneity exists (BLP) and WHETHER the ranking has targeting value (RATE). That is the right granularity for UK personal lines pricing decisions.
# MAGIC
# MAGIC **Key outputs:**
# MAGIC - `HeterogeneousElasticityEstimator.fit()` — CausalForestDML with CatBoost nuisance, honest sample splitting
# MAGIC - `est.gate(df, by=segment)` — group average treatment effects with confidence intervals
# MAGIC - `HeterogeneousInference.run()` — BLP/GATES/CLAN formal tests
# MAGIC - `TargetingEvaluator.fit()` — RATE/AUTOC for targeting validation
