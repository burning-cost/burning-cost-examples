"""
Heterogeneous treatment effect estimation for insurance pricing.

The problem: you want to know which customers respond most to a price change.
Not just "is there a causal effect on average?" but "is the price elasticity
larger for PCW customers than direct? Does it vary by NCD band?"

Standard regression gives biased answers. Higher-risk customers get higher
premiums — the risk-premium correlation confounds naive elasticity estimates.
Double Machine Learning (DML) via CausalForestDML removes this confounding
using cross-fitting.

The key practical insight this script demonstrates: individual-level CATEs
from a causal forest are noisy. You cannot reliably rank individual customers
by their price sensitivity from a single year of renewal data. What you can do
reliably is estimate group average treatment effects (GATES) for well-defined
segments. GATES across NCD bands are the right granularity for a rate review
discussion.

This script demonstrates:

    1. Generate synthetic UK motor renewal data with known heterogeneous
       price elasticity (NCD band 0 = most elastic, NCD band 5 = least)
    2. Fit HeterogeneousElasticityEstimator and compute per-customer CATEs
    3. Formal heterogeneity test via BLP (Best Linear Predictor)
    4. GATES: group average effects by CATE quantile — the key segment view
    5. CLAN: which risk factors drive the heterogeneity
    6. RATE/AUTOC: does the CATE ranking identify genuinely high-effect customers

Libraries used
--------------
    insurance-causal  — HeterogeneousElasticityEstimator, HeterogeneousInference,
                        TargetingEvaluator, make_hte_renewal_data, true_cate_by_ncd

Dependencies
------------
    uv add "insurance-causal"

Approximate runtime: 5-10 minutes (CausalForestDML with 5-fold cross-fitting).
For faster iteration, reduce n_estimators and catboost_iterations.
"""

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

# ---------------------------------------------------------------------------
# Step 1: Synthetic UK motor renewal data
# ---------------------------------------------------------------------------
#
# make_hte_renewal_data generates a renewal book with known CATE structure:
#   NCD=0: tau = -0.30  (most price-elastic — young, price-constrained)
#   NCD=5: tau = -0.10  (least elastic — loyal, value their NCD protection)
#
# The treatment log_price_change is confounded: risk factors (NCD, age, channel)
# determine most of the re-rate. DML cross-fitting identifies the causal effect
# from the exogenous portion of price variation.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Heterogeneous price elasticity via CausalForestDML")
print("insurance-causal v0.4 — causal_forest subpackage")
print("=" * 70)
print()
print("Step 1: Generate synthetic UK motor renewal data")
print("-" * 70)

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
print()
print("  Interpretation: NCD=0 drivers reduce renewal probability by 0.30")
print("  per unit increase in log_price_change. NCD=5 drivers reduce by 0.10.")
print("  This 3x heterogeneity is economically meaningful for retention targeting.")

# ---------------------------------------------------------------------------
# Step 2: Fit HeterogeneousElasticityEstimator
# ---------------------------------------------------------------------------
#
# HeterogeneousElasticityEstimator wraps CausalForestDML with insurance defaults:
#   - CatBoost nuisance models (handles categoricals natively)
#   - honest=True (Athey & Imbens 2016 sample splitting for valid inference)
#   - min_samples_leaf=20 (sparse rating segments need larger leaves)
#
# fit() accepts polars DataFrames directly.
# outcome: binary renewal indicator
# treatment: log_price_change (the confounded but partially exogenous variable)
# confounders: risk factors that jointly determine price AND outcome
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 2: Fit HeterogeneousElasticityEstimator")
print("-" * 70)
print("  Fitting CausalForestDML with 5-fold cross-fitting...")
print("  (CatBoost nuisance models, honest=True, min_samples_leaf=20)")
print()

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
print()
print("  Individual CATE estimates (per customer):")
cates = est.cate(df)
print(f"    Mean:  {cates.mean():.4f}")
print(f"    Std:   {cates.std():.4f}")
print(f"    Range: [{cates.min():.4f}, {cates.max():.4f}]")
print()
print("  Note: individual CATEs are noisy. The std > true heterogeneity is")
print("  expected — individual estimates carry sampling uncertainty. Use GATES")
print("  for segment-level inference, not individual CATEs for customer ranking.")

# ---------------------------------------------------------------------------
# Step 3: GATES by NCD band — the key segment view
# ---------------------------------------------------------------------------
#
# est.gate(df, by="ncd_years") computes group average treatment effects for
# each level of a categorical variable. This is where the actionable insight
# lives: the difference in price elasticity between NCD=0 and NCD=5 directly
# informs how much retention discount is justified per band.
#
# Wide CIs on small segments are expected and honest — don't report them as
# if they were reliable.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 3: GATES by NCD band (group average treatment effects)")
print("-" * 70)
print()

gates_ncd = est.gate(df, by="ncd_years")

print(
    f"  {'NCD':>5}  {'GATE':>10}  {'95% CI':>28}  {'n':>7}  {'True CATE':>10}"
)
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
print("  The pattern should be monotone: NCD=0 most negative (most elastic),")
print("  NCD=5 least negative (least elastic). CIs overlap due to estimation")
print("  uncertainty but the direction is identifiable with 8,000 records.")
print()
print("  Practical implication: a 5% retention discount is worth offering to")
print("  NCD=0 customers but not to NCD=5 customers whose renewal decision")
print("  is much less price-sensitive. Using the ATE to set a uniform discount")
print("  policy would over-discount loyal customers and under-invest in elastic ones.")

# ---------------------------------------------------------------------------
# Step 4: Formal HTE inference via BLP, GATES, CLAN
# ---------------------------------------------------------------------------
#
# HeterogeneousInference runs the Chernozhukov et al. (2020/2025) procedure:
#
# BLP (Best Linear Predictor): Regresses Y on (W - e_hat) and
#   (W - e_hat) * (S - S_bar) where S = tau_hat. beta_2 > 0 and p < 0.05
#   formally confirms that the CATE proxy explains real variation in treatment
#   effects. This is the test you cite in your model governance documentation.
#
# GATES: Partitions customers into K=5 CATE quantile groups and estimates
#   the group average effect for each. These should increase monotonically.
#
# CLAN: Compares covariate means between the highest and lowest GATE groups.
#   Tells you WHICH risk factors drive the heterogeneity (not just that it exists).
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 4: Formal HTE inference — BLP, GATES, CLAN")
print("-" * 70)
print("  Running 50 data splits (standard: 100; reduced here for speed)...")
print()

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
    print("  The CATE proxy S(x) explains real variation in individual treatment")
    print("  effects. The GATES are therefore interpretable as causal estimates,")
    print("  not just noise.")
else:
    print("  BLP did not detect significant heterogeneity at p < 0.05.")
    print("  This could be a power issue (need more data) or genuine homogeneity.")
    print("  Do not use individual CATEs for customer-level targeting.")

print()
print("  GATES monotonicity check:", hte_result.gates.gates_increasing)
if hte_result.gates.gates_increasing:
    print("  GATE estimates increase across quantile groups — the causal forest")
    print("  correctly orders customers by their price sensitivity.")
else:
    print("  GATE estimates are not monotone. This can happen with limited data.")
    print("  The BLP result is the primary inference; GATES order is secondary.")

# ---------------------------------------------------------------------------
# Step 5: RATE/AUTOC — does the CATE ranking have targeting value?
# ---------------------------------------------------------------------------
#
# RATE asks: if you offer a retention discount to the top-q fraction of
# customers ranked by predicted elasticity, do they actually have larger
# causal effects than average?
#
# AUTOC > 0 and p < 0.05: yes, the ranking has targeting value. The model
# correctly identifies who is most responsive to price.
#
# AUTOC not significant: the individual CATE ranking is noise. Offer discounts
# based on segment-level GATES (e.g., all NCD=0 PCW customers), not individual
# scores.
#
# This is the honest test most practitioners skip. Running it and reporting
# the result — even when negative — is good practice.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 5: RATE/AUTOC — does the CATE ranking have targeting value?")
print("-" * 70)
print("  Computing doubly-robust scores and bootstrap SE (200 samples)...")
print()

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
    print("  AUTOC > 0 and significant: the CATE ranking identifies real targeting")
    print("  value. Customers in the top CATE quantile genuinely have larger causal")
    print("  effects. You can use the CATE ranking to prioritise retention offers.")
else:
    print("  AUTOC not significant at p=0.05. The individual CATE ranking does not")
    print("  demonstrate targeting value beyond noise. Use segment-level GATES for")
    print("  targeting decisions (NCD band, channel) rather than individual CATEs.")

print()
print("  TOC curve (targeting gain vs fraction targeted):")
print(f"  {'q':>8}  {'TOC(q)':>10}  {'95% band':>22}")
print(f"  {'-'*8}  {'-'*10}  {'-'*22}")
for _, row in rate_result.toc_curve.iterrows():
    if row["q"] in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        print(
            f"  {row['q']:>8.2f}  {row['toc']:>+10.4f}"
            f"  [{row['se_lower']:+.4f}, {row['se_upper']:+.4f}]"
        )

# ---------------------------------------------------------------------------
# Step 6: Simple GATE by channel — actionable segment analysis
# ---------------------------------------------------------------------------
#
# The formal inference machinery is for governance and model validation.
# For day-to-day pricing decisions, a simple gate() call by a business
# segment is often more useful than the full BLP/CLAN output.
#
# Here: GATE by channel. PCW customers are most price-elastic in this DGP
# (they are actively comparison-shopping). Direct customers are more loyal.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 6: Simple GATE by channel — for the rate review meeting")
print("-" * 70)
print()

gates_channel = est.gate(df, by="channel")
print(f"  {'Channel':>10}  {'GATE':>10}  {'95% CI':>28}  {'n':>7}")
print(f"  {'-'*10}  {'-'*10}  {'-'*28}  {'-'*7}")
for row in gates_channel.sort("cate").iter_rows(named=True):
    print(
        f"  {row['channel']:>10}  {row['cate']:>+10.4f}  "
        f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]  "
        f"{row['n']:>7,}"
    )

print()
print("  PCW customers (comparison shoppers) should show the most negative GATE.")
print("  Direct customers less negative. Broker somewhere between.")
print("  This table informs channel-specific renewal discount strategy directly.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Demo complete")
print("=" * 70)
print(f"""
What was demonstrated:

  1. make_hte_renewal_data()
     Synthetic UK motor book, {len(df):,} policies.
     True CATE spans {df['true_cate'].min():.2f} (NCD=0) to {df['true_cate'].max():.2f} (NCD=5).

  2. HeterogeneousElasticityEstimator.fit()
     CausalForestDML with CatBoost nuisance, 5-fold cross-fitting.
     ATE estimate: {ate:.4f}  [true mean: {df['true_cate'].mean():.4f}]

  3. GATES by NCD band (est.gate())
     Segment estimates are reliable; individual CATEs are noisy.
     Use GATES for pricing decisions, not individual customer scores.

  4. Formal HTE inference (HeterogeneousInference)
     BLP heterogeneity test: beta_2={hte_result.blp.beta_2:.4f}  p={hte_result.blp.beta_2_pvalue:.4f}
     GATES monotone: {hte_result.gates.gates_increasing}
     CLAN identifies features driving heterogeneity.

  5. RATE/AUTOC (TargetingEvaluator)
     AUTOC={rate_result.rate:.4f}  p={rate_result.p_value:.4f}
     {"Ranking has targeting value — CATEs useful for prioritisation." if rate_result.p_value < 0.05
      else "Ranking not significant — use segment GATES, not individual CATEs."}

  Key design decision:
    The causal_forest subpackage does NOT try to give reliable individual
    CATE estimates for every customer. It gives reliable GROUP estimates
    (GATES) and formal tests for WHETHER heterogeneity exists (BLP) and
    WHETHER the ranking has targeting value (RATE). That is the right
    granularity for UK personal lines pricing decisions.
""")
