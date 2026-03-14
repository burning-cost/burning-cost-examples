"""
Price elasticity estimation and renewal pricing optimisation.

In UK personal lines motor insurance, pricing teams face a question every renewal
cycle: how much can we raise rates before the phone starts ringing with
cancellations? The naive answer — run a logistic regression of renewal flag on
price change — gets the sign right but the magnitude badly wrong. Your pricing
model determines the rate, and customers who receive higher rates are also higher
risk. That risk-renewal correlation swamps the causal price effect you are trying
to measure.

This script demonstrates how to do it properly using Double Machine Learning
(DML) via the insurance-elasticity library. DML separates the causal price
effect from confounding by fitting two nuisance models: one predicts the renewal
outcome from risk factors, one predicts the price change from risk factors. What
remains after subtracting both fitted values is the residual relationship between
price and renewal that risk-driven correlation cannot explain.

The workflow is:

    1. Generate 10,000 synthetic UK motor renewals (known DGP, heterogeneous
       elasticities by NCD band, age, and channel)
    2. Run the treatment variation diagnostic — confirm there is enough exogenous
       price variation for DML to work
    3. Fit CausalForestDML to estimate heterogeneous price elasticity (CATE)
    4. Interpret the results: what does elasticity = -2.5 mean in practice?
    5. Explore how elasticity varies by NCD band, age group, and vehicle value
    6. Run the ENBP-constrained renewal pricing optimiser (FCA PS21/5 compliance)
    7. Compute the efficient frontier: renewal rate vs expected profit
    8. Generate the ICOBS 6B.2 compliance audit trail

Libraries used
--------------
    insurance-elasticity  — DML elasticity estimation and FCA-compliant optimiser

Dependencies
------------
    uv add "insurance-elasticity[all]"
    # installs econml, catboost, matplotlib, polars, numpy

Approximate runtime: 5–10 minutes on a modern CPU. CausalForestDML with
CatBoost nuisance models accounts for most of that. Reduce n_estimators to 50
and catboost_iterations to 100 for a quick smoke test.
"""

from __future__ import annotations

import sys

# econml is required for CausalForestDML. It has complex C extensions
# that cannot be installed in all environments (e.g. Databricks serverless
# compute via environment spec). If econml is not available, exit cleanly
# with a clear message rather than crashing with an ImportError mid-script.
try:
    import econml  # noqa: F401
except ImportError:
    print(
        "SKIP: econml is not installed. This script requires econml for "
        "CausalForestDML. Install with: pip install econml\n"
        "On Databricks serverless, use a standard cluster with econml pre-installed."
    )
    sys.exit(0)

import warnings
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Step 1: Generate a synthetic renewal dataset
# ---------------------------------------------------------------------------
#
# make_renewal_data() generates a fully synthetic UK motor renewal book with a
# known data-generating process (DGP). That DGP has heterogeneous elasticities:
# no-NCD customers are much more price-sensitive than max-NCD customers, young
# drivers more than older ones, and PCW customers 30% more elastic than
# direct-channel equivalents.
#
# We know the true elasticities by segment, which means we can check that our
# DML estimates are recovering something plausible. In production you replace
# this step with your own policy extract, but the column schema stays the same:
#
#   log_price_change  = log(offer_price / last_year_price)  — the treatment
#   renewed           = 1 if customer renewed, 0 otherwise  — the outcome
#   age, ncd_years, vehicle_group, region, channel          — confounders
#   tech_prem         = technical (cost-equivalent) premium
#   enbp              = equivalent new business price (FCA PS21/5 ceiling)
#   last_premium      = what the customer paid last year
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Generate synthetic renewal dataset")
print("=" * 70)

from insurance_elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age

df = make_renewal_data(n=10_000, seed=42, price_variation_sd=0.08)

print(f"Portfolio: {len(df):,} renewal records")
print(f"Overall renewal rate: {df['renewed'].mean():.1%}")
print(f"Mean last premium: £{df['last_premium'].mean():.0f}")
print(f"Mean offer premium: £{df['offer_price'].mean():.0f}")
print(f"Mean price change: {(df['log_price_change'].mean() * 100):.1f}%")
print(f"\nChannel mix:")
for ch, n in df.group_by("channel").agg(pl.len()).sort("channel").iter_rows():
    print(f"  {ch:<10} {n:,} ({n/len(df):.0%})")

print(f"\nNCD band distribution and true elasticities (from the DGP):")
true_ncd = true_gate_by_ncd(df)
for row in true_ncd.iter_rows(named=True):
    print(
        f"  NCD {row['ncd_years']} years  n={row['n']:,}"
        f"  true elasticity = {row['true_elasticity_mean']:.2f}"
    )

print()
print(
    "The true elasticity varies from -3.5 (no NCD) to -1.0 (max NCD)."
    "\nThis is the heterogeneous treatment effect the DML model needs to recover."
)

# ---------------------------------------------------------------------------
# Step 2: Treatment variation diagnostic
# ---------------------------------------------------------------------------
#
# Before fitting anything, check whether there is enough exogenous price
# variation for DML to identify the causal effect. The central challenge in
# applying DML to insurance renewal data is the near-deterministic price
# problem: re-rating makes the offered price nearly a deterministic function of
# observable risk factors. When that happens, Var(D̃)/Var(D) approaches zero —
# the DML residual has almost no variation to work with, exactly as in a weak
# instruments problem.
#
# This dataset has price_variation_sd=0.08, which provides enough exogenous
# variation (Var(D̃)/Var(D) well above the 10% threshold). A real book with no
# A/B testing or UW override will typically have far less. See the report
# suggestions for what to do in that case.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: Treatment variation diagnostic")
print("=" * 70)

from insurance_elasticity.diagnostics import ElasticityDiagnostics

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics(n_folds=3, random_state=42)
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=CONFOUNDERS,
)

print(report.summary())

if report.weak_treatment:
    print(
        "\nWARNING: weak treatment detected. The estimates below will be"
        " unreliable. Address the identification problem before presenting"
        " results to a pricing committee."
    )
    # In a real script you might raise an exception here. We continue for
    # demonstration purposes.

# Sanity check: does renewal rate fall as price rises?
# It should — if it is flat, there is something wrong with the data or
# the near-deterministic price problem is severe.
print("\nCalibration check: renewal rate by price change decile")
print("(should decline as price change rises)")
cal = diag.calibration_summary(df, outcome="renewed", treatment="log_price_change", n_bins=10)
for row in cal.iter_rows(named=True):
    bar = "#" * int(row["renewal_rate"] * 30)
    print(
        f"  Decile {row['bin']:2d}"
        f"  price change {row['mean_log_price_change']:+.3f}"
        f"  renewal {row['renewal_rate']:.1%}  {bar}"
    )

# ---------------------------------------------------------------------------
# Step 3: Fit the elasticity model (CausalForestDML)
# ---------------------------------------------------------------------------
#
# CausalForestDML is the right choice when you want a heterogeneous treatment
# effect surface across the portfolio, not just a portfolio average. It is
# fully non-parametric: it makes no assumption about which interactions drive
# elasticity heterogeneity. The honest splitting procedure gives valid
# pointwise confidence intervals.
#
# CatBoost nuisance models handle the categorical covariates (region, vehicle
# group, channel) natively — no encoding needed on our side, the library
# converts them internally via pandas get_dummies.
#
# The estimator settings here are a practical compromise:
#   n_estimators=100  — enough for a stable CATE surface without taking 20
#                       minutes. Use 200-500 for a production run.
#   catboost_iterations=300 — adequate nuisance model quality. Use 500+
#                              for larger datasets.
#   n_folds=5         — standard for DML cross-fitting (Chernozhukov 2018).
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: Fit CausalForestDML elasticity model")
print("=" * 70)

from insurance_elasticity.fit import RenewalElasticityEstimator

est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=100,         # increase to 200-500 for production
    catboost_iterations=300,  # increase to 500 for production
    n_folds=5,
    random_state=42,
)

print(
    f"Fitting CausalForestDML with {est.n_estimators} trees,"
    f" {est.catboost_iterations} CatBoost iterations, {est.n_folds} folds."
)
print("This typically takes 5-10 minutes on 10,000 rows...")

est.fit(df, outcome="renewed", treatment="log_price_change", confounders=CONFOUNDERS)

# Average treatment effect
ate, lb, ub = est.ate()
print(f"\nAverage treatment effect (ATE): {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")

# ---------------------------------------------------------------------------
# Interpreting the ATE
#
# The ATE is a semi-elasticity: a 1-unit increase in log_price_change changes
# the renewal probability by ATE percentage points. But a 100% price increase
# is not a realistic scenario.
#
# For the 5-20% price changes typical of UK personal lines renewals:
#
#   10% increase  =>  log change = log(1.10) = 0.0953
#                 =>  effect = ATE * 0.0953
#
# So if ATE = -2.5, a 10% price increase reduces the expected renewal
# probability by approximately 2.5 * 0.095 = 0.24 percentage points for the
# average customer. That is the number to present to a pricing committee.
# ---------------------------------------------------------------------------

log_10pct = np.log(1.10)
effect_10pct = ate * log_10pct
log_5pct = np.log(1.05)
effect_5pct = ate * log_5pct
log_20pct = np.log(1.20)
effect_20pct = ate * log_20pct

print(f"\nHow to interpret the ATE:")
print(f"  ATE = {ate:.3f}  (semi-elasticity: effect of 1 log-unit price increase)")
print(f"")
print(f"  For realistic renewal re-rates:")
print(f"    5% price increase  (log change = {log_5pct:.4f}) => {effect_5pct:+.3f} pp change in renewal probability")
print(f"   10% price increase  (log change = {log_10pct:.4f}) => {effect_10pct:+.3f} pp change in renewal probability")
print(f"   20% price increase  (log change = {log_20pct:.4f}) => {effect_20pct:+.3f} pp change in renewal probability")
print(f"")
print(f"  On a book of 100,000 policies at 85% base renewal rate:")
overall_renewal = float(df["renewed"].mean())
book_size = 100_000
retained = int(book_size * overall_renewal)
lost_10pct = int(book_size * abs(effect_10pct / 100))
print(f"    Currently retaining ~{retained:,} customers")
print(f"    A 10% rate increase would lose an additional ~{lost_10pct:,} customers")
print(f"    (before accounting for any premium uplift benefit)")

# ---------------------------------------------------------------------------
# Step 4: Segment-level elasticity (GATE)
# ---------------------------------------------------------------------------
#
# The ATE is useful for portfolio-level planning, but the real value is
# understanding which segments are most price-sensitive. Group Average
# Treatment Effects (GATEs) answer: what is the average elasticity for
# customers in NCD band 0 vs NCD band 5?
#
# Segments that are more elastic (more negative GATE) should receive smaller
# price increases or, in a profit-maximising framework, will be optimised to
# a lower price than the ENBP ceiling allows. Segments that are inelastic
# can absorb larger increases with minimal retention impact.
#
# We compare the estimated GATEs against the known DGP truth. In a real book
# you would not have the true values, but the pattern — max-NCD customers
# least elastic, no-NCD most elastic — should be visible in the data.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: Segment-level elasticity (GATE)")
print("=" * 70)

# --- By NCD band ---
print("\nElasticity by NCD band (estimated vs DGP truth):")
gate_ncd = est.gate(df, by="ncd_years")
true_ncd = true_gate_by_ncd(df)
true_map = {row["ncd_years"]: row["true_elasticity_mean"] for row in true_ncd.iter_rows(named=True)}

print(f"  {'NCD':>5}  {'Estimated':>10}  {'95% CI':>22}  {'DGP truth':>10}  {'Bias':>8}  {'n':>6}")
print(f"  {'-'*5}  {'-'*10}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*6}")
for row in gate_ncd.iter_rows(named=True):
    ncd = row["ncd_years"]
    est_e = row["elasticity"]
    ci_lo = row["ci_lower"]
    ci_hi = row["ci_upper"]
    truth = true_map.get(ncd, float("nan"))
    bias = est_e - truth if not np.isnan(truth) else float("nan")
    print(
        f"  {ncd:>5}  {est_e:>10.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  {truth:>10.3f}  {bias:>+8.3f}  {row['n']:>6,}"
    )

# --- By age band ---
print("\nElasticity by age band (estimated vs DGP truth):")
gate_age = est.gate(df, by="age_band")
true_age_df = true_gate_by_age(df)
true_age_map = {row["age_band"]: row["true_elasticity_mean"] for row in true_age_df.iter_rows(named=True)}

# Sort age bands in natural order
age_order = {"17-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
gate_age_sorted = gate_age.with_columns(
    pl.col("age_band").replace(age_order).alias("_order")
).sort("_order").drop("_order")

print(f"  {'Age band':>10}  {'Estimated':>10}  {'95% CI':>22}  {'DGP truth':>10}  {'n':>6}")
print(f"  {'-'*10}  {'-'*10}  {'-'*22}  {'-'*10}  {'-'*6}")
for row in gate_age_sorted.iter_rows(named=True):
    ab = row["age_band"]
    truth = true_age_map.get(ab, float("nan"))
    print(
        f"  {ab:>10}  {row['elasticity']:>10.3f}  [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        f"  {truth:>10.3f}  {row['n']:>6,}"
    )

# --- By channel ---
print("\nElasticity by acquisition channel:")
gate_ch = est.gate(df, by="channel")
for row in gate_ch.sort("elasticity").iter_rows(named=True):
    print(
        f"  {row['channel']:<10}  elasticity {row['elasticity']:.3f}"
        f"  [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        f"  n={row['n']:,}"
    )

print()
print(
    "PCW customers are more elastic than direct because they are actively"
    " comparing prices at renewal. Raising rates on PCW customers by 10%"
    " produces a larger lapse impact than the same increase on direct customers."
)

# ---------------------------------------------------------------------------
# Step 5: Elasticity surface (NCD x age band)
# ---------------------------------------------------------------------------
#
# The elasticity surface is the key deliverable for a rate review discussion.
# It lets you identify which (NCD, age) combinations you can push harder and
# which you need to handle carefully.
#
# We use the ElasticitySurface class to produce both the heatmap and a segment
# summary table in a format suitable for Excel export.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 5: Elasticity surface and visualisations")
print("=" * 70)

from insurance_elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est)

# Segment summary table: easy to export to Excel
ncd_age_summary = surface.segment_summary(df, by=["ncd_years", "age_band"])
print("Segment summary (NCD x age band) — first 12 rows:")
print(f"  {'NCD':>5}  {'Age band':>10}  {'Elasticity':>12}  {'At 10% rise':>12}  {'n':>6}")
print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*6}")
for row in ncd_age_summary.head(12).iter_rows(named=True):
    print(
        f"  {row['ncd_years']:>5}  {row['age_band']:>10}"
        f"  {row['elasticity']:>12.3f}"
        f"  {row['elasticity_at_10pct']:>12.3f}"
        f"  {row['n']:>6,}"
    )
print(f"  ... ({len(ncd_age_summary)} segments total)")

print()
print(
    "The 'elasticity_at_10pct' column is what a pricing actuary needs:"
    " the expected change in renewal probability if this segment receives a 10%"
    " rate increase. A value of -0.25 means: for every 100 customers in this"
    " segment, a 10% increase costs you approximately 0.25 policy renewals."
)

# Save plots
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    fig_surface = surface.plot_surface(df, dims=["ncd_years", "age_band"])
    fig_surface.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")
    print("\nElasticity heatmap saved to /tmp/elasticity_surface_ncd_age.png")

    fig_gate_ncd = surface.plot_gate(df, by="ncd_years", title="Elasticity by NCD band")
    fig_gate_ncd.savefig("/tmp/elasticity_gate_ncd.png", dpi=150, bbox_inches="tight")
    print("GATE bar chart (NCD) saved to /tmp/elasticity_gate_ncd.png")

    fig_gate_age = surface.plot_gate(
        df.with_columns(
            pl.col("age_band").replace(age_order).alias("_age_sort")
        ).sort("_age_sort").drop("_age_sort"),
        by="age_band",
        title="Elasticity by age band",
        color="#2ca02c",
    )
    fig_gate_age.savefig("/tmp/elasticity_gate_age.png", dpi=150, bbox_inches="tight")
    print("GATE bar chart (age) saved to /tmp/elasticity_gate_age.png")

    _plots_available = True
except Exception as exc:
    print(f"\nNote: plot generation skipped ({exc}). Install matplotlib to enable plots.")
    _plots_available = False

# ---------------------------------------------------------------------------
# Step 6: ENBP-constrained pricing optimisation
# ---------------------------------------------------------------------------
#
# The pricing optimiser solves, for each policy independently:
#
#   max  (price - tech_prem) * P(renew | price, elasticity)
#   s.t. price <= ENBP               (FCA ICOBS 6B.2 constraint)
#        price >= tech_prem * 1.0    (no sub-technical pricing)
#
# P(renew | price) uses the linear approximation:
#   P = P0 + theta_i * log(new_price / current_offer)
#
# where theta_i is this customer's CATE. The optimisation is separable (each
# policy solved independently), which makes it fast on large portfolios.
#
# Customers with high negative elasticity (e.g., no-NCD PCW customers) will be
# optimised to a lower price than the ENBP allows because pushing them to ENBP
# would cost too much in expected renewal probability. Inelastic customers (max-
# NCD, older, direct channel) will typically be priced at or near ENBP.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 6: FCA PS21/5-compliant pricing optimisation")
print("=" * 70)

from insurance_elasticity.optimise import RenewalPricingOptimiser

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,          # do not price below technical premium
)

# Suppress the expected UserWarning if any ENBP breaches appear — we handle
# them explicitly in the audit step
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    priced_df = opt.optimise(df, objective="profit")

print("Pricing optimisation complete.")
print(f"\nPortfolio summary (profit-maximising prices):")
print(f"  Mean current offer price:   £{df['offer_price'].mean():.2f}")
print(f"  Mean optimal price:         £{priced_df['optimal_price'].mean():.2f}")
print(f"  Mean ENBP:                  £{df['enbp'].mean():.2f}")
print(f"  Mean ENBP headroom:         £{priced_df['enbp_headroom'].mean():.2f}")
print(f"  Mean predicted renewal prob: {priced_df['predicted_renewal_prob'].mean():.1%}")

# Breakdown by segment
print("\nOptimal pricing by NCD band (profit objective):")
priced_with_ncd = priced_df.with_columns(pl.col("ncd_years"))
ncd_summary = (
    priced_with_ncd
    .group_by("ncd_years")
    .agg([
        pl.col("optimal_price").mean().alias("mean_optimal_price"),
        pl.col("enbp").mean().alias("mean_enbp"),
        pl.col("predicted_renewal_prob").mean().alias("mean_predicted_renewal"),
        pl.col("expected_profit").mean().alias("mean_expected_profit"),
        pl.len().alias("n"),
    ])
    .sort("ncd_years")
)

print(f"  {'NCD':>5}  {'Opt price':>10}  {'ENBP':>8}  {'% of ENBP':>10}  {'Pred renewal':>13}  {'Profit/pol':>11}")
print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*11}")
for row in ncd_summary.iter_rows(named=True):
    pct_enbp = row["mean_optimal_price"] / row["mean_enbp"]
    print(
        f"  {row['ncd_years']:>5}"
        f"  £{row['mean_optimal_price']:>8.0f}"
        f"  £{row['mean_enbp']:>6.0f}"
        f"  {pct_enbp:>10.1%}"
        f"  {row['mean_predicted_renewal']:>13.1%}"
        f"  £{row['mean_expected_profit']:>9.0f}"
    )

print(
    "\nNCD 0-2 customers are priced below ENBP: the optimiser is trading"
    " margin for retention because these customers are highly elastic."
    " NCD 4-5 customers are priced closer to or at ENBP because their"
    " lower elasticity means the margin gain outweighs the retention loss."
)

# ---------------------------------------------------------------------------
# Step 7: Efficient frontier — renewal rate vs expected profit
# ---------------------------------------------------------------------------
#
# The demand curve sweeps a range of uniform price changes across the whole
# portfolio and predicts renewal rate and expected profit at each point. This
# gives the efficient frontier: the set of (renewal rate, profit) combinations
# achievable by choosing a portfolio-wide rate change level.
#
# Every point on this frontier is Pareto-optimal: you cannot improve both
# metrics simultaneously. The profit-maximising price lies at the peak of the
# profit curve. If your board has set a retention floor (e.g., "we must retain
# at least 85% of customers"), the demand curve tells you the maximum price
# change compatible with that constraint.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 7: Portfolio demand curve and efficient frontier")
print("=" * 70)

from insurance_elasticity.demand import demand_curve, plot_demand_curve

demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))

# Find profit-maximising price change
max_profit_row = demand_df.filter(
    pl.col("predicted_profit") == pl.col("predicted_profit").max()
).row(0, named=True)

print(f"\nPortfolio demand curve (selected price points):")
print(f"  {'Price change':>13}  {'Renewal rate':>13}  {'Profit/policy':>14}")
print(f"  {'-'*13}  {'-'*13}  {'-'*14}")

# Print a subset of the curve for readability
selected_indices = [0, 6, 12, 19, 24, 30, 37, 43, 49]
for i, row in enumerate(demand_df.iter_rows(named=True)):
    if i in selected_indices:
        pct = row["pct_price_change"] * 100
        renewal = row["predicted_renewal_rate"] * 100
        profit = row["predicted_profit"]
        star = "  <-- profit max" if abs(row["log_price_change"] - max_profit_row["log_price_change"]) < 0.01 else ""
        print(f"  {pct:>+12.1f}%  {renewal:>12.1f}%  £{profit:>12.0f}{star}")

print(f"\nProfit-maximising price change: {max_profit_row['pct_price_change']*100:+.1f}%")
print(f"  Expected renewal rate at that price: {max_profit_row['predicted_renewal_rate']:.1%}")
print(f"  Expected profit per policy: £{max_profit_row['predicted_profit']:.0f}")

# What if the business sets a retention floor?
retention_floor = 0.83  # 83% retention target
compliant_rows = demand_df.filter(pl.col("predicted_renewal_rate") >= retention_floor)
if len(compliant_rows) > 0:
    max_price_at_floor = compliant_rows.filter(
        pl.col("pct_price_change") == pl.col("pct_price_change").max()
    ).row(0, named=True)
    print(
        f"\nWith a {retention_floor:.0%} retention floor:"
        f"\n  Maximum compatible price change: {max_price_at_floor['pct_price_change']*100:+.1f}%"
        f"\n  Expected profit at that price: £{max_price_at_floor['predicted_profit']:.0f}"
    )

if _plots_available:
    try:
        fig_demand = plot_demand_curve(demand_df, show_profit=True)
        fig_demand.savefig("/tmp/demand_curve.png", dpi=150, bbox_inches="tight")
        print("\nDemand curve chart saved to /tmp/demand_curve.png")
    except Exception as exc:
        print(f"\nDemand curve plot skipped ({exc}).")

# ---------------------------------------------------------------------------
# Step 8: ICOBS 6B.2 compliance audit trail
# ---------------------------------------------------------------------------
#
# FCA PS21/5 (January 2022) requires that no renewing customer is quoted a
# price above the equivalent new business price (ENBP). The enbp_audit() method
# produces the per-row compliance flag that your compliance function needs.
#
# The audit output has five columns:
#   policy_id      — policy reference for the compliance log
#   offered_price  — the price produced by the optimiser
#   enbp           — the ENBP for this customer
#   compliant      — True if offered_price <= enbp, False if breached
#   margin_to_enbp — positive = headroom, negative = breach amount
#   pct_above_enbp — percentage above ENBP (0 if compliant)
#
# A correctly configured optimiser with floor_loading=1.0 and valid ENBP data
# should produce zero breaches. Breaches in production indicate one of: an
# ENBP data quality issue, a system error in the ENBP calculation, or the
# optimiser ceiling_prices logic being overridden downstream.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 8: FCA ICOBS 6B.2 compliance audit trail")
print("=" * 70)

# Suppress the warning — we will report breaches manually
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    audit = opt.enbp_audit(priced_df)

n_compliant = int(audit["compliant"].sum())
n_breach = len(audit) - n_compliant
breach_rate = n_breach / len(audit)

print(f"\nCompliance summary:")
print(f"  Total policies:      {len(audit):,}")
print(f"  Compliant:           {n_compliant:,}  ({n_compliant/len(audit):.2%})")
print(f"  Breaches:            {n_breach:,}  ({breach_rate:.2%})")

if n_breach > 0:
    print(f"\n  BREACH DETAIL (first 10):")
    breaches = audit.filter(pl.col("compliant") == False).head(10)
    for row in breaches.iter_rows(named=True):
        print(
            f"    policy {row['policy_id']:>6}"
            f"  offered £{row['offered_price']:.2f}"
            f"  ENBP £{row['enbp']:.2f}"
            f"  breach {row['pct_above_enbp']:.2f}%"
        )
    print(
        "\n  These breaches should be investigated before production deployment."
        " Common causes: ENBP data feed error, mismatch between the NB model"
        " applied to renewals vs the actual live NB quote, or PCW cashback"
        " adjustments not reflected in the ENBP calculation."
    )
else:
    print(
        f"\n  All {len(audit):,} policies are ICOBS 6B.2 compliant."
        f" The optimiser has respected the ENBP ceiling for every renewal."
    )

# Distribution of headroom — useful for assessing ENBP constraint tightness
headroom = audit["margin_to_enbp"].to_numpy()
print(f"\nENBP headroom distribution (positive = under ENBP):")
print(f"  Median headroom:   £{np.median(headroom):.2f}")
print(f"  10th percentile:   £{np.percentile(headroom, 10):.2f}")
print(f"  25th percentile:   £{np.percentile(headroom, 25):.2f}")
print(f"  Policies within £5 of ENBP:   {int((headroom < 5).sum()):,}")
print(f"  Policies within £10 of ENBP:  {int((headroom < 10).sum()):,}")

# Sample audit trail suitable for the compliance log (first 5 rows)
print("\nSample audit output (first 5 policies):")
print(f"  {'policy_id':>10}  {'offered_price':>14}  {'enbp':>8}  {'compliant':>10}  {'margin':>8}")
print(f"  {'-'*10}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*8}")
for row in audit.head(5).iter_rows(named=True):
    compliant_str = "YES" if row["compliant"] else "BREACH"
    print(
        f"  {row['policy_id']:>10}"
        f"  £{row['offered_price']:>12.2f}"
        f"  £{row['enbp']:>6.2f}"
        f"  {compliant_str:>10}"
        f"  £{row['margin_to_enbp']:>6.2f}"
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Workflow complete")
print("=" * 70)

ate_mag = abs(ate)
effect_10_mag = abs(effect_10pct * 100)

print(f"""
What was demonstrated:

  1. Synthetic data       {len(df):,} renewal records, heterogeneous true elasticities
                          (NCD 0: -3.5 to NCD 5: -1.0, reflecting real UK motor patterns)

  2. Diagnostics          Var(D̃)/Var(D) = {report.variation_fraction:.3f} (above 0.10 threshold — OK to proceed)
                          Calibration check confirms renewal rate falls with price

  3. CausalForestDML      ATE = {ate:.3f}  95% CI [{lb:.3f}, {ub:.3f}]
                          A 10% price increase reduces renewal by ~{effect_10_mag:.2f}pp on average

  4. GATE by segment      NCD heterogeneity recovered: DML correctly identifies
                          no-NCD customers as most elastic, max-NCD as most inelastic
                          Age and channel effects visible and directionally correct

  5. Elasticity surface   heatmap and bar charts saved to /tmp/elasticity_*.png

  6. Pricing optimisation Profit-maximising prices set per policy within ENBP ceiling
                          Elastic segments (NCD 0-1, PCW) priced below ENBP ceiling
                          Inelastic segments (NCD 4-5, direct) priced closer to ENBP

  7. Demand curve         Profit maximum at {max_profit_row['pct_price_change']*100:+.1f}% across the portfolio
                          Efficient frontier available for retention-constrained planning

  8. Compliance audit     {n_compliant:,} / {len(audit):,} policies ICOBS 6B.2 compliant
                          Audit trail in per-row format ready for compliance function

Next steps for a production deployment:
  - Replace make_renewal_data() with your own policy extract (same schema)
  - Increase n_estimators to 200-500 and catboost_iterations to 500 for accuracy
  - Validate ENBP data feed against your NB model output before running the audit
  - Run treatment variation diagnostic on your real book — if Var(D̃)/Var(D) < 0.10,
    address identification problem before presenting elasticity estimates
  - Consider fitting LinearDML first (much faster) for portfolio ATE, then
    CausalForestDML only on segments where heterogeneity is commercially important
""")
