# Databricks notebook source

# MAGIC %md
# MAGIC # Price Elasticity Estimation and FCA PS21/5-Compliant Renewal Optimisation
# MAGIC
# MAGIC In UK personal lines motor insurance, pricing teams face a question every renewal cycle: how much can you raise rates before the phone starts ringing with cancellations? The naive answer — run a logistic regression of renewal flag on price change — gets the sign right but the magnitude badly wrong. Your pricing model determines the rate, and customers who receive higher rates are also higher risk. That risk-renewal correlation swamps the causal price effect you are trying to measure.
# MAGIC
# MAGIC The right tool is Double Machine Learning (DML). It separates the causal price effect from confounding by fitting two nuisance models: one predicts the renewal outcome from risk factors, one predicts the price change from risk factors. What remains after subtracting both fitted values is the residual relationship between price and renewal that risk-driven correlation cannot explain.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Generate 10,000 synthetic UK motor renewals with known heterogeneous elasticities
# MAGIC 2. Run the treatment variation diagnostic — confirm enough exogenous price variation
# MAGIC 3. Fit CausalForestDML and estimate heterogeneous price elasticity (CATE)
# MAGIC 4. Segment-level GATEs: elasticity by NCD band, age group, and channel
# MAGIC 5. Elasticity surface and visualisations
# MAGIC 6. ENBP-constrained profit-maximising pricing (FCA PS21/5)
# MAGIC 7. Portfolio demand curve and efficient frontier
# MAGIC 8. ICOBS 6B.2 compliance audit trail
# MAGIC
# MAGIC **Library:** `insurance-elasticity` — DML elasticity estimation and FCA-compliant renewal optimiser

# COMMAND ----------

# MAGIC %pip install "insurance-elasticity[all]" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate Synthetic Renewal Dataset
# MAGIC
# MAGIC `make_renewal_data()` generates a synthetic UK motor renewal book with a known data-generating process. The DGP has heterogeneous elasticities: no-NCD customers are much more price-sensitive than max-NCD customers, young drivers more than older ones, and PCW customers 30% more elastic than direct-channel equivalents.
# MAGIC
# MAGIC The true elasticities by segment are exported so we can check the DML estimates are plausible. In production, replace this with your own policy extract — the schema is the same.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Treatment Variation Diagnostic
# MAGIC
# MAGIC Before fitting anything, check whether there is enough exogenous price variation for DML to identify the causal effect. The central challenge in applying DML to renewal data is the near-deterministic price problem: re-rating makes the offered price nearly a deterministic function of observable risk factors. When that happens, Var(D̃)/Var(D) approaches zero — exactly as in a weak instruments problem.
# MAGIC
# MAGIC This dataset has `price_variation_sd=0.08`, which provides enough exogenous variation (Var(D̃)/Var(D) well above the 10% threshold). A real book with no A/B testing or UW override will typically have far less.

# COMMAND ----------

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

# Calibration check: renewal rate should fall as price rises
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fit CausalForestDML Elasticity Model
# MAGIC
# MAGIC `CausalForestDML` is the right choice when you want a heterogeneous treatment effect surface across the portfolio, not just a portfolio average. It is fully non-parametric: it makes no assumption about which interactions drive elasticity heterogeneity. Honest splitting gives valid pointwise confidence intervals.
# MAGIC
# MAGIC CatBoost nuisance models handle categorical covariates natively. Estimator settings here are a practical compromise — increase `n_estimators` to 200-500 for production runs.

# COMMAND ----------

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

ate, lb, ub = est.ate()
print(f"\nAverage treatment effect (ATE): {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")

# Interpret the ATE in practical terms
log_10pct = np.log(1.10)
log_5pct = np.log(1.05)
log_20pct = np.log(1.20)

print(f"\nHow to interpret the ATE (semi-elasticity):")
print(f"    5% price increase  (log change = {log_5pct:.4f}) => {ate * log_5pct:+.3f} pp change in renewal probability")
print(f"   10% price increase  (log change = {log_10pct:.4f}) => {ate * log_10pct:+.3f} pp change in renewal probability")
print(f"   20% price increase  (log change = {log_20pct:.4f}) => {ate * log_20pct:+.3f} pp change in renewal probability")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Segment-Level Elasticity (GATE)
# MAGIC
# MAGIC Group Average Treatment Effects (GATEs) answer: what is the average elasticity for customers in NCD band 0 vs NCD band 5? Segments that are more elastic (more negative GATE) should receive smaller price increases. Inelastic segments can absorb larger increases with minimal retention impact.
# MAGIC
# MAGIC We compare the estimated GATEs against the known DGP truth. In production you would not have the true values, but the pattern — max-NCD customers least elastic, no-NCD most elastic — should be visible in the data.

# COMMAND ----------

# By NCD band
print("Elasticity by NCD band (estimated vs DGP truth):")
gate_ncd = est.gate(df, by="ncd_years")
true_ncd = true_gate_by_ncd(df)
true_map = {row["ncd_years"]: row["true_elasticity_mean"] for row in true_ncd.iter_rows(named=True)}

print(f"  {'NCD':>5}  {'Estimated':>10}  {'95% CI':>22}  {'DGP truth':>10}  {'Bias':>8}  {'n':>6}")
print(f"  {'-'*5}  {'-'*10}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*6}")
for row in gate_ncd.iter_rows(named=True):
    ncd = row["ncd_years"]
    est_e = row["elasticity"]
    ci_lo, ci_hi = row["ci_lower"], row["ci_upper"]
    truth = true_map.get(ncd, float("nan"))
    bias = est_e - truth if not np.isnan(truth) else float("nan")
    print(
        f"  {ncd:>5}  {est_e:>10.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  {truth:>10.3f}  {bias:>+8.3f}  {row['n']:>6,}"
    )

# By age band
print("\nElasticity by age band (estimated vs DGP truth):")
gate_age = est.gate(df, by="age_band")
true_age_df = true_gate_by_age(df)
true_age_map = {row["age_band"]: row["true_elasticity_mean"] for row in true_age_df.iter_rows(named=True)}

age_order = {"17-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
gate_age_sorted = gate_age.with_columns(
    pl.col("age_band").replace(age_order).alias("_order")
).sort("_order").drop("_order")

for row in gate_age_sorted.iter_rows(named=True):
    ab = row["age_band"]
    truth = true_age_map.get(ab, float("nan"))
    print(
        f"  {ab:>10}  {row['elasticity']:>10.3f}  [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        f"  {truth:>10.3f}  n={row['n']:>6,}"
    )

# By channel
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
    " comparing prices at renewal. A 10% increase on PCW customers produces"
    " a larger lapse impact than the same increase on direct customers."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Elasticity Surface
# MAGIC
# MAGIC The elasticity surface is the key deliverable for a rate review discussion. It lets you identify which (NCD, age) combinations you can push harder and which need handling carefully. The `elasticity_at_10pct` column is what a pricing actuary needs: the expected change in renewal probability for a 10% rate increase.

# COMMAND ----------

from insurance_elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est)

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

# Save plots if matplotlib is available
try:
    import matplotlib
    matplotlib.use("Agg")
    fig = surface.plot_surface(df, dims=["ncd_years", "age_band"])
    fig.savefig("/tmp/elasticity_surface_ncd_age.png", dpi=150, bbox_inches="tight")
    print("\nElasticity heatmap saved to /tmp/elasticity_surface_ncd_age.png")
except Exception as exc:
    print(f"\nNote: plot skipped ({exc}). Install matplotlib to enable plots.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: FCA PS21/5-Compliant Pricing Optimisation
# MAGIC
# MAGIC The pricing optimiser solves, for each policy independently:
# MAGIC
# MAGIC ```
# MAGIC   max  (price - tech_prem) * P(renew | price, elasticity)
# MAGIC   s.t. price <= ENBP               (FCA ICOBS 6B.2 constraint)
# MAGIC        price >= tech_prem * 1.0    (no sub-technical pricing)
# MAGIC ```
# MAGIC
# MAGIC Customers with high negative elasticity (no-NCD PCW customers) will be optimised to a price below the ENBP because pushing them to ENBP costs too much in expected renewal probability. Inelastic customers (max-NCD, older, direct channel) will typically be priced at or near ENBP.

# COMMAND ----------

from insurance_elasticity.optimise import RenewalPricingOptimiser

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    priced_df = opt.optimise(df, objective="profit")

print("Profit-maximising pricing complete.")
print(f"\nPortfolio summary:")
print(f"  Mean current offer price:    £{df['offer_price'].mean():.2f}")
print(f"  Mean optimal price:          £{priced_df['optimal_price'].mean():.2f}")
print(f"  Mean ENBP:                   £{df['enbp'].mean():.2f}")
print(f"  Mean predicted renewal prob: {priced_df['predicted_renewal_prob'].mean():.1%}")

print("\nOptimal pricing by NCD band:")
ncd_summary = (
    priced_df
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Portfolio Demand Curve and Efficient Frontier
# MAGIC
# MAGIC The demand curve sweeps a range of uniform price changes across the portfolio and predicts renewal rate and expected profit at each point. Every point on this frontier is Pareto-optimal: you cannot improve both metrics simultaneously. If your board sets a retention floor (e.g., "retain at least 83% of customers"), the demand curve tells you the maximum price change compatible with that constraint.

# COMMAND ----------

from insurance_elasticity.demand import demand_curve, plot_demand_curve

demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))

# Find profit-maximising price change
max_profit_row = demand_df.filter(
    pl.col("predicted_profit") == pl.col("predicted_profit").max()
).row(0, named=True)

print(f"Portfolio demand curve (selected price points):")
print(f"  {'Price change':>13}  {'Renewal rate':>13}  {'Profit/policy':>14}")
print(f"  {'-'*13}  {'-'*13}  {'-'*14}")

selected_indices = [0, 6, 12, 19, 24, 30, 37, 43, 49]
for i, row in enumerate(demand_df.iter_rows(named=True)):
    if i in selected_indices:
        pct = row["pct_price_change"] * 100
        renewal = row["predicted_renewal_rate"] * 100
        profit = row["predicted_profit"]
        star = "  <-- profit max" if abs(row["log_price_change"] - max_profit_row["log_price_change"]) < 0.01 else ""
        print(f"  {pct:>+12.1f}%  {renewal:>12.1f}%  £{profit:>12.0f}{star}")

print(f"\nProfit-maximising price change: {max_profit_row['pct_price_change']*100:+.1f}%")
print(f"  Expected renewal rate: {max_profit_row['predicted_renewal_rate']:.1%}")
print(f"  Expected profit per policy: £{max_profit_row['predicted_profit']:.0f}")

# With a retention floor
retention_floor = 0.83
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: FCA ICOBS 6B.2 Compliance Audit Trail
# MAGIC
# MAGIC FCA PS21/5 (January 2022) requires that no renewing customer is quoted a price above the equivalent new business price (ENBP). The `enbp_audit()` method produces the per-row compliance flag for your compliance function. A correctly configured optimiser with `floor_loading=1.0` and valid ENBP data should produce zero breaches.

# COMMAND ----------

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    audit = opt.enbp_audit(priced_df)

n_compliant = int(audit["compliant"].sum())
n_breach = len(audit) - n_compliant

print(f"Compliance summary:")
print(f"  Total policies: {len(audit):,}")
print(f"  Compliant:      {n_compliant:,}  ({n_compliant/len(audit):.2%})")
print(f"  Breaches:       {n_breach:,}  ({n_breach/len(audit):.2%})")

if n_breach > 0:
    print(f"\n  BREACH DETAIL (first 10):")
    for row in audit.filter(pl.col("compliant") == False).head(10).iter_rows(named=True):
        print(
            f"    policy {row['policy_id']:>6}"
            f"  offered £{row['offered_price']:.2f}"
            f"  ENBP £{row['enbp']:.2f}"
            f"  breach {row['pct_above_enbp']:.2f}%"
        )
else:
    print(f"\n  All {len(audit):,} policies are ICOBS 6B.2 compliant.")

headroom = audit["margin_to_enbp"].to_numpy()
print(f"\nENBP headroom distribution:")
print(f"  Median headroom:          £{np.median(headroom):.2f}")
print(f"  10th percentile:          £{np.percentile(headroom, 10):.2f}")
print(f"  Policies within £5 of ENBP:  {int((headroom < 5).sum()):,}")
print(f"  Policies within £10 of ENBP: {int((headroom < 10).sum()):,}")

print("\nSample audit output (first 5 policies):")
print(f"  {'policy_id':>10}  {'offered_price':>14}  {'enbp':>8}  {'compliant':>10}  {'margin':>8}")
for row in audit.head(5).iter_rows(named=True):
    compliant_str = "YES" if row["compliant"] else "BREACH"
    print(
        f"  {row['policy_id']:>10}"
        f"  £{row['offered_price']:>12.2f}"
        f"  £{row['enbp']:>6.2f}"
        f"  {compliant_str:>10}"
        f"  £{row['margin_to_enbp']:>6.2f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was demonstrated:**
# MAGIC 1. Synthetic data with heterogeneous true elasticities (NCD 0: -3.5 to NCD 5: -1.0)
# MAGIC 2. Treatment variation diagnostic (Var(D̃)/Var(D) check before fitting)
# MAGIC 3. CausalForestDML ATE with confidence intervals
# MAGIC 4. GATE by NCD band, age band, and channel — with DGP truth comparison
# MAGIC 5. Elasticity surface (NCD x age band) for rate review discussions
# MAGIC 6. Profit-maximising pricing within ENBP ceiling (FCA PS21/5)
# MAGIC 7. Demand curve and efficient frontier for retention-constrained planning
# MAGIC 8. ICOBS 6B.2 compliance audit trail
# MAGIC
# MAGIC **In production:**
# MAGIC - Replace `make_renewal_data()` with your policy extract (same schema)
# MAGIC - Increase `n_estimators` to 200-500 and `catboost_iterations` to 500
# MAGIC - Validate ENBP data feed against your NB model output before running the audit
# MAGIC - If Var(D̃)/Var(D) < 0.10 on your real book, address the identification problem before presenting results
