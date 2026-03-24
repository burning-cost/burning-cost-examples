# Databricks notebook source

# MAGIC %md
# MAGIC # 3-Objective Pareto Optimisation: Profit, Retention, and Fairness
# MAGIC
# MAGIC In UK personal lines pricing, the standard workflow is to maximise profit subject to a retention floor. That fails badly when two things happen simultaneously: the FCA starts asking about Consumer Duty compliance, and the board wants to understand what it costs to improve the fairness of your pricing structure.
# MAGIC
# MAGIC Single-objective optimisation does not answer that question. The Pareto surface gives you the whole trade-off: here is what happens to premium disparity across deprivation quintiles as you slide from maximum profit to maximum retention. Here is the point where accepting 3% less profit reduces your fairness disparity ratio from 1.42 to 1.12. That is the conversation a pricing committee can have.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Generate a synthetic 500-policy UK renewal portfolio
# MAGIC 2. Single-objective profit maximisation — establish a baseline
# MAGIC 3. 3-objective ParetoFrontier (profit x retention x fairness)
# MAGIC 4. TOPSIS selection of best compromise
# MAGIC 5. Board summary table

# COMMAND ----------

# MAGIC %pip install "insurance-optimise" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings
from functools import partial

import numpy as np

warnings.filterwarnings("ignore")

from insurance_optimise import (
    ConstraintConfig, EfficientFrontier, ParetoFrontier,
    PortfolioOptimiser, premium_disparity_ratio,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic UK Motor Renewal Portfolio
# MAGIC
# MAGIC 500 policies with technical prices drawn from a realistic UK motor range. Group labels represent deprivation quintiles (1=least deprived, 5=most). The premium disparity ratio = mean(most expensive quintile) / mean(cheapest quintile). FCA Consumer Duty PS22/9 does not set a hard limit but expects firms to justify the level.

# COMMAND ----------

N = 500
SEED = 42
N_QUINTILES = 5

rng = np.random.default_rng(SEED)

tc = rng.uniform(200, 800, N)
cost = tc * rng.uniform(0.55, 0.75, N)
p_demand = rng.uniform(0.70, 0.95, N)
elasticity = rng.uniform(-2.5, -0.8, N)
renewal_flag = np.ones(N, dtype=bool)
enbp = tc * rng.uniform(1.0, 1.3, N)
group_labels = np.clip(
    rng.integers(1, N_QUINTILES + 1, N) + (tc > 600).astype(int),
    1, N_QUINTILES,
)

print(f"Portfolio: {N} policies, {N_QUINTILES} deprivation quintiles")
print(f"  Technical price: £{tc.mean():.0f} mean  (£{tc.min():.0f}–£{tc.max():.0f})")
print(f"  Expected LR:     {(cost / tc).mean():.3f} mean")
print(f"  Base renewal:    {p_demand.mean():.3f} mean probability at tech price")
print(f"  Elasticity:      {elasticity.mean():.2f} mean semi-elasticity")

fairness_fn = partial(premium_disparity_ratio, technical_price=tc, group_labels=group_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Single-Objective Baseline (Profit Maximisation)
# MAGIC
# MAGIC Standard setup: maximise profit, subject to a 90% retention floor and a 72% loss ratio cap. This is how most UK pricing teams run optimisation today. The fairness number is reported post-hoc — nobody optimised it, nobody chose it.

# COMMAND ----------

config_base = ConstraintConfig(lr_max=0.72, retention_min=0.90, max_rate_change=0.25)

opt_single = PortfolioOptimiser(
    technical_price=tc, expected_loss_cost=cost, p_demand=p_demand,
    elasticity=elasticity, renewal_flag=renewal_flag, enbp=enbp,
    constraints=config_base, seed=SEED,
)

result_single = opt_single.optimise()
fair_single = fairness_fn(result_single.multipliers)
ret_single = result_single.expected_retention or 0.0

print(f"Single-objective result:")
print(f"  Profit:    £{result_single.expected_profit:,.0f}")
print(f"  Retention: {ret_single * 100:.1f}%")
print(f"  LR:        {result_single.expected_loss_ratio:.3f}")
print(f"  Fairness (disparity ratio): {fair_single:.3f}")
print(f"\nThe fairness ratio of {fair_single:.3f} means the highest-premium quintile pays")
print(f"{(fair_single - 1) * 100:.0f}% more on average than the lowest. This was never")
print("an objective — it just happened.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: 3-Objective Pareto Surface (8x8 Grid)
# MAGIC
# MAGIC We sweep over: retention floor (0.80 to 0.98, 8 points) and fairness cap (1.05 to 2.50, 8 points). At each combination we solve the SLSQP problem and record profit, retention, fairness. The Pareto filter identifies non-dominated solutions.

# COMMAND ----------

config_no_ret = ConstraintConfig(lr_max=0.72, retention_min=None, max_rate_change=0.25)

opt_pareto = PortfolioOptimiser(
    technical_price=tc, expected_loss_cost=cost, p_demand=p_demand,
    elasticity=elasticity, renewal_flag=renewal_flag, enbp=enbp,
    constraints=config_no_ret, seed=SEED,
)

pf = ParetoFrontier(
    optimiser=opt_pareto,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(0.80, 0.98),
    sweep_y="fairness_max",
    sweep_y_range=(1.05, 2.50),
    n_points_x=8, n_points_y=8, n_jobs=1,
)

print("Running 64 SLSQP solves (profit x retention x fairness)...")
pareto_result = pf.run()

n_total = len(pareto_result.surface)
n_converged = int(pareto_result.surface["converged"].sum())
n_nondom = len(pareto_result.pareto_df)

print(f"Grid: {n_total} solves, {n_converged} converged, {n_nondom} non-dominated")

if n_nondom > 0:
    summary = pareto_result.summary()
    print("\nNon-dominated solutions (sorted by profit):")
    print(f"  {'Profit':>12}  {'Retention':>10}  {'Fairness':>10}  {'LR':>8}")
    for row in summary.head(10).iter_rows(named=True):
        print(f"  £{row['profit']:>10,.0f}  {row['retention'] * 100:>9.1f}%  {row['fairness']:>10.3f}  {row['loss_ratio']:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: TOPSIS Selection
# MAGIC
# MAGIC TOPSIS selects the single "best compromise" from the Pareto surface. The weights encode preferences: 0.5 profit (primary commercial objective), 0.3 retention (regulatory/competitive), 0.2 fairness (Consumer Duty / ESG). The weights are inputs to a governance discussion, not magic numbers.

# COMMAND ----------

if n_nondom > 0:
    pareto_result.select(method="topsis", weights=(0.5, 0.3, 0.2))
    sel = pareto_result.selected

    profit_topsis = sel.expected_profit
    ret_topsis = sel.expected_retention or 0.0
    fair_topsis = sel.audit_trail.get("fairness", float("nan"))

    profit_diff_pct = (profit_topsis - result_single.expected_profit) / abs(result_single.expected_profit) * 100
    fair_improvement = fair_single - fair_topsis
    ret_diff_pp = (ret_topsis - ret_single) * 100

    print(f"TOPSIS compromise (weights: profit=0.5, retention=0.3, fairness=0.2):")
    print(f"  Profit:    £{profit_topsis:,.0f}  ({profit_diff_pct:+.1f}% vs single-objective)")
    print(f"  Retention: {ret_topsis * 100:.1f}%  ({ret_diff_pp:+.1f}pp vs single-objective)")
    print(f"  Fairness:  {fair_topsis:.3f}  (was {fair_single:.3f} — improvement of {fair_improvement:.3f})")

    if abs(profit_diff_pct) < 0.5:
        print(f"\nA {fair_improvement:.2f}-point fairness improvement costs essentially nothing.")
        print(f"The profit difference is within rounding error ({profit_diff_pct:+.2f}%).")
    else:
        print(f"\nThe fairness improvement costs {abs(profit_diff_pct):.1f}% of profit.")
        print("Whether that is acceptable is a governance decision, not a technical one.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Board Summary Table
# MAGIC
# MAGIC This is the table you bring to the pricing committee. It shows what single-objective optimisation gets you, the extremes of the Pareto surface, and the TOPSIS compromise. Under FCA Consumer Duty, you need to justify your pricing structure. "We ran the Pareto analysis, understood the trade-off, and made an informed decision" is defensible. "We didn't know what the fairness impact was" is not.

# COMMAND ----------

if n_nondom > 0:
    pf_data = pareto_result.pareto_df
    profits = pf_data["profit"].to_list()
    retentions = pf_data["retention"].to_list()
    fairnesses = pf_data["fairness"].to_list()

    max_profit_row = pf_data.sort("profit", descending=True).row(0, named=True)
    max_ret_row = pf_data.sort("retention", descending=True).row(0, named=True)
    min_fair_row = pf_data.sort("fairness").row(0, named=True)

    print(f"  {'Scenario':<40} {'Profit':>10} {'Retention':>10} {'Fairness':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Baseline (single-objective)':<40} £{result_single.expected_profit:>8,.0f} {ret_single * 100:>9.1f}% {fair_single:>10.3f}")
    print(f"  {'Pareto: max profit':<40} £{max_profit_row['profit']:>8,.0f} {max_profit_row['retention'] * 100:>9.1f}% {max_profit_row['fairness']:>10.3f}")
    print(f"  {'Pareto: max retention':<40} £{max_ret_row['profit']:>8,.0f} {max_ret_row['retention'] * 100:>9.1f}% {max_ret_row['fairness']:>10.3f}")
    print(f"  {'Pareto: min fairness disparity':<40} £{min_fair_row['profit']:>8,.0f} {min_fair_row['retention'] * 100:>9.1f}% {min_fair_row['fairness']:>10.3f}")
    if n_nondom > 0:
        print(f"  {'TOPSIS compromise (0.5/0.3/0.2)':<40} £{profit_topsis:>8,.0f} {ret_topsis * 100:>9.1f}% {fair_topsis:>10.3f}")

    try:
        pareto_result.save_audit("/tmp/pareto_audit.json")
        print("\nAudit trail saved to /tmp/pareto_audit.json (suitable for Consumer Duty evidence)")
    except Exception as exc:
        print(f"\n(Audit trail save skipped: {exc})")
