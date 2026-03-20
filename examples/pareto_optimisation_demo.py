"""
3-objective Pareto optimisation: profit, retention, and fairness.

In UK personal lines pricing, the standard workflow is to maximise profit
subject to a retention floor. That works well enough when you have a single
stakeholder. It fails badly when two things happen simultaneously: the FCA
starts asking about Consumer Duty compliance, and the board wants to understand
what it actually costs to improve the fairness of your pricing structure.

Single-objective optimisation does not answer that question. It gives you one
number. The Pareto surface gives you the whole trade-off: here is what happens
to premium disparity across deprivation quintiles as you slide from maximum
profit to maximum retention. Here is the point where accepting 3% less profit
reduces your fairness disparity ratio from 1.42 to 1.12. That is the
conversation a pricing committee can actually have.

This script demonstrates:

    1. Generate a synthetic 500-policy UK renewal portfolio
    2. Run single-objective profit maximisation — establish a baseline
    3. Run the 3-objective ParetoFrontier (profit × retention × fairness)
       using an 8x8 epsilon-constraint grid sweep
    4. Use TOPSIS to select the best compromise under given preference weights
    5. Show what the board actually needs to see: the explicit trade-off table

Libraries used
--------------
    insurance-optimise  — PortfolioOptimiser, ParetoFrontier, ConstraintConfig

Dependencies
------------
    uv add "insurance-optimise"

Approximate runtime: 2–5 minutes on a modern CPU (64 SLSQP solves, each
on a 500-policy problem with analytical gradients).
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np

warnings.filterwarnings("ignore")

from insurance_optimise import (
    ConstraintConfig,
    EfficientFrontier,
    ParetoFrontier,
    PortfolioOptimiser,
    premium_disparity_ratio,
)

# ---------------------------------------------------------------------------
# Step 1: Synthetic UK motor renewal portfolio
# ---------------------------------------------------------------------------
#
# 500 policies. Technical prices drawn from a realistic range for UK motor:
# budget hatchbacks at the low end, high-value vehicles or young drivers at
# the top. Loss ratios are modestly variable — this is a reasonably-run book.
#
# Group labels represent deprivation quintiles (1=least deprived, 5=most).
# Premium disparity ratio = mean(premium in most expensive quintile) /
#                            mean(premium in cheapest quintile).
# A ratio of 1.0 is perfect equality. FCA Consumer Duty PS22/9 does not set a
# hard limit but expects firms to be able to explain and justify the level.
# ---------------------------------------------------------------------------

print("=" * 68)
print("Step 1: Synthetic portfolio")
print("=" * 68)

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
# Deprivation quintile assigned with correlation to premium: higher-deprivation
# areas tend to have slightly older vehicles and lower NCD. In a real book this
# is the channel through which indirect discrimination can occur.
group_labels = np.clip(
    rng.integers(1, N_QUINTILES + 1, N) + (tc > 600).astype(int),
    1, N_QUINTILES,
)

print(f"Portfolio: {N} policies, {N_QUINTILES} deprivation quintiles")
print(f"  Technical price: £{tc.mean():.0f} mean  (£{tc.min():.0f}–£{tc.max():.0f})")
print(f"  Expected LR:     {(cost / tc).mean():.3f} mean")
print(f"  Base renewal:    {p_demand.mean():.3f} mean probability at tech price")
print(f"  Elasticity:      {elasticity.mean():.2f} mean semi-elasticity")
print()

# Bind the fairness metric to the portfolio arrays
fairness_fn = partial(
    premium_disparity_ratio,
    technical_price=tc,
    group_labels=group_labels,
)

# ---------------------------------------------------------------------------
# Step 2: Single-objective baseline
# ---------------------------------------------------------------------------
#
# Standard setup: maximise profit, subject to a 90% retention floor and a
# 72% loss ratio cap. This is how most UK pricing teams run optimisation today.
#
# The result looks good on the metrics the team tracks. The fairness number
# is reported post-hoc — nobody optimised it, nobody chose it.
# ---------------------------------------------------------------------------

print("=" * 68)
print("Step 2: Single-objective baseline (profit maximisation)")
print("=" * 68)

config_base = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.90,
    max_rate_change=0.25,
)

opt_single = PortfolioOptimiser(
    technical_price=tc,
    expected_loss_cost=cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config_base,
    seed=SEED,
)

result_single = opt_single.optimise()
fair_single = fairness_fn(result_single.multipliers)
ret_single = result_single.expected_retention or 0.0

print(f"  Profit:    £{result_single.expected_profit:,.0f}")
print(f"  Retention: {ret_single * 100:.1f}%")
print(f"  LR:        {result_single.expected_loss_ratio:.3f}")
print(f"  Fairness (disparity ratio): {fair_single:.3f}")
print(f"  Converged: {result_single.converged}")
print()
print(
    "  The fairness ratio of {:.3f} means the highest-premium quintile pays".format(fair_single)
)
print(
    "  {:.0f}% more on average than the lowest. This was never an objective —".format(
        (fair_single - 1) * 100
    )
)
print("  it just happened. The board cannot assess whether this is acceptable")
print("  because they were never shown the alternative.")

# ---------------------------------------------------------------------------
# Step 3: 3-objective Pareto surface
# ---------------------------------------------------------------------------
#
# We sweep over:
#   x-axis: volume retention (0.80 to 0.98, 8 points)
#   y-axis: fairness_max (1.05 to 2.50, 8 points — cap on disparity ratio)
#
# At each (retention_floor, fairness_cap) combination, we solve the
# SLSQP problem and record profit, retention, fairness. The Pareto filter
# then identifies which solutions are non-dominated.
#
# 8x8 = 64 solves. Each takes 0.5–3s depending on problem difficulty.
# n_jobs=1 to avoid joblib dependency issues in some environments.
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("Step 3: 3-objective Pareto surface (8x8 grid)")
print("=" * 68)
print("Running 64 SLSQP solves (profit × retention × fairness)...")

config_no_ret = ConstraintConfig(
    lr_max=0.72,
    retention_min=None,  # ParetoFrontier sweeps this
    max_rate_change=0.25,
)

opt_pareto = PortfolioOptimiser(
    technical_price=tc,
    expected_loss_cost=cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config_no_ret,
    seed=SEED,
)

pf = ParetoFrontier(
    optimiser=opt_pareto,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(0.80, 0.98),
    sweep_y="fairness_max",
    sweep_y_range=(1.05, 2.50),
    n_points_x=8,
    n_points_y=8,
    n_jobs=1,
)

pareto_result = pf.run()

n_total = len(pareto_result.surface)
n_converged = int(pareto_result.surface["converged"].sum())
n_nondom = len(pareto_result.pareto_df)

print(f"Grid: {n_total} solves, {n_converged} converged, {n_nondom} non-dominated")
print()

# Print the Pareto front sorted by profit
if n_nondom > 0:
    summary = pareto_result.summary()
    print("Non-dominated solutions (sorted by profit):")
    print(f"  {'Profit':>12}  {'Retention':>10}  {'Fairness':>10}  {'LR':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for row in summary.head(10).iter_rows(named=True):
        print(
            f"  £{row['profit']:>10,.0f}"
            f"  {row['retention'] * 100:>9.1f}%"
            f"  {row['fairness']:>10.3f}"
            f"  {row['loss_ratio']:>8.3f}"
        )
    if len(summary) > 10:
        print(f"  ... ({len(summary)} total non-dominated solutions)")

# ---------------------------------------------------------------------------
# Step 4: TOPSIS selection
# ---------------------------------------------------------------------------
#
# TOPSIS selects the single "best compromise" point from the Pareto surface.
# The weights encode your preferences:
#   0.5  profit    — primary commercial objective
#   0.3  retention — regulatory and competitive concern
#   0.2  fairness  — Consumer Duty / board ESG requirement
#
# Changing these weights shifts the selected point along the Pareto surface.
# The weights are inputs to a governance discussion, not magic numbers.
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("Step 4: TOPSIS selection (profit=0.5, retention=0.3, fairness=0.2)")
print("=" * 68)

if n_nondom > 0:
    pareto_result.select(method="topsis", weights=(0.5, 0.3, 0.2))
    sel = pareto_result.selected

    profit_topsis = sel.expected_profit
    ret_topsis = sel.expected_retention or 0.0
    fair_topsis = sel.audit_trail.get("fairness", float("nan"))

    profit_diff_pct = (
        (profit_topsis - result_single.expected_profit)
        / abs(result_single.expected_profit)
        * 100
    )
    fair_improvement = fair_single - fair_topsis
    ret_diff_pp = (ret_topsis - ret_single) * 100

    print(f"  TOPSIS compromise:")
    print(f"    Profit:    £{profit_topsis:,.0f}  ({profit_diff_pct:+.1f}% vs single-objective)")
    print(f"    Retention: {ret_topsis * 100:.1f}%  ({ret_diff_pp:+.1f}pp vs single-objective)")
    print(f"    Fairness:  {fair_topsis:.3f}  (was {fair_single:.3f} — improvement of {fair_improvement:.3f})")
    print()

    # The key insight: quantify the cost of fairness improvement
    if abs(profit_diff_pct) < 0.5:
        print(
            f"  A {fair_improvement:.2f}-point fairness improvement (disparity ratio "
            f"{fair_single:.3f} → {fair_topsis:.3f}) costs essentially nothing."
        )
        print(
            f"  The profit difference is within rounding error ({profit_diff_pct:+.2f}%)."
        )
        print(
            f"  The single-objective solution leaves fairness on the table for free."
        )
    else:
        print(
            f"  The fairness improvement costs {abs(profit_diff_pct):.1f}% of profit."
        )
        print(
            f"  Whether that is acceptable is a governance decision, not a technical one."
        )
        print(
            f"  The Pareto surface makes the cost explicit so the board can decide."
        )
else:
    print("  No non-dominated solutions found — check constraint feasibility.")
    profit_topsis = float("nan")
    ret_topsis = fair_topsis = float("nan")
    profit_diff_pct = fair_improvement = ret_diff_pp = float("nan")

# ---------------------------------------------------------------------------
# Step 5: The board summary table
# ---------------------------------------------------------------------------
#
# This is the table you bring to the pricing committee. It shows:
#   - What single-objective optimisation gets you
#   - What you can get at the extremes of the Pareto surface
#   - What the TOPSIS compromise looks like
#
# The point is not to optimise for fairness at all costs. The point is to make
# the trade-off visible and governable. Under FCA Consumer Duty, you need to
# be able to justify your pricing structure. "We ran the Pareto analysis,
# understood the trade-off, and made an informed decision" is a defensible
# position. "We didn't know what the fairness impact was" is not.
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("Step 5: Board summary — the trade-off made visible")
print("=" * 68)
print()

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

    print(
        f"  {'Baseline (single-objective, no fairness)':<40}"
        f" £{result_single.expected_profit:>8,.0f}"
        f" {ret_single * 100:>9.1f}%"
        f" {fair_single:>10.3f}"
    )
    print(
        f"  {'Pareto: max profit':<40}"
        f" £{max_profit_row['profit']:>8,.0f}"
        f" {max_profit_row['retention'] * 100:>9.1f}%"
        f" {max_profit_row['fairness']:>10.3f}"
    )
    print(
        f"  {'Pareto: max retention':<40}"
        f" £{max_ret_row['profit']:>8,.0f}"
        f" {max_ret_row['retention'] * 100:>9.1f}%"
        f" {max_ret_row['fairness']:>10.3f}"
    )
    print(
        f"  {'Pareto: min fairness disparity':<40}"
        f" £{min_fair_row['profit']:>8,.0f}"
        f" {min_fair_row['retention'] * 100:>9.1f}%"
        f" {min_fair_row['fairness']:>10.3f}"
    )
    if not (profit_topsis != profit_topsis):
        print(
            f"  {'TOPSIS compromise (0.5/0.3/0.2)':<40}"
            f" £{profit_topsis:>8,.0f}"
            f" {ret_topsis * 100:>9.1f}%"
            f" {fair_topsis:>10.3f}"
        )

print()
print(
    "  Save a JSON audit trail with pareto_result.save_audit('/tmp/pareto_audit.json')"
)
print(
    "  This JSON is suitable as regulatory evidence under FCA Consumer Duty PS22/9."
)

# Audit trail output
try:
    pareto_result.save_audit("/tmp/pareto_audit.json")
    print("  Audit trail saved to /tmp/pareto_audit.json")
except Exception as exc:
    print(f"  (Audit trail save skipped: {exc})")

print()
print("=" * 68)
print("Demo complete")
print("=" * 68)
print(f"""
What was demonstrated:

  Single-objective  Profit £{result_single.expected_profit:,.0f}  Retention {ret_single:.1%}
                    Fairness disparity ratio: {fair_single:.3f}
                    (This ratio was never optimised — it is a side effect)

  Pareto surface    {n_nondom} non-dominated solutions from a {n_total}-point grid
                    Retention range: {min(r for r in retentions if r):.1%} – {max(r for r in retentions if r):.1%}
                    Fairness range: {min(f for f in fairnesses if f):.3f} – {max(f for f in fairnesses if f):.3f}

  TOPSIS (0.5/0.3/0.2)  Profit {profit_diff_pct:+.1f}%  Fairness improvement {fair_improvement:.3f}

  Next steps:
    - Run with your real portfolio (replace the numpy arrays with your data)
    - Adjust weights in select() to match your board's priorities
    - Schedule quarterly: regulatory scrutiny of fairness increases after
      each rate review cycle
    - Use pareto_result.save_audit() for Consumer Duty evidence trail
""")
