# Databricks notebook source

# MAGIC %md
# MAGIC # Constrained Portfolio Rate Optimisation with insurance-optimise
# MAGIC
# MAGIC **The problem:** Your pricing model produces technically correct loss costs for every policy in the book. That number tells you what you should charge actuarially. But it does not tell you what you *can* charge. Between the actuarial answer and the rate you actually file sit five overlapping constraints:
# MAGIC
# MAGIC - FCA PS21/11 (ENBP): renewal premiums must not exceed what a new customer would be quoted. This is enforceable, not advisory.
# MAGIC - A portfolio-level rate change target (+3% here) set by the CFO.
# MAGIC - Rate change guardrails — you cannot shock a policyholder with a 40% increase even if the model demands it.
# MAGIC - A loss ratio ceiling you have told Lloyd's about.
# MAGIC - Retention — the underwriting team gets nervous below 88% renewal rate.
# MAGIC
# MAGIC The typical approach is to run scenarios in a spreadsheet, pick one, and hope it satisfies all the constraints. It usually does not. This notebook shows what SLSQP constrained optimisation looks like on a realistic UK motor portfolio and why it materially outperforms both unconstrained greedy repricing and manual scenario selection.
# MAGIC
# MAGIC **What we benchmark:**
# MAGIC
# MAGIC | Method | Description |
# MAGIC |---|---|
# MAGIC | Baseline | Current premiums — no repricing at all |
# MAGIC | Unconstrained greedy | Reprice every policy to loss cost + fixed margin |
# MAGIC | Constrained SLSQP | Maximise profit subject to ENBP, rate cap, GWP target, LR ceiling |
# MAGIC
# MAGIC **Portfolio:** 50,000 synthetic UK motor renewal policies with realistic segment heterogeneity — some segments profitable at current rates, others loss-making, and ENBP bindings distributed non-uniformly across age bands and regions.

# COMMAND ----------

# MAGIC %pip install "insurance-optimise>=0.3.2" polars matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate Synthetic Portfolio
# MAGIC
# MAGIC We build 50,000 UK motor renewal policies with a structure that reflects a real book:
# MAGIC
# MAGIC - Age bands: 17-25 (young, high loss cost), 26-39, 40-59, 60+ (mature, more price-sensitive to absolute premium, ENBP often binding)
# MAGIC - Regions: London, South East, Midlands, North, Scotland, Wales
# MAGIC - NCD: 0, 1, 2, 3, 4, 5 years
# MAGIC - Technical price: GLM output (loss cost + expenses), varying £200-£2,000
# MAGIC - Expected loss cost: 55-75% of technical price
# MAGIC - ENBP: the new business quote, which is typically close to but not always at technical price — FCA requires renewal price <= ENBP
# MAGIC - Price elasticity: heterogeneous by age and channel (young drivers are more elastic on PCW)
# MAGIC - Current premium: some segments are over-priced (profitable), some under-priced (loss-making)
# MAGIC
# MAGIC The data-generating process is intentionally constructed so that:
# MAGIC 1. ENBP is binding for ~18% of renewals (realistic for a book that has been price-optimised over several cycles)
# MAGIC 2. The young driver segment is loss-making at current rates (adverse selection)
# MAGIC 3. Mature drivers have headroom above technical but are rate-sensitive

# COMMAND ----------

from __future__ import annotations

import json
import warnings
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", category=UserWarning)

# --- Reproducible synthetic data ---
rng = np.random.default_rng(42)
N = 50_000

# Segment definitions
AGE_BANDS = ["17-25", "26-39", "40-59", "60+"]
REGIONS = ["London", "South East", "Midlands", "North", "Scotland", "Wales"]
NCD_YEARS = [0, 1, 2, 3, 4, 5]

age_band = rng.choice(AGE_BANDS, N, p=[0.12, 0.28, 0.35, 0.25])
region = rng.choice(REGIONS, N, p=[0.18, 0.20, 0.22, 0.22, 0.10, 0.08])
ncd = rng.choice(NCD_YEARS, N, p=[0.10, 0.12, 0.14, 0.16, 0.20, 0.28])
is_pcw = rng.choice([True, False], N, p=[0.55, 0.45])

# Technical price: GLM output (loss cost + 20% expense loading)
# Young: high base cost; London loading; zero-NCD loading
age_loading = np.where(
    age_band == "17-25", 1.70,
    np.where(age_band == "26-39", 1.15,
    np.where(age_band == "40-59", 0.95, 0.90))
)
region_loading = np.where(
    region == "London", 1.30,
    np.where(region == "South East", 1.10,
    np.where(region == "Midlands", 1.00,
    np.where(region == "North", 0.95,
    np.where(region == "Scotland", 0.88, 0.85))))
)
ncd_discount = 1.0 - ncd * 0.08  # 8% per year of NCD

base_loss_cost = 420.0  # mean loss cost before loadings
loss_cost_noise = rng.lognormal(0, 0.25, N)
expected_loss_cost = (
    base_loss_cost * age_loading * region_loading * ncd_discount * loss_cost_noise
)
# Technical price = loss cost / (1 - expense ratio)
expense_ratio = 0.22
technical_price = expected_loss_cost / (1.0 - expense_ratio)

# Clamp to reasonable range
technical_price = np.clip(technical_price, 200.0, 2500.0)
expected_loss_cost = technical_price * (1.0 - expense_ratio)

# Current premium: simulate a book with historical pricing decisions
# Some segments deliberately over-priced (cheap NCD, mature) or under-priced (young)
current_premium_multiplier = np.ones(N)

# Young drivers: under-priced relative to technical by 5-20%
young_mask = age_band == "17-25"
current_premium_multiplier[young_mask] *= rng.uniform(0.80, 0.95, young_mask.sum())

# Mature drivers: at or slightly above technical
mature_mask = age_band == "60+"
current_premium_multiplier[mature_mask] *= rng.uniform(1.00, 1.10, mature_mask.sum())

# London: slight over-pricing legacy
london_mask = region == "London"
current_premium_multiplier[london_mask] *= rng.uniform(1.00, 1.08, london_mask.sum())

# Add noise to the rest
other_mask = ~(young_mask | mature_mask)
current_premium_multiplier[other_mask] *= rng.uniform(0.95, 1.10, other_mask.sum())

current_premium = current_premium_multiplier * technical_price

# ENBP: the new business equivalent price
# FCA PS21/11 requires renewal price <= ENBP.
# In practice, new business is priced more aggressively (lower) in some segments.
# ENBP is typically 95-115% of technical price, but tighter for mature/London.
enbp_ratio = np.ones(N)

# Mature drivers: ENBP close to technical (new business is competitive)
enbp_ratio[mature_mask] = rng.uniform(0.97, 1.05, mature_mask.sum())

# Young drivers: ENBP has more headroom (new business is less aggressively priced)
enbp_ratio[young_mask] = rng.uniform(1.05, 1.20, young_mask.sum())

# London: competitive new business means tighter ENBP
enbp_ratio[london_mask] = np.minimum(
    enbp_ratio[london_mask], rng.uniform(0.98, 1.08, london_mask.sum())
)

# General noise
enbp_ratio *= rng.uniform(0.98, 1.02, N)
enbp = technical_price * enbp_ratio

# ENBP must be positive
enbp = np.maximum(enbp, technical_price * 0.85)

# Price elasticity: heterogeneous by segment
# Log-linear demand: x(m) = x0 * m^epsilon
# Negative; more negative = more price-sensitive
base_elasticity = -1.80
elasticity = np.full(N, base_elasticity)

# Young PCW: most elastic
pcw_young_mask = is_pcw & young_mask
elasticity[pcw_young_mask] += rng.uniform(-0.60, -0.20, pcw_young_mask.sum())

# Mature direct: least elastic (captive)
direct_mature_mask = (~is_pcw) & mature_mask
elasticity[direct_mature_mask] += rng.uniform(0.20, 0.60, direct_mature_mask.sum())

# PCW generally more elastic than direct
elasticity[is_pcw] += rng.uniform(-0.30, -0.10, is_pcw.sum())

# Clamp to plausible range
elasticity = np.clip(elasticity, -3.5, -0.5)

# Renewal probability at current price
# Modelled as function of rate change, elasticity, and segment loyalty
# At current price (multiplier=1.0), baseline renewal probability
base_renewal_prob = 0.85
# Adjust by NCD (higher NCD = more loyal)
ncd_loyalty = 0.025 * ncd
# Adjust by channel (direct more loyal than PCW)
channel_loyalty = np.where(is_pcw, -0.04, 0.03)
# Young drivers: lower baseline loyalty
age_loyalty = np.where(
    age_band == "17-25", -0.06,
    np.where(age_band == "60+", 0.04, 0.00)
)

p_renewal = np.clip(
    base_renewal_prob + ncd_loyalty + channel_loyalty + age_loyalty
    + rng.normal(0, 0.02, N),
    0.55, 0.99
)

# Prior year multiplier: all renewals start at 1.0 (technical price IS the reference)
prior_multiplier = np.ones(N)

print(f"Portfolio: {N:,} UK motor renewal policies")
print(f"\nAge band distribution:")
for band in AGE_BANDS:
    mask = age_band == band
    mean_tc = technical_price[mask].mean()
    mean_lc = expected_loss_cost[mask].mean()
    raw_lr = expected_loss_cost[mask].sum() / (current_premium[mask]).sum()
    print(f"  {band:<8}  n={mask.sum():,}  mean_tc=£{mean_tc:.0f}  "
          f"current_LR={raw_lr:.2%}  mean_elasticity={elasticity[mask].mean():.2f}")

print(f"\nRegion distribution:")
for reg in REGIONS:
    mask = region == reg
    print(f"  {reg:<12}  n={mask.sum():,}  mean_tc=£{technical_price[mask].mean():.0f}")

print(f"\nPortfolio summary at current rates:")
total_gwp_current = current_premium.sum()
total_claims = expected_loss_cost.sum()
total_profit_current = total_gwp_current - total_claims
print(f"  Current GWP:       £{total_gwp_current:>12,.0f}")
print(f"  Expected claims:   £{total_claims:>12,.0f}")
print(f"  Expected profit:   £{total_profit_current:>12,.0f}")
print(f"  Current LR:        {total_claims/total_gwp_current:.2%}")
print(f"  ENBP violations:   {(current_premium > enbp).sum():,}  "
      f"({(current_premium > enbp).mean():.1%} of book)")
print(f"  Mean elasticity:   {elasticity.mean():.2f}")
print(f"  Mean renewal prob: {p_renewal.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Unconstrained Greedy Optimisation
# MAGIC
# MAGIC The simplest repricing strategy: set each policy's premium to its technical price, i.e., price = loss cost / (1 - expense ratio). This recovers the expense loading exactly and targets zero profit margin beyond expenses.
# MAGIC
# MAGIC In practice this means: if the loss cost says £600, charge £769. Do this for every policy with no regard for what anyone else in the portfolio pays.
# MAGIC
# MAGIC The result is almost always infeasible for regulatory and commercial reasons:
# MAGIC - ENBP violations in mature/London segments where the technical price exceeds the new business equivalent
# MAGIC - Extreme rate increases on the young driver book (currently under-priced)
# MAGIC - Portfolio-level rate change that the CFO did not approve
# MAGIC
# MAGIC We compute it here as a theoretical benchmark — the maximum profit achievable with no constraints.

# COMMAND ----------

# Unconstrained greedy: price = technical_price (multiplier = 1.0)
# But we want loss cost + margin, so price = loss_cost / (1 - target_margin)
# We use a 15% profit margin above loss cost (approximates pricing at technical rate)
# In multiplier space: m_greedy = 1.0 (technical_price already includes expense loading)
greedy_multiplier = np.ones(N)
greedy_premium = greedy_multiplier * technical_price

# Demand at greedy prices: use log-linear demand model
# x(m) = x0 * m^epsilon, where m = greedy_price / technical_price = 1.0
# At m=1.0, demand = p_renewal (by definition)
greedy_demand = p_renewal.copy()  # m=1.0, so demand = x0 * 1^epsilon = x0

# Portfolio metrics
greedy_gwp = np.dot(greedy_premium, greedy_demand)
greedy_claims = np.dot(expected_loss_cost, greedy_demand)
greedy_profit = greedy_gwp - greedy_claims
greedy_lr = greedy_claims / greedy_gwp
greedy_retention = greedy_demand.mean()
greedy_enbp_violations = np.sum(greedy_premium > enbp)

# Rate changes vs current
greedy_rate_change = (greedy_premium - current_premium) / current_premium
greedy_max_increase = greedy_rate_change.max()
greedy_pct_above_15 = (greedy_rate_change > 0.15).mean()

print("Unconstrained greedy (price to technical rate):")
print(f"  Expected GWP:      £{greedy_gwp:>12,.0f}")
print(f"  Expected claims:   £{greedy_claims:>12,.0f}")
print(f"  Expected profit:   £{greedy_profit:>12,.0f}")
print(f"  Loss ratio:        {greedy_lr:.2%}")
print(f"  Retention:         {greedy_retention:.2%}")
print(f"  ENBP violations:   {greedy_enbp_violations:,} ({greedy_enbp_violations/N:.1%})")
print(f"  Max rate increase: +{greedy_max_increase:.1%}")
print(f"  % policies >+15%:  {greedy_pct_above_15:.1%}")
print()
print("The ENBP violations and rate shock would not be acceptable.")
print("This gives the upper bound on profit — we will pay in constraint violations to get there.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Constrained SLSQP Optimisation
# MAGIC
# MAGIC Now we run `PortfolioOptimiser` with a realistic constraint set:
# MAGIC
# MAGIC | Constraint | Value | Rationale |
# MAGIC |---|---|---|
# MAGIC | ENBP | Hard ceiling per policy | FCA PS21/11, non-negotiable |
# MAGIC | ENBP buffer | 1% | Safety margin to avoid edge cases |
# MAGIC | Max rate change | ±15% | CFO guardrail — no shock increases |
# MAGIC | GWP min | +3% vs current | Portfolio target rate increase |
# MAGIC | Loss ratio max | 72% | Underwriting target |
# MAGIC | Retention min | 88% | Underwriting floor |
# MAGIC | Technical floor | True | Price >= loss cost + expenses always |
# MAGIC
# MAGIC The optimiser maximises expected profit subject to all of these simultaneously. The decision variables are multipliers `m_i = p_i / tc_i` (price as a fraction of technical price) for each of the 50,000 policies.

# COMMAND ----------

from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier

# GWP target: +3% rate increase over current book
gwp_target = current_premium.sum() * 1.03

config = ConstraintConfig(
    lr_max=0.72,                # loss ratio ceiling
    retention_min=0.88,         # renewal retention floor
    gwp_min=gwp_target,         # +3% rate increase target
    max_rate_change=0.15,       # ±15% rate change per policy
    enbp_buffer=0.01,           # 1% safety margin below ENBP
    technical_floor=True,       # price >= loss cost + expenses
    min_multiplier=0.85,        # absolute floor at 85% of technical
    max_multiplier=1.50,        # absolute ceiling at 150% of technical
)

print("Building PortfolioOptimiser with 50,000 policies...")
print(f"GWP target (current +3%): £{gwp_target:,.0f}")

opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_renewal,
    elasticity=elasticity,
    renewal_flag=np.ones(N, dtype=bool),  # all renewals
    enbp=enbp,
    prior_multiplier=prior_multiplier,
    constraints=config,
    demand_model="log_linear",
    solver="slsqp",
    n_restarts=1,
    ftol=1e-8,
    maxiter=500,
)

print(f"Active (non-bound) constraints: {opt.n_constraints}")
print("Running SLSQP optimisation...")
result = opt.optimise()
print()
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Results — Constrained vs Unconstrained vs Baseline
# MAGIC
# MAGIC The benchmark table captures the full picture. Unconstrained greedy maximises profit in theory but is not deliverable — it has regulatory violations and rate shocks. The constrained solution is the actually achievable optimum.

# COMMAND ----------

# Baseline metrics (current premiums, apply demand at current multiplier)
# Current multiplier relative to technical
current_multiplier = current_premium / technical_price

# Demand at current multiplier (demand model: x = x0 * m^epsilon)
# At current_multiplier m = current_premium/tc:
current_demand = p_renewal * (current_multiplier ** elasticity)
current_demand = np.clip(current_demand, 0.01, 0.99)

baseline_gwp = np.dot(current_premium, current_demand)
baseline_claims = np.dot(expected_loss_cost, current_demand)
baseline_profit = baseline_gwp - baseline_claims
baseline_lr = baseline_claims / baseline_gwp
baseline_retention = current_demand.mean()
baseline_enbp_viol = np.sum(current_premium > enbp)

# Constrained result
constrained_premium = result.new_premiums
constrained_demand = result.expected_demand
constrained_rate_change = (constrained_premium - current_premium) / current_premium
constrained_enbp_viol = np.sum(constrained_premium > enbp * (1 + 1e-4))
constrained_max_increase = constrained_rate_change.max()
constrained_pct_above_15 = (constrained_rate_change > 0.15).mean()

# Greedy rate change vs current
greedy_enbp_viol_count = int(greedy_enbp_violations)

# Summary table
rows = [
    {
        "Method": "Baseline (current rates)",
        "GWP (£M)": baseline_gwp / 1e6,
        "Profit (£M)": baseline_profit / 1e6,
        "Loss Ratio": baseline_lr,
        "Retention": baseline_retention,
        "ENBP Violations": baseline_enbp_viol,
        "Max Rate Increase": greedy_rate_change.min(),  # baseline: no change
        "% Policies >+15%": 0.0,
        "Converged": True,
    },
    {
        "Method": "Unconstrained greedy",
        "GWP (£M)": greedy_gwp / 1e6,
        "Profit (£M)": greedy_profit / 1e6,
        "Loss Ratio": greedy_lr,
        "Retention": greedy_retention,
        "ENBP Violations": greedy_enbp_viol_count,
        "Max Rate Increase": greedy_max_increase,
        "% Policies >+15%": greedy_pct_above_15,
        "Converged": True,
    },
    {
        "Method": "Constrained SLSQP",
        "GWP (£M)": result.expected_gwp / 1e6,
        "Profit (£M)": result.expected_profit / 1e6,
        "Loss Ratio": result.expected_loss_ratio,
        "Retention": result.expected_retention,
        "ENBP Violations": constrained_enbp_viol,
        "Max Rate Increase": constrained_max_increase,
        "% Policies >+15%": constrained_pct_above_15,
        "Converged": result.converged,
    },
]

benchmark_df = pl.DataFrame(rows)
print("=" * 80)
print("BENCHMARK: Unconstrained vs Constrained Rate Optimisation")
print("=" * 80)
print(f"\n{'Method':<28} {'GWP (£M)':>10} {'Profit (£M)':>12} {'LR':>6} "
      f"{'Ret%':>6} {'ENBP Viol':>10} {'Converged':>10}")
print("-" * 85)
for row in benchmark_df.iter_rows(named=True):
    print(
        f"{row['Method']:<28} "
        f"{row['GWP (£M)']:>10.2f} "
        f"{row['Profit (£M)']:>12.2f} "
        f"{row['Loss Ratio']:>6.2%} "
        f"{row['Retention']:>6.2%} "
        f"{row['ENBP Violations']:>10,} "
        f"{'Yes' if row['Converged'] else 'No':>10}"
    )
print()
print(f"Constraint satisfaction at SLSQP solution:")
for name, val in result.shadow_prices.items():
    status = "BINDING" if abs(val) > 1e-4 else "slack"
    print(f"  {name:<20}: shadow price = {val:.4f}  [{status}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Efficient Frontier
# MAGIC
# MAGIC The efficient frontier tells the pricing team the cost of tightening the retention floor. If we demand 90% retention we get less profit than at 85%. The question is: how much less, and is that an acceptable trade-off?
# MAGIC
# MAGIC We sweep `retention_min` from 83% to 93% with the other constraints fixed. Each point is an independent SLSQP solve.

# COMMAND ----------

print("Running efficient frontier sweep (retention 83% → 93%, 10 points)...")
print("Each point is an independent SLSQP solve — this will take 2-3 minutes.")

frontier = EfficientFrontier(
    opt,
    sweep_param="volume_retention",
    sweep_range=(0.83, 0.93),
    n_points=10,
    n_jobs=1,
)
frontier_result = frontier.run()
frontier_data = frontier_result.data

print(f"\nFrontier sweep complete. Converged points: "
      f"{frontier_data['converged'].sum()}/{len(frontier_data)}")
print()
print(f"{'Retention target':>18} {'Profit (£M)':>14} {'GWP (£M)':>10} "
      f"{'LR':>6} {'Converged':>10}")
print("-" * 65)
for row in frontier_data.iter_rows(named=True):
    print(
        f"{row['retention']:>17.1%}  "
        f"{row['profit']/1e6:>13.2f}  "
        f"{row['gwp']/1e6:>9.2f}  "
        f"{row['loss_ratio']:>5.2%}  "
        f"{'Yes' if row['converged'] else 'No':>10}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Plots
# MAGIC
# MAGIC Four panels:
# MAGIC 1. Efficient frontier: profit vs retention trade-off
# MAGIC 2. Rate change distribution: constrained vs unconstrained
# MAGIC 3. ENBP headroom: where is the constraint binding?
# MAGIC 4. Segment profit contribution: which segments drive the result?

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
ax1, ax2, ax3, ax4 = (
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
)

gbp_fmt = FuncFormatter(lambda x, _: f"£{x:.1f}M")

# ---- Panel 1: Efficient frontier ----
pareto = frontier_result.pareto_data()
if len(pareto) > 0:
    ax1.plot(
        pareto["retention"].to_numpy() * 100,
        pareto["profit"].to_numpy() / 1e6,
        "o-",
        color="#1f77b4",
        linewidth=2.0,
        markersize=7,
        label="Constrained frontier",
    )
# Mark the chosen operating point
ax1.axhline(result.expected_profit / 1e6, color="#d62728", linestyle="--",
            linewidth=1.2, alpha=0.7, label=f"Base constraint (88% ret.)")
ax1.axvline(88, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.set_xlabel("Retention rate (%)", fontsize=11)
ax1.set_ylabel("Expected profit (£M)", fontsize=11)
ax1.set_title("Efficient Frontier: Profit vs Retention", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(gbp_fmt)

# ---- Panel 2: Rate change distribution ----
bins = np.linspace(-0.20, 0.25, 40)
ax2.hist(
    constrained_rate_change * 100,
    bins=bins * 100,
    color="#1f77b4",
    alpha=0.70,
    label="Constrained SLSQP",
    density=True,
)
ax2.hist(
    greedy_rate_change * 100,
    bins=bins * 100,
    color="#ff7f0e",
    alpha=0.55,
    label="Unconstrained greedy",
    density=True,
)
ax2.axvline(15, color="#d62728", linestyle="--", linewidth=1.5, label="±15% guardrail")
ax2.axvline(-15, color="#d62728", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Rate change (%)", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Rate Change Distribution", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ---- Panel 3: ENBP headroom by age band ----
enbp_headroom_pct = (enbp - constrained_premium) / enbp * 100  # positive = room below ENBP
enbp_data = []
for band in AGE_BANDS:
    mask = age_band == band
    enbp_data.append({
        "band": band,
        "median_headroom": np.median(enbp_headroom_pct[mask]),
        "p10_headroom": np.percentile(enbp_headroom_pct[mask], 10),
        "p90_headroom": np.percentile(enbp_headroom_pct[mask], 90),
        "pct_binding": (enbp_headroom_pct[mask] < 1.5).mean() * 100,
    })

band_names = [d["band"] for d in enbp_data]
medians = [d["median_headroom"] for d in enbp_data]
pct_binding = [d["pct_binding"] for d in enbp_data]

x_pos = np.arange(len(band_names))
bars = ax3.bar(x_pos, medians, color=["#d62728" if m < 3 else "#1f77b4" for m in medians],
               alpha=0.8, width=0.6)

# Add error bars for p10-p90 range
p10 = [d["p10_headroom"] for d in enbp_data]
p90 = [d["p90_headroom"] for d in enbp_data]
ax3.errorbar(x_pos, medians,
             yerr=[np.array(medians) - np.array(p10), np.array(p90) - np.array(medians)],
             fmt="none", color="black", capsize=4, linewidth=1.2)

# Annotate with % binding
for i, (bar, pct) in enumerate(zip(bars, pct_binding)):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{pct:.0f}% binding", ha="center", va="bottom", fontsize=8.5)

ax3.axhline(1, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
            label="1% buffer threshold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(band_names, fontsize=10)
ax3.set_xlabel("Age band", fontsize=11)
ax3.set_ylabel("Median ENBP headroom (%)", fontsize=11)
ax3.set_title("ENBP Headroom by Age Band\n(% below ENBP ceiling)", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

# ---- Panel 4: Segment profit contribution (constrained vs baseline) ----
seg_data = []
for band in AGE_BANDS:
    mask = age_band == band
    # Constrained
    c_profit = np.dot(
        (constrained_premium[mask] - expected_loss_cost[mask]),
        constrained_demand[mask]
    )
    # Baseline
    b_profit = np.dot(
        (current_premium[mask] - expected_loss_cost[mask]),
        current_demand[mask]
    )
    seg_data.append({
        "band": band,
        "constrained_profit_M": c_profit / 1e6,
        "baseline_profit_M": b_profit / 1e6,
    })

band_names2 = [d["band"] for d in seg_data]
constrained_profits = [d["constrained_profit_M"] for d in seg_data]
baseline_profits = [d["baseline_profit_M"] for d in seg_data]

x_pos2 = np.arange(len(band_names2))
width = 0.35
ax4.bar(x_pos2 - width / 2, baseline_profits, width, label="Baseline", color="#aec7e8", alpha=0.9)
ax4.bar(x_pos2 + width / 2, constrained_profits, width, label="Constrained SLSQP",
        color="#1f77b4", alpha=0.9)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_xticks(x_pos2)
ax4.set_xticklabels(band_names2, fontsize=10)
ax4.set_xlabel("Age band", fontsize=11)
ax4.set_ylabel("Expected profit (£M)", fontsize=11)
ax4.set_title("Segment Profit: Baseline vs Constrained", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(gbp_fmt)
ax4.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "insurance-optimise: Constrained Portfolio Rate Optimisation — UK Motor (50k policies)",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.savefig("/tmp/insurance_optimise_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to /tmp/insurance_optimise_demo.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Segment-Level Deep Dive
# MAGIC
# MAGIC The policy-level output from `result.summary_df` lets you drill into any segment. Here we look at the young driver cohort — the segment that was most under-priced at baseline and where ENBP headroom is largest.

# COMMAND ----------

# Attach segment labels to the result summary DataFrame
summary = result.summary_df.with_columns([
    pl.Series("age_band", age_band),
    pl.Series("region", region),
    pl.Series("ncd_years", ncd.astype(int)),
    pl.Series("is_pcw", is_pcw),
    pl.Series("current_premium", current_premium),
    pl.Series("technical_price", technical_price),
    pl.Series("enbp", enbp),
    pl.Series("expected_loss_cost", expected_loss_cost),
    pl.Series("elasticity", elasticity),
])

# Rate change by age band
print("Rate change distribution by age band (constrained SLSQP):")
print()
age_rate_summary = (
    summary
    .group_by("age_band")
    .agg([
        pl.len().alias("n_policies"),
        pl.col("rate_change_pct").mean().alias("mean_rate_change_pct"),
        pl.col("rate_change_pct").median().alias("median_rate_change_pct"),
        pl.col("rate_change_pct").max().alias("max_rate_change_pct"),
        pl.col("enbp_binding").mean().alias("pct_enbp_binding"),
        pl.col("contribution").sum().alias("total_contribution"),
    ])
    .sort("age_band")
)

print(f"{'Age Band':<10} {'N':>6} {'Mean Δ%':>9} {'Median Δ%':>11} "
      f"{'Max Δ%':>8} {'ENBP bind%':>11} {'Contribution £':>14}")
print("-" * 76)
for row in age_rate_summary.iter_rows(named=True):
    print(
        f"{row['age_band']:<10} "
        f"{row['n_policies']:>6,} "
        f"{row['mean_rate_change_pct']:>8.1f}% "
        f"{row['median_rate_change_pct']:>10.1f}% "
        f"{row['max_rate_change_pct']:>7.1f}% "
        f"{row['pct_enbp_binding']:>10.1%} "
        f"£{row['total_contribution']:>12,.0f}"
    )

print()
print("Rate change by region (mean):")
region_rate = (
    summary
    .group_by("region")
    .agg([
        pl.len().alias("n"),
        pl.col("rate_change_pct").mean().alias("mean_rate_change_pct"),
        pl.col("enbp_binding").mean().alias("pct_enbp_binding"),
    ])
    .sort("region")
)
for row in region_rate.iter_rows(named=True):
    print(f"  {row['region']:<14}  n={row['n']:,}  "
          f"mean Δ={row['mean_rate_change_pct']:.1f}%  "
          f"ENBP binding={row['pct_enbp_binding']:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Scenario Analysis — Elasticity Uncertainty
# MAGIC
# MAGIC Elasticity estimates carry confidence intervals. A central estimate of -1.80 might have 90% CI from -2.30 to -1.30. The correct response is not to pick one point and optimise — it is to show the pricing team the spread across scenarios and let them choose a strategy that is robust to uncertainty.
# MAGIC
# MAGIC Here we run three scenarios: pessimistic (customers more price-sensitive than assumed), central, and optimistic (customers less price-sensitive).

# COMMAND ----------

print("Running elasticity scenario analysis...")
print("Pessimistic: elasticity * 1.25 (customers 25% more price-sensitive)")
print("Central:     elasticity * 1.00 (base case)")
print("Optimistic:  elasticity * 0.75 (customers 25% less price-sensitive)")
print()

scenario_result = opt.optimise_scenarios(
    elasticity_scenarios=[
        elasticity * 1.25,   # pessimistic: underestimated sensitivity
        elasticity,          # central
        elasticity * 0.75,   # optimistic: overestimated sensitivity
    ],
    scenario_names=["pessimistic", "central", "optimistic"],
)

summary_df = scenario_result.summary()
print(summary_df)
print()
print(f"Profit spread across scenarios:")
print(f"  Pessimistic:  £{scenario_result.profit_p10/1e6:.2f}M")
print(f"  Central:      £{scenario_result.profit_mean/1e6:.2f}M")
print(f"  Optimistic:   £{scenario_result.profit_p90/1e6:.2f}M")
print(f"  Spread (P90-P10): £{(scenario_result.profit_p90 - scenario_result.profit_p10)/1e6:.2f}M")
print()
print("The spread shows how sensitive the profit outcome is to elasticity uncertainty.")
print("If the spread is large relative to the mean, the pricing recommendation needs")
print("to be stress-tested before filing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: FCA Compliance — ENBP Audit Trail
# MAGIC
# MAGIC Under FCA PS21/11 and Consumer Duty, firms must be able to demonstrate that:
# MAGIC 1. ENBP was enforced at the individual policy level
# MAGIC 2. The methodology used to set prices is documented
# MAGIC 3. The decision was made on a specific date with specific parameters
# MAGIC
# MAGIC `result.audit_trail` is a JSON-serialisable dict that contains all of this. Save it to Unity Catalog or a documented storage location alongside the ratebook update. When the FCA asks how you set renewal prices, hand them this file.

# COMMAND ----------

audit = result.audit_trail

print("FCA Audit Trail — key fields:")
print(f"  Library:             {audit.get('library', 'N/A')}")
print(f"  Timestamp (UTC):     {audit.get('timestamp_utc', 'N/A')}")
print(f"  Demand model:        {audit.get('demand_model', 'N/A')}")
print(f"  Solver:              {audit['solver']['method']}")
print(f"  SLSQP tolerance:     {audit['solver']['options']['ftol']}")
print()
print("Input summary:")
for k, v in audit["inputs"].items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:.4f}")
    else:
        print(f"  {k:<35}: {v}")
print()
print("Constraint configuration:")
for k, v in audit["constraints"].items():
    if v is not None and v is not False:
        print(f"  {k:<35}: {v}")
print()
print("Portfolio metrics at solution:")
for k, v in audit["portfolio_metrics"].items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:,.4f}")
    else:
        print(f"  {k:<35}: {v}")
print()
print("Convergence:")
for k, v in audit["convergence"].items():
    print(f"  {k:<35}: {v}")
print()
print("Constraint evaluation (positive = satisfied, near-zero = binding):")
for k, v in audit["constraint_evaluation"].items():
    status = "BINDING" if abs(v) < 1e-3 else "SLACK"
    print(f"  {k:<35}: {v:.6f}  [{status}]")
print()
print("Shadow prices (Lagrange multipliers at solution):")
for k, v in audit["shadow_prices"].items():
    print(f"  {k:<35}: {v:.6f}")

# Save audit trail
audit_path = "/tmp/renewal_optimisation_audit.json"
result.save_audit(audit_path)
print(f"\nFull audit trail saved to: {audit_path}")
print("In production: write to Unity Catalog with policy_run_id and effective_date.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: ENBP Compliance Report
# MAGIC
# MAGIC A focused view of ENBP enforcement — the specific regulatory requirement most likely to attract FCA scrutiny. For every renewal policy we show: the proposed premium, the ENBP ceiling, the headroom, and whether the constraint was binding.
# MAGIC
# MAGIC The ENBP binding flag in `result.summary_df` marks policies where the optimal multiplier was at (or within 0.01%) of the ENBP upper bound. These are the policies where the optimiser had to cap the price to comply with PS21/11 — and where a human analyst would have exceeded it without the constraint.

# COMMAND ----------

enbp_report = (
    summary
    .with_columns([
        ((pl.col("enbp") - pl.col("new_premium")) / pl.col("enbp") * 100)
            .alias("enbp_headroom_pct"),
        (pl.col("new_premium") > pl.col("enbp")).alias("would_exceed_enbp"),
    ])
    .select([
        "age_band",
        "region",
        "ncd_years",
        "technical_price",
        "new_premium",
        "enbp",
        "enbp_headroom_pct",
        "enbp_binding",
        "would_exceed_enbp",
        "rate_change_pct",
    ])
)

n_binding = enbp_report.filter(pl.col("enbp_binding")).height
n_violations = enbp_report.filter(pl.col("would_exceed_enbp")).height

print(f"ENBP Compliance Report")
print(f"{'='*60}")
print(f"Total renewal policies:          {N:,}")
print(f"ENBP constraint binding:         {n_binding:,} ({n_binding/N:.1%})")
print(f"Policies above ENBP (violations): {n_violations:,} ({n_violations/N:.1%})")
print()

if n_binding > 0:
    print("Binding ENBP policies by age band:")
    binding_by_age = (
        enbp_report
        .filter(pl.col("enbp_binding"))
        .group_by("age_band")
        .agg([
            pl.len().alias("n_binding"),
            pl.col("enbp_headroom_pct").mean().alias("mean_headroom_pct"),
            pl.col("rate_change_pct").mean().alias("mean_rate_change_pct"),
            pl.col("technical_price").mean().alias("mean_tc"),
        ])
        .sort("age_band")
    )
    for row in binding_by_age.iter_rows(named=True):
        print(f"  {row['age_band']:<8}: {row['n_binding']:,} binding  "
              f"mean headroom={row['mean_headroom_pct']:.2f}%  "
              f"mean rate Δ={row['mean_rate_change_pct']:.1f}%  "
              f"mean TC=£{row['mean_tc']:.0f}")

print()
print("Sample of policies where ENBP was binding (capped at ENBP ceiling):")
binding_sample = (
    enbp_report
    .filter(pl.col("enbp_binding"))
    .head(10)
    .select([
        "age_band", "region", "technical_price", "new_premium",
        "enbp", "enbp_headroom_pct", "rate_change_pct"
    ])
)
print(
    f"{'Age':>8} {'Region':>12} {'Tech Price':>10} "
    f"{'New Premium':>11} {'ENBP':>8} {'Headroom%':>10} {'Rate Δ%':>8}"
)
print("-" * 75)
for row in binding_sample.iter_rows(named=True):
    print(
        f"{row['age_band']:>8} {row['region']:>12} "
        f"£{row['technical_price']:>9.0f} "
        f"£{row['new_premium']:>10.0f} "
        f"£{row['enbp']:>7.0f} "
        f"{row['enbp_headroom_pct']:>9.2f}% "
        f"{row['rate_change_pct']:>7.1f}%"
    )

if n_violations == 0:
    print()
    print("PASS: Zero ENBP violations at the constrained optimum.")
    print("The 1% enbp_buffer provides a 1% safety margin below each policy's ceiling.")
else:
    print()
    print(f"WARNING: {n_violations} ENBP violations detected.")
    print("This should not occur with the 1% buffer — investigate constraint tolerance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we showed:**
# MAGIC
# MAGIC - Unconstrained greedy repricing (price every policy to technical rate) is theoretically optimal but undeliverable: it creates ENBP violations, rate shocks, and ignores the portfolio-level rate change target.
# MAGIC
# MAGIC - Constrained SLSQP (`PortfolioOptimiser`) finds the best achievable solution subject to all of those constraints simultaneously. For a 50,000-policy book it runs in under 5 minutes on a single Databricks worker.
# MAGIC
# MAGIC - The efficient frontier tells you the actual cost of tightening or relaxing the retention floor — not a guess, but the result of solving the constrained optimisation at each point.
# MAGIC
# MAGIC - Scenario analysis quantifies how sensitive the profit outcome is to elasticity uncertainty. If the pessimistic/optimistic spread is wide, the strategy needs to be stress-tested before implementation.
# MAGIC
# MAGIC - The JSON audit trail is a complete record of what the optimiser was asked to do and what it produced. ENBP binding policies are flagged individually. This is the evidence document for FCA Consumer Duty review.
# MAGIC
# MAGIC **When to use this:**
# MAGIC - Annual or quarterly ratebook updates for renewal books of 5,000+ policies
# MAGIC - Any portfolio where the ENBP constraint is expected to be binding on more than ~5% of policies
# MAGIC - Pricing reviews where the pricing team needs to show the regulator that PS21/11 was enforced systematically rather than through manual case-by-case review
# MAGIC
# MAGIC **When not to use this:**
# MAGIC - New business pricing (no ENBP constraint; use a demand curve model with revenue maximisation instead)
# MAGIC - Portfolios smaller than ~1,000 policies (the SLSQP overhead is not worth it; segment-level pricing is fine)
# MAGIC - When elasticity estimates are completely unknown (the optimiser will produce near-flat recommendations anyway)
# MAGIC
# MAGIC **Library:** [insurance-optimise on PyPI](https://pypi.org/project/insurance-optimise/)
# MAGIC
# MAGIC **FCA reference:** [PS21/11 — General insurance pricing practices](https://www.fca.org.uk/publication/policy/ps21-11.pdf)
