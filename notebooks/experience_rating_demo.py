# Databricks notebook source

# MAGIC %md
# MAGIC # Experience Rating: NCD Systems, Optimal Thresholds, and EMFs
# MAGIC
# MAGIC ## Why NCD systems need quantitative analysis
# MAGIC
# MAGIC No-Claims Discount systems are ubiquitous in UK motor insurance, but they are
# MAGIC almost always managed by intuition and convention rather than by calculation.
# MAGIC The ABI standard scale has been in widespread use for decades, and most pricing
# MAGIC teams can tell you what the NCD levels are. Very few can answer the questions
# MAGIC that actually matter for pricing and customer management:
# MAGIC
# MAGIC **Stationary distribution** — if claim frequency stays at 10%, what fraction of
# MAGIC your book eventually settles at each NCD level? This is not a rhetorical question.
# MAGIC It determines your long-run average discount and therefore your required base rate.
# MAGIC A book where 80% of policyholders are at maximum NCD is very different from one
# MAGIC split evenly across levels. The transition matrix tells you the answer analytically.
# MAGIC
# MAGIC **Optimal claim threshold** — should a policyholder with 65% NCD (paying £280/yr
# MAGIC after discount) claim a £450 windscreen repair? The naive answer is "it's below the
# MAGIC excess, so yes". The correct answer depends on what they pay in extra premium over
# MAGIC the next few years to rebuild NCD. For many customers at high NCD levels, the NCD
# MAGIC cost of claiming exceeds the claim value — but only up to a threshold that shifts
# MAGIC with premium level and time horizon. Most insurer call centre scripts get this wrong.
# MAGIC
# MAGIC **Experience modification factors** — for fleet and commercial risks, where the same
# MAGIC entity generates multiple claims over multiple years, you have enough data to
# MAGIC credibility-weight individual experience against the portfolio. The ballast parameter
# MAGIC is the key actuarial choice: too small and a single large loss dominates the mod;
# MAGIC too large and you dilute all signal. This notebook makes that trade-off explicit.
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC - UK motor NCD scale (10 levels, ABI-standard) with full transition rules
# MAGIC - Stationary distribution at varying claim frequencies
# MAGIC - Optimal claiming threshold by NCD level and premium
# MAGIC - Bühlmann-style EMFs with credibility weight sensitivity
# MAGIC - Benchmark: NCD-adjusted rates vs flat portfolio rate vs EMF-adjusted rates

# COMMAND ----------

# MAGIC %pip install experience-rating polars matplotlib

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
warnings.filterwarnings("ignore")

# Databricks display helpers
try:
    display  # noqa: F821 — Databricks built-in
except NameError:
    def display(x):
        print(x)

try:
    displayHTML  # noqa: F821
except NameError:
    def displayHTML(html: str):
        print(html)

from experience_rating import BonusMalusScale, BonusMalusSimulator, ClaimThreshold
from experience_rating import ExperienceModFactor
from experience_rating.experience_mod import CredibilityParams
import experience_rating

print(f"experience-rating version: {experience_rating.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. UK Motor NCD Scale
# MAGIC
# MAGIC We use the ABI-standard scale: 10 levels (0 to 9 years), discounts from 0% to 65%.
# MAGIC Transition rules: one claim-free year moves up one level (capped at max), one claim
# MAGIC drops back two levels (minimum 0), two or more claims resets to level 0.
# MAGIC
# MAGIC The standard scale's discount schedule is deliberately nonlinear — the jump from 8
# MAGIC to 9 years (60% to 65%) is worth less in absolute premium terms than earlier jumps.
# MAGIC This matters for the claiming threshold calculation: the marginal value of protecting
# MAGIC NCD falls as you get closer to the top.

# COMMAND ----------

scale = BonusMalusScale.from_uk_standard()
scale_df = scale.summary()

# Build a rich HTML table of the scale
rows = []
for row in scale_df.iter_rows(named=True):
    bar_width = int(row["ncd_percent"] * 1.8)
    bar = f'<div style="background:#2563eb;height:14px;width:{bar_width}px;border-radius:3px;"></div>'
    rows.append(
        f"<tr>"
        f"<td style='text-align:center;font-weight:bold'>{row['index']}</td>"
        f"<td>{row['name']}</td>"
        f"<td style='text-align:center'>{row['ncd_percent']}%</td>"
        f"<td style='text-align:center'>{row['premium_factor']:.2f}</td>"
        f"<td>{bar}</td>"
        f"</tr>"
    )

html = f"""
<h3 style='font-family:Arial;color:#1e293b'>ABI-Standard UK Motor NCD Scale</h3>
<table style='font-family:Arial;font-size:13px;border-collapse:collapse;width:600px'>
  <thead>
    <tr style='background:#1e293b;color:white'>
      <th style='padding:8px'>Level</th>
      <th style='padding:8px'>Name</th>
      <th style='padding:8px'>NCD</th>
      <th style='padding:8px'>Premium Factor</th>
      <th style='padding:8px;text-align:left'>Discount</th>
    </tr>
  </thead>
  <tbody>
    {''.join(f"<tr style='background:{'#f8fafc' if i%2==0 else 'white'}'>{r[3:]}" for i, r in enumerate(['x'+r for r in rows]))}
  </tbody>
</table>
"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transition Matrix
# MAGIC
# MAGIC The transition matrix encodes the probability of moving from level i to level j in
# MAGIC one year. Under Poisson claim arrival with frequency lambda, the probability of zero
# MAGIC claims is exp(-lambda), one claim is lambda*exp(-lambda), and two or more is the
# MAGIC complement. Each row sums to 1.
# MAGIC
# MAGIC This matrix is the foundation for everything else: stationary distribution,
# MAGIC expected premium factor, and the time-discounted NCD cost used in threshold analysis.

# COMMAND ----------

claim_freq = 0.10
sim = BonusMalusSimulator(scale, claim_frequency=claim_freq)
T = scale.transition_matrix(claim_frequency=claim_freq)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(T.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
n = T.shape[0]
for i in range(n):
    for j in range(n):
        val = T[i, j]
        if val > 0.01:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if val > 0.5 else "#1e293b")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f"L{j}" for j in range(n)], fontsize=9)
ax.set_yticklabels([f"L{i}" for i in range(n)], fontsize=9)
ax.set_xlabel("Next year level", fontsize=11)
ax.set_ylabel("Current level", fontsize=11)
ax.set_title(f"NCD Transition Matrix  (claim frequency = {claim_freq:.0%})", fontsize=12)
plt.colorbar(im, ax=ax, label="Transition probability")
plt.tight_layout()
plt.show()
print(f"Row sums (should all be 1.0): {T.to_numpy().sum(axis=1).round(6)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stationary Distribution
# MAGIC
# MAGIC The stationary distribution tells us where the book settles in the long run.
# MAGIC We compute it analytically (left eigenvector corresponding to eigenvalue 1) across
# MAGIC a range of claim frequencies: 5%, 10%, 15%, and 20%.
# MAGIC
# MAGIC The key insight: at low claim frequency (5%), most of the book concentrates at the
# MAGIC top NCD level. At higher frequencies (15-20%), the distribution spreads down the
# MAGIC scale and the long-run average discount shrinks substantially. This directly affects
# MAGIC the required base rate — a book that writes better risks needs a higher base rate to
# MAGIC achieve the same GWP mix, because the average NCD erosion is larger.

# COMMAND ----------

frequencies = [0.05, 0.10, 0.15, 0.20]
colors = ["#16a34a", "#2563eb", "#d97706", "#dc2626"]
labels = ["5% frequency", "10% frequency", "15% frequency", "20% frequency"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stat_results = {}
for freq, col, label in zip(frequencies, colors, labels):
    sim_f = BonusMalusSimulator(scale, claim_frequency=freq)
    dist = sim_f.stationary_distribution(method="analytical")
    stat_results[freq] = dist
    probs = [dist[i] for i in range(len(dist))]
    axes[0].plot(range(len(dist)), probs, "o-", color=col, label=label, linewidth=2, markersize=5)

axes[0].set_xlabel("NCD Level", fontsize=11)
axes[0].set_ylabel("Steady-state proportion", fontsize=11)
axes[0].set_title("Stationary Distribution by Claim Frequency", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].set_xticks(range(10))
axes[0].set_xticklabels([f"L{i}\n({int(scale_df['ncd_percent'][i])}%)" for i in range(10)], fontsize=8)

# Expected premium factor at each frequency
epfs = []
for freq in frequencies:
    sim_f = BonusMalusSimulator(scale, claim_frequency=freq)
    epf = sim_f.expected_premium_factor()
    epfs.append(epf)
    avg_ncd = (1 - epf) * 100

bar_colors = colors
bars = axes[1].bar(labels, [(1 - e) * 100 for e in epfs], color=bar_colors, alpha=0.85, width=0.5)
axes[1].set_ylabel("Average NCD at steady state (%)", fontsize=11)
axes[1].set_title("Long-run Average NCD by Claim Frequency", fontsize=12)
axes[1].set_ylim(0, 70)
for bar, epf in zip(bars, epfs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{(1-epf)*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.show()

# Summary
print("Stationary distribution summary:")
print(f"{'Frequency':>12} | {'Avg NCD':>10} | {'Avg Prem Factor':>16} | {'% at max NCD (L9)':>18}")
print("-" * 65)
for freq, epf in zip(frequencies, epfs):
    dist = stat_results[freq]
    pct_max = dist[9] * 100
    print(f"{freq:>11.0%}  | {(1-epf)*100:>9.1f}% | {epf:>16.4f} | {pct_max:>17.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optimal Claim Threshold
# MAGIC
# MAGIC The question every NCD policyholder faces: should I claim, or absorb the loss?
# MAGIC
# MAGIC The correct answer requires comparing two quantities over a multi-year horizon:
# MAGIC - **Cost of claiming:** the discounted extra premium paid at lower NCD levels until
# MAGIC   the policyholder rebuilds to their pre-claim level
# MAGIC - **Benefit of claiming:** the claim payout received
# MAGIC
# MAGIC If the claim payout exceeds the NCD cost, claim. If not, absorb it. The crossover
# MAGIC point is the threshold.
# MAGIC
# MAGIC The threshold is non-trivial because it depends on current NCD level, annual premium,
# MAGIC and the planning horizon. We compute `full_analysis()` to see thresholds at every
# MAGIC NCD level for a representative policyholder, then show how it varies with premium.

# COMMAND ----------

ct = ClaimThreshold(scale, discount_rate=0.05)

# Full threshold analysis at a representative premium (£300/yr after NCD)
# Note: this is the base premium before the NCD level's own discount is applied
# We use annual_premium as the currently-paid premium
annual_premium = 300.0
years_horizon = 3

threshold_df = ct.full_analysis(annual_premium=annual_premium, years_horizon=years_horizon)

# Plot thresholds by NCD level
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

levels = threshold_df["level"].to_list()
thresholds = threshold_df["threshold"].to_list()
ncd_pcts = [int(scale_df["ncd_percent"][lv]) for lv in levels]

bar_colors_t = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(levels)))
bars = axes[0].bar(
    [f"L{lv}\n{ncd}%" for lv, ncd in zip(levels, ncd_pcts)],
    thresholds,
    color=bar_colors_t,
    alpha=0.9,
    width=0.6,
)
axes[0].axhline(y=annual_premium * 0.5, color="#94a3b8", linestyle="--", linewidth=1.5,
                label=f"50% of annual premium (£{annual_premium*0.5:.0f})")
axes[0].set_ylabel("Claim threshold (£)", fontsize=11)
axes[0].set_xlabel("NCD Level", fontsize=11)
axes[0].set_title(f"Optimal Claim Threshold by NCD Level\n(£{annual_premium:.0f}/yr premium, {years_horizon}-year horizon)", fontsize=11)
axes[0].legend(fontsize=9)
for bar, t in zip(bars, thresholds):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"£{t:.0f}", ha="center", va="bottom", fontsize=8)
axes[0].tick_params(axis="x", labelsize=8)

# Threshold vs horizon for a policyholder at max NCD (level 9)
max_level = 9
horizon_df = ct.threshold_curve(
    current_level=max_level,
    annual_premium=annual_premium,
    max_horizon=7,
)
horizons = horizon_df["years_horizon"].to_list()
thresh_curve = horizon_df["threshold"].to_list()

axes[1].plot(horizons, thresh_curve, "o-", color="#2563eb", linewidth=2.5, markersize=7)
axes[1].fill_between(horizons, thresh_curve, alpha=0.12, color="#2563eb")
for h, t in zip(horizons, thresh_curve):
    axes[1].text(h, t + 8, f"£{t:.0f}", ha="center", fontsize=8.5, color="#1e40af")
axes[1].set_xlabel("Planning horizon (years)", fontsize=11)
axes[1].set_ylabel("Claim threshold (£)", fontsize=11)
axes[1].set_title(f"Threshold vs Horizon at Level {max_level} (65% NCD)\n£{annual_premium:.0f}/yr premium", fontsize=11)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Threshold Sensitivity to Premium Level
# MAGIC
# MAGIC The NCD cost scales with the annual premium — a £600 policy has double the NCD cost
# MAGIC of a £300 policy for the same level drop. So the threshold scales roughly linearly
# MAGIC with premium, but not exactly, because the transition dynamics are the same.
# MAGIC
# MAGIC This matters for customer communications: the standard advice ("don't claim if it's
# MAGIC under £500") is only valid for a specific premium range. A young driver paying £900/yr
# MAGIC faces a much higher threshold than a middle-aged driver paying £350/yr.

# COMMAND ----------

premiums = [200, 300, 450, 600, 900]
premium_colors = ["#6366f1", "#2563eb", "#0891b2", "#059669", "#d97706"]

fig, ax = plt.subplots(figsize=(10, 5))

for prem, col in zip(premiums, premium_colors):
    ta = ct.full_analysis(annual_premium=float(prem), years_horizon=3)
    ax.plot(
        [f"L{lv}\n{int(scale_df['ncd_percent'][lv])}%" for lv in ta["level"].to_list()],
        ta["threshold"].to_list(),
        "o-", color=col, linewidth=2, markersize=5, label=f"£{prem}/yr"
    )

ax.set_xlabel("NCD Level", fontsize=11)
ax.set_ylabel("Claim threshold (£)", fontsize=11)
ax.set_title("Claim Threshold by NCD Level and Annual Premium\n(3-year horizon, 5% discount rate)", fontsize=11)
ax.legend(title="Annual premium", fontsize=10, title_fontsize=10)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.show()

# Key numbers for a representative case (level 9, 3yr horizon)
print("Threshold at Level 9 (65% NCD), 3-year horizon:")
for prem in premiums:
    t = ct.threshold(current_level=9, annual_premium=float(prem), years_horizon=3)
    print(f"  £{prem:>4}/yr  ->  claim threshold £{t:>6.0f}  ({t/prem*100:.0f}% of annual premium)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Experience Modification Factors (Fleet / Commercial)
# MAGIC
# MAGIC For fleet and commercial risks, the NCD mechanism is supplemented (or replaced) by
# MAGIC experience modification factors: a multiplicative adjustment applied to the risk's
# MAGIC technical premium based on its own loss history relative to expectation.
# MAGIC
# MAGIC The formula is Bühlmann-style:
# MAGIC
# MAGIC ```
# MAGIC EMF = Z * (Actual / Expected) + (1 - Z)
# MAGIC ```
# MAGIC
# MAGIC where Z is the credibility weight. The credibility weight is derived from:
# MAGIC
# MAGIC ```
# MAGIC Z = Expected / (Expected + Ballast)
# MAGIC ```
# MAGIC
# MAGIC **Ballast** is the key parameter. It represents the amount of expected losses that
# MAGIC would make a risk "fully credible" (Z = 0.5). Set it too low and any single bad year
# MAGIC produces a large mod. Set it too high and you fail to price the risk on its own
# MAGIC experience even when you have years of data.
# MAGIC
# MAGIC We construct a synthetic fleet portfolio to demonstrate.

# COMMAND ----------

np.random.seed(2024)
n_risks = 40

# Simulate a commercial motor fleet portfolio
# True individual frequencies drawn from Gamma(shape=2.5, mean=0.10)
true_freqs = np.random.gamma(shape=2.5, scale=0.10 / 2.5, size=n_risks)
fleet_sizes = np.random.randint(8, 80, size=n_risks)
annual_premiums_fleet = fleet_sizes * np.random.uniform(800, 1500, size=n_risks)
years_history = np.random.randint(2, 6, size=n_risks)

# Expected losses: fleet size * vehicle premium * expected frequency
# Actual losses: drawn from Poisson claim counts * Gamma claim amounts
expected_losses = annual_premiums_fleet * 0.65  # 65% loss ratio as technical expectation
actual_losses = np.array([
    np.random.poisson(true_freqs[i] * fleet_sizes[i] * years_history[i])
    * np.random.gamma(shape=3, scale=3500 / 3)
    for i in range(n_risks)
])
# Scale to multi-year: expected over history period
expected_losses_hist = expected_losses * years_history

fleet_df = pl.DataFrame({
    "risk_id": [f"FLEET-{i+1:03d}" for i in range(n_risks)],
    "fleet_size": fleet_sizes.tolist(),
    "years_history": years_history.tolist(),
    "expected_losses": expected_losses_hist.tolist(),
    "actual_losses": actual_losses.tolist(),
    "true_freq": true_freqs.tolist(),
})

# Apply EMF with market-standard params
params = CredibilityParams(credibility_weight=0.65, ballast=12_000.0)
emod = ExperienceModFactor(params)
result_df = emod.predict_batch(fleet_df.select(["risk_id", "expected_losses", "actual_losses"]),
                               cap=2.0, floor=0.5)
fleet_result = fleet_df.join(result_df, on="risk_id")

# Plot: EMF distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

emfs = fleet_result["emf"].to_list()
axes[0].hist(emfs, bins=18, color="#2563eb", alpha=0.8, edgecolor="white")
axes[0].axvline(x=1.0, color="#dc2626", linewidth=2, linestyle="--", label="EMF = 1.0 (neutral)")
axes[0].axvline(x=float(np.mean(emfs)), color="#16a34a", linewidth=2, linestyle="-",
                label=f"Mean EMF = {np.mean(emfs):.3f}")
axes[0].set_xlabel("Experience Modification Factor", fontsize=11)
axes[0].set_ylabel("Number of risks", fontsize=11)
axes[0].set_title("EMF Distribution — 40-Risk Fleet Portfolio", fontsize=12)
axes[0].legend(fontsize=10)

# Scatter: actual/expected loss ratio vs EMF (coloured by fleet size)
ae_ratios = fleet_result["actual_losses"] / fleet_result["expected_losses"]
scatter = axes[1].scatter(
    ae_ratios.to_list(), emfs,
    c=fleet_result["fleet_size"].to_list(),
    cmap="viridis", alpha=0.8, s=60, edgecolors="white", linewidths=0.5
)
axes[1].axhline(y=1.0, color="#94a3b8", linestyle="--", linewidth=1)
axes[1].axvline(x=1.0, color="#94a3b8", linestyle="--", linewidth=1)
axes[1].set_xlabel("Actual / Expected Loss Ratio", fontsize=11)
axes[1].set_ylabel("EMF", fontsize=11)
axes[1].set_title("EMF vs A/E Ratio (colour = fleet size)", fontsize=12)
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label("Fleet size (vehicles)", fontsize=9)

plt.tight_layout()
plt.show()

print(f"\nPortfolio summary:")
print(f"  Risks: {n_risks}")
print(f"  Mean EMF: {np.mean(emfs):.4f}")
print(f"  EMF > 1.1 (surcharge): {sum(e > 1.1 for e in emfs)} risks")
print(f"  EMF < 0.9 (credit):    {sum(e < 0.9 for e in emfs)} risks")
print(f"  Capped at 2.0:         {sum(e >= 1.99 for e in emfs)} risks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. EMF Sensitivity to Ballast
# MAGIC
# MAGIC The ballast parameter is the most consequential choice in EMF design, and it is
# MAGIC almost never documented or justified. We show how the ballast determines the
# MAGIC credibility-weight curve and how it changes the spread of EMFs across the portfolio.
# MAGIC
# MAGIC Low ballast: individual experience dominates quickly. A risk with £5k expected losses
# MAGIC gets significant credibility. Good if you trust your experience; dangerous if claims
# MAGIC are volatile (one lorry accident blows up the mod).
# MAGIC
# MAGIC High ballast: only large fleets with significant expected losses get meaningful
# MAGIC individual weighting. Safe from noise but slow to price in genuine deterioration.
# MAGIC
# MAGIC The sensitivity plot shows the Z-weight curve and the resulting EMF spread for
# MAGIC the portfolio across a range of ballast values.

# COMMAND ----------

ballast_values = [3_000, 6_000, 12_000, 25_000, 50_000]
ballast_colors = ["#dc2626", "#d97706", "#2563eb", "#059669", "#6366f1"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: credibility weight Z as function of expected losses
exp_range = np.linspace(1_000, 100_000, 300)
for ballast, col in zip(ballast_values, ballast_colors):
    z_curve = exp_range / (exp_range + ballast)
    axes[0].plot(exp_range / 1000, z_curve, color=col, linewidth=2,
                 label=f"Ballast £{ballast:,}")

axes[0].axhline(y=0.5, color="#94a3b8", linestyle="--", linewidth=1, label="Z = 0.5 (half-credible)")
axes[0].set_xlabel("Expected losses (£k)", fontsize=11)
axes[0].set_ylabel("Credibility weight Z", fontsize=11)
axes[0].set_title("Credibility Weight vs Expected Losses by Ballast", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.05)

# Right: EMF interquartile range across portfolio for each ballast
emf_distributions = {}
for ballast in ballast_values:
    p = CredibilityParams(credibility_weight=0.65, ballast=float(ballast))
    em = ExperienceModFactor(p)
    r = em.predict_batch(
        fleet_df.select(["risk_id", "expected_losses", "actual_losses"]),
        cap=2.0, floor=0.5
    )
    emf_distributions[ballast] = r["emf"].to_list()

means = [np.mean(emf_distributions[b]) for b in ballast_values]
stds = [np.std(emf_distributions[b]) for b in ballast_values]
p10s = [np.percentile(emf_distributions[b], 10) for b in ballast_values]
p90s = [np.percentile(emf_distributions[b], 90) for b in ballast_values]

x_pos = range(len(ballast_values))
axes[1].bar(x_pos, [p90 - p10 for p90, p10 in zip(p90s, p10s)],
            bottom=p10s, color=ballast_colors, alpha=0.7, width=0.5,
            label="P10–P90 range")
axes[1].plot(x_pos, means, "D", color="#1e293b", markersize=8, zorder=5, label="Mean EMF")
axes[1].axhline(y=1.0, color="#94a3b8", linestyle="--", linewidth=1)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f"£{b//1000}k" for b in ballast_values], fontsize=10)
axes[1].set_xlabel("Ballast", fontsize=11)
axes[1].set_ylabel("EMF", fontsize=11)
axes[1].set_title("EMF Spread (P10–P90) by Ballast Setting", fontsize=12)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()

print("EMF distribution by ballast:")
print(f"{'Ballast':>10} | {'Mean':>7} | {'Std':>7} | {'P10':>7} | {'P90':>7} | {'P90-P10':>8}")
print("-" * 60)
for ballast in ballast_values:
    d = emf_distributions[ballast]
    print(f"£{ballast:>8,} | {np.mean(d):>7.4f} | {np.std(d):>7.4f} | "
          f"{np.percentile(d,10):>7.4f} | {np.percentile(d,90):>7.4f} | "
          f"{np.percentile(d,90)-np.percentile(d,10):>8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Benchmark: Flat Rate vs NCD-Adjusted vs EMF-Adjusted
# MAGIC
# MAGIC We now compare three pricing approaches on a synthetic motor portfolio with known
# MAGIC true individual frequencies (drawn from a Gamma distribution, producing realistic
# MAGIC heterogeneity). The portfolio has 2,000 policyholders over 4 history years.
# MAGIC
# MAGIC - **Flat rate:** everyone gets the portfolio mean frequency. No individual adjustment.
# MAGIC - **NCD-adjusted:** premium adjusted by the NCD level at the end of the history period.
# MAGIC   NCD level is a lossy encoding of history — it discards claim size and counts beyond
# MAGIC   the transition rules.
# MAGIC - **EMF-adjusted:** credibility-weighted individual experience over the history period.
# MAGIC   Retains full loss information.
# MAGIC
# MAGIC We measure predictive performance against true frequency (Gamma DGP, shape=2,
# MAGIC mean=10%) using MSE and a Gini-style discrimination metric.

# COMMAND ----------

np.random.seed(42)
N_POLS = 2_000
N_HIST = 4
MEAN_FREQ = 0.10
BASE_PREMIUM = 600.0

# True individual frequencies from Gamma(shape=2, mean=0.10)
true_freq = np.random.gamma(shape=2.0, scale=MEAN_FREQ / 2.0, size=N_POLS)

# Simulate NCD history
ncd_levels = np.zeros(N_POLS, dtype=int)  # everyone starts at level 0

for year in range(N_HIST):
    year_claims = np.array([np.random.poisson(true_freq[i]) for i in range(N_POLS)])
    new_levels = np.zeros(N_POLS, dtype=int)
    T_mat = scale.transition_matrix(claim_frequency=MEAN_FREQ).to_numpy()
    for i in range(N_POLS):
        cl = year_claims[i]
        lv = ncd_levels[i]
        if cl == 0:
            # move up one level
            claim_free_col = None
            # Use the transition rules: find next level from transition matrix row
            # The highest probability non-current level in the matrix is the claim-free level
            # For zero claims: level = min(lv+1, max_level)
            new_levels[i] = min(lv + 1, 9)
        elif cl == 1:
            new_levels[i] = max(lv - 2, 0)
        else:
            new_levels[i] = 0
    ncd_levels = new_levels

# Simulate aggregate losses over history
total_claims_hist = np.array([
    sum(np.random.poisson(true_freq[i]) for _ in range(N_HIST))
    for i in range(N_POLS)
])
avg_claim_size = 3_500.0
actual_losses_total = total_claims_hist * np.random.gamma(shape=3, scale=avg_claim_size / 3, size=N_POLS)
expected_losses_total = BASE_PREMIUM * 0.65 * N_HIST * np.ones(N_POLS)

# Holdout: simulate one future year
holdout_freq = np.array([np.random.poisson(true_freq[i]) for i in range(N_POLS)]) / 1.0

# --- Method 1: Flat rate
flat_rate = np.full(N_POLS, MEAN_FREQ)

# --- Method 2: NCD-adjusted
ncd_discount = np.array([float(scale_df["ncd_percent"][lv]) / 100 for lv in ncd_levels])
ncd_rate = MEAN_FREQ * (1 - ncd_discount * 0.5)  # NCD discount partly reflects risk level

# --- Method 3: EMF-adjusted
emf_df_bench = pl.DataFrame({
    "risk_id": [f"P{i}" for i in range(N_POLS)],
    "expected_losses": expected_losses_total.tolist(),
    "actual_losses": actual_losses_total.tolist(),
})
params_bench = CredibilityParams(credibility_weight=0.65, ballast=10_000.0)
emod_bench = ExperienceModFactor(params_bench)
emf_result_bench = emod_bench.predict_batch(emf_df_bench, cap=2.0, floor=0.5)
emf_vals = np.array(emf_result_bench["emf"].to_list())
emf_rate = MEAN_FREQ * emf_vals

# --- Evaluation: MSE vs true frequency
mse_flat = np.mean((flat_rate - true_freq) ** 2)
mse_ncd  = np.mean((ncd_rate  - true_freq) ** 2)
mse_emf  = np.mean((emf_rate  - true_freq) ** 2)

# Gini-style: rank policyholders by predicted rate, measure discrimination vs holdout claims
def lorenz_gini(predicted, actual):
    order = np.argsort(predicted)
    actual_sorted = actual[order]
    cum_actual = np.cumsum(actual_sorted) / actual_sorted.sum()
    n = len(actual_sorted)
    cum_pop = np.arange(1, n + 1) / n
    gini = 1 - 2 * np.trapz(cum_actual, cum_pop)
    return gini

gini_flat = lorenz_gini(flat_rate, holdout_freq)
gini_ncd  = lorenz_gini(ncd_rate,  holdout_freq)
gini_emf  = lorenz_gini(emf_rate,  holdout_freq)

print("Benchmark results:")
print(f"{'Method':>15} | {'MSE vs true freq':>18} | {'Gini vs holdout':>16} | {'MSE improve':>12} | {'Gini improve':>13}")
print("-" * 82)
for name, mse, gini in [("Flat rate", mse_flat, gini_flat),
                         ("NCD-adjusted", mse_ncd, gini_ncd),
                         ("EMF-adjusted", mse_emf, gini_emf)]:
    mse_imp = f"+{(mse_flat - mse)/mse_flat*100:.1f}%" if mse < mse_flat else "baseline"
    gini_imp = f"+{(gini - gini_flat)*100:.1f}pp" if gini > gini_flat else "baseline"
    print(f"{name:>15} | {mse:>18.6f} | {gini:>16.4f} | {mse_imp:>12} | {gini_imp:>13}")

# COMMAND ----------

# Plot benchmark comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

methods = ["Flat rate", "NCD-adjusted", "EMF-adjusted"]
mse_vals = [mse_flat, mse_ncd, mse_emf]
gini_vals = [gini_flat, gini_ncd, gini_emf]
bar_cols = ["#94a3b8", "#2563eb", "#16a34a"]

b1 = axes[0].bar(methods, [m * 1e4 for m in mse_vals], color=bar_cols, alpha=0.85, width=0.5)
axes[0].set_ylabel("MSE vs true frequency (×10⁻⁴)", fontsize=11)
axes[0].set_title("Predictive Accuracy: MSE vs True Frequency\n(lower is better)", fontsize=11)
for bar, val in zip(b1, mse_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val*1e4:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

b2 = axes[1].bar(methods, gini_vals, color=bar_cols, alpha=0.85, width=0.5)
axes[1].set_ylabel("Gini coefficient (vs holdout claims)", fontsize=11)
axes[1].set_title("Discrimination: Gini vs Holdout Year Claims\n(higher is better)", fontsize=11)
for bar, val in zip(b2, gini_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary Table
# MAGIC
# MAGIC Key metrics from the full analysis.

# COMMAND ----------

# Stationary distribution summary at 10% frequency
sim_10 = BonusMalusSimulator(scale, claim_frequency=0.10)
dist_10 = sim_10.stationary_distribution(method="analytical")
epf_10 = sim_10.expected_premium_factor()
pct_at_max_10 = dist_10[9] * 100

# Threshold at key NCD levels, representative premium £300/yr, 3yr horizon
ct_ref = ClaimThreshold(scale, discount_rate=0.05)
thresh_l0  = ct_ref.threshold(current_level=0, annual_premium=300.0, years_horizon=3)
thresh_l5  = ct_ref.threshold(current_level=5, annual_premium=300.0, years_horizon=3)
thresh_l9  = ct_ref.threshold(current_level=9, annual_premium=300.0, years_horizon=3)

rows_summary = [
    ("Scale", "NCD levels", "10", "ABI-standard (0%-65%)"),
    ("Scale", "Discount range", "0% – 65%", "10 levels"),
    ("Stationary (10% freq)", "Avg NCD at steady state", f"{(1-epf_10)*100:.1f}%", "Analytical eigenvector"),
    ("Stationary (10% freq)", "% of book at max NCD (L9)", f"{pct_at_max_10:.1f}%", "Long-run"),
    ("Stationary (10% freq)", "Expected premium factor", f"{epf_10:.4f}", "Premium-weighted avg"),
    ("Claiming threshold", "Level 0 (0% NCD), £300/yr, 3yr", f"£{thresh_l0:.0f}", "Claim if loss > threshold"),
    ("Claiming threshold", "Level 5 (45% NCD), £300/yr, 3yr", f"£{thresh_l5:.0f}", ""),
    ("Claiming threshold", "Level 9 (65% NCD), £300/yr, 3yr", f"£{thresh_l9:.0f}", ""),
    ("EMF benchmark", "Ballast used", "£12,000", "Market-standard for mid-fleets"),
    ("EMF benchmark", "Credibility weight param", "0.65", "Bühlmann-style"),
    ("EMF benchmark", "Risks with surcharge (>1.1)", f"{sum(e > 1.1 for e in emfs)}/{n_risks}", ""),
    ("Benchmark vs flat rate", "NCD MSE improvement", f"{(mse_flat-mse_ncd)/mse_flat*100:.1f}%", "Lower MSE vs true freq"),
    ("Benchmark vs flat rate", "EMF MSE improvement", f"{(mse_flat-mse_emf)/mse_flat*100:.1f}%", "Full history retained"),
    ("Benchmark vs flat rate", "EMF Gini improvement", f"+{(gini_emf-gini_flat)*100:.2f}pp", "vs holdout year"),
]

table_rows = ""
for i, (cat, metric, value, note) in enumerate(rows_summary):
    bg = "#f8fafc" if i % 2 == 0 else "white"
    table_rows += (
        f"<tr style='background:{bg}'>"
        f"<td style='padding:7px 12px;color:#64748b;font-size:12px'>{cat}</td>"
        f"<td style='padding:7px 12px;font-weight:500'>{metric}</td>"
        f"<td style='padding:7px 12px;text-align:center;font-weight:bold;color:#1d4ed8'>{value}</td>"
        f"<td style='padding:7px 12px;color:#64748b;font-size:11px'>{note}</td>"
        f"</tr>"
    )

summary_html = f"""
<h3 style='font-family:Arial;color:#1e293b'>Experience Rating — Key Results Summary</h3>
<table style='font-family:Arial;font-size:13px;border-collapse:collapse;width:800px'>
  <thead>
    <tr style='background:#1e293b;color:white'>
      <th style='padding:8px 12px;text-align:left'>Category</th>
      <th style='padding:8px 12px;text-align:left'>Metric</th>
      <th style='padding:8px 12px;text-align:center'>Value</th>
      <th style='padding:8px 12px;text-align:left'>Notes</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>
<p style='font-family:Arial;font-size:11px;color:#94a3b8;margin-top:8px'>
  Synthetic data. DGP: Gamma(shape=2, mean=10%) individual frequencies. 2,000 policyholders, 4-year history.
</p>
"""
displayHTML(summary_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key findings
# MAGIC
# MAGIC **Stationary distribution** — at 10% claim frequency, the ABI-standard NCD scale
# MAGIC concentrates roughly 40-50% of the book at the maximum level (L9, 65% NCD) at
# MAGIC steady state. The average NCD is around 50-55%, meaning the required base rate
# MAGIC must be set ~2x the final premium to recover the correct technical rate. This is
# MAGIC a structural feature of the scale, not a portfolio selection effect.
# MAGIC
# MAGIC **Claiming thresholds** — at 65% NCD on a £300/yr premium over a 3-year horizon,
# MAGIC the rational threshold is typically in the £150-250 range (roughly 50-80% of annual
# MAGIC premium). The threshold at lower NCD levels is much smaller because losing one or
# MAGIC two levels costs less in absolute terms. Standard call-centre scripts that apply a
# MAGIC fixed £500 threshold are often wrong in both directions.
# MAGIC
# MAGIC **EMF ballast** — moving from a £3k ballast to a £50k ballast roughly halves the
# MAGIC P10-P90 spread of EMFs across the portfolio. For mid-sized fleets (£10k-£30k
# MAGIC expected losses), the £12k ballast is a reasonable default: it provides meaningful
# MAGIC credibility to fleets with 2+ years of history while limiting the impact of
# MAGIC single-year volatility.
# MAGIC
# MAGIC **Benchmark** — both NCD adjustment and EMF adjustment improve on the flat rate,
# MAGIC with EMF outperforming NCD because it retains full loss history rather than
# MAGIC encoding it through the lossy level-transition mechanism. The ordering is
# MAGIC consistent: EMF > NCD > flat, as expected from the data-generating process.
