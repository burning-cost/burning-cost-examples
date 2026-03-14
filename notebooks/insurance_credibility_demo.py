# Databricks notebook source

# MAGIC %md
# MAGIC # Bühlmann-Straub Credibility: Blending Segment Experience with Portfolio
# MAGIC
# MAGIC ## The thin-data problem in segment pricing
# MAGIC
# MAGIC Every motor book has the same structural problem: you have 30 rating segments
# MAGIC (territories, vehicle groups, NCD classes) but most of the exposure is concentrated
# MAGIC in a handful of them. Three of your territories might each have 50 policies.
# MAGIC You still need a rate for those three.
# MAGIC
# MAGIC The failure modes are symmetric:
# MAGIC
# MAGIC **Raw experience** — just use what you observed. If a thin territory had 6 claims
# MAGIC on 45 policies, your burning cost is 13.3%. But the true rate might be 7%. You are
# MAGIC pricing noise. The observed rate for thin segments is essentially random — the signal
# MAGIC is buried under sampling variance.
# MAGIC
# MAGIC **Portfolio average** — ignore the segment entirely. Use the portfolio-wide rate for
# MAGIC everyone. No variance problem, but you throw away real signal. A territory that
# MAGIC genuinely runs at 14% loss cost (motorway-adjacent, commercial traffic, known flood
# MAGIC risk) should not get priced at the 8% portfolio average. You will be adversely
# MAGIC selected against instantly.
# MAGIC
# MAGIC **Bühlmann-Straub credibility** — the mathematically optimal blend. The formula
# MAGIC `P_i = Z_i * X̄_i + (1 - Z_i) * mu` gives each segment a weight `Z_i` determined
# MAGIC by the ratio of within-segment noise (EPV) to genuine between-segment signal (VHM).
# MAGIC Thin segments get low Z and regress heavily toward the portfolio mean. Thick segments
# MAGIC get high Z and are trusted on their own experience. The parameters are estimated from
# MAGIC the data — you do not need to choose Z manually.
# MAGIC
# MAGIC This is the classical credibility theory result, and it is not a soft recommendation.
# MAGIC Bühlmann-Straub is the minimum-variance unbiased linear estimator under the model
# MAGIC assumptions. If the assumptions hold, you cannot do better with a linear estimator.
# MAGIC
# MAGIC **What this notebook demonstrates:**
# MAGIC - Synthetic motor dataset: 50k policies across 30 segments, deliberate thin tails
# MAGIC - Data-generating process with known true segment-level loss costs
# MAGIC - Three estimators: raw experience, portfolio average, Bühlmann-Straub
# MAGIC - Benchmark table: MAE and RMSE broken out by segment size tier
# MAGIC - Visualisation: credibility weight vs exposure, shrinkage plot
# MAGIC
# MAGIC The `insurance-credibility` library handles the structural parameter estimation
# MAGIC (EPV, VHM) and credibility factor computation. The math is in the library; this
# MAGIC notebook is about showing what it means in practice.

# COMMAND ----------

# MAGIC %pip install insurance-credibility polars numpy matplotlib

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

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

from insurance_credibility import BuhlmannStraub
import insurance_credibility

print(f"insurance-credibility {insurance_credibility.__version__}")

rng = np.random.default_rng(seed=2025)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Motor Dataset
# MAGIC
# MAGIC The data-generating process is designed to look like a real UK motor book:
# MAGIC
# MAGIC - **Base loss cost**: £420 per vehicle-year (mid-market personal motor)
# MAGIC - **30 rating segments**: territory × vehicle group combinations, each with a
# MAGIC   true underlying loss cost drawn from LogNormal(log(420), 0.3). This gives a
# MAGIC   realistic spread — roughly ±35% around the portfolio mean.
# MAGIC - **Deliberate thin tails**: 10 of the 30 segments are rural or specialist risks
# MAGIC   with fewer than 200 policies. These are the segments that break naïve estimators.
# MAGIC - **Volume distribution**: log-normal policy counts so a few urban segments dominate
# MAGIC   and thin segments get 30-180 policies each.
# MAGIC - **5 accident years**: each policy contributes one row per year of exposure,
# MAGIC   giving the model panel data to estimate EPV (within-segment year-to-year noise)
# MAGIC   and VHM (genuine between-segment spread).
# MAGIC
# MAGIC Loss costs per policy-year are drawn from Gamma(alpha, scale) with the true
# MAGIC segment mean equal to the latent loss cost. Individual claim severity and
# MAGIC frequency are combined — the loss_rate column is incurred cost per vehicle-year.

# COMMAND ----------

# ── Data-generating parameters ─────────────────────────────────────────────────
N_POLICIES    = 50_000
N_SEGMENTS    = 30
N_YEARS       = 5
BASE_LOSS     = 420.0        # £ per vehicle-year
SEGMENT_SIGMA = 0.30         # log-scale spread — realistic for UK motor territories
GAMMA_SHAPE   = 2.5          # within-segment claim volatility (lower = noisier)

# True segment-level loss costs — fixed seed for reproducibility
rng_dgp = np.random.default_rng(seed=42)
segment_ids = [f"SEG{i:02d}" for i in range(1, N_SEGMENTS + 1)]

true_log_offsets = rng_dgp.normal(0.0, SEGMENT_SIGMA, size=N_SEGMENTS)
true_loss_costs  = BASE_LOSS * np.exp(true_log_offsets)

# ── Allocate policies to segments ──────────────────────────────────────────────
# Log-normal weights produce the realistic skew: a few thick urban segments,
# many thin rural ones.
raw_weights = rng_dgp.lognormal(mean=4.8, sigma=1.1, size=N_SEGMENTS)

# Force 10 segments to be thin (rural / specialist)
thin_idx = rng_dgp.choice(N_SEGMENTS, size=10, replace=False)
raw_weights[thin_idx] = rng_dgp.uniform(6, 40, size=10)

policy_counts = np.round(raw_weights / raw_weights.sum() * N_POLICIES).astype(int)
policy_counts[-1] += N_POLICIES - policy_counts.sum()
policy_counts = np.maximum(policy_counts, 5)

# ── Segment size tiers ─────────────────────────────────────────────────────────
def size_tier(n: int) -> str:
    if n < 200:
        return "thin"
    elif n < 1_500:
        return "medium"
    else:
        return "thick"

tier_labels = [size_tier(n) for n in policy_counts]

true_df = pl.DataFrame({
    "segment":    segment_ids,
    "true_loss":  true_loss_costs,
    "n_policies": policy_counts.tolist(),
    "tier":       tier_labels,
})

print(f"Total policies: {policy_counts.sum():,}")
print(f"Thin (<200):    {sum(1 for t in tier_labels if t == 'thin')} segments")
print(f"Medium (200-1499): {sum(1 for t in tier_labels if t == 'medium')} segments")
print(f"Thick (1500+):  {sum(1 for t in tier_labels if t == 'thick')} segments")
print(f"\nTrue loss cost range: £{true_loss_costs.min():.0f} – £{true_loss_costs.max():.0f}")
print(f"Portfolio baseline:   £{BASE_LOSS:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate panel data: one row per (segment, accident year)
# MAGIC
# MAGIC Bühlmann-Straub requires panel data — multiple periods per group. The within-segment
# MAGIC year-to-year variance estimates EPV; the between-segment variance of period means
# MAGIC estimates VHM. With only one period per segment you cannot separate EPV from VHM
# MAGIC and the model is not identified.
# MAGIC
# MAGIC Each row holds the aggregate loss rate for that segment in that year:
# MAGIC `loss_rate = total_incurred / earned_vehicle_years`. This is the natural format
# MAGIC from a claims system join — you do not need policy-level rows.

# COMMAND ----------

panel_rows = []
rng_sim = np.random.default_rng(seed=123)

for i, seg in enumerate(segment_ids):
    n  = policy_counts[i]
    mu = true_loss_costs[i]
    scale = mu / GAMMA_SHAPE   # Gamma parameterisation: mean = shape * scale

    for yr in range(2020, 2020 + N_YEARS):
        # Each policy contributes ~1 vehicle-year of exposure (slight variation)
        exposures = rng_sim.uniform(0.85, 1.0, size=n)
        total_exp = exposures.sum()

        # Aggregate incurred from Gamma losses for each policy-year
        incurred = rng_sim.gamma(GAMMA_SHAPE, scale, size=n) * exposures
        total_inc = incurred.sum()

        panel_rows.append({
            "segment":   seg,
            "year":      yr,
            "exposure":  round(total_exp, 2),
            "incurred":  round(total_inc, 2),
            "loss_rate": total_inc / total_exp,
        })

panel = pl.DataFrame(panel_rows)

print(f"Panel: {len(panel)} rows ({N_SEGMENTS} segments × {N_YEARS} years)")
print(f"Mean loss rate across all rows: £{panel['loss_rate'].mean():.2f}")

display(
    panel
    .join(true_df.select(["segment", "n_policies", "tier"]), on="segment")
    .sort(["n_policies", "year"])
    .head(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Raw Experience Estimates
# MAGIC
# MAGIC The naïve approach: for each segment, take the exposure-weighted average loss rate
# MAGIC across all available years. No pooling, no shrinkage. The estimate is unbiased but
# MAGIC has variance proportional to 1/exposure — so thin segments are all over the place.
# MAGIC
# MAGIC This is what you get if you build a rate table by querying your claims system and
# MAGIC dividing incurred by earned without any credibility adjustment.

# COMMAND ----------

# Observed (exposure-weighted) mean per segment
raw_est = (
    panel
    .group_by("segment")
    .agg([
        pl.col("exposure").sum().alias("total_exposure"),
        pl.col("incurred").sum().alias("total_incurred"),
    ])
    .with_columns(
        (pl.col("total_incurred") / pl.col("total_exposure")).alias("raw_loss_rate")
    )
)

# ── Portfolio average ──────────────────────────────────────────────────────────
portfolio_avg = (
    panel["incurred"].sum() / panel["exposure"].sum()
)
print(f"Portfolio-wide average loss rate: £{portfolio_avg:.2f}")

# Attach segment metadata for benchmarking
raw_est = (
    raw_est
    .join(true_df, on="segment")
    .with_columns(
        pl.lit(portfolio_avg).alias("portfolio_avg")
    )
)

display(
    raw_est
    .sort("n_policies")
    .select(["segment", "tier", "n_policies", "true_loss", "raw_loss_rate", "portfolio_avg"])
    .head(12)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bühlmann-Straub Credibility
# MAGIC
# MAGIC `BuhlmannStraub.fit()` takes the panel DataFrame and estimates three structural
# MAGIC parameters from the data:
# MAGIC
# MAGIC - **mu** — collective mean: the grand exposure-weighted average across all groups
# MAGIC - **v (EPV)** — expected process variance: within-segment year-to-year noise
# MAGIC - **a (VHM)** — variance of hypothetical means: genuine between-segment spread
# MAGIC
# MAGIC From these it computes **k = v / a** (Bühlmann's k), which is the exposure a segment
# MAGIC needs to achieve Z = 0.5. A segment with total exposure w_i gets Z_i = w_i / (w_i + k).
# MAGIC
# MAGIC The credibility premium is then: `P_i = Z_i * X̄_i + (1 - Z_i) * mu`
# MAGIC
# MAGIC Thin segments (small w_i) get low Z and are pulled toward mu. Thick segments
# MAGIC get high Z and are trusted on their own experience.

# COMMAND ----------

bs = BuhlmannStraub()
bs.fit(
    panel,
    group_col="segment",
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

print(bs.summary())

# COMMAND ----------

# Attach credibility premiums to our benchmarking table
cred_premiums = bs.premiums_.rename({"group": "segment"})

results = (
    raw_est
    .join(
        cred_premiums.select(["segment", "Z", "credibility_premium"]),
        on="segment",
    )
)

display(
    results
    .sort("n_policies")
    .select([
        "segment", "tier", "n_policies", "true_loss",
        "raw_loss_rate", "portfolio_avg", "credibility_premium", "Z"
    ])
    .head(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark Table: MAE and RMSE by Segment Tier
# MAGIC
# MAGIC We evaluate each estimator against the true DGP loss costs, broken out by
# MAGIC segment size tier. This is the key table: it shows where each approach fails
# MAGIC and why credibility dominates.
# MAGIC
# MAGIC **What to expect:**
# MAGIC - Raw experience: low error on thick segments, high error on thin (variance dominates)
# MAGIC - Portfolio average: moderate error everywhere, systematically biased on thick segments
# MAGIC - Bühlmann-Straub: best of both — thick segments get Z near 1.0 (trusted),
# MAGIC   thin segments get shrunk toward the portfolio (variance controlled)
# MAGIC
# MAGIC The RMSE gap between raw and credibility on thin segments is the dollar value
# MAGIC of the credibility model.

# COMMAND ----------

def mae(pred: pl.Series, true: pl.Series) -> float:
    return float((pred - true).abs().mean())

def rmse(pred: pl.Series, true: pl.Series) -> float:
    return float(((pred - true) ** 2).mean() ** 0.5)

tiers = ["thin", "medium", "thick", "all"]
estimators = {
    "raw_experience": "raw_loss_rate",
    "portfolio_avg":  "portfolio_avg",
    "buhlmann_straub": "credibility_premium",
}

benchmark_rows = []
for tier in tiers:
    if tier == "all":
        subset = results
    else:
        subset = results.filter(pl.col("tier") == tier)

    n_segs = len(subset)
    if n_segs == 0:
        continue

    row = {"tier": tier, "n_segments": n_segs}
    for label, col in estimators.items():
        pred = subset[col]
        true = subset["true_loss"]
        row[f"{label}_mae"]  = round(mae(pred, true), 2)
        row[f"{label}_rmse"] = round(rmse(pred, true), 2)

    benchmark_rows.append(row)

bench = pl.DataFrame(benchmark_rows)

# Format as HTML table for display
html_rows = ""
for r in bench.iter_rows(named=True):
    html_rows += f"""
    <tr>
      <td><strong>{r['tier']}</strong></td>
      <td>{r['n_segments']}</td>
      <td>{r['raw_experience_mae']:.2f}</td>
      <td>{r['raw_experience_rmse']:.2f}</td>
      <td>{r['portfolio_avg_mae']:.2f}</td>
      <td>{r['portfolio_avg_rmse']:.2f}</td>
      <td style="background:#e8f5e9"><strong>{r['buhlmann_straub_mae']:.2f}</strong></td>
      <td style="background:#e8f5e9"><strong>{r['buhlmann_straub_rmse']:.2f}</strong></td>
    </tr>"""

html = f"""
<style>
  table.bench {{border-collapse: collapse; font-family: monospace; font-size: 13px;}}
  table.bench th, table.bench td {{border: 1px solid #ccc; padding: 6px 12px; text-align: right;}}
  table.bench th {{background: #f5f5f5;}}
  table.bench td:first-child {{text-align: left;}}
</style>
<table class="bench">
  <thead>
    <tr>
      <th rowspan="2">Tier</th>
      <th rowspan="2">Segments</th>
      <th colspan="2">Raw Experience</th>
      <th colspan="2">Portfolio Average</th>
      <th colspan="2" style="background:#c8e6c9">Bühlmann-Straub</th>
    </tr>
    <tr>
      <th>MAE (£)</th><th>RMSE (£)</th>
      <th>MAE (£)</th><th>RMSE (£)</th>
      <th style="background:#c8e6c9">MAE (£)</th>
      <th style="background:#c8e6c9">RMSE (£)</th>
    </tr>
  </thead>
  <tbody>{html_rows}</tbody>
</table>
"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations
# MAGIC
# MAGIC Two plots that tell the story:
# MAGIC
# MAGIC **Left: Credibility weight Z vs segment exposure.** The S-curve from the formula
# MAGIC Z = w / (w + k). Every segment sits on this curve — position determined entirely
# MAGIC by exposure. Colour shows segment tier. The value of k (the half-credibility point)
# MAGIC is labelled — it is the exposure at which Z = 0.5.
# MAGIC
# MAGIC **Right: Shrinkage plot.** X-axis is the raw observed loss rate; Y-axis is the
# MAGIC credibility estimate. Points above the 45° line are shrunk upward (thin segments
# MAGIC with below-average raw rates pulled toward portfolio mean); points below are shrunk
# MAGIC downward. The degree of shrinkage is exactly 1 - Z_i.
# MAGIC
# MAGIC The portfolio average is a horizontal line at £{portfolio_avg:.0f} — every segment
# MAGIC gets the same estimate regardless of what they observed. Credibility is the better
# MAGIC compromise.

# COMMAND ----------

matplotlib.use("Agg")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Bühlmann-Straub Credibility: Motor Portfolio", fontsize=14, y=1.01)

k_val = bs.k_
mu_val = bs.mu_hat_

colour_map = {"thin": "#e74c3c", "medium": "#f39c12", "thick": "#27ae60"}

# ── Plot 1: Z vs Exposure ──────────────────────────────────────────────────────
ax1 = axes[0]

# Theoretical curve
exp_range = np.linspace(0, results["total_exposure"].max() * 1.05, 400)
z_curve   = exp_range / (exp_range + k_val)
ax1.plot(exp_range, z_curve, "k-", lw=1.5, alpha=0.4, label="Z = w / (w + k)")

for tier, colour in colour_map.items():
    sub = results.filter(pl.col("tier") == tier)
    ax1.scatter(
        sub["total_exposure"].to_numpy(),
        sub["Z"].to_numpy(),
        c=colour, s=60, alpha=0.85, label=tier, zorder=3,
    )

# Mark k on x-axis
ax1.axvline(k_val, color="grey", ls="--", lw=1, alpha=0.6)
ax1.text(k_val * 1.02, 0.05, f"k = {k_val:,.0f}\n(Z = 0.5)", fontsize=9, color="grey")
ax1.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.5)

ax1.set_xlabel("Total exposure (vehicle-years)", fontsize=11)
ax1.set_ylabel("Credibility factor Z", fontsize=11)
ax1.set_title("Credibility weight vs segment volume", fontsize=11)
ax1.set_ylim(-0.02, 1.05)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Shrinkage plot ─────────────────────────────────────────────────────
ax2 = axes[1]

raw_arr  = results["raw_loss_rate"].to_numpy()
cred_arr = results["credibility_premium"].to_numpy()
true_arr = results["true_loss"].to_numpy()
tier_arr = results["tier"].to_list()

# 45° reference line (no shrinkage)
lim_min = min(raw_arr.min(), cred_arr.min()) * 0.92
lim_max = max(raw_arr.max(), cred_arr.max()) * 1.08
ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k-", lw=1, alpha=0.3, label="No shrinkage")

# Portfolio average line
ax2.axhline(portfolio_avg, color="#3498db", ls="--", lw=1.2, alpha=0.7,
            label=f"Portfolio avg £{portfolio_avg:.0f}")

for tier, colour in colour_map.items():
    mask = [t == tier for t in tier_arr]
    idx  = [i for i, m in enumerate(mask) if m]
    if not idx:
        continue
    ax2.scatter(
        raw_arr[idx], cred_arr[idx],
        c=colour, s=60, alpha=0.85, label=tier, zorder=3,
    )

# Draw shrinkage arrows for thin segments
thin_mask = [t == "thin" for t in tier_arr]
for i, is_thin in enumerate(thin_mask):
    if is_thin:
        ax2.annotate(
            "",
            xy=(raw_arr[i], cred_arr[i]),
            xytext=(raw_arr[i], raw_arr[i]),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8, alpha=0.5),
        )

ax2.set_xlabel("Raw observed loss rate (£)", fontsize=11)
ax2.set_ylabel("Credibility estimate (£)", fontsize=11)
ax2.set_title("Shrinkage: raw → credibility estimate", fontsize=11)
ax2.set_xlim(lim_min, lim_max)
ax2.set_ylim(lim_min, lim_max)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Segment-Level Detail Table
# MAGIC
# MAGIC Full results for all 30 segments, sorted by exposure. This is the table you
# MAGIC would hand to an underwriter: raw rate, credibility estimate, true rate (known
# MAGIC here from the simulation but unknown in practice), and the credibility factor.
# MAGIC
# MAGIC Pay attention to the thin segments at the top. Their raw rates are erratic —
# MAGIC some are far above true, some below. The credibility estimates are all pulled
# MAGIC back toward £420, tightly clustered. For thick segments at the bottom, the
# MAGIC credibility estimates closely track the raw rates (Z near 1.0) but with residual
# MAGIC shrinkage removing the most extreme one-year fluctuations.

# COMMAND ----------

detail = (
    results
    .sort("n_policies")
    .select([
        "segment", "tier", "n_policies",
        pl.col("total_exposure").round(1),
        pl.col("true_loss").round(2),
        pl.col("raw_loss_rate").round(2),
        pl.col("portfolio_avg").round(2),
        pl.col("credibility_premium").round(2),
        pl.col("Z").round(3),
    ])
    .rename({
        "n_policies": "policies",
        "total_exposure": "exposure",
        "true_loss": "true_£",
        "raw_loss_rate": "raw_£",
        "portfolio_avg": "portf_£",
        "credibility_premium": "cred_£",
    })
)

# HTML table with colour coding by tier
tier_colours = {"thin": "#fdecea", "medium": "#fff8e1", "thick": "#e8f5e9"}
html_rows2   = ""
for r in detail.iter_rows(named=True):
    bg = tier_colours.get(r["tier"], "#ffffff")
    raw_err  = abs(r["raw_£"]   - r["true_£"])
    cred_err = abs(r["cred_£"]  - r["true_£"])
    better   = "✓" if cred_err < raw_err else ""
    html_rows2 += f"""
    <tr style="background:{bg}">
      <td>{r['segment']}</td>
      <td>{r['tier']}</td>
      <td style="text-align:right">{r['policies']:,}</td>
      <td style="text-align:right">{r['exposure']:,.1f}</td>
      <td style="text-align:right">£{r['true_£']:.2f}</td>
      <td style="text-align:right">£{r['raw_£']:.2f}</td>
      <td style="text-align:right">£{r['portf_£']:.2f}</td>
      <td style="text-align:right"><strong>£{r['cred_£']:.2f}</strong></td>
      <td style="text-align:right">{r['Z']:.3f}</td>
      <td style="text-align:center; color:green">{better}</td>
    </tr>"""

html2 = f"""
<style>
  table.detail {{border-collapse:collapse; font-family:monospace; font-size:12px;}}
  table.detail th, table.detail td {{border:1px solid #ddd; padding:5px 10px;}}
  table.detail th {{background:#f5f5f5;}}
</style>
<table class="detail">
  <thead>
    <tr>
      <th>Segment</th><th>Tier</th><th>Policies</th><th>Exposure</th>
      <th>True £</th><th>Raw £</th><th>Portfolio £</th>
      <th>Credibility £</th><th>Z</th><th>Cred wins?</th>
    </tr>
  </thead>
  <tbody>{html_rows2}</tbody>
</table>
"""
displayHTML(html2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Practitioner Notes
# MAGIC
# MAGIC ### What the numbers show
# MAGIC
# MAGIC The benchmark table will show credibility dominating on thin and medium segments
# MAGIC with a clear RMSE reduction versus raw experience. On thick segments the gap closes
# MAGIC because Z approaches 1.0 and the credibility estimate is essentially the raw rate
# MAGIC anyway. This is the correct behaviour: when you have enough data, use it.
# MAGIC
# MAGIC ### What Bühlmann's k means
# MAGIC
# MAGIC The fitted k is the exposure (in vehicle-years) at which a segment achieves Z = 0.5
# MAGIC — equal weight between own experience and the portfolio mean. Read it as: "a segment
# MAGIC needs k vehicle-years before its own data gets 50% of the vote." For a portfolio
# MAGIC with moderate between-segment spread (realistic UK motor), k typically runs
# MAGIC 500–3,000 vehicle-years. A segment with 200 vehicle-years therefore gets Z around
# MAGIC 0.1–0.3 — heavily pooled.
# MAGIC
# MAGIC ### EPV vs VHM
# MAGIC
# MAGIC High EPV (v) relative to VHM (a) means the portfolio looks homogeneous — segments
# MAGIC bounce around a lot year to year, but there is little genuine structural difference
# MAGIC between them. In that case k is large and you lean toward the portfolio mean even
# MAGIC for moderately-sized segments. Low EPV/VHM ratio means segments genuinely differ
# MAGIC and you should trust their experience sooner.
# MAGIC
# MAGIC The ratio v/a is the data's answer to the question "is this segmentation variable
# MAGIC actually doing any work?" A fitted a near zero means the variable probably should
# MAGIC not be a rating factor.
# MAGIC
# MAGIC ### Where this fits in a pricing workflow
# MAGIC
# MAGIC Bühlmann-Straub is not a replacement for a GLM. It is what you use when your GLM
# MAGIC segments have insufficient volume to be trusted individually. The standard workflow:
# MAGIC
# MAGIC 1. Fit a GLM to the full portfolio — this estimates your main relativities
# MAGIC 2. Compute GLM-predicted loss rates per segment as the "complement" (mu)
# MAGIC 3. Apply Bühlmann-Straub to blend each segment's own experience with the GLM prediction
# MAGIC
# MAGIC In this notebook mu is the simple portfolio average, which is the correct baseline
# MAGIC when you have no prior model. In a real book, replace `complement` in `premiums_`
# MAGIC with your GLM predictions.
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC - Assumes the within-group variance structure is homoscedastic across groups.
# MAGIC   If young-driver segments have fundamentally higher volatility than experienced-driver
# MAGIC   segments, fit separate models by stratum.
# MAGIC - Needs at least 2 periods per group to estimate EPV. With only one year of data,
# \#   use a Bayesian hierarchical model (see `bayesian_pricing_demo.py`).
# MAGIC - The panel must have the same time structure across groups for the EPV estimator
# \#   to be unbiased. Missing periods for some groups will degrade the v_hat estimate.
# \#   `BuhlmannStraub` handles unbalanced panels correctly, but be aware of the implications.

# COMMAND ----------

print("Notebook complete.")
print(f"Fitted k = {bs.k_:,.1f} vehicle-years to half-credibility")
print(f"EPV (v)  = {bs.v_hat_:.4f}")
print(f"VHM (a)  = {bs.a_hat_:.4f}")
print(f"Ratio v/a = k = {bs.k_:,.1f}")
print()
overall_mae_raw  = mae(results["raw_loss_rate"],       results["true_loss"])
overall_mae_port = mae(results["portfolio_avg"],        results["true_loss"])
overall_mae_cred = mae(results["credibility_premium"],  results["true_loss"])
print("Overall MAE across all segments:")
print(f"  Raw experience:  £{overall_mae_raw:.2f}")
print(f"  Portfolio avg:   £{overall_mae_port:.2f}")
print(f"  Bühlmann-Straub: £{overall_mae_cred:.2f}")
