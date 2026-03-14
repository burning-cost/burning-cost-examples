# Databricks notebook source

# MAGIC %md
# MAGIC # Hierarchical Bayesian Pricing: Thin Segment Benchmark
# MAGIC
# MAGIC ## The problem this notebook exists to answer
# MAGIC
# MAGIC Every UK motor book has thin segments. You have 30 regions, and three of them
# MAGIC have fewer than 50 policies. You need a rate for those regions. What do you do?
# MAGIC
# MAGIC Three approaches are in common use:
# MAGIC
# MAGIC 1. **Raw experience**: use what you observed. If a thin region had 3 claims on
# MAGIC    40 policies, your burning cost is 7.5%. Simple. Also wildly noisy.
# MAGIC
# MAGIC 2. **Portfolio average**: ignore region entirely. Use the portfolio-wide rate.
# MAGIC    No variance problem, but you throw away real signal where segments genuinely differ.
# MAGIC
# MAGIC 3. **Hierarchical Bayesian partial pooling**: the Bayesian posterior tells you
# MAGIC    exactly how much weight to put on the thin segment's own experience vs the
# MAGIC    portfolio mean. That weight is determined by the data — specifically by the
# MAGIC    ratio of within-segment sampling noise to between-segment signal variance.
# MAGIC    This is the credibility theory result, done properly.
# MAGIC
# MAGIC The punchline: partial pooling dominates both alternatives. It outperforms raw
# MAGIC experience on thin segments (less variance) and portfolio average everywhere
# MAGIC (less bias where segments genuinely differ). This is not surprising — it is
# MAGIC the classical James-Stein result — but it is worth showing on data that looks
# MAGIC like what UK pricing teams actually handle.
# MAGIC
# MAGIC **What you will see:**
# MAGIC - Synthetic motor dataset: ~50k policies, 30 regions with deliberate thin tails
# MAGIC - Three estimators fit and evaluated against the true DGP rates
# MAGIC - MAE and RMSE broken out by segment size: thin (<100 policies) vs dense
# MAGIC - Shrinkage plot: Bayesian estimates pulling thin segments toward the portfolio mean
# MAGIC - Credibility weight vs segment size: the relationship that drives it all

# COMMAND ----------

# MAGIC %pip install "bayesian-pricing[pymc]" pymc arviz matplotlib pandas polars "numpy>=2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*ArviZ.*")

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

import pymc as pm
import arviz as az

from bayesian_pricing import HierarchicalFrequency, BayesianRelativities
from bayesian_pricing.frequency import SamplerConfig
from bayesian_pricing.diagnostics import convergence_summary

rng = np.random.default_rng(seed=2025)

print(f"PyMC {pm.__version__}")
print(f"ArviZ {az.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Motor Dataset
# MAGIC
# MAGIC The data-generating process encodes what we know about UK personal motor:
# MAGIC
# MAGIC - **Base frequency**: 8% annual claim rate (mid-market UK motor)
# MAGIC - **30 regions**: region effects drawn from Normal(0, 0.35) on the log scale —
# MAGIC   a realistic spread, implying roughly ±40% relativity across regions
# MAGIC - **Deliberate thin tails**: 8 of the 30 regions are rural/remote and receive
# MAGIC   a small policy count. These are the segments that break naïve estimators.
# MAGIC - **Policy count by region**: log-normally distributed to mimic real books
# MAGIC   where London + Birmingham dominate and rural regions get a trickle
# MAGIC
# MAGIC The true region-level claim rate is `BASE_FREQ * exp(region_effect)`. We
# MAGIC simulate policy-level data (one row per policy), then aggregate to segments.
# MAGIC The "true rate" for each region is what we benchmark against.

# COMMAND ----------

# ── Data-generating parameters ────────────────────────────────────────────────
N_POLICIES = 50_000
N_REGIONS = 30
BASE_FREQ = 0.08          # 8% annual claim frequency
REGION_SIGMA = 0.35       # log-scale spread across regions — realistic for UK
THIN_THRESHOLD = 100      # policies: below this we call a segment "thin"

# True region effects — fixed seed so the benchmark is reproducible
rng_dgp = np.random.default_rng(seed=42)
region_ids = [f"R{i:02d}" for i in range(1, N_REGIONS + 1)]

# Draw true log-rate offsets from the hierarchical prior
true_log_offsets = rng_dgp.normal(0.0, REGION_SIGMA, size=N_REGIONS)
true_rates = BASE_FREQ * np.exp(true_log_offsets)

true_rates_df = pd.DataFrame({
    "region": region_ids,
    "true_log_offset": true_log_offsets,
    "true_rate": true_rates,
})

# ── Allocate policies to regions ───────────────────────────────────────────────
# Log-normal policy count ensures a heavy-tailed allocation: a few large regions
# get thousands of policies; remote regions get tens.
raw_weights = rng_dgp.lognormal(mean=4.5, sigma=1.2, size=N_REGIONS)

# Force 8 regions to be thin (remote rural): cap their weight at a low value
thin_region_indices = rng_dgp.choice(N_REGIONS, size=8, replace=False)
raw_weights[thin_region_indices] = rng_dgp.uniform(5, 30, size=8)

policy_counts = np.round(raw_weights / raw_weights.sum() * N_POLICIES).astype(int)
# Adjust for rounding to hit exactly N_POLICIES
policy_counts[-1] += N_POLICIES - policy_counts.sum()
policy_counts = np.maximum(policy_counts, 1)

true_rates_df["n_policies"] = policy_counts
true_rates_df["is_thin"] = policy_counts < THIN_THRESHOLD

print(f"Total policies: {policy_counts.sum():,}")
print(f"Thin regions (<{THIN_THRESHOLD} policies): {(policy_counts < THIN_THRESHOLD).sum()}")
print(f"Thinnest region: {policy_counts.min()} policies")
print(f"Largest region:  {policy_counts.max():,} policies")
print(f"\nTrue rate range: {true_rates.min():.1%} – {true_rates.max():.1%}")
print(f"Portfolio-implied mean rate: {BASE_FREQ:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate policy-level claims and aggregate to segments
# MAGIC
# MAGIC Each policy gets a Bernoulli draw against its region's true frequency. We
# MAGIC then aggregate to segment-level sufficient statistics: claim count + exposure.
# MAGIC This is the input format `HierarchicalFrequency` expects — it operates on
# MAGIC segment-level statistics, not policy-level rows, which is both computationally
# MAGIC sensible and the natural production interface (aggregate from your claims system).

# COMMAND ----------

# Simulate policy-level data
policy_rows = []
for i, region in enumerate(region_ids):
    n = policy_counts[i]
    # Each policy has 1 year of earned exposure and a Bernoulli claim indicator
    claims = rng_dgp.binomial(1, true_rates[i], size=n)
    for c in claims:
        policy_rows.append({"region": region, "claims": int(c), "exposure": 1.0})

policies_df = pd.DataFrame(policy_rows)

# Aggregate to segment level (one row per region)
segments = (
    policies_df
    .groupby("region", as_index=False)
    .agg(claims=("claims", "sum"), exposure=("exposure", "sum"))
    .merge(true_rates_df[["region", "true_rate", "n_policies", "is_thin"]], on="region")
)

segments["raw_rate"] = segments["claims"] / segments["exposure"]

print(f"Segment table: {len(segments)} rows")
display(
    pl.from_pandas(segments)
    .sort("n_policies")
    .select(["region", "n_policies", "is_thin", "true_rate", "claims", "raw_rate"])
    .head(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit Three Approaches
# MAGIC
# MAGIC ### Approach 1 — Raw experience
# MAGIC Just `claims / exposure` per segment. No pooling. Maximally noisy for thin cells.
# MAGIC
# MAGIC ### Approach 2 — Portfolio average
# MAGIC Ignore region altogether. Use the portfolio-wide `total_claims / total_exposure`.
# MAGIC Zero variance but high bias where regions genuinely differ.
# MAGIC
# MAGIC ### Approach 3 — Hierarchical Bayesian (bayesian-pricing)
# MAGIC Fits a Poisson hierarchical model with partial pooling across regions. The
# MAGIC model learns `sigma_region` — how much between-region variation exists — and
# MAGIC uses it to set the credibility weight for each segment automatically.
# MAGIC
# MAGIC We use Pathfinder (variational inference) rather than NUTS here for speed.
# MAGIC For production rate tables, switch to `SamplerConfig(method="nuts")`.

# COMMAND ----------

# ── Approach 1: Raw experience ─────────────────────────────────────────────────
segments["pred_raw"] = segments["raw_rate"]

# ── Approach 2: Portfolio average ─────────────────────────────────────────────
portfolio_rate = segments["claims"].sum() / segments["exposure"].sum()
segments["pred_portfolio"] = portfolio_rate

print(f"Portfolio-wide claim rate: {portfolio_rate:.4f} ({portfolio_rate:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit the Hierarchical Bayesian model
# MAGIC
# MAGIC `HierarchicalFrequency` takes a segment-level DataFrame with claim counts and
# MAGIC exposure. The `group_cols` list tells it which columns to pool across — here
# MAGIC just `region`. In a real motor book you would add `veh_group`, `age_band`, etc.
# MAGIC
# MAGIC The `prior_mean_rate=BASE_FREQ` anchors the global intercept prior. In
# MAGIC production, set this from your portfolio statistics (it is not a sensitive
# MAGIC choice — the data quickly overrides a mildly wrong prior).

# COMMAND ----------

freq_model = HierarchicalFrequency(
    group_cols=["region"],
    prior_mean_rate=BASE_FREQ,
    variance_prior_sigma=0.4,   # allow reasonably wide region spread a priori
)

sampler_cfg = SamplerConfig(
    method="pathfinder",        # fast VI — use nuts for production
    draws=2_000,
    random_seed=2025,
)

freq_model.fit(
    segments[["region", "claims", "exposure"]],
    claim_count_col="claims",
    exposure_col="exposure",
    sampler_config=sampler_cfg,
)

print("Model fitted.")

# COMMAND ----------

# Posterior predictions: segment-level claim rates
preds_pl = freq_model.predict(quantiles=(0.05, 0.5, 0.95))
preds_pd = preds_pl.to_pandas()

segments = segments.merge(
    preds_pd[["region", "mean", "p5", "p95", "credibility_factor"]].rename(
        columns={"mean": "pred_bayes", "p5": "bayes_p5", "p95": "bayes_p95"}
    ),
    on="region",
)

display(
    pl.from_pandas(
        segments[["region", "n_policies", "is_thin", "true_rate",
                   "pred_raw", "pred_portfolio", "pred_bayes", "credibility_factor"]]
        .sort_values("n_policies")
    ).head(12)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Benchmark: MAE and RMSE Against True DGP Rates
# MAGIC
# MAGIC We evaluate each estimator against `true_rate` — the known ground truth.
# MAGIC The benchmark is split into thin (<100 policies) and dense segments because
# MAGIC that is where the story is: the Bayesian estimator wins most strongly on thin.

# COMMAND ----------

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

approaches = {
    "Raw experience":   "pred_raw",
    "Portfolio average":"pred_portfolio",
    "Hierarchical Bayes":"pred_bayes",
}

rows = []
for name, col in approaches.items():
    for label, mask in [
        ("All segments",   np.ones(len(segments), dtype=bool)),
        ("Thin (<100)",    segments["is_thin"].values),
        ("Dense (≥100)",   ~segments["is_thin"].values),
    ]:
        y_true = segments["true_rate"].values[mask]
        y_pred = segments[col].values[mask]
        rows.append({
            "Approach": name,
            "Segment group": label,
            "n_segments": int(mask.sum()),
            "MAE": round(mae(y_true, y_pred), 5),
            "RMSE": round(rmse(y_true, y_pred), 5),
        })

benchmark_df = pd.DataFrame(rows)
benchmark_pl = pl.from_pandas(benchmark_df)

display(benchmark_pl)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benchmark interpretation
# MAGIC
# MAGIC The table above should show:
# MAGIC
# MAGIC - **Raw experience on thin segments**: highest RMSE. Small sample = high variance
# MAGIC   = noisy estimates that may be far from truth.
# MAGIC - **Portfolio average on dense segments**: highest MAE. Dense segments have
# MAGIC   enough data to identify real regional differences; ignoring them is costly.
# MAGIC - **Hierarchical Bayes**: lowest or competitive RMSE everywhere. It borrows
# MAGIC   more on thin segments (where own experience is unreliable) and borrows less
# MAGIC   on dense segments (where own experience is trustworthy).
# MAGIC
# MAGIC This is the bias-variance trade-off resolved optimally by the data.

# COMMAND ----------

# Print a readable summary
print("=" * 72)
print(f"{'Approach':<22} {'Segment group':<18} {'n':>6} {'MAE':>10} {'RMSE':>10}")
print("-" * 72)
for _, row in benchmark_df.iterrows():
    print(
        f"{row['Approach']:<22} {row['Segment group']:<18} "
        f"{row['n_segments']:>6} {row['MAE']:>10.5f} {row['RMSE']:>10.5f}"
    )
print("=" * 72)

# Compute relative improvement of Bayes over raw on thin segments
thin_raw  = benchmark_df[(benchmark_df["Approach"] == "Raw experience")   & (benchmark_df["Segment group"] == "Thin (<100)")]["RMSE"].values[0]
thin_bayes= benchmark_df[(benchmark_df["Approach"] == "Hierarchical Bayes") & (benchmark_df["Segment group"] == "Thin (<100)")]["RMSE"].values[0]
pct_improvement = (thin_raw - thin_bayes) / thin_raw * 100
print(f"\nBayes vs raw on thin segments: {pct_improvement:.1f}% RMSE reduction")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Shrinkage Plot
# MAGIC
# MAGIC The central diagnostic for partial pooling: where does the Bayesian estimate
# MAGIC sit relative to the raw rate and the portfolio mean?
# MAGIC
# MAGIC - **Thin segments** (small dots) should pull hard toward the portfolio mean —
# MAGIC   their own data is too noisy to trust.
# MAGIC - **Dense segments** (large dots) should sit close to their own raw rate —
# MAGIC   they have enough data to self-identify.
# MAGIC
# MAGIC The slope of the Bayesian line relative to the 45-degree "no pooling" line
# MAGIC quantifies the global degree of shrinkage.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Bayesian Partial Pooling: Thin Segment Benchmark", fontsize=14, fontweight="bold")

# ── Left: Raw rate vs Bayes estimate ─────────────────────────────────────────
ax = axes[0]
is_thin = segments["is_thin"].values
sizes = np.where(is_thin, 40, 120)
colors = np.where(is_thin, "#d62728", "#1f77b4")

ax.scatter(
    segments["raw_rate"], segments["pred_bayes"],
    s=sizes, c=colors, alpha=0.7, edgecolors="none",
)

# 45-degree "no pooling" line
lims = [
    min(segments["raw_rate"].min(), segments["pred_bayes"].min()) * 0.9,
    max(segments["raw_rate"].max(), segments["pred_bayes"].max()) * 1.1,
]
ax.plot(lims, lims, "k--", lw=1.2, label="No pooling (45°)")

# Portfolio mean line
ax.axhline(portfolio_rate, color="grey", lw=1.0, ls=":", label=f"Portfolio mean ({portfolio_rate:.1%})")

# Annotate a few thin segments
thin_segs = segments[segments["is_thin"]].nlargest(3, "raw_rate")
for _, row in thin_segs.iterrows():
    ax.annotate(
        row["region"],
        xy=(row["raw_rate"], row["pred_bayes"]),
        xytext=(8, -12), textcoords="offset points",
        fontsize=8, color="#d62728",
    )

ax.set_xlabel("Raw observed rate (claims / exposure)")
ax.set_ylabel("Bayesian posterior mean rate")
ax.set_title("Shrinkage: raw vs Bayesian estimate")

red_patch  = mpatches.Patch(color="#d62728", label=f"Thin (<{THIN_THRESHOLD} policies)")
blue_patch = mpatches.Patch(color="#1f77b4", label=f"Dense (≥{THIN_THRESHOLD} policies)")
ax.legend(handles=[red_patch, blue_patch, plt.Line2D([0],[0],color="k",ls="--",lw=1.2,label="No pooling"),
                   plt.Line2D([0],[0],color="grey",ls=":",lw=1.0,label="Portfolio mean")],
          fontsize=8, loc="upper left")

# ── Right: Bayes estimate vs True rate ───────────────────────────────────────
ax2 = axes[1]

# Plot error bars for credible intervals on thin segments only
for _, row in segments[segments["is_thin"]].iterrows():
    ax2.plot(
        [row["true_rate"], row["true_rate"]],
        [row["bayes_p5"], row["bayes_p95"]],
        color="#d62728", alpha=0.4, lw=1.5,
    )

ax2.scatter(
    segments["true_rate"], segments["pred_bayes"],
    s=sizes, c=colors, alpha=0.75, edgecolors="none",
    zorder=3,
)
ax2.scatter(
    segments["true_rate"], segments["pred_raw"],
    s=40, marker="x", c=colors, alpha=0.5, zorder=2, linewidths=1.2,
)

# True rate parity line
lims2 = [
    min(segments["true_rate"].min(), segments[["pred_bayes","pred_raw"]].min().min()) * 0.9,
    max(segments["true_rate"].max(), segments[["pred_bayes","pred_raw"]].max().max()) * 1.1,
]
ax2.plot(lims2, lims2, "k--", lw=1.2, label="Perfect estimate")

ax2.set_xlabel("True DGP rate")
ax2.set_ylabel("Estimated rate")
ax2.set_title("Bayes (circles) vs Raw (×) against truth\n90% credible interval shown for thin segments")

circle_patch = mpatches.Patch(color="grey", label="Bayes estimate (circle)")
ax2.legend(handles=[
    red_patch, blue_patch,
    plt.Line2D([0],[0], color="k", ls="--", lw=1.2, label="Perfect estimate"),
    plt.scatter([], [], marker="o", color="grey", label="Bayes"),
    plt.scatter([], [], marker="x", color="grey", s=40, linewidths=1.2, label="Raw"),
], fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Credibility Weight vs Segment Size
# MAGIC
# MAGIC The credibility factor tells you what fraction of the Bayesian estimate comes
# MAGIC from the segment's own experience (vs the portfolio mean). It should rise
# MAGIC monotonically with exposure — that is exactly the Bühlmann-Straub credibility
# MAGIC result. The Bayesian model recovers this from first principles without needing
# MAGIC to specify a credibility formula.
# MAGIC
# MAGIC In practice, segments with credibility < 0.3 should be treated as "needing
# MAGIC review" — the estimate is dominated by the prior, not the segment's own data.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 5))

cred_data = segments.sort_values("n_policies")
sc = ax.scatter(
    cred_data["n_policies"],
    cred_data["credibility_factor"],
    c=cred_data["credibility_factor"],
    cmap="RdYlGn",
    s=80,
    alpha=0.8,
    edgecolors="none",
    vmin=0, vmax=1,
)

# Reference lines
ax.axvline(THIN_THRESHOLD, color="#d62728", ls="--", lw=1.2,
           label=f"Thin threshold ({THIN_THRESHOLD} policies)")
ax.axhline(0.3, color="orange", ls=":", lw=1.2,
           label="Review threshold (credibility = 0.3)")

# Annotate extreme cases
for _, row in cred_data.nsmallest(3, "credibility_factor").iterrows():
    ax.annotate(
        f"{row['region']} (n={int(row['n_policies'])})",
        xy=(row["n_policies"], row["credibility_factor"]),
        xytext=(12, 5), textcoords="offset points",
        fontsize=8, color="#d62728",
    )

cb = plt.colorbar(sc, ax=ax)
cb.set_label("Credibility factor")

ax.set_xscale("log")
ax.set_xlabel("Number of policies in segment (log scale)")
ax.set_ylabel("Credibility factor (0 = all prior, 1 = all data)")
ax.set_title("Credibility weight vs segment size\n(Bayesian partial pooling recovers Bühlmann-Straub automatically)")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Variance Components
# MAGIC
# MAGIC The model estimates `sigma_region` — how much the true claim rates vary across
# MAGIC regions. This is the key hyperparameter. It controls:
# MAGIC
# MAGIC - How much pooling to apply globally (high sigma = less pooling)
# MAGIC - The width of the credible intervals for each region
# MAGIC
# MAGIC In a Bühlmann-Straub analogy, `sigma_region` on the log scale corresponds to
# MAGIC `sqrt(a)` (the between-group variance). The Bayesian model estimates this from
# MAGIC data rather than requiring you to set it externally.

# COMMAND ----------

var_comp = freq_model.variance_components()
display(var_comp)

# Extract estimated sigma for display
sigma_est = var_comp.filter(pl.col("parameter") == "sigma_region")["mean"][0]
print(f"\nEstimated sigma_region: {sigma_est:.3f}")
print(f"True DGP sigma_region:  {REGION_SIGMA:.3f}")
print(f"\nTypical relativity spread (±1 sigma): {(np.exp(sigma_est) - 1) * 100:.1f}% from portfolio mean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary Table: Three-Way Comparison
# MAGIC
# MAGIC An actuary-ready summary showing, for each segment, where each approach lands
# MAGIC relative to truth. The "error vs truth" columns make the bias-variance story
# MAGIC concrete. Sorted by segment size so thin segments appear at the top.

# COMMAND ----------

summary = segments[[
    "region", "n_policies", "is_thin",
    "true_rate", "pred_raw", "pred_portfolio", "pred_bayes",
    "credibility_factor",
]].copy()

summary["err_raw"]       = (summary["pred_raw"]       - summary["true_rate"]).abs()
summary["err_portfolio"] = (summary["pred_portfolio"]  - summary["true_rate"]).abs()
summary["err_bayes"]     = (summary["pred_bayes"]      - summary["true_rate"]).abs()
summary["best"] = summary[["err_raw", "err_portfolio", "err_bayes"]].idxmin(axis=1).str.replace("err_", "")

summary_pl = (
    pl.from_pandas(summary)
    .sort("n_policies")
    .select([
        "region", "n_policies", "is_thin",
        pl.col("true_rate").round(4),
        pl.col("pred_raw").round(4),
        pl.col("pred_portfolio").round(4),
        pl.col("pred_bayes").round(4),
        pl.col("credibility_factor").round(3),
        pl.col("err_raw").round(4),
        pl.col("err_portfolio").round(4),
        pl.col("err_bayes").round(4),
        "best",
    ])
)

display(summary_pl)

# COMMAND ----------

# Count how often each approach wins
winner_counts = summary["best"].value_counts()
print("Which estimator has lowest absolute error by segment:")
print(winner_counts.to_string())
print()

thin_summary = summary[summary["is_thin"]]
thin_winner_counts = thin_summary["best"].value_counts()
print("Same, restricted to thin segments only:")
print(thin_winner_counts.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MAE/RMSE Bar Chart
# MAGIC
# MAGIC Visual summary of the benchmark. The left panel shows all segments; the right
# MAGIC shows only thin segments. The Bayesian estimator's advantage is most visible
# MAGIC on thin segments where partial pooling does the most work.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Error vs True DGP Rate: Three Approaches", fontsize=13, fontweight="bold")

metric_labels = ["MAE", "RMSE"]
approach_labels = ["Raw experience", "Portfolio average", "Hierarchical Bayes"]
colors_bar = ["#e377c2", "#bcbd22", "#17becf"]

for ax_idx, (ax, group_label) in enumerate(zip(axes, ["All segments", "Thin (<100)"])):
    subset = benchmark_df[benchmark_df["Segment group"] == group_label]

    for metric_idx, metric in enumerate(metric_labels):
        x = np.arange(len(approach_labels)) + metric_idx * (len(approach_labels) + 1) * 0.3
        vals = [
            subset[subset["Approach"] == ap][metric].values[0]
            for ap in approach_labels
        ]
        bars = ax.bar(x, vals, width=0.28, color=colors_bar, alpha=0.85,
                      label=[f"{ap} ({metric})" for ap in approach_labels] if metric_idx == 0 else [""] * 3)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_title(f"{group_label} (n={subset['n_segments'].iloc[0]})")
    ax.set_xticks([])
    ax.set_ylabel("Error rate (claim rate scale)")

    if ax_idx == 0:
        patches = [mpatches.Patch(color=c, label=ap) for c, ap in zip(colors_bar, approach_labels)]
        mae_line  = mpatches.Patch(color="white", label="Left cluster: MAE")
        rmse_line = mpatches.Patch(color="white", label="Right cluster: RMSE")
        ax.legend(handles=patches + [mae_line, rmse_line], fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Findings
# MAGIC
# MAGIC **The credibility result holds exactly as theory predicts:**
# MAGIC
# MAGIC 1. **Raw experience** is unbeatable if you have enough data — on dense segments
# MAGIC    it is competitive with Bayes. But for thin segments it is unreliable: a
# MAGIC    region with 40 policies and 4 claims has a "rate" of 10%, but the true rate
# MAGIC    could plausibly be anywhere from 3% to 20% given sampling noise alone.
# MAGIC
# MAGIC 2. **Portfolio average** makes thin segments precise but biased. If a remote
# MAGIC    rural region genuinely has 5% frequency and the portfolio average is 8%,
# MAGIC    you are 60% overpriced on that segment. That is adverse selection risk —
# MAGIC    better risks will shop away, and you will be left with worse ones.
# MAGIC
# MAGIC 3. **Hierarchical Bayes** wins by resolving the bias-variance trade-off
# MAGIC    adaptively. The credibility weight rises from near 0 for single-digit-policy
# MAGIC    segments to near 1 for segments with hundreds of policies. The model learns
# MAGIC    the amount of between-region variation (`sigma_region`) from the dense
# MAGIC    segments and uses it to calibrate shrinkage on the thin ones.
# MAGIC
# MAGIC **On thin segments specifically**: expect 20–50% RMSE reduction vs raw
# MAGIC experience (varies with true sigma and the thinness of the tail).
# MAGIC
# MAGIC **Practical implication for UK pricing teams**: any region, territory, or
# MAGIC rating cell with <100 policies is a candidate for Bayesian pooling. The
# MAGIC `credibility_factor` output from `HierarchicalFrequency.predict()` directly
# MAGIC flags which segments are prior-dominated and therefore require sign-off before
# MAGIC large rate movements are applied.
# MAGIC
# MAGIC **Further reading:**
# MAGIC - Bühlmann & Straub (1970): the original credibility paper
# MAGIC - Gelman et al. BDA3, Chapter 5: hierarchical models as partial pooling
# MAGIC - `bayesian-pricing` library: `pip install bayesian-pricing[pymc]`
