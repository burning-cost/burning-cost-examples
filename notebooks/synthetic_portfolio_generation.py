# Databricks notebook source

# MAGIC %md
# MAGIC # Synthetic Motor Portfolio Generation with Vine Copulas
# MAGIC
# MAGIC **The problem:** Your pricing team needs to share portfolio data with a vendor
# MAGIC for a model build. Your data science team wants a public benchmark dataset.
# MAGIC Your new hire needs realistic data to develop a frequency model before they
# MAGIC have production access. In all three cases, handing over real policyholder
# MAGIC data is either prohibited by your data governance policy or inadvisable.
# MAGIC
# MAGIC **The naive solution** — sample each rating factor independently from its
# MAGIC marginal distribution — produces data that looks plausible in isolation but
# MAGIC is statistically broken. Young drivers with 20 years of NCD. High-mileage
# MAGIC commuters in rural Scotland. Claim frequencies that don't track exposure.
# MAGIC Any model trained on this data will learn the wrong relationships.
# MAGIC
# MAGIC **What this notebook shows:** `insurance-synthetic` uses R-vine copulas to
# MAGIC generate synthetic portfolios that preserve the full dependence structure of
# MAGIC the original data — including tail dependence that a Gaussian copula misses.
# MAGIC We run a direct benchmark between vine-copula synthesis and independent
# MAGIC (naive) sampling, and show quantitatively where the naive approach fails.
# MAGIC
# MAGIC **What you will cover:**
# MAGIC 1. Build a realistic seed portfolio with correlated rating factors
# MAGIC 2. Fit `InsuranceSynthesizer` (vine copula + automatic marginals)
# MAGIC 3. Generate 50,000 synthetic policies with domain constraints
# MAGIC 4. Build a naive independent baseline (the thing we're beating)
# MAGIC 5. Run `SyntheticFidelityReport`: KS, Wasserstein, Spearman matrix
# MAGIC 6. Correlation heatmap: vine vs naive — visually see what the copula preserves
# MAGIC 7. Marginal distribution comparison plots (matplotlib)
# MAGIC 8. Fidelity metrics table (HTML): vine copula vs naive side by side
# MAGIC 9. Exposure/frequency structure validation
# MAGIC 10. Tail dependence: the case where Gaussian copulas fail

# COMMAND ----------

# MAGIC %pip install "insurance-synthetic[vine,fidelity]" pyvinecopulib matplotlib seaborn

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and Version Check

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
import io
import base64

import insurance_synthetic
from insurance_synthetic import (
    InsuranceSynthesizer,
    SyntheticFidelityReport,
    uk_motor_schema,
    fit_marginal,
)

print(f"insurance-synthetic: {insurance_synthetic.__version__}")
print(f"numpy: {np.__version__}")
print(f"polars: {pl.__version__}")

SEED = 42
rng_global = np.random.default_rng(SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the Seed Portfolio
# MAGIC
# MAGIC We construct a 10,000-policy UK motor book with realistic correlations
# MAGIC baked in. This is the data we're treating as "real" — the thing the
# MAGIC synthesiser has to learn from.
# MAGIC
# MAGIC Key structural dependencies:
# MAGIC - NCD is bounded by driving age (can't have 15 years NCD at age 25)
# MAGIC - Young drivers cluster in higher vehicle groups (hot hatches, performance cars)
# MAGIC - Monthly payment correlates with higher vehicle group and lower NCD
# MAGIC - Claim frequency falls steeply with NCD, rises with vehicle group
# MAGIC - Annual mileage follows a log-normal with a right tail (high-mileage commuters)
# MAGIC
# MAGIC This isn't a toy dataset. These correlations are the actuarial problem.

# COMMAND ----------

rng = np.random.default_rng(SEED)
N_SEED = 10_000

# Driver demographics
driver_age = rng.integers(17, 82, size=N_SEED)

# NCD: bounded by driving years, beta-distributed toward higher end for older drivers
max_ncd = np.minimum(driver_age - 17, 25)
ncd_raw = max_ncd * rng.beta(2.5, 1.2, size=N_SEED)
ncd_years = np.clip(ncd_raw.astype(int), 0, 25)

# Vehicle group: younger drivers skew toward higher groups
age_effect = np.clip((35 - driver_age) * 0.4, -8, 12)
vg_noise = rng.normal(0, 8, size=N_SEED)
vehicle_group = np.clip((22 + age_effect + vg_noise).astype(int), 1, 50)

# Vehicle age: slight negative correlation with vehicle group (newer expensive cars)
veh_age_base = 5 + 0.1 * (25 - vehicle_group) + rng.exponential(3, size=N_SEED)
vehicle_age = np.clip(veh_age_base.astype(int), 0, 20)

# Region: realistic UK distribution, London-heavy
region_weights = [0.18, 0.15, 0.10, 0.10, 0.09, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02]
regions = [
    "London", "South East", "East of England", "South West",
    "West Midlands", "East Midlands", "Yorkshire", "North West",
    "North East", "Scotland", "Wales", "Northern Ireland",
]
region = rng.choice(regions, size=N_SEED, p=region_weights)

# Cover type: most are comprehensive, TPO skews young/low-value
cover_weights = [0.72, 0.20, 0.08]
cover_type = rng.choice(
    ["Comprehensive", "Third Party Fire & Theft", "Third Party Only"],
    size=N_SEED, p=cover_weights,
)

# Payment method: monthly payers skew younger and higher-risk
young_mask = driver_age < 30
monthly_prob = np.where(young_mask, 0.55, 0.35)
payment_method = np.where(
    rng.uniform(size=N_SEED) < monthly_prob,
    rng.choice(["Monthly Direct Debit", "Monthly Credit"], size=N_SEED, p=[0.65, 0.35]),
    "Annual",
)

# Annual mileage: log-normal, modestly correlated with vehicle group
mileage_mean = np.log(9_000 + 150 * vehicle_group)
annual_mileage = rng.lognormal(mileage_mean, 0.45, size=N_SEED).astype(int)
annual_mileage = np.clip(annual_mileage, 1_000, 50_000)

# Exposure: Beta(8, 1.5) — most policies near full year, tail of mid-terms
exposure = np.clip(rng.beta(8, 1.5, size=N_SEED), 0.01, 1.0)

# Claim frequency: structured Poisson
#   Base: 0.08, rising with vehicle group, falling sharply with NCD
#   Young-driver uplift: significant for <25
base_freq = (
    0.07
    + 0.0025 * vehicle_group
    - 0.006 * ncd_years
    + 0.04 * (driver_age < 25).astype(float)
    + 0.01 * (driver_age > 70).astype(float)
)
base_freq = np.clip(base_freq, 0.01, 0.55)
claim_count = rng.poisson(base_freq * exposure)
claim_count = np.clip(claim_count, 0, 6)

# Severity: log-normal, zero for no-claim policies
#   Severity rises with vehicle group (more expensive to repair)
sev_log_mean = np.log(1_800 + 55 * vehicle_group)
claim_amount = np.where(
    claim_count > 0,
    np.maximum(0.0, rng.lognormal(sev_log_mean, 0.85, size=N_SEED)),
    0.0,
)

seed_df = pl.DataFrame({
    "driver_age": driver_age.tolist(),
    "vehicle_age": vehicle_age.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "region": region.tolist(),
    "ncd_years": ncd_years.tolist(),
    "cover_type": cover_type.tolist(),
    "payment_method": payment_method.tolist(),
    "annual_mileage": annual_mileage.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
    "claim_amount": claim_amount.tolist(),
})

print(f"Seed portfolio: {len(seed_df):,} policies, {len(seed_df.columns)} columns")
print(f"Claim frequency: {seed_df['claim_count'].sum() / seed_df['exposure'].sum():.4f} per policy year")
print(f"Claim rate (policies with >=1 claim): {(seed_df['claim_count'] > 0).mean():.3f}")
print(f"\nCorrelation sanity checks (Spearman):")
rho_ncd_freq, _ = stats.spearmanr(seed_df["ncd_years"].to_numpy(), seed_df["claim_count"].to_numpy())
rho_vg_freq, _ = stats.spearmanr(seed_df["vehicle_group"].to_numpy(), seed_df["claim_count"].to_numpy())
rho_age_ncd, _ = stats.spearmanr(seed_df["driver_age"].to_numpy(), seed_df["ncd_years"].to_numpy())
print(f"  NCD <-> claim_count:    {rho_ncd_freq:+.3f}  (should be negative)")
print(f"  vehicle_group <-> claim_count: {rho_vg_freq:+.3f}  (should be positive)")
print(f"  driver_age <-> ncd_years:      {rho_age_ncd:+.3f}  (should be positive)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit the Vine Copula Synthesiser
# MAGIC
# MAGIC The synthesiser does three things during fitting:
# MAGIC
# MAGIC 1. **Marginal fitting**: for each column, it fits a battery of parametric
# MAGIC    distributions (Gamma, LogNormal, Poisson, NegBin, Beta, Normal) and selects
# MAGIC    by AIC. Categoricals use their empirical distribution.
# MAGIC
# MAGIC 2. **PIT transform**: each column is mapped to uniform [0,1] via its fitted
# MAGIC    CDF. Discrete columns get jitter to break ties at CDF steps — the standard
# MAGIC    approach for fitting copulas to count data.
# MAGIC
# MAGIC 3. **Vine copula fitting**: an R-vine copula is fitted to the uniform-margin
# MAGIC    data using `pyvinecopulib`. The vine structure is selected by maximising the
# MAGIC    log-likelihood of bivariate copula pairs. `trunc_lvl=3` limits the vine to
# MAGIC    3 trees — sufficient for capturing the main dependencies without overfitting
# MAGIC    on 10,000 rows.
# MAGIC
# MAGIC The exposure/frequency separation is critical: claim counts are regenerated
# MAGIC at `.generate()` time using `Poisson(lambda * exposure)` where `lambda` is the
# MAGIC fitted annualised rate. This is not the same as sampling from the marginal —
# MAGIC it preserves the actuarial relationship between exposure and frequency.

# COMMAND ----------

synth = InsuranceSynthesizer(
    method="vine",
    family_set="all",
    trunc_lvl=3,
    n_threads=4,
    random_state=SEED,
)

synth.fit(
    seed_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
)

synth.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate the Synthetic Portfolio
# MAGIC
# MAGIC We generate 50,000 policies — 5x the seed size. Constraints from
# MAGIC `uk_motor_schema()` enforce hard domain bounds: no 16-year-old drivers,
# MAGIC no negative NCD, no mileage below 1,000. The synthesiser uses
# MAGIC rejection-resample to enforce these without distorting the copula.

# COMMAND ----------

schema = uk_motor_schema()

synthetic_df = synth.generate(
    50_000,
    constraints=schema["constraints"],
    max_resample_attempts=15,
)

print(f"Synthetic portfolio: {len(synthetic_df):,} policies")
print(f"\nFrequency check:")
print(f"  Seed:      {seed_df['claim_count'].sum() / seed_df['exposure'].sum():.4f} claims/year")
print(f"  Synthetic: {synthetic_df['claim_count'].sum() / synthetic_df['exposure'].sum():.4f} claims/year")
print(f"\nExposure distribution:")
print(f"  Seed mean:      {seed_df['exposure'].mean():.4f}")
print(f"  Synthetic mean: {synthetic_df['exposure'].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build the Naive Independent Baseline
# MAGIC
# MAGIC The naive approach: sample each column independently from its marginal
# MAGIC distribution, ignoring all correlations between columns.
# MAGIC
# MAGIC This is what you'd get from:
# MAGIC - Drawing driver_age from its empirical distribution
# MAGIC - Drawing ncd_years from its empirical distribution (independently)
# MAGIC - etc.
# MAGIC
# MAGIC The marginal distributions are preserved by construction — that's not
# MAGIC the test. The test is whether the *relationships between columns* survive.
# MAGIC A 22-year-old driver is much more likely to have 0 NCD and a high vehicle
# MAGIC group than the marginal distributions suggest in isolation. The naive
# MAGIC approach destroys this structure.

# COMMAND ----------

rng_naive = np.random.default_rng(SEED + 1)
N_SYNTH = 50_000

numeric_cols = ["driver_age", "vehicle_age", "vehicle_group", "ncd_years",
                "annual_mileage", "exposure", "claim_count", "claim_amount"]

# Sample each numeric column independently from its empirical distribution
naive_data = {}
for col in numeric_cols:
    source = seed_df[col].to_numpy()
    naive_data[col] = rng_naive.choice(source, size=N_SYNTH, replace=True)

# Sample categorical columns independently
for col in ["region", "cover_type", "payment_method"]:
    source = seed_df[col].to_list()
    naive_data[col] = rng_naive.choice(source, size=N_SYNTH, replace=True).tolist()

# Re-generate frequency using the same exposure/rate as vine method (fair comparison:
# same overall frequency preservation mechanism, just no dependence structure)
naive_freq_rate = float(seed_df["claim_count"].sum() / seed_df["exposure"].sum())
naive_claims = rng_naive.poisson(naive_freq_rate * naive_data["exposure"])
naive_data["claim_count"] = np.clip(naive_claims, 0, 6)

naive_df = pl.DataFrame({
    "driver_age": naive_data["driver_age"].astype(int).tolist(),
    "vehicle_age": naive_data["vehicle_age"].astype(int).tolist(),
    "vehicle_group": naive_data["vehicle_group"].astype(int).tolist(),
    "region": naive_data["region"],
    "ncd_years": naive_data["ncd_years"].astype(int).tolist(),
    "cover_type": naive_data["cover_type"],
    "payment_method": naive_data["payment_method"],
    "annual_mileage": naive_data["annual_mileage"].astype(int).tolist(),
    "exposure": naive_data["exposure"].astype(float).tolist(),
    "claim_count": naive_data["claim_count"].astype(int).tolist(),
    "claim_amount": naive_data["claim_amount"].astype(float).tolist(),
})

print(f"Naive baseline: {len(naive_df):,} policies")
print(f"\nKey correlation check — this is where naive fails:")
for label, df in [("Seed (real)", seed_df), ("Vine (synthetic)", synthetic_df), ("Naive (independent)", naive_df)]:
    rho_ncd, _ = stats.spearmanr(df["ncd_years"].to_numpy(), df["claim_count"].to_numpy())
    rho_vg, _ = stats.spearmanr(df["vehicle_group"].to_numpy(), df["claim_count"].to_numpy())
    rho_age_ncd, _ = stats.spearmanr(df["driver_age"].to_numpy(), df["ncd_years"].to_numpy())
    print(f"\n  {label}:")
    print(f"    NCD <-> claims:        {rho_ncd:+.4f}")
    print(f"    vehicle_group <-> claims: {rho_vg:+.4f}")
    print(f"    driver_age <-> NCD:    {rho_age_ncd:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run the Fidelity Report (Vine Copula)
# MAGIC
# MAGIC `SyntheticFidelityReport` quantifies fidelity at three levels:
# MAGIC - **Marginal**: KS test + Wasserstein distance, per column
# MAGIC - **Correlation**: Spearman correlation matrix comparison (Frobenius distance)
# MAGIC - **Tail**: TVaR ratio at the 99th percentile
# MAGIC
# MAGIC Interpretation benchmarks:
# MAGIC - KS statistic < 0.05: excellent marginal preservation
# MAGIC - Wasserstein (normalised) < 0.10: good shape match
# MAGIC - Frobenius norm < 2.0: dependence structure well-preserved
# MAGIC - TVaR ratio in [0.80, 1.20]: tail risk not inflated or deflated

# COMMAND ----------

report_vine = SyntheticFidelityReport(
    seed_df,
    synthetic_df,
    exposure_col="exposure",
    target_col="claim_count",
)

report_naive = SyntheticFidelityReport(
    seed_df,
    naive_df,
    exposure_col="exposure",
    target_col="claim_count",
)

marg_vine = report_vine.marginal_report()
marg_naive = report_naive.marginal_report()
corr_vine = report_vine.correlation_report()
corr_naive = report_naive.correlation_report()

frob_vine = float(corr_vine["frobenius_norm"][0])
frob_naive = float(corr_naive["frobenius_norm"][0])

print(f"Frobenius norm (Spearman distance): vine={frob_vine:.4f}  naive={frob_naive:.4f}")
print(f"Lower is better. The naive approach should be substantially worse.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fidelity Metrics Table — Vine vs Naive (HTML)

# COMMAND ----------

def _fmt(val, precision=4):
    if val is None:
        return "<em>n/a</em>"
    return f"{val:.{precision}f}"

def _ks_badge(ks):
    if ks is None:
        return "<span style='color:#888'>n/a</span>"
    color = "#1a7a1a" if ks < 0.05 else ("#c07700" if ks < 0.10 else "#b22222")
    label = "excellent" if ks < 0.05 else ("acceptable" if ks < 0.10 else "poor")
    return f"<span style='color:{color}'>{ks:.4f} <em>({label})</em></span>"

# Join vine and naive marginal reports
vine_dict = {r["column"]: r for r in marg_vine.iter_rows(named=True)}
naive_dict = {r["column"]: r for r in marg_naive.iter_rows(named=True)}

rows_html = []
for col in [r["column"] for r in marg_vine.iter_rows(named=True)]:
    v = vine_dict.get(col, {})
    n = naive_dict.get(col, {})
    rows_html.append(f"""
    <tr>
      <td style='font-family:monospace'>{col}</td>
      <td>{_ks_badge(v.get('ks_statistic'))}</td>
      <td>{_ks_badge(n.get('ks_statistic'))}</td>
      <td>{_fmt(v.get('wasserstein'))}</td>
      <td>{_fmt(n.get('wasserstein'))}</td>
      <td>{_fmt(v.get('mean_real'), 3)}</td>
      <td>{_fmt(v.get('mean_synthetic'), 3)}</td>
      <td>{_fmt(n.get('mean_synthetic'), 3)}</td>
    </tr>""")

table_html = f"""
<style>
  .fid-table {{ border-collapse: collapse; font-size: 13px; font-family: sans-serif; }}
  .fid-table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  .fid-table td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  .fid-table tr:nth-child(even) {{ background: #f9f9f9; }}
  .fid-table caption {{ font-weight: bold; font-size: 15px; padding: 8px; color: #2c3e50; text-align: left; }}
</style>
<table class='fid-table'>
  <caption>Marginal Fidelity: Vine Copula vs Independent Sampling</caption>
  <thead>
    <tr>
      <th>Column</th>
      <th>KS (Vine)</th>
      <th>KS (Naive)</th>
      <th>Wasserstein (Vine)</th>
      <th>Wasserstein (Naive)</th>
      <th>Mean (Real)</th>
      <th>Mean (Vine)</th>
      <th>Mean (Naive)</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
  <tfoot>
    <tr>
      <td colspan='8' style='padding: 8px 12px; font-style: italic; color: #555;'>
        Frobenius norm (Spearman matrix distance):
        Vine = <strong>{frob_vine:.4f}</strong> |
        Naive = <strong>{frob_naive:.4f}</strong>.
        Smaller = better dependence preservation.
      </td>
    </tr>
  </tfoot>
</table>
"""
displayHTML(table_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Correlation Heatmaps: Vine vs Naive
# MAGIC
# MAGIC The heatmap makes the structural failure of naive sampling immediately visible.
# MAGIC Vine copula should show a heatmap that closely resembles the seed data.
# MAGIC Naive should show near-zero off-diagonal correlations — because we
# MAGIC sampled each column independently.

# COMMAND ----------

numeric_rating_cols = [
    "driver_age", "vehicle_age", "vehicle_group",
    "ncd_years", "annual_mileage", "exposure", "claim_count",
]

def compute_spearman_matrix(df, cols):
    n = len(cols)
    mat = np.zeros((n, n))
    arrays = {c: df[c].to_numpy().astype(float) for c in cols}
    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(arrays[cols[i]], arrays[cols[j]])
            mat[i, j] = rho
    return mat

mat_seed = compute_spearman_matrix(seed_df, numeric_rating_cols)
mat_vine = compute_spearman_matrix(synthetic_df, numeric_rating_cols)
mat_naive = compute_spearman_matrix(naive_df, numeric_rating_cols)

col_labels = ["age", "veh_age", "veh_grp", "ncd", "mileage", "exposure", "claims"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin, vmax = -0.7, 0.7

for ax, mat, title in zip(
    axes,
    [mat_seed, mat_vine, mat_naive],
    ["Seed (Real)", "Vine Copula (Synthetic)", "Naive Independent (Synthetic)"],
):
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=col_labels,
        linewidths=0.5,
        square=True,
        cbar=ax is axes[2],
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

fig.suptitle(
    "Spearman Correlation Matrices: Seed vs Synthetic vs Naive",
    fontsize=14, y=1.02,
)
plt.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)

displayHTML(f"""
<div style='margin: 16px 0;'>
  <img src='data:image/png;base64,{img_b64}' style='max-width:100%;border:1px solid #ddd;border-radius:4px;'/>
  <p style='font-size:12px;color:#666;margin-top:6px;'>
    The vine copula (centre) closely replicates the correlation structure of the seed data (left).
    Independent sampling (right) correctly preserves marginals but destroys all relationships
    between columns — off-diagonal cells collapse toward zero.
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Marginal Distribution Comparison Plots
# MAGIC
# MAGIC For continuous and integer columns we compare the empirical CDFs (seed vs
# MAGIC vine vs naive). ECDFs are more honest than histograms: any divergence in
# MAGIC the tails is visible without bin-width choices masking it.

# COMMAND ----------

plot_cols = [
    ("driver_age", "Driver Age", "years"),
    ("ncd_years", "NCD Years", "years"),
    ("vehicle_group", "Vehicle Group", "ABI group"),
    ("annual_mileage", "Annual Mileage", "miles"),
    ("exposure", "Exposure", "policy years"),
    ("claim_amount", "Claim Amount (non-zero only)", "GBP"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (col, label, unit) in zip(axes, plot_cols):
    if col == "claim_amount":
        r = seed_df.filter(pl.col(col) > 0)[col].to_numpy()
        v = synthetic_df.filter(pl.col(col) > 0)[col].to_numpy()
        n_arr = naive_df.filter(pl.col(col) > 0)[col].to_numpy()
    else:
        r = seed_df[col].to_numpy().astype(float)
        v = synthetic_df[col].to_numpy().astype(float)
        n_arr = naive_df[col].to_numpy().astype(float)

    # ECDF
    for arr, lbl, color, lw, ls in [
        (r, "Seed (real)", "#2c3e50", 2.5, "-"),
        (v, "Vine copula", "#27ae60", 1.8, "--"),
        (n_arr, "Naive", "#e74c3c", 1.5, ":"),
    ]:
        sorted_arr = np.sort(arr)
        y = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, y, label=lbl, color=color, linewidth=lw, linestyle=ls, alpha=0.85)

    ks_vine, _ = stats.ks_2samp(r, v)
    ks_naive, _ = stats.ks_2samp(r, n_arr)

    ax.set_xlabel(unit, fontsize=10)
    ax.set_ylabel("ECDF", fontsize=10)
    ax.set_title(f"{label}\nKS: vine={ks_vine:.3f}  naive={ks_naive:.3f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if col == "claim_amount":
        ax.set_xscale("log")

plt.suptitle(
    "Empirical CDFs: Seed vs Vine Copula vs Naive Independent Sampling",
    fontsize=13, y=1.01,
)
plt.tight_layout()

buf2 = io.BytesIO()
fig.savefig(buf2, format="png", dpi=110, bbox_inches="tight")
buf2.seek(0)
img2_b64 = base64.b64encode(buf2.read()).decode()
plt.close(fig)

displayHTML(f"""
<div style='margin: 16px 0;'>
  <img src='data:image/png;base64,{img2_b64}' style='max-width:100%;border:1px solid #ddd;border-radius:4px;'/>
  <p style='font-size:12px;color:#666;margin-top:6px;'>
    ECDFs for six columns. Both vine and naive preserve marginals well (by design for naive —
    it samples from the empirical marginal directly). The vine copula's advantage is the
    dependence structure shown in the correlation heatmaps, not the marginals.
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. The Structural Failure of Naive: NCD vs Driver Age
# MAGIC
# MAGIC This scatter shows the most obvious actuarial problem with naive sampling.
# MAGIC In the real data, young drivers cannot have high NCD — they haven't been
# MAGIC driving long enough. Naive sampling produces 19-year-olds with 20 years of
# MAGIC NCD. That's not just unrealistic; a model trained on it will learn the
# MAGIC wrong NCD relativities for young drivers.
# MAGIC
# MAGIC The vine copula preserves the bounded joint region.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, df, title, color in [
    (axes[0], seed_df, "Seed (Real)", "#2c3e50"),
    (axes[1], synthetic_df, "Vine Copula", "#27ae60"),
    (axes[2], naive_df, "Naive Independent", "#e74c3c"),
]:
    ages = df["driver_age"].to_numpy()
    ncds = df["ncd_years"].to_numpy()

    # Scatter a sample for clarity
    idx = np.random.default_rng(0).choice(len(ages), size=min(3000, len(ages)), replace=False)
    ax.scatter(ages[idx], ncds[idx], alpha=0.15, s=4, color=color)

    # Theoretical upper bound: NCD = min(age - 17, 25)
    age_range = np.linspace(17, 82, 200)
    ax.plot(age_range, np.minimum(age_range - 17, 25), "k--", linewidth=1.5,
            label="max possible NCD")

    ax.set_xlabel("Driver Age", fontsize=10)
    ax.set_ylabel("NCD Years", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(15, 85)
    ax.set_ylim(-1, 28)
    ax.grid(True, alpha=0.3)

# Annotate naive failure
axes[2].annotate(
    "Physically impossible\n(young driver, high NCD)",
    xy=(22, 18), xytext=(35, 22),
    arrowprops=dict(arrowstyle="->", color="#b22222"),
    fontsize=9, color="#b22222",
)

plt.suptitle("NCD Years vs Driver Age: Joint Distribution Preservation", fontsize=13, y=1.02)
plt.tight_layout()

buf3 = io.BytesIO()
fig.savefig(buf3, format="png", dpi=110, bbox_inches="tight")
buf3.seek(0)
img3_b64 = base64.b64encode(buf3.read()).decode()
plt.close(fig)

displayHTML(f"""
<div style='margin: 16px 0;'>
  <img src='data:image/png;base64,{img3_b64}' style='max-width:100%;border:1px solid #ddd;border-radius:4px;'/>
  <p style='font-size:12px;color:#666;margin-top:6px;'>
    The dashed line shows the physical upper bound (NCD cannot exceed driving years).
    The vine copula (centre) respects this constraint. The naive approach (right) scatters
    policies across the entire grid including physically impossible combinations.
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Exposure / Frequency Structure Validation
# MAGIC
# MAGIC The actuarial relationship we must preserve: annualised claim frequency
# MAGIC should be stable regardless of exposure bucket. A portfolio of short-term
# MAGIC policies should show the same claims per policy year as full-year policies.
# MAGIC
# MAGIC This fails if frequency generation ignores the exposure offset.

# COMMAND ----------

def exposure_freq_table(df, label):
    return (
        df.with_columns(
            pl.when(pl.col("exposure") < 0.25).then(pl.lit("<0.25"))
             .when(pl.col("exposure") < 0.50).then(pl.lit("0.25-0.50"))
             .when(pl.col("exposure") < 0.75).then(pl.lit("0.50-0.75"))
             .otherwise(pl.lit("0.75-1.00"))
             .alias("exp_bucket")
        )
        .group_by("exp_bucket")
        .agg([
            pl.len().alias("policies"),
            pl.col("claim_count").sum().alias("claims"),
            pl.col("exposure").sum().alias("total_exposure"),
        ])
        .with_columns(
            (pl.col("claims") / pl.col("total_exposure")).alias("freq_per_year")
        )
        .sort("exp_bucket")
        .with_columns(pl.lit(label).alias("dataset"))
    )

tbl_seed = exposure_freq_table(seed_df, "Seed")
tbl_vine = exposure_freq_table(synthetic_df, "Vine")
tbl_naive = exposure_freq_table(naive_df, "Naive")

combined_tbl = pl.concat([tbl_seed, tbl_vine, tbl_naive])
print("Annualised frequency by exposure bucket:")
print(combined_tbl.select(["dataset", "exp_bucket", "policies", "claims", "freq_per_year"]))

# HTML version
rows_exp = []
for r in combined_tbl.iter_rows(named=True):
    rows_exp.append(f"""
    <tr>
      <td>{r['dataset']}</td>
      <td style='font-family:monospace'>{r['exp_bucket']}</td>
      <td style='text-align:right'>{r['policies']:,}</td>
      <td style='text-align:right'>{r['claims']:,}</td>
      <td style='text-align:right'>{r['freq_per_year']:.4f}</td>
    </tr>""")

displayHTML(f"""
<style>
  .exp-table {{ border-collapse:collapse; font-size:13px; font-family:sans-serif; }}
  .exp-table th {{ background:#34495e; color:white; padding:8px 14px; text-align:left; }}
  .exp-table td {{ padding:6px 14px; border-bottom:1px solid #ddd; }}
  .exp-table tr:nth-child(even) {{ background:#f4f4f4; }}
  .exp-table caption {{ font-weight:bold; font-size:14px; padding:8px; text-align:left; color:#2c3e50; }}
</style>
<table class='exp-table'>
  <caption>Annualised Frequency by Exposure Bucket</caption>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Exposure Bucket</th>
      <th>Policies</th>
      <th>Claims</th>
      <th>Freq/Year</th>
    </tr>
  </thead>
  <tbody>{''.join(rows_exp)}</tbody>
</table>
<p style='font-size:12px;color:#555;margin-top:6px;'>
  Freq/Year should be stable across exposure buckets for both seed and vine data.
  The vine synthesiser regenerates counts via Poisson(lambda * exposure), preserving this.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Tail Dependence: Where Gaussian Copulas Fail
# MAGIC
# MAGIC A Gaussian copula has zero tail dependence by construction: the probability
# MAGIC of two variables being simultaneously extreme goes to zero as you approach
# MAGIC the tails. For insurance data this is wrong — young + high-vehicle-group +
# MAGIC zero-NCD is a cluster of extreme risk that co-occurs more often than
# MAGIC independence would predict.
# MAGIC
# MAGIC We demonstrate this by comparing the joint tail behaviour of the vine
# MAGIC copula (which can capture tail dependence) against a Gaussian copula fit.

# COMMAND ----------

synth_gaussian = InsuranceSynthesizer(
    method="gaussian",
    random_state=SEED,
)
synth_gaussian.fit(
    seed_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
)
gaussian_df = synth_gaussian.generate(
    50_000,
    constraints=schema["constraints"],
    max_resample_attempts=10,
)

print("Gaussian copula synthesiser fitted and 50,000 policies generated.")

# Check tail co-occurrence: proportion of policies in top decile of both
# vehicle_group AND claim_count
def tail_co_occurrence(df, col_a, col_b, pct=0.90):
    """Fraction of rows where both columns exceed their pct-th quantile."""
    a = df[col_a].to_numpy().astype(float)
    b = df[col_b].to_numpy().astype(float)
    q_a = np.quantile(a, pct)
    q_b = np.quantile(b, pct)
    return float(np.mean((a >= q_a) & (b >= q_b)))

for label, df in [
    ("Seed (real)", seed_df),
    ("Vine copula", synthetic_df),
    ("Gaussian copula", gaussian_df),
    ("Naive", naive_df),
]:
    tco_vg_claims = tail_co_occurrence(df, "vehicle_group", "claim_count")
    tco_age_claims = tail_co_occurrence(
        df.with_columns((pl.col("driver_age") < 25).cast(pl.Int8).alias("young")),
        "young", "claim_count", pct=0.5
    )
    print(f"{label:25s}  tail-co-occur(veh_grp, claims @90%): {tco_vg_claims:.4f}  "
          f"  young-driver claim-rate: {tco_age_claims:.4f}")

print(f"\nExpected tail co-occurrence under independence: {0.10 * 0.10:.4f}")
print("Values above this indicate positive tail dependence — the vine copula should match seed most closely.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Correlation Pair Differences: Top Divergences
# MAGIC
# MAGIC Which column pairs show the largest Spearman correlation difference between
# MAGIC real and synthetic? This is the per-pair breakdown of the Frobenius norm.
# MAGIC For a well-fitted vine, no pair should diverge by more than ~0.05.

# COMMAND ----------

corr_vine_clean = corr_vine.select(["col_a", "col_b", "spearman_real", "spearman_synthetic", "delta"])
corr_naive_clean = corr_naive.select(["col_a", "col_b", "spearman_real", "spearman_synthetic", "delta"])

joined = corr_vine_clean.rename({
    "spearman_synthetic": "vine_rho",
    "delta": "vine_delta",
}).join(
    corr_naive_clean.select(["col_a", "col_b", "spearman_synthetic", "delta"]).rename({
        "spearman_synthetic": "naive_rho",
        "delta": "naive_delta",
    }),
    on=["col_a", "col_b"],
    how="inner",
).sort("vine_delta", descending=True)

rows_pair = []
for r in joined.head(12).iter_rows(named=True):
    vine_color = "#1a7a1a" if r["vine_delta"] < 0.05 else ("#c07700" if r["vine_delta"] < 0.10 else "#b22222")
    naive_color = "#1a7a1a" if r["naive_delta"] < 0.05 else ("#c07700" if r["naive_delta"] < 0.10 else "#b22222")
    rows_pair.append(f"""
    <tr>
      <td style='font-family:monospace'>{r['col_a']}</td>
      <td style='font-family:monospace'>{r['col_b']}</td>
      <td style='text-align:right'>{r['spearman_real']:+.3f}</td>
      <td style='text-align:right'>{r['vine_rho']:+.3f}</td>
      <td style='text-align:right;color:{vine_color}'>{r['vine_delta']:.3f}</td>
      <td style='text-align:right'>{r['naive_rho']:+.3f}</td>
      <td style='text-align:right;color:{naive_color}'>{r['naive_delta']:.3f}</td>
    </tr>""")

displayHTML(f"""
<style>
  .pair-table {{ border-collapse:collapse; font-size:13px; font-family:sans-serif; }}
  .pair-table th {{ background:#2c3e50; color:white; padding:8px 12px; text-align:left; }}
  .pair-table td {{ padding:6px 12px; border-bottom:1px solid #ddd; }}
  .pair-table tr:nth-child(even) {{ background:#f9f9f9; }}
  .pair-table caption {{ font-weight:bold; font-size:14px; padding:8px; text-align:left; color:#2c3e50; }}
</style>
<table class='pair-table'>
  <caption>Pairwise Spearman Correlation Differences (top 12 by vine delta)</caption>
  <thead>
    <tr>
      <th>Col A</th><th>Col B</th>
      <th>Rho (Real)</th>
      <th>Rho (Vine)</th><th>Delta (Vine)</th>
      <th>Rho (Naive)</th><th>Delta (Naive)</th>
    </tr>
  </thead>
  <tbody>{''.join(rows_pair)}</tbody>
  <tfoot>
    <tr>
      <td colspan='7' style='padding:8px;font-style:italic;color:#555;'>
        Frobenius norm: Vine = {frob_vine:.4f}, Naive = {frob_naive:.4f}.
        Colour: green &lt; 0.05 | amber 0.05-0.10 | red &gt; 0.10
      </td>
    </tr>
  </tfoot>
</table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Fitted Marginals Audit
# MAGIC
# MAGIC Quick audit of which distribution family was selected for each column.
# MAGIC This confirms the synthesiser is using sensible families — not forcing
# MAGIC a normal distribution onto a log-normal severity column, for instance.

# COMMAND ----------

marginal_rows = []
for col, m in synth._fitted_marginals.items():
    aic_str = f"{m.aic:.1f}" if (hasattr(m, 'aic') and np.isfinite(m.aic)) else "n/a"
    marginal_rows.append({
        "column": col,
        "kind": m.kind,
        "family": m.family_name(),
        "aic": aic_str,
    })

displayHTML(f"""
<style>
  .marg-table {{ border-collapse:collapse; font-size:13px; font-family:sans-serif; }}
  .marg-table th {{ background:#16a085; color:white; padding:8px 12px; text-align:left; }}
  .marg-table td {{ padding:6px 12px; border-bottom:1px solid #ddd; font-family:monospace; }}
  .marg-table tr:nth-child(even) {{ background:#f0faf8; }}
  .marg-table caption {{ font-weight:bold; font-size:14px; padding:8px; text-align:left; color:#16a085; }}
</style>
<table class='marg-table'>
  <caption>Fitted Marginal Distributions (selected by AIC)</caption>
  <thead>
    <tr><th>Column</th><th>Kind</th><th>Family</th><th>AIC</th></tr>
  </thead>
  <tbody>
    {''.join(f"<tr><td>{r['column']}</td><td>{r['kind']}</td><td>{r['family']}</td><td>{r['aic']}</td></tr>" for r in marginal_rows)}
  </tbody>
</table>
<p style='font-size:12px;color:#555;margin-top:6px;'>
  The synthesiser fits Gamma, LogNormal, Poisson, NegBin, Beta, and Normal to each column
  and selects by AIC. Severity columns should be Gamma or LogNormal. Count columns should
  be Poisson or NegBin. Exposure should be Beta.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Benchmark Summary
# MAGIC
# MAGIC A single-screen summary of how vine copula synthesis compares to naive
# MAGIC independent sampling across the key fidelity dimensions.

# COMMAND ----------

tvar_vine = report_vine.tvar_ratio("claim_count", percentile=0.99)
tvar_naive = report_naive.tvar_ratio("claim_count", percentile=0.99)

freq_seed = float(seed_df["claim_count"].sum() / seed_df["exposure"].sum())
freq_vine = float(synthetic_df["claim_count"].sum() / synthetic_df["exposure"].sum())
freq_naive = float(naive_df["claim_count"].sum() / naive_df["exposure"].sum())

ew_ks_vine = report_vine.exposure_weighted_ks("driver_age")
ew_ks_naive = report_naive.exposure_weighted_ks("driver_age")

def _pct_err(a, b):
    return abs(a - b) / max(abs(b), 1e-9) * 100

summary_rows = [
    ("Frobenius norm (Spearman matrix)", f"{frob_vine:.4f}", f"{frob_naive:.4f}", "Lower = better dependence"),
    ("TVaR ratio @ 99th pct (claims)", f"{tvar_vine:.4f}", f"{tvar_naive:.4f}", "Target: 1.0"),
    ("Annualised frequency (seed)", f"{freq_seed:.4f}", f"{freq_seed:.4f}", "Reference"),
    ("Annualised frequency (synthetic)", f"{freq_vine:.4f}", f"{freq_naive:.4f}", f"Seed: {freq_seed:.4f}"),
    ("Freq error %", f"{_pct_err(freq_vine, freq_seed):.2f}%", f"{_pct_err(freq_naive, freq_seed):.2f}%", "Lower = better"),
    ("Exposure-weighted KS (driver_age)", f"{ew_ks_vine:.4f}", f"{ew_ks_naive:.4f}", "Lower = better"),
    ("Rho(NCD, claims) — seed", f"{rho_ncd_freq:+.4f}", f"{rho_ncd_freq:+.4f}", "Reference"),
    ("Rho(NCD, claims) — synthetic", "", "", ""),
]

# Fill in the synthetic rho values
rho_ncd_vine, _ = stats.spearmanr(synthetic_df["ncd_years"].to_numpy(), synthetic_df["claim_count"].to_numpy())
rho_ncd_naive, _ = stats.spearmanr(naive_df["ncd_years"].to_numpy(), naive_df["claim_count"].to_numpy())
summary_rows[-1] = ("Rho(NCD, claims) — synthetic", f"{rho_ncd_vine:+.4f}", f"{rho_ncd_naive:+.4f}",
                    f"Target: {rho_ncd_freq:+.4f}")

rho_vg_vine, _ = stats.spearmanr(synthetic_df["vehicle_group"].to_numpy(), synthetic_df["claim_count"].to_numpy())
rho_vg_naive, _ = stats.spearmanr(naive_df["vehicle_group"].to_numpy(), naive_df["claim_count"].to_numpy())
summary_rows.append(("Rho(vehicle_group, claims)", f"{rho_vg_vine:+.4f}", f"{rho_vg_naive:+.4f}",
                     f"Target: {rho_vg_freq:+.4f}"))

rho_age_ncd_vine, _ = stats.spearmanr(synthetic_df["driver_age"].to_numpy(), synthetic_df["ncd_years"].to_numpy())
rho_age_ncd_naive, _ = stats.spearmanr(naive_df["driver_age"].to_numpy(), naive_df["ncd_years"].to_numpy())
summary_rows.append(("Rho(driver_age, ncd_years)", f"{rho_age_ncd_vine:+.4f}", f"{rho_age_ncd_naive:+.4f}",
                     f"Target: {rho_age_ncd:+.4f}"))

sr_html = ""
for metric, vine_val, naive_val, note in summary_rows:
    sr_html += f"<tr><td>{metric}</td><td style='text-align:right;font-weight:bold;color:#27ae60'>{vine_val}</td><td style='text-align:right;color:#c0392b'>{naive_val}</td><td style='color:#777;font-style:italic'>{note}</td></tr>"

displayHTML(f"""
<style>
  .sum-table {{ border-collapse:collapse; font-size:13px; font-family:sans-serif; min-width:700px; }}
  .sum-table th {{ background:#2c3e50; color:white; padding:10px 14px; text-align:left; }}
  .sum-table td {{ padding:7px 14px; border-bottom:1px solid #e0e0e0; }}
  .sum-table tr:nth-child(even) {{ background:#f7f7f7; }}
  .sum-table caption {{ font-weight:bold; font-size:16px; padding:10px 0; text-align:left; color:#2c3e50; }}
</style>
<table class='sum-table'>
  <caption>Benchmark Summary: Vine Copula vs Naive Independent Sampling</caption>
  <thead>
    <tr><th>Metric</th><th>Vine Copula</th><th>Naive</th><th>Notes</th></tr>
  </thead>
  <tbody>{sr_html}</tbody>
</table>
<p style='font-size:12px;color:#555;margin-top:8px;'>
  Vine copula preserves marginals <em>and</em> dependence structure.
  Naive preserves marginals only. For any model trained on this data —
  a frequency GLM, a severity GBM, a Tweedie — the missing correlations
  in the naive dataset will produce biased relativities.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. When to Use Which Approach
# MAGIC
# MAGIC **Use vine copula synthesis when:**
# MAGIC - The downstream task involves a model that uses multiple correlated features
# MAGIC - You need to share data with vendors or collaborators (privacy use case)
# MAGIC - You're building a benchmark where relative model performance matters
# MAGIC - Tail events matter (vine handles tail dependence; Gaussian copula does not)
# MAGIC
# MAGIC **Use Gaussian copula when:**
# MAGIC - You have very limited compute and many dimensions (vine fitting is O(d^2))
# MAGIC - Tail dependence is genuinely not a concern for your use case
# MAGIC - You just need more rows from a roughly-right distribution
# MAGIC
# MAGIC **Never use independent sampling when:**
# MAGIC - Training any model with multiple features — the missing correlations
# MAGIC   will corrupt all interaction effects
# MAGIC - The synthetic data will be used for pricing relativities
# MAGIC - Physical constraints between columns matter (NCD vs age, mileage vs vehicle type)
# MAGIC
# MAGIC **Practical thresholds for fit quality:**
# MAGIC - KS statistic < 0.05 per column: excellent
# MAGIC - Frobenius norm < 2.0: dependence structure well-preserved
# MAGIC - TVaR ratio in [0.80, 1.20]: tail risk not materially distorted
# MAGIC - Annualised frequency ratio within 5% of 1.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Export Synthetic Portfolio
# MAGIC
# MAGIC Write the synthetic portfolio to the Databricks Unity Catalog or a volume
# MAGIC path for downstream use. Using Delta format for compatibility with Spark
# MAGIC workflows in the rest of the platform.

# COMMAND ----------

# Uncomment and configure the path for your workspace:
#
# OUTPUT_PATH = "/Volumes/your_catalog/your_schema/your_volume/synthetic_motor_50k.parquet"
# synthetic_df.write_parquet(OUTPUT_PATH)
# print(f"Written: {OUTPUT_PATH}")
#
# Or as a Delta table:
# spark_df = spark.createDataFrame(synthetic_df.to_pandas())
# spark_df.write.format("delta").mode("overwrite").saveAsTable("your_catalog.your_schema.synthetic_motor")

print("Export cells above — configure OUTPUT_PATH for your workspace.")
print(f"\nDataFrame ready for export: {len(synthetic_df):,} rows x {len(synthetic_df.columns)} columns")
print(f"Schema:")
for col_name, dtype in zip(synthetic_df.columns, synthetic_df.dtypes):
    print(f"  {col_name:<25} {dtype}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was demonstrated:**
# MAGIC
# MAGIC - `InsuranceSynthesizer` with vine copulas preserves the dependence structure
# MAGIC   of a UK motor portfolio — marginals, pairwise correlations, and joint tail
# MAGIC   behaviour
# MAGIC - The fidelity report gives quantitative evidence: KS statistics, Wasserstein
# MAGIC   distances, Frobenius norm on the Spearman matrix, TVaR ratio
# MAGIC - Independent (naive) sampling correctly preserves marginals but destroys all
# MAGIC   relationships between columns — physically impossible combinations appear,
# MAGIC   and any model trained on the data learns biased relativities
# MAGIC - The NCD/age joint distribution is the clearest illustration of this failure:
# MAGIC   naive synthesis produces 19-year-olds with 20 years of NCD
# MAGIC - Vine copulas handle this correctly without explicit constraint specification,
# MAGIC   because the dependence structure is learnt from data
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Try with your own real portfolio: replace `seed_df` with your data
# MAGIC - Run `report.tstr_score()` for the full Train-on-Synthetic-Test-on-Real Gini gap
# MAGIC - Use `uk_employer_liability_schema()` for commercial lines data
# MAGIC - Adjust `trunc_lvl` for high-dimensional portfolios (>15 columns)
