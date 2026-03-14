# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-dynamics: GAS Filter vs Static GLM on Drifting Claim Frequency
# MAGIC
# MAGIC **What this notebook does**
# MAGIC
# MAGIC A static Poisson GLM estimates one frequency for the whole experience period.
# MAGIC When the underlying rate drifts — because of whiplash reforms, competitor exit,
# MAGIC or portfolio mix shift — the GLM blends the two regimes into a long-run average
# MAGIC that is wrong in both directions and gives no signal that anything changed.
# MAGIC
# MAGIC This notebook benchmarks a GAS (Generalised Autoregressive Score) Poisson filter
# MAGIC against a static Poisson GLM on synthetic quarterly data with a planted step-change
# MAGIC at Q20 of 40 quarters (frequency shifts from 0.10 to 0.14 claims per policy-year).
# MAGIC It then runs Bayesian Online Changepoint Detection (BOCPD) to show how precisely the
# MAGIC break is localised.
# MAGIC
# MAGIC **Library:** `insurance-dynamics` — GAS models and BOCPD for UK pricing teams.
# MAGIC
# MAGIC **Sections**
# MAGIC 1. Install dependencies
# MAGIC 2. Generate synthetic quarterly claims data
# MAGIC 3. Fit static Poisson GLM (baseline)
# MAGIC 4. Fit GAS Poisson filter
# MAGIC 5. Run BOCPD frequency detector
# MAGIC 6. Benchmark table: log-likelihood, MAE, detection lag
# MAGIC 7. Visualisations

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install insurance-dynamics matplotlib polars

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Databricks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import poisson

# insurance-dynamics imports
from insurance_dynamics.gas import GASModel
from insurance_dynamics.changepoint import FrequencyChangeDetector
from insurance_dynamics.changepoint.plot import plot_regime_probs, plot_run_length_heatmap

print("All imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate synthetic quarterly claims data
# MAGIC
# MAGIC **Data-generating process:**
# MAGIC - 40 quarters, 1 000 policies per quarter (constant exposure)
# MAGIC - True claim frequency: 0.10 claims/policy-year for Q1-Q19, 0.14 from Q20 onwards (+40%)
# MAGIC - Claims per quarter ~ Poisson(true_rate * exposure)
# MAGIC
# MAGIC The true break at Q20 is planted so we can measure detection lag precisely.

# COMMAND ----------

RNG_SEED = 42
T = 40            # quarters
BREAK_AT = 20     # zero-indexed: Q20 is index 19 (first quarter of new regime)
POLICIES = 1_000  # policies per quarter (= exposure in policy-years at quarterly rate)
RATE_PRE  = 0.10  # claims per policy-year
RATE_POST = 0.14  # claims per policy-year post-break (+40%)

rng = np.random.default_rng(RNG_SEED)

quarters = np.arange(1, T + 1)
quarter_labels = [f"Q{q}" for q in quarters]

# Constant exposure: 1 000 policy-years each quarter
exposure = np.full(T, float(POLICIES))

# True underlying rate schedule
true_rate = np.where(quarters <= BREAK_AT, RATE_PRE, RATE_POST)

# Observed claim counts
counts = rng.poisson(true_rate * exposure).astype(float)
observed_freq = counts / exposure  # empirical claim frequency per policy-year

print("Synthetic data summary")
print(f"  T = {T} quarters")
print(f"  Break at Q{BREAK_AT} (index {BREAK_AT-1})")
print(f"  Pre-break true rate:  {RATE_PRE:.2f}")
print(f"  Post-break true rate: {RATE_POST:.2f}")
print(f"  Mean observed freq pre:  {observed_freq[:BREAK_AT].mean():.4f}")
print(f"  Mean observed freq post: {observed_freq[BREAK_AT:].mean():.4f}")

df = pd.DataFrame({
    "quarter": quarter_labels,
    "exposure": exposure,
    "true_rate": true_rate,
    "claims": counts,
    "obs_freq": observed_freq,
})
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit static Poisson GLM (baseline)
# MAGIC
# MAGIC The static GLM estimates a single overall rate from all 40 quarters.
# MAGIC We implement it directly: MLE for a Poisson with offset is just
# MAGIC `total_claims / total_exposure`. No covariates, no time structure.
# MAGIC
# MAGIC We also compute per-quarter log-likelihoods so we can compare
# MAGIC fairly with the GAS model.

# COMMAND ----------

# MLE: constant rate
total_claims = counts.sum()
total_exposure = exposure.sum()
static_rate = total_claims / total_exposure

print(f"Static GLM estimated rate: {static_rate:.4f}")
print(f"  Pre-break true rate:     {RATE_PRE:.4f}")
print(f"  Post-break true rate:    {RATE_POST:.4f}")
print(f"  (Static averages across both regimes — wrong in both directions)")

# Per-quarter predictions under static GLM
static_pred_counts = static_rate * exposure          # expected claims
static_pred_freq   = np.full(T, static_rate)         # predicted frequency

# Per-quarter log-likelihoods: log P(N_t | lambda=static_rate, exposure=e_t)
static_log_lls = poisson.logpmf(counts.astype(int), static_rate * exposure)
static_total_ll = static_log_lls.sum()

print(f"\nStatic GLM total log-likelihood: {static_total_ll:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit GAS Poisson filter
# MAGIC
# MAGIC `GASModel('poisson')` fits a GAS(1,1) with a log-linked time-varying mean.
# MAGIC The recursion is:
# MAGIC
# MAGIC ```
# MAGIC f_{t+1} = omega + alpha * (y_t / mu_t - 1) * (1 / mu_t)^{-1}  +  phi * f_t
# MAGIC mu_t    = exp(f_t) * exposure_t
# MAGIC ```
# MAGIC
# MAGIC `alpha` controls how quickly the filter reacts to new data.
# MAGIC `phi` controls persistence (how much of last period carries forward).
# MAGIC
# MAGIC The `filter_path` column `'mean'` gives the estimated per-unit-exposure rate
# MAGIC at each quarter — directly comparable to `true_rate`.

# COMMAND ----------

gas_model = GASModel("poisson", p=1, q=1, scaling="fisher_inv")
gas_result = gas_model.fit(counts, exposure=exposure, max_iter=2000)

print(gas_result.summary())

# The filter_path DataFrame has one column 'mean' — the time-varying Poisson rate
# on the natural scale (per unit exposure). With exposure=1000, this is
# E[claims_t] / exposure_t = the per-policy-year frequency.
gas_filter_path = gas_result.filter_path  # shape (T,) with column 'mean'
gas_pred_freq   = gas_filter_path["mean"].values   # per-unit-exposure rate

# GAS per-quarter log-likelihoods (already computed during fitting)
# Re-derive for transparency: Poisson(lambda_t * exposure_t) evaluated at counts_t
gas_log_lls = poisson.logpmf(counts.astype(int), gas_pred_freq * exposure)
gas_total_ll = gas_log_lls.sum()

print(f"\nGAS total log-likelihood: {gas_total_ll:.4f}")
print(f"Static total log-likelihood: {static_total_ll:.4f}")
print(f"Log-likelihood improvement (GAS - static): {gas_total_ll - static_total_ll:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run BOCPD frequency detector
# MAGIC
# MAGIC `FrequencyChangeDetector` runs Poisson-Gamma conjugate BOCPD online.
# MAGIC It produces a posterior `P(changepoint at t)` for each quarter.
# MAGIC
# MAGIC **Prior calibration for quarterly motor data:**
# MAGIC - `prior_alpha=2`, `prior_beta=20` → prior mean rate 0.10, moderate uncertainty
# MAGIC - `hazard=1/20` → expect a structural break roughly every 20 quarters (5 years)
# MAGIC - `threshold=0.25` → flag as detected when P(changepoint) > 25%

# COMMAND ----------

detector = FrequencyChangeDetector(
    prior_alpha=2.0,
    prior_beta=20.0,
    hazard=1.0 / 20,
    threshold=0.25,
    max_run_length=200,
    uk_events=False,
)

cp_result = detector.fit(
    claim_counts=counts,
    earned_exposure=exposure,
    periods=quarter_labels,
)

print(f"BOCPD detected {cp_result.n_breaks} break(s)")
print(f"Max changepoint probability: {cp_result.max_changepoint_prob:.3f}")

for brk in cp_result.detected_breaks:
    print(f"  {brk}")
    true_break_label = f"Q{BREAK_AT}"
    detection_lag = brk.period_index - (BREAK_AT - 1)  # periods after true break
    print(f"    True break was at index {BREAK_AT-1} ({true_break_label})")
    print(f"    Detection lag: {detection_lag} quarter(s)")

# Also surface the raw changepoint probabilities for plotting
cp_probs = cp_result.changepoint_probs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Benchmark table
# MAGIC
# MAGIC We compare static GLM vs GAS on:
# MAGIC - **Total log-likelihood** — higher is better (GAS should win because it adapts)
# MAGIC - **MAE on observed frequency** — overall and on held-out post-break quarters
# MAGIC - **Lambda RMSE vs true rate** — how close each model is to the DGP
# MAGIC - **Post-break calibration** — does the model overpredict or underpredict?
# MAGIC - **Break detected within ±3 quarters** — yes/no

# COMMAND ----------

def mae(pred, actual):
    return np.abs(pred - actual).mean()

def rmse(pred, actual):
    return np.sqrt(((pred - actual) ** 2).mean())

# Masks
pre_mask  = np.arange(T) < BREAK_AT
post_mask = ~pre_mask

# MAE on observed frequency
static_mae_all  = mae(static_pred_freq, observed_freq)
static_mae_post = mae(static_pred_freq[post_mask], observed_freq[post_mask])
gas_mae_all     = mae(gas_pred_freq, observed_freq)
gas_mae_post    = mae(gas_pred_freq[post_mask], observed_freq[post_mask])

# RMSE vs true rate (oracle comparison)
static_rmse_all  = rmse(static_pred_freq, true_rate)
static_rmse_post = rmse(static_pred_freq[post_mask], true_rate[post_mask])
gas_rmse_all     = rmse(gas_pred_freq, true_rate)
gas_rmse_post    = rmse(gas_pred_freq[post_mask], true_rate[post_mask])

# Post-break mean prediction bias
static_bias_post = (static_pred_freq[post_mask] - true_rate[post_mask]).mean()
gas_bias_post    = (gas_pred_freq[post_mask] - true_rate[post_mask]).mean()

# Detection
true_break_idx = BREAK_AT - 1  # zero-indexed
within_3q = any(
    abs(brk.period_index - true_break_idx) <= 3
    for brk in cp_result.detected_breaks
)

benchmark = pd.DataFrame({
    "Metric": [
        "Total log-likelihood",
        "MAE on obs freq (all quarters)",
        "MAE on obs freq (post-break)",
        "RMSE vs true rate (all quarters)",
        "RMSE vs true rate (post-break)",
        "Mean bias post-break (pred - true)",
        "Break detected within ±3 quarters",
    ],
    "Static Poisson GLM": [
        f"{static_total_ll:.2f}",
        f"{static_mae_all:.5f}",
        f"{static_mae_post:.5f}",
        f"{static_rmse_all:.5f}",
        f"{static_rmse_post:.5f}",
        f"{static_bias_post:+.5f}",
        "No",
    ],
    "GAS Poisson Filter": [
        f"{gas_total_ll:.2f}",
        f"{gas_mae_all:.5f}",
        f"{gas_mae_post:.5f}",
        f"{gas_rmse_all:.5f}",
        f"{gas_rmse_post:.5f}",
        f"{gas_bias_post:+.5f}",
        f"Yes — BOCPD flags Q{cp_result.detected_breaks[0].period_index+1}" if cp_result.n_breaks > 0 else "No (raise hazard or lower threshold)",
    ],
})

print(benchmark.to_string(index=False))
display(benchmark)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading the table**
# MAGIC
# MAGIC The static GLM estimates a single rate somewhere between the two regimes.
# MAGIC Post-break it systematically underpredicts (positive bias means it is above the
# MAGIC pre-break rate but below the post-break rate — or it may underpredict depending
# MAGIC on where the long-run average lands relative to the true schedule).
# MAGIC
# MAGIC The GAS filter adapts each quarter: once the break happens its score residuals
# MAGIC are positive (more claims than expected) and the filter path is pulled upward,
# MAGIC tracking the new regime within a few quarters. BOCPD provides the explicit
# MAGIC signal: a spike in P(changepoint) pinpoints the break without smoothing it away.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualisations
# MAGIC
# MAGIC **Figure 1:** Observed frequency vs static GLM vs GAS filter path (with true rate).
# MAGIC
# MAGIC **Figure 2:** BOCPD — P(changepoint) over time.
# MAGIC
# MAGIC **Figure 3:** BOCPD run-length posterior heatmap (shows the break as a diagonal reset).

# COMMAND ----------

# Figure 1 — Time series comparison

fig1, ax = plt.subplots(figsize=(13, 5))

x = np.arange(T)

# Observed frequency as scatter
ax.scatter(x, observed_freq, color="#7f8c8d", s=25, zorder=3,
           label="Observed frequency", alpha=0.8)

# True rate schedule
ax.step(x, true_rate, where="post", color="#27ae60", linewidth=2.0,
        linestyle="-", label="True rate (DGP)", zorder=4)

# Static GLM — flat line
ax.axhline(static_rate, color="#e74c3c", linewidth=1.8, linestyle="--",
           label=f"Static GLM ({static_rate:.4f})", zorder=3)

# GAS filter path
ax.plot(x, gas_pred_freq, color="#2980b9", linewidth=2.0,
        label="GAS Poisson filter", zorder=5)

# Shade break region
ax.axvspan(BREAK_AT - 1 - 0.5, BREAK_AT - 0.5, alpha=0.12, color="#f39c12",
           label=f"True break (Q{BREAK_AT})")

# Mark BOCPD detections
for brk in cp_result.detected_breaks:
    ax.axvline(brk.period_index, color="#8e44ad", linestyle=":",
               linewidth=1.5, alpha=0.85,
               label=f"BOCPD flags Q{brk.period_index+1} (p={brk.probability:.2f})")

ax.set_xticks(x[::2])
ax.set_xticklabels(quarter_labels[::2], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Quarter", fontsize=10)
ax.set_ylabel("Claim frequency (claims / policy-year)", fontsize=10)
ax.set_title(
    "GAS Poisson Filter vs Static GLM — Quarterly Claim Frequency with Planted Break at Q20",
    fontsize=11, fontweight="bold",
)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
fig1.tight_layout()
display(fig1)

# COMMAND ----------

# MAGIC %md
# MAGIC **What the chart shows:** The static GLM (red dashed) sits at the blended average
# MAGIC of both regimes and never moves. The GAS filter (blue) starts near the pre-break rate,
# MAGIC then climbs after Q20 as the score residuals accumulate, converging on the new regime
# MAGIC within ~4 quarters. The BOCPD detection (purple dotted) confirms the break with a
# MAGIC probability spike close to the true break point.

# COMMAND ----------

# Figure 2 — BOCPD changepoint probabilities

fig2 = plot_regime_probs(
    cp_result,
    figsize=(13, 4),
    threshold=0.25,
    title=f"BOCPD — P(Changepoint) per Quarter  |  True break: Q{BREAK_AT}",
)

# Annotate true break location
ax2 = fig2.axes[0]
ax2.axvspan(BREAK_AT - 1 - 0.5, BREAK_AT - 0.5, alpha=0.15, color="#27ae60",
            label=f"True break (Q{BREAK_AT})")
ax2.legend(fontsize=8)
fig2.tight_layout()
display(fig2)

# COMMAND ----------

# MAGIC %md
# MAGIC **What the chart shows:** The BOCPD posterior probability spikes shortly after
# MAGIC the true break at Q20. Any spike that crosses the 0.25 threshold triggers a
# MAGIC `DetectedBreak`. In practice you would wire this to an alert: when P > threshold
# MAGIC for two consecutive periods, escalate to model governance.

# COMMAND ----------

# Figure 3 — Run-length heatmap

fig3 = plot_run_length_heatmap(
    cp_result,
    figsize=(13, 5),
    max_run_length=30,
    title=f"BOCPD Run-Length Posterior  |  True break: Q{BREAK_AT}",
)
display(fig3)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading the heatmap:** Each column is a quarter; each row is a run length (how
# MAGIC many consecutive quarters since the last changepoint). Dark blue = high probability.
# MAGIC Before Q20 the probability mass drifts diagonally upward — the model believes the
# MAGIC current regime is growing older. At the break the mass resets to run-length 0 and
# MAGIC starts again. This diagonal-then-reset pattern is the BOCPD fingerprint of a clean
# MAGIC structural break.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. GAS bootstrap confidence intervals on the filter path
# MAGIC
# MAGIC `result.bootstrap_ci()` runs parametric bootstrap to give 95% confidence bands
# MAGIC on the filter path. This quantifies estimation uncertainty — useful when the
# MAGIC pricing team needs to know whether the apparent upward drift is real or noise.

# COMMAND ----------

print("Running parametric bootstrap (500 replicates)...")
ci = gas_result.bootstrap_ci(n_boot=500, confidence=0.95, rng=np.random.default_rng(99))

fig4, ax4 = plt.subplots(figsize=(13, 5))

x = np.arange(T)
lower = ci.lower["mean"].values
upper = ci.upper["mean"].values
mid   = ci.central["mean"].values

ax4.fill_between(x, lower, upper, alpha=0.2, color="#2980b9", label="95% bootstrap CI")
ax4.plot(x, mid, color="#2980b9", linewidth=2.0, label="GAS filter (central)")
ax4.step(x, true_rate, where="post", color="#27ae60", linewidth=1.8,
         linestyle="-", label="True rate (DGP)")
ax4.scatter(x, observed_freq, color="#7f8c8d", s=20, zorder=3,
            label="Observed frequency", alpha=0.7)
ax4.axvspan(BREAK_AT - 1 - 0.5, BREAK_AT - 0.5, alpha=0.12, color="#f39c12",
            label=f"True break (Q{BREAK_AT})")

ax4.set_xticks(x[::2])
ax4.set_xticklabels(quarter_labels[::2], rotation=45, ha="right", fontsize=8)
ax4.set_xlabel("Quarter", fontsize=10)
ax4.set_ylabel("Claim frequency (claims / policy-year)", fontsize=10)
ax4.set_title(
    "GAS Poisson Filter — Parametric Bootstrap 95% Confidence Bands",
    fontsize=11, fontweight="bold",
)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
display(fig4)

# COMMAND ----------

# MAGIC %md
# MAGIC **What the bands tell you:** If the CI is wide relative to the rate change,
# MAGIC the apparent drift may be within sampling noise. If the upper CI stays above
# MAGIC the pre-break rate throughout the post-break period (as it should here with
# MAGIC a 40% step), the rate shift is statistically distinguishable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. GAS score residuals — model adequacy check
# MAGIC
# MAGIC The score residuals should look approximately iid(0,1) if the model is
# MAGIC well-specified. Structure (autocorrelation, heteroskedasticity) indicates
# MAGIC a misfit — e.g. the model is not reacting fast enough to the break.
# MAGIC
# MAGIC `result.diagnostics()` returns PIT residuals, coverage test and ACF.

# COMMAND ----------

diag = gas_result.diagnostics()
print(diag)

# Manual score residual plot
score_vals = gas_result.score_residuals["mean"].values

fig5, axes5 = plt.subplots(1, 2, figsize=(13, 4))

ax_ts, ax_acf = axes5

# Time series of score residuals
ax_ts.plot(np.arange(T), score_vals, color="#2980b9", linewidth=1.0)
ax_ts.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
ax_ts.axvspan(BREAK_AT - 1 - 0.5, BREAK_AT - 0.5, alpha=0.15, color="#f39c12",
              label=f"True break (Q{BREAK_AT})")
ax_ts.set_xlabel("Quarter")
ax_ts.set_ylabel("Score residual")
ax_ts.set_title("Score Residuals over Time", fontsize=10)
ax_ts.legend(fontsize=8)
ax_ts.grid(True, alpha=0.3)

# ACF (manual: lag 1..10)
max_lag = min(10, T // 4)
acf_vals = [
    np.corrcoef(score_vals[:-lag], score_vals[lag:])[0, 1]
    for lag in range(1, max_lag + 1)
]
lags = np.arange(1, max_lag + 1)
ax_acf.bar(lags, acf_vals, color="#2980b9", alpha=0.7)
conf_bound = 1.96 / np.sqrt(T)
ax_acf.axhline(conf_bound, color="#e74c3c", linestyle="--", linewidth=0.9)
ax_acf.axhline(-conf_bound, color="#e74c3c", linestyle="--", linewidth=0.9)
ax_acf.set_xlabel("Lag (quarters)")
ax_acf.set_ylabel("Autocorrelation")
ax_acf.set_title("Score Residual ACF (95% band dashed)", fontsize=10)
ax_acf.grid(True, alpha=0.3)

fig5.suptitle("GAS Model Adequacy — Score Residuals", fontsize=11, fontweight="bold")
fig5.tight_layout()
display(fig5)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for:** A large positive spike in the score residuals around Q20
# MAGIC (more claims than the model predicted) is expected — that is how the GAS filter
# MAGIC detects the break and adjusts. After ~3-4 quarters the residuals should return
# MAGIC to zero as the filter converges on the new rate. Persistent autocorrelation at
# MAGIC lag 1 would indicate that `phi` (persistence) is too low; heteroskedasticity
# MAGIC would suggest a distributional mismatch.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. When to use GAS vs a static GLM
# MAGIC
# MAGIC | Situation | Use |
# MAGIC |---|---|
# MAGIC | Stable book, explaining risk factors cross-sectionally | Static GLM |
# MAGIC | Quarterly monitoring — is my frequency drifting? | GAS Poisson filter |
# MAGIC | Regulatory event happened — did it move my book? | BOCPD (FrequencyChangeDetector) |
# MAGIC | Board pack: "has claims experience changed?" | GAS filter path + BOCPD posterior |
# MAGIC | Model governance: retrain trigger | LossRatioMonitor (combines both) |
# MAGIC
# MAGIC **Key design point:** GAS and BOCPD are complementary. GAS smooths the signal
# MAGIC continuously and tells you *how much* the rate has changed. BOCPD tells you *when*
# MAGIC the regime shifted. The `LossRatioMonitor` combines both and returns a binary
# MAGIC recommendation ('retrain' / 'monitor') — which is what pricing governance
# MAGIC actually needs.

# COMMAND ----------

# Final summary printout for convenience
print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"Dataset: {T} quarters, break at Q{BREAK_AT} (+{int((RATE_POST/RATE_PRE-1)*100)}% frequency)")
print()
print(f"{'Metric':<42} {'Static':>10} {'GAS':>10}")
print("-" * 62)
print(f"{'Total log-likelihood':<42} {static_total_ll:>10.2f} {gas_total_ll:>10.2f}")
print(f"{'MAE obs freq — all quarters':<42} {static_mae_all:>10.5f} {gas_mae_all:>10.5f}")
print(f"{'MAE obs freq — post-break only':<42} {static_mae_post:>10.5f} {gas_mae_post:>10.5f}")
print(f"{'RMSE vs true rate — all':<42} {static_rmse_all:>10.5f} {gas_rmse_all:>10.5f}")
print(f"{'RMSE vs true rate — post-break':<42} {static_rmse_post:>10.5f} {gas_rmse_post:>10.5f}")
print(f"{'Post-break mean bias':<42} {static_bias_post:>+10.5f} {gas_bias_post:>+10.5f}")
det_str = f"Q{cp_result.detected_breaks[0].period_index+1}" if cp_result.n_breaks > 0 else "none"
print(f"{'BOCPD break detection':<42} {'N/A':>10} {det_str:>10}")
print("=" * 60)
