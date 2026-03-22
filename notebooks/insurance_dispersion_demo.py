# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-dispersion: Double GLM for Joint Mean-Dispersion Modelling
# MAGIC
# MAGIC **Package:** `insurance-dispersion`
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC Every standard Gamma GLM you fit makes one assumption you probably haven't thought about: it assigns the same scalar dispersion parameter phi to every policy in your portfolio. A broker-sourced fleet account and a direct-channel private car policy share identical uncertainty around their fitted means. That assumption is wrong, and it costs you in three places:
# MAGIC
# MAGIC 1. **Risk selection** — your confidence intervals are too wide for low-volatility risks and too narrow for high-volatility ones. You're mispricing certainty, not just expected cost.
# MAGIC
# MAGIC 2. **Reinsurance pricing** — XL layers are driven by variance. A policy with correct mean but wrong phi produces an incorrect excess loss premium. The DGLM gives you per-policy variance estimates.
# MAGIC
# MAGIC 3. **Reserving volatility** — when you sign off on an IBNR range for a segment, the width of that range depends on phi. Aggregating policies with heterogeneous phi using a single number overstates diversification for high-volatility segments and understates it for low-volatility ones.
# MAGIC
# MAGIC The **Double GLM** (Smyth 1989) solves this by fitting two linked regression models simultaneously:
# MAGIC
# MAGIC ```
# MAGIC Mean submodel:        log(mu_i)  = x_i^T beta    [standard GLM]
# MAGIC Dispersion submodel:  log(phi_i) = z_i^T alpha   [new: each policy gets its own phi]
# MAGIC
# MAGIC Var[Y_i] = phi_i * V(mu_i) = phi_i * mu_i^2      [Gamma]
# MAGIC ```
# MAGIC
# MAGIC The algorithm (alternating IRLS, Smyth & Verbyla 1999) is robust, pure numpy, and converges in 8-15 iterations for typical insurance datasets.
# MAGIC
# MAGIC **This notebook**: 50k synthetic UK motor policies. Young drivers have both higher mean severity AND higher variance. High-value vehicles have higher variance. We benchmark the DGLM against a constant-phi Gamma GLM and show where constant-phi fails.

# COMMAND ----------

# MAGIC %pip install insurance-dispersion formulaic scipy pandas numpy matplotlib --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC *Restart the Python environment after pip install before importing.*

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from insurance_dispersion import DGLM, DGLMResult, diagnostics
import insurance_dispersion.families as fam

print(f"insurance-dispersion loaded OK")
np.random.seed(2025)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Motor Portfolio
# MAGIC
# MAGIC **Data-generating process (known truth):**
# MAGIC
# MAGIC Mean severity depends on: age band, vehicle group, region.
# MAGIC Dispersion depends on: age band, vehicle value band, distribution channel.
# MAGIC
# MAGIC The key design choice is that **age band appears in both submodels** — young drivers are both more expensive (higher mean) AND more volatile (higher phi). This mimics real motor books where young driver frequency is predictable but severity is erratic (minor bumps vs write-offs). High-value vehicles also carry higher phi because large repair bills and total-loss valuations are harder to estimate.

# COMMAND ----------

def simulate_uk_motor(n: int = 50_000, seed: int = 2025) -> pd.DataFrame:
    """
    Simulate 50k UK motor policies with realistic heterogeneous variance.

    True DGP:
      log(mu_i) = 7.8
                  + {0.0, 0.35, 0.55}         [vehicle_group A/B/C]
                  + {0.0, -0.12, 0.22, 0.08}  [region SE/NW/Scotland/Midlands]
                  - 0.15 * I(age=17-24)        [young drivers cheaper — but more volatile]
                  + 0.10 * I(age=55+)
                  + log(exposure)              [offset]

      log(phi_i) = -1.5
                   + 0.90 * I(age=17-24)      [young = 2.5x dispersion]
                   + 0.40 * I(age=55+)        [older also more volatile: claims rare/large]
                   + 0.65 * I(vehicle_value > 25000)  [high-value = 1.9x dispersion]
                   + 0.80 * I(channel=broker) [broker heterogeneity: 2.2x]
                   + 0.45 * I(channel=aggregator)

    Note: mean differences by age are modest (-15%, +10%); dispersion differences
    are large (2.5x). This is the kind of structure a standard GLM will miss.
    """
    rng = np.random.default_rng(seed)

    vehicle_group = rng.choice(["A", "B", "C"], n, p=[0.45, 0.35, 0.20])
    region = rng.choice(["SE", "NW", "Scotland", "Midlands"], n, p=[0.30, 0.28, 0.17, 0.25])
    age_band = rng.choice(["17-24", "25-54", "55+"], n, p=[0.12, 0.68, 0.20])
    channel = rng.choice(["direct", "broker", "aggregator"], n, p=[0.38, 0.32, 0.30])
    vehicle_value = rng.lognormal(np.log(16_000), 0.55, n).round(-2)
    earned_exposure = rng.uniform(0.25, 1.0, n)

    # ------------------------------------------------------------------
    # True mean
    # ------------------------------------------------------------------
    log_mu = (
        7.8
        + 0.35 * (vehicle_group == "B").astype(float)
        + 0.55 * (vehicle_group == "C").astype(float)
        + 0.00 * (region == "SE").astype(float)
        - 0.12 * (region == "NW").astype(float)
        + 0.22 * (region == "Scotland").astype(float)
        + 0.08 * (region == "Midlands").astype(float)
        - 0.15 * (age_band == "17-24").astype(float)
        + 0.10 * (age_band == "55+").astype(float)
    )
    # per-exposure expected severity
    mu = np.exp(log_mu) * earned_exposure

    # ------------------------------------------------------------------
    # True dispersion — the DGLM signal
    # ------------------------------------------------------------------
    log_phi = (
        -1.5
        + 0.90 * (age_band == "17-24").astype(float)
        + 0.40 * (age_band == "55+").astype(float)
        + 0.65 * (vehicle_value > 25_000).astype(float)
        + 0.80 * (channel == "broker").astype(float)
        + 0.45 * (channel == "aggregator").astype(float)
    )
    phi = np.exp(log_phi)

    # Gamma: shape = 1/phi, scale = mu*phi
    shape = 1.0 / phi
    claim_amount = rng.gamma(shape, mu * phi)

    # High-value indicator for dispersion formula
    hv = (vehicle_value > 25_000).astype(int)

    return pd.DataFrame({
        "claim_amount": claim_amount.round(2),
        "vehicle_group": vehicle_group,
        "region": region,
        "age_band": age_band,
        "channel": channel,
        "vehicle_value": vehicle_value,
        "high_value_vehicle": hv,
        "earned_exposure": earned_exposure,
        "true_mu": mu,
        "true_phi": phi,
        "true_log_phi": log_phi,
    })


df = simulate_uk_motor(n=50_000, seed=2025)

print(f"Portfolio: {len(df):,} policies")
print(f"Claim amount range: £{df.claim_amount.min():,.0f} – £{df.claim_amount.max():,.0f}")
print()

# Show dispersion truly varies — the problem we're solving
summary = df.groupby("age_band").agg(
    n=("claim_amount", "count"),
    mean_claim=("claim_amount", "mean"),
    std_claim=("claim_amount", "std"),
    cv=("claim_amount", lambda x: x.std() / x.mean()),
    true_phi=("true_phi", "mean"),
).round(3)
print("Observed statistics by age band:")
print(summary.to_string())
print()
print("True phi by channel:")
print(df.groupby("channel")["true_phi"].mean().round(3).to_string())
print()
print("True phi by vehicle value band:")
df["value_band"] = pd.cut(df["vehicle_value"], bins=[0, 15000, 25000, 1e9],
                           labels=["<15k", "15-25k", ">25k"])
print(df.groupby("value_band")["true_phi"].mean().round(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploratory: Variance Heterogeneity is Real
# MAGIC
# MAGIC Before fitting anything, visualise that the variance-to-mean relationship differs across segments. If the standard GLM is correct, the coefficient of variation (std/mean) should be constant across groups. It is not.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Observed Variance Heterogeneity in Synthetic UK Motor Portfolio",
             fontsize=13, fontweight="bold", y=1.01)

# --- CV by age band ---
cv_age = df.groupby("age_band").apply(lambda g: g["claim_amount"].std() / g["claim_amount"].mean())
ax = axes[0]
bars = ax.bar(cv_age.index, cv_age.values, color=["#e74c3c", "#3498db", "#2ecc71"], alpha=0.85)
ax.set_title("Coefficient of Variation by Age Band", fontsize=11)
ax.set_xlabel("Age band")
ax.set_ylabel("CV = std / mean")
ax.set_ylim(0, cv_age.max() * 1.3)
for bar, val in zip(bars, cv_age.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(cv_age.mean(), color="black", linestyle="--", lw=1.2, label=f"Overall mean CV={cv_age.mean():.2f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- CV by channel ---
cv_ch = df.groupby("channel").apply(lambda g: g["claim_amount"].std() / g["claim_amount"].mean())
ax = axes[1]
bars = ax.bar(cv_ch.index, cv_ch.values, color=["#9b59b6", "#e67e22", "#1abc9c"], alpha=0.85)
ax.set_title("Coefficient of Variation by Channel", fontsize=11)
ax.set_xlabel("Distribution channel")
ax.set_ylabel("CV = std / mean")
ax.set_ylim(0, cv_ch.max() * 1.3)
for bar, val in zip(bars, cv_ch.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(cv_ch.mean(), color="black", linestyle="--", lw=1.2, label=f"Overall mean CV={cv_ch.mean():.2f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- CV by vehicle value band ---
cv_vb = df.groupby("value_band", observed=True).apply(
    lambda g: g["claim_amount"].std() / g["claim_amount"].mean()
)
ax = axes[2]
bars = ax.bar(cv_vb.index.astype(str), cv_vb.values, color=["#27ae60", "#f39c12", "#c0392b"], alpha=0.85)
ax.set_title("Coefficient of Variation by Vehicle Value Band", fontsize=11)
ax.set_xlabel("Vehicle value")
ax.set_ylabel("CV = std / mean")
ax.set_ylim(0, cv_vb.max() * 1.3)
for bar, val in zip(bars, cv_vb.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(cv_vb.mean(), color="black", linestyle="--", lw=1.2, label=f"Overall mean CV={cv_vb.mean():.2f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
display(fig)
plt.close()

print("Key observation: CV varies 2-3x across segments.")
print("A constant-phi GLM assumes CV is identical everywhere — clearly wrong.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Baseline: Constant-Phi Gamma GLM
# MAGIC
# MAGIC This is equivalent to a standard Gamma GLM. We use the DGLM API with `dformula="~1"` (intercept-only dispersion), so the comparison is apples-to-apples on the same IRLS fitting engine.

# COMMAND ----------

glm_model = DGLM(
    formula="claim_amount ~ C(vehicle_group) + C(region) + C(age_band)",
    dformula="~ 1",   # intercept only = constant phi across all policies
    family=fam.Gamma(),
    data=df,
    exposure="earned_exposure",
    method="ml",      # use ML for fair AIC comparison with DGLM
)

print("Fitting constant-phi Gamma GLM...")
glm_result = glm_model.fit(verbose=True)

print(f"\nConstant-phi GLM:")
print(f"  Fitted phi (single value): {glm_result.phi_.mean():.4f}")
print(f"  Log-likelihood:            {glm_result.loglik:.1f}")
print(f"  AIC:                       {glm_result.aic:.1f}")
print(f"  Mean params:               {len(glm_result.mean_model.coef)}")
print(f"  Dispersion params:         {len(glm_result.dispersion_model.coef)} (intercept only)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit DGLM: Model Both Mean and Dispersion
# MAGIC
# MAGIC The dispersion submodel uses: age band, vehicle value (high/low), distribution channel.
# MAGIC Notice that `age_band` appears in **both** submodels — it affects both the expected severity
# MAGIC and how volatile that severity is. That is the key insight of the DGLM.

# COMMAND ----------

dglm_model = DGLM(
    formula="claim_amount ~ C(vehicle_group) + C(region) + C(age_band)",
    dformula="~ C(age_band) + C(high_value_vehicle) + C(channel)",
    family=fam.Gamma(),
    data=df,
    exposure="earned_exposure",
    method="reml",    # REML corrects for estimating beta — recommended
)

print("Fitting DGLM (joint mean + dispersion)...")
dglm_result = dglm_model.fit(verbose=True)

print("\n" + dglm_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Comparison: Log-likelihood, AIC, LRT
# MAGIC
# MAGIC The LRT tests H0: phi is constant vs H1: phi depends on age band, vehicle value, channel.

# COMMAND ----------

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\nConstant-phi GLM:   loglik = {glm_result.loglik:>10.1f}   AIC = {glm_result.aic:>10.1f}")
print(f"DGLM (phi varies):  loglik = {dglm_result.loglik:>10.1f}   AIC = {dglm_result.aic:>10.1f}")
print(f"\nDelta AIC: {glm_result.aic - dglm_result.aic:.1f}  (positive = DGLM is better)")
print(f"Delta loglik (x2): {2*(dglm_result.loglik - glm_result.loglik):.1f}")

print("\n" + "-" * 60)
print("Likelihood Ratio Test: constant phi vs. phi = f(age, vehicle_value, channel)")
print("-" * 60)
test = dglm_result.overdispersion_test()
print(f"  LRT statistic: {test['statistic']:.2f}")
print(f"  Degrees of freedom: {test['df']}")
print(f"  p-value: {test['p_value']:.2e}")
print(f"  Conclusion: {test['conclusion']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dispersion Factor Tables
# MAGIC
# MAGIC The dispersion submodel gives multiplicative relativities for phi. Read these as:
# MAGIC
# MAGIC - A factor with `exp_coef = 2.2` means that segment has 2.2x the dispersion of the base level.
# MAGIC - Higher phi means wider prediction intervals around the fitted mean.
# MAGIC - This is independent of the mean relativities — a segment can be cheap (low mu) but volatile (high phi), or expensive but predictable.

# COMMAND ----------

print("MEAN SUBMODEL — Severity relativities (exp_coef = multiplicative factor on mean)")
print("=" * 70)
mean_rel = dglm_result.mean_relativities()
print(mean_rel[["coef", "exp_coef", "se", "z", "p_value"]].round(4).to_string())

print()
print("DISPERSION SUBMODEL — Phi relativities (exp_coef = multiplicative factor on variance)")
print("=" * 70)
disp_rel = dglm_result.dispersion_relativities()
print(disp_rel[["coef", "exp_coef", "se", "z", "p_value"]].round(4).to_string())

print()
print("Interpretation:")
print("  - A 17-24 driver: mean is exp(mean coef) x base, phi is exp(dispersion coef) x base phi")
print("  - The dispersion multiplier is what determines prediction interval WIDTH")
print("  - High-value vehicles may have similar mean severity but much wider intervals")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Per-Policy Dispersion Estimates
# MAGIC
# MAGIC The DGLM gives each policy its own phi. Compare the distribution of fitted phi
# MAGIC against the true phi (which we know from the DGP) — and against the constant-phi estimate.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Per-Policy Dispersion Estimates: DGLM vs Constant-Phi GLM",
             fontsize=13, fontweight="bold", y=1.01)

# --- Plot 1: Fitted phi distribution ---
ax = axes[0]
ax.hist(dglm_result.phi_, bins=80, density=True, alpha=0.7,
        color="#3498db", label="DGLM fitted phi_i", edgecolor="white", lw=0.5)
ax.hist(df["true_phi"], bins=80, density=True, alpha=0.5,
        color="#e74c3c", label="True phi_i (DGP)", edgecolor="white", lw=0.5)
ax.axvline(glm_result.phi_.mean(), color="black", lw=2, linestyle="--",
           label=f"GLM constant phi = {glm_result.phi_.mean():.3f}")
ax.set_xlabel("Dispersion (phi)")
ax.set_ylabel("Density")
ax.set_title("Distribution of phi Estimates")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: DGLM fitted vs true phi ---
ax = axes[1]
sample = np.random.choice(len(df), size=5000, replace=False)
ax.scatter(df["true_phi"].values[sample], dglm_result.phi_[sample],
           alpha=0.15, s=5, color="#3498db")
lim = max(df["true_phi"].max(), dglm_result.phi_.max())
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect recovery")
ax.set_xlabel("True phi (DGP)")
ax.set_ylabel("DGLM fitted phi")
ax.set_title("DGLM: Fitted vs. True Dispersion")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Phi by segment (boxplots) ---
ax = axes[2]
segment_labels = []
dglm_phis = []
true_phis = []

for channel in ["direct", "broker", "aggregator"]:
    mask = df["channel"] == channel
    segment_labels.append(channel)
    dglm_phis.append(dglm_result.phi_[mask])
    true_phis.append(df["true_phi"][mask].values)

positions = np.array([1, 2, 3])
bp1 = ax.boxplot(dglm_phis, positions=positions - 0.2, widths=0.3,
                  patch_artist=True,
                  boxprops=dict(facecolor="#3498db", alpha=0.7),
                  medianprops=dict(color="white", lw=2),
                  whiskerprops=dict(lw=1.2),
                  capprops=dict(lw=1.2),
                  flierprops=dict(marker=".", ms=2, alpha=0.3))
bp2 = ax.boxplot(true_phis, positions=positions + 0.2, widths=0.3,
                  patch_artist=True,
                  boxprops=dict(facecolor="#e74c3c", alpha=0.7),
                  medianprops=dict(color="white", lw=2),
                  whiskerprops=dict(lw=1.2),
                  capprops=dict(lw=1.2),
                  flierprops=dict(marker=".", ms=2, alpha=0.3))
ax.set_xticks(positions)
ax.set_xticklabels(segment_labels)
ax.set_xlabel("Channel")
ax.set_ylabel("Dispersion (phi)")
ax.set_title("Fitted vs. True Phi by Channel")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["DGLM fitted", "True phi"],
          fontsize=9)
ax.axhline(glm_result.phi_.mean(), color="black", linestyle="--", lw=1.5,
           label=f"Constant GLM phi")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prediction Interval Coverage Analysis
# MAGIC
# MAGIC This is the core diagnostic for whether constant-phi is causing pricing harm.
# MAGIC
# MAGIC For a Gamma with parameters (shape=1/phi, scale=mu*phi), the 90% prediction interval is
# MAGIC the 5th and 95th percentiles of Gamma(1/phi, mu*phi). We compute nominal coverage
# MAGIC by segment — a correctly specified model should hit 90% coverage in every segment.
# MAGIC The constant-phi GLM will systematically **under-cover** high-phi segments and
# MAGIC **over-cover** low-phi segments.

# COMMAND ----------

from scipy.stats import gamma as gamma_dist

def compute_coverage(y: np.ndarray, mu: np.ndarray, phi: np.ndarray,
                     level: float = 0.90) -> float:
    """Fraction of observations inside the (1-level)/2 to (1+level)/2 prediction interval."""
    alpha_tail = (1.0 - level) / 2.0
    shape = 1.0 / np.clip(phi, 1e-9, None)
    scale = mu * phi
    lo = gamma_dist.ppf(alpha_tail, a=shape, scale=scale)
    hi = gamma_dist.ppf(1.0 - alpha_tail, a=shape, scale=scale)
    return float(np.mean((y >= lo) & (y <= hi)))


y = df["claim_amount"].values
mu_glm = glm_result.mu_
phi_glm = glm_result.phi_   # constant across all policies
mu_dglm = dglm_result.mu_
phi_dglm = dglm_result.phi_

# Compute coverage by segment
segments = {
    "Age: 17-24":        df["age_band"] == "17-24",
    "Age: 25-54":        df["age_band"] == "25-54",
    "Age: 55+":          df["age_band"] == "55+",
    "Channel: direct":   df["channel"] == "direct",
    "Channel: broker":   df["channel"] == "broker",
    "Channel: aggregator": df["channel"] == "aggregator",
    "Vehicle value: <15k":  df["vehicle_value"] < 15_000,
    "Vehicle value: 15-25k": (df["vehicle_value"] >= 15_000) & (df["vehicle_value"] <= 25_000),
    "Vehicle value: >25k":  df["vehicle_value"] > 25_000,
}

TARGET = 0.90
rows = []
for seg_name, mask in segments.items():
    mask_arr = mask.values
    y_seg = y[mask_arr]
    cov_glm = compute_coverage(y_seg, mu_glm[mask_arr], phi_glm[mask_arr])
    cov_dglm = compute_coverage(y_seg, mu_dglm[mask_arr], phi_dglm[mask_arr])
    true_phi_seg = df["true_phi"].values[mask_arr].mean()
    rows.append({
        "Segment": seg_name,
        "N": int(mask_arr.sum()),
        "True phi (mean)": round(true_phi_seg, 3),
        "Coverage: GLM (const phi)": round(cov_glm, 3),
        "Coverage: DGLM": round(cov_dglm, 3),
        "GLM miss": round(abs(cov_glm - TARGET), 3),
        "DGLM miss": round(abs(cov_dglm - TARGET), 3),
    })

coverage_df = pd.DataFrame(rows)
print("90% Prediction Interval Coverage by Segment")
print("=" * 80)
print(coverage_df.to_string(index=False))
print()
print(f"GLM  mean absolute coverage error: {coverage_df['GLM miss'].mean():.3f}")
print(f"DGLM mean absolute coverage error: {coverage_df['DGLM miss'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage deviation chart
# MAGIC
# MAGIC Green bars close to zero are good. The constant-phi GLM will under-cover
# MAGIC high-variance segments and over-cover low-variance ones.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("90% Prediction Interval Coverage Deviation by Segment\n(Target: 0.0 — positive means over-coverage, negative means under-coverage)",
             fontsize=12, fontweight="bold")

seg_names = coverage_df["Segment"].tolist()
glm_dev = (coverage_df["Coverage: GLM (const phi)"] - TARGET).tolist()
dglm_dev = (coverage_df["Coverage: DGLM"] - TARGET).tolist()

colors_glm = ["#e74c3c" if d < 0 else "#e67e22" for d in glm_dev]
colors_dglm = ["#2ecc71" if abs(d) < 0.015 else "#f39c12" for d in dglm_dev]

for ax, devs, colors, title in [
    (axes[0], glm_dev, colors_glm, "Constant-phi Gamma GLM"),
    (axes[1], dglm_dev, colors_dglm, "DGLM (varying phi)"),
]:
    y_pos = np.arange(len(seg_names))
    bars = ax.barh(y_pos, devs, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(seg_names, fontsize=10)
    ax.axvline(0, color="black", lw=1.5)
    ax.axvline(-0.02, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0.02, color="grey", lw=0.8, linestyle="--", alpha=0.6, label="+/- 2pp tolerance")
    ax.set_xlabel("Coverage deviation from 90% target")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, devs):
        xpos = val + 0.001 if val >= 0 else val - 0.001
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.2%}", va="center", ha=ha, fontsize=8)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Predicted Variance Comparison by Segment
# MAGIC
# MAGIC For the Gamma family, `Var[Y_i] = phi_i * mu_i^2`. Compare predicted variance
# MAGIC from the GLM (phi constant) vs DGLM (phi varies) vs truth.

# COMMAND ----------

var_glm = phi_glm * mu_glm**2
var_dglm = phi_dglm * mu_dglm**2
var_true = df["true_phi"].values * df["true_mu"].values**2

print("Variance calibration by segment — ratio of predicted to true mean variance")
print("(closer to 1.0 is better)\n")

for seg_name, mask in segments.items():
    mask_arr = mask.values
    ratio_glm = var_glm[mask_arr].mean() / var_true[mask_arr].mean()
    ratio_dglm = var_dglm[mask_arr].mean() / var_true[mask_arr].mean()
    direction = "OVER" if ratio_glm > 1.05 else ("UNDER" if ratio_glm < 0.95 else "OK  ")
    print(f"  {seg_name:<26}  GLM: {ratio_glm:.2f} ({direction})   DGLM: {ratio_dglm:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Predictions on New Risks
# MAGIC
# MAGIC Predict mean, phi, and variance for a set of representative policies.
# MAGIC Notice how the broker/aggregator and young-driver rows have similar mean
# MAGIC severity but very different uncertainty.

# COMMAND ----------

new_risks = pd.DataFrame({
    "vehicle_group":      ["A",       "A",       "B",       "C",       "C",       "B"],
    "region":             ["SE",      "SE",      "NW",      "Scotland","SE",      "Midlands"],
    "age_band":           ["17-24",   "25-54",   "25-54",   "25-54",   "55+",     "17-24"],
    "channel":            ["direct",  "broker",  "direct",  "aggregator","broker", "broker"],
    "high_value_vehicle": [0,         0,         0,         1,          1,         1],
    "vehicle_value":      [9000,      12000,     14000,     32000,      28000,     35000],
    "earned_exposure":    [1.0,       1.0,       1.0,       1.0,        1.0,       1.0],
})

mu_pred  = dglm_result.predict(new_risks, which="mean")
phi_pred = dglm_result.predict(new_risks, which="dispersion")
var_pred = dglm_result.predict(new_risks, which="variance")

# 80% prediction interval using Gamma CDF
from scipy.stats import gamma as gamma_dist
lo80 = gamma_dist.ppf(0.10, a=1.0/phi_pred, scale=mu_pred*phi_pred)
hi80 = gamma_dist.ppf(0.90, a=1.0/phi_pred, scale=mu_pred*phi_pred)

new_risks["mu_hat"]       = mu_pred.round(0)
new_risks["phi_hat"]      = phi_pred.round(3)
new_risks["sigma_hat"]    = np.sqrt(var_pred).round(0)
new_risks["cv"]           = (np.sqrt(var_pred) / mu_pred).round(3)
new_risks["PI_80_lo"]     = lo80.round(0)
new_risks["PI_80_hi"]     = hi80.round(0)

display_cols = ["age_band", "channel", "vehicle_group", "high_value_vehicle",
                "mu_hat", "phi_hat", "sigma_hat", "cv", "PI_80_lo", "PI_80_hi"]

print("Predictions on representative new risks:")
print("mu_hat = expected severity; phi_hat = dispersion; cv = std/mean")
print("PI_80_lo/hi = 80% prediction interval bounds\n")
print(new_risks[display_cols].to_string(index=False))
print()
print("Note: rows 1 and 2 (both direct/SE/25-54) differ only in vehicle_group.")
print("Their means differ but their CVs are similar — mean difference, same volatility.")
print()
print("Row 5 (55+/broker/high-value): modest mean premium but phi=", round(phi_pred[4], 2),
      "-> wide interval. Reinsurance pricing needs this number.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Business Interpretation: Volatile vs. Expensive
# MAGIC
# MAGIC The key distinction a DGLM enables: segments that are **genuinely volatile** (high phi,
# MAGIC prediction intervals wide relative to mean) vs segments that are simply **high expected cost**
# MAGIC (high mu, intervals proportionally tight).
# MAGIC
# MAGIC This matters because:
# MAGIC - An XL reinsurance layer for a volatile segment costs more than the same layer for a predictable segment with the same mu.
# MAGIC - Capital loading should reflect uncertainty, not just mean cost.
# MAGIC - Credibility allocation: low-phi risks can be priced with less volume than high-phi risks.

# COMMAND ----------

# Portfolio segmentation: separate high-mu from high-phi
df["fitted_mu"]  = mu_dglm
df["fitted_phi"] = phi_dglm
df["fitted_cv"]  = np.sqrt(phi_dglm * mu_dglm**2) / mu_dglm  # = sqrt(phi) for Gamma

mu_median  = np.median(mu_dglm)
phi_median = np.median(phi_dglm)

df["risk_type"] = "Low cost, low vol"
df.loc[(df["fitted_mu"] > mu_median) & (df["fitted_phi"] <= phi_median), "risk_type"] = "High cost, low vol"
df.loc[(df["fitted_mu"] <= mu_median) & (df["fitted_phi"] > phi_median), "risk_type"] = "Low cost, high vol"
df.loc[(df["fitted_mu"] > mu_median) & (df["fitted_phi"] > phi_median),  "risk_type"] = "High cost, high vol"

# Quadrant counts and profile
quadrant_summary = (
    df.groupby("risk_type")
    .agg(
        n=("claim_amount", "count"),
        mean_mu=("fitted_mu", "mean"),
        mean_phi=("fitted_phi", "mean"),
        mean_cv=("fitted_cv", "mean"),
        pct_young=("age_band", lambda x: (x == "17-24").mean()),
        pct_broker=("channel", lambda x: (x == "broker").mean()),
        pct_high_value=("high_value_vehicle", "mean"),
    )
    .round(3)
)

print("Portfolio Quadrant Analysis: Expensive vs. Volatile")
print("=" * 70)
print(quadrant_summary.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quadrant scatter plot

# COMMAND ----------

fig, ax = plt.subplots(figsize=(11, 8))

colors = {
    "Low cost, low vol":   "#2ecc71",
    "High cost, low vol":  "#3498db",
    "Low cost, high vol":  "#e67e22",
    "High cost, high vol": "#e74c3c",
}

sample_idx = np.random.choice(len(df), size=6000, replace=False)
df_sample = df.iloc[sample_idx]

for risk_type, color in colors.items():
    mask = df_sample["risk_type"] == risk_type
    ax.scatter(
        df_sample.loc[mask, "fitted_mu"],
        df_sample.loc[mask, "fitted_phi"],
        alpha=0.25, s=7, c=color, label=f"{risk_type} (n={mask.sum():,})"
    )

ax.axvline(mu_median, color="black", lw=1.2, linestyle="--", alpha=0.6)
ax.axhline(phi_median, color="black", lw=1.2, linestyle="--", alpha=0.6)
ax.set_xlabel("Fitted mean severity (mu_i)", fontsize=12)
ax.set_ylabel("Fitted dispersion (phi_i)", fontsize=12)
ax.set_title("Portfolio Segmentation: Expected Cost vs. Volatility\n"
             "Dashed lines at portfolio medians", fontsize=12)
ax.legend(fontsize=10, markerscale=3)
ax.grid(True, alpha=0.2)

# Annotate quadrants
ax.text(mu_median * 0.4, phi_median * 1.8,  "Low cost\nHigh vol\n(e.g. young/aggregator)",
        fontsize=9, ha="center", color="#e67e22", fontweight="bold")
ax.text(mu_median * 1.6, phi_median * 1.8, "High cost\nHigh vol\n(e.g. C-group/broker/high-value)",
        fontsize=9, ha="center", color="#e74c3c", fontweight="bold")
ax.text(mu_median * 0.4, phi_median * 0.35, "Low cost\nLow vol\n(direct/standard)",
        fontsize=9, ha="center", color="#2ecc71", fontweight="bold")
ax.text(mu_median * 1.6, phi_median * 0.35, "High cost\nLow vol\n(premium/predictable)",
        fontsize=9, ha="center", color="#3498db", fontweight="bold")

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Diagnostics
# MAGIC
# MAGIC Standard GLM diagnostics adjusted for per-observation phi_i.

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("DGLM Diagnostics — UK Motor Severity (50k policies)",
             fontsize=13, fontweight="bold")

# --- 1. QQ plot of quantile residuals ---
ax = axes[0, 0]
qq = diagnostics.qq_plot_data(dglm_result)
sample_qq = np.random.choice(len(qq), size=min(10_000, len(qq)), replace=False)
ax.scatter(qq["theoretical"].values[sample_qq], qq["observed"].values[sample_qq],
           alpha=0.12, s=5, color="#3498db")
lim = max(abs(qq["theoretical"]).max(), abs(qq["observed"]).max())
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5, label="45-degree line")
ax.set_xlabel("N(0,1) quantiles")
ax.set_ylabel("Quantile residuals")
ax.set_title("QQ Plot (quantile residuals)\nShould follow N(0,1) under true model")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- 2. Fitted phi vs. true phi ---
ax = axes[0, 1]
sample_phi = np.random.choice(len(df), size=5000, replace=False)
ax.scatter(df["true_phi"].values[sample_phi], dglm_result.phi_[sample_phi],
           alpha=0.15, s=5, color="#9b59b6")
ax.scatter(df["true_phi"].values[sample_phi], np.full(5000, glm_result.phi_.mean()),
           alpha=0.05, s=3, color="grey", label=f"GLM const phi={glm_result.phi_.mean():.3f}")
lim = max(df["true_phi"].max(), dglm_result.phi_.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect recovery")
ax.set_xlabel("True phi (DGP)")
ax.set_ylabel("Fitted phi")
ax.set_title("Dispersion Recovery\nDGLM (purple) vs. constant GLM (grey)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- 3. Scaled unit deviances (should be chi2(1)) ---
ax = axes[1, 0]
diag_df = diagnostics.dispersion_diagnostic(dglm_result)
scaled_dev = diag_df["scaled_deviance"].clip(0, 15)
ax.hist(scaled_dev, bins=80, density=True, alpha=0.75,
        color="#3498db", edgecolor="white", lw=0.4, label="DGLM scaled deviances")
x_chi = np.linspace(0.001, 15, 500)
ax.plot(x_chi, stats.chi2.pdf(x_chi, df=1), "r-", lw=2, label="chi2(1) density (expected)")
ax.axvline(1.0, color="black", linestyle="--", lw=1.2, label="E[delta_i]=1 under model")
ax.set_xlabel("Scaled unit deviance (d_i / phi_i)")
ax.set_ylabel("Density")
ax.set_title("Dispersion Pseudo-Response Distribution\nShould follow chi2(1)")
ax.legend(fontsize=9)
ax.set_xlim(0, 12)
ax.grid(True, alpha=0.3)

# --- 4. Log-likelihood convergence ---
ax = axes[1, 1]
iters = range(1, len(dglm_result.loglik_history) + 1)
ax.plot(iters, dglm_result.loglik_history, "o-", color="#3498db",
        markersize=6, markerfacecolor="white", markeredgewidth=2, lw=2)
ax.set_xlabel("Outer iteration (alternating IRLS)")
ax.set_ylabel("Joint log-likelihood")
ax.set_title(f"Convergence: {dglm_result.n_iter} iterations\n"
             f"Converged: {dglm_result.converged}")
ax.grid(True, alpha=0.3)
ax.annotate(f"Final: {dglm_result.loglik_history[-1]:.0f}",
            xy=(len(dglm_result.loglik_history), dglm_result.loglik_history[-1]),
            xytext=(-30, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->"), fontsize=9)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary HTML Table

# COMMAND ----------

# Build a clean summary for displayHTML
summary_rows = [
    ("Observations", f"{dglm_result.n_obs:,}"),
    ("Family", str(dglm_result._dglm.family)),
    ("Method", dglm_result._dglm.method.upper()),
    ("Converged", f"{dglm_result.converged} ({dglm_result.n_iter} iterations)"),
    ("Mean params (p)", str(len(dglm_result.mean_model.coef))),
    ("Dispersion params (q)", str(len(dglm_result.dispersion_model.coef))),
    ("Total params (p+q)", str(len(dglm_result.mean_model.coef) + len(dglm_result.dispersion_model.coef))),
    ("Log-likelihood (GLM)", f"{glm_result.loglik:,.1f}"),
    ("Log-likelihood (DGLM)", f"{dglm_result.loglik:,.1f}"),
    ("Delta log-lik (x2)", f"{2*(dglm_result.loglik - glm_result.loglik):,.1f}"),
    ("AIC (GLM)", f"{glm_result.aic:,.1f}"),
    ("AIC (DGLM)", f"{dglm_result.aic:,.1f}"),
    ("Delta AIC", f"{glm_result.aic - dglm_result.aic:,.1f} (DGLM better)"),
    ("LRT p-value", f"{test['p_value']:.2e}"),
    ("Phi range (DGLM)", f"[{dglm_result.phi_.min():.3f}, {dglm_result.phi_.max():.3f}]"),
    ("GLM coverage error (mean |dev| across segs)", f"{coverage_df['GLM miss'].mean():.3f}"),
    ("DGLM coverage error (mean |dev| across segs)", f"{coverage_df['DGLM miss'].mean():.3f}"),
]

rows_html = "".join(
    f"<tr><td style='padding:6px 14px;font-weight:bold;background:#f8f9fa'>{k}</td>"
    f"<td style='padding:6px 14px'>{v}</td></tr>"
    for k, v in summary_rows
)

html = f"""
<style>
  table.bc-summary {{ border-collapse: collapse; font-family: monospace; font-size: 13px; }}
  table.bc-summary td {{ border: 1px solid #dee2e6; }}
  table.bc-summary caption {{ font-size: 15px; font-weight: bold; margin-bottom: 8px;
    font-family: sans-serif; text-align: left; color: #2c3e50; }}
</style>
<table class="bc-summary">
  <caption>DGLM vs Constant-Phi GLM — 50k UK Motor Policies</caption>
  <tbody>{rows_html}</tbody>
</table>
"""

displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Takeaways for Pricing Teams
# MAGIC
# MAGIC **When to use the DGLM:**
# MAGIC
# MAGIC - You suspect phi varies by distribution channel, policy limit, fleet size, or risk type.
# MAGIC - You are pricing reinsurance layers or capital loadings where variance matters, not just expected cost.
# MAGIC - Your standard GLM's prediction intervals are too wide for some segments (causing hesitancy to quote) and too narrow for others (causing losses on bad risks).
# MAGIC
# MAGIC **What the DGLM gives you that the standard GLM cannot:**
# MAGIC
# MAGIC - A per-policy phi_i — the uncertainty multiplier for that specific risk.
# MAGIC - Correctly calibrated prediction intervals that hit 90% coverage in each segment, not just on average.
# MAGIC - A formal LRT to decide whether modelling dispersion is worth the complexity.
# MAGIC - Separate factor tables for volatility (phi relativities) vs. mean cost (severity relativities).
# MAGIC
# MAGIC **What the DGLM does NOT do:**
# MAGIC
# MAGIC - Fix a misspecified mean model. If your mean formula is wrong, the dispersion fit will absorb some of the residual but the mean predictions will still be biased.
# MAGIC - Replace scenario analysis or cat modelling for extreme events.
# MAGIC - Make slow convergence impossible. On very small datasets or heavily parameterised dispersion submodels, check `converged` and `n_iter`.
# MAGIC
# MAGIC **Practical tip:** Start with `dformula="~ C(channel)"` — just the distribution channel.
# MAGIC If the LRT is significant (and in most multi-channel UK motor books it will be), add vehicle value band and limit band. That usually captures the bulk of the dispersion variation.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Package:** `insurance-dispersion` | **Algorithm:** Alternating IRLS (Smyth 1989, Smyth & Verbyla 1999) | **Fitting engine:** Pure numpy/scipy, no ML framework dependency
