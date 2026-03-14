# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-distributional-glm: GAMLSS vs Standard Gamma GLM
# MAGIC
# MAGIC **Library:** `insurance-distributional-glm` — GAMLSS (Generalised Additive Models
# MAGIC for Location, Scale and Shape) for insurance pricing in Python.
# MAGIC
# MAGIC ## The problem with standard GLMs
# MAGIC
# MAGIC A standard Gamma GLM fits one thing: `E[Y|X]`, the conditional mean. The dispersion
# MAGIC parameter (sigma) is a single scalar shared across every policy in the portfolio.
# MAGIC
# MAGIC That assumption is almost always wrong in practice. Consider two UK motor policies
# MAGIC with the same expected claim of £2,500:
# MAGIC
# MAGIC - Policy A: 22-year-old driving a modified hatchback, no claims discount 0
# MAGIC - Policy B: 48-year-old driving a standard saloon, NCD 5 years
# MAGIC
# MAGIC Both have the same point prediction. But Policy A has a much wider claim
# MAGIC distribution — the uncertainty is higher. A standard GLM cannot tell them apart
# MAGIC on this dimension. GAMLSS can.
# MAGIC
# MAGIC ## What GAMLSS does
# MAGIC
# MAGIC GAMLSS models every distribution parameter as a function of covariates. For a
# MAGIC Gamma severity model, sigma is the coefficient of variation (CV = sd/mean):
# MAGIC
# MAGIC ```
# MAGIC log(mu_i)    = x_i^T beta_mu        # mean: all standard risk factors
# MAGIC log(sigma_i) = z_i^T beta_sigma      # CV: age and vehicle value drive heterogeneity
# MAGIC ```
# MAGIC
# MAGIC The RS algorithm (Rigby & Stasinopoulos 2005) cycles through each parameter,
# MAGIC updating it via IRLS while holding all others fixed. This is coordinate descent
# MAGIC on the joint log-likelihood with closed-form weighted least squares steps.
# MAGIC
# MAGIC ## What this notebook demonstrates
# MAGIC
# MAGIC 1. Synthetic DGP: 50k policies where sigma genuinely depends on age and vehicle value
# MAGIC 2. Standard Gamma GLM (statsmodels) as baseline — constant sigma
# MAGIC 3. GAMLSS Gamma with sigma as covariate function
# MAGIC 4. Comparison: prediction interval coverage, log-likelihood, sigma estimates by segment
# MAGIC 5. Business interpretation: which risks need wider pricing confidence bands

# COMMAND ----------

# MAGIC %pip install insurance-distributional-glm statsmodels polars scipy matplotlib --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm

from insurance_distributional_glm import DistributionalGLM, quantile_residuals, choose_distribution
from insurance_distributional_glm.families import Gamma, LogNormal, InverseGaussian
import insurance_distributional_glm

print(f"insurance-distributional-glm version: {insurance_distributional_glm.__version__}")
print("Setup complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data: UK motor claim severity with heterogeneous variance
# MAGIC
# MAGIC The DGP reflects a realistic UK motor portfolio where:
# MAGIC
# MAGIC **Mean structure** (what both models can recover):
# MAGIC - Age: older drivers have higher average severity (more expensive vehicles, more cautious)
# MAGIC - Vehicle value: higher-value vehicles cost more to repair
# MAGIC - Region: London/SE premium on repairs
# MAGIC - NCD level: proxy for driving quality, mild effect on severity
# MAGIC
# MAGIC **Sigma (CV) structure** (what only GAMLSS can recover):
# MAGIC - Young drivers (age < 28): CV 2-3x higher — impulsive driving, inconsistent severity
# MAGIC - High-value vehicles (> £20k): CV 1.5-2x higher — repair variance is wide (write-off vs repair)
# MAGIC - All other segments: baseline CV
# MAGIC
# MAGIC The standard GLM gets the mean right but assigns the same sigma to everyone.
# MAGIC Its prediction intervals will be too wide for mature low-value risks and too
# MAGIC narrow for young high-value risks.

# COMMAND ----------

def simulate_motor_severity(n: int = 50_000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate UK motor claim severity with covariate-driven heterogeneity of variance.

    True DGP:
        log(mu_i)    = 7.5 + 0.008*(age-30) + 0.000018*vehicle_value
                        + region_effect + 0.04*ncd_level
        log(sigma_i) = -1.4 + 0.035*max(0, 28-age) + 0.00004*max(0, vehicle_value-20000)
        y_i ~ Gamma(mu_i, sigma_i)   [sigma = CV, not shape]

    Returns
    -------
    df : pd.DataFrame
        Covariate matrix. Continuous covariates, no encoding — DistributionalGLM
        takes column names directly and builds design matrix with intercept.
    y : np.ndarray
        Claim severity (GBP), positive continuous.
    """
    rng = np.random.default_rng(seed)
    n_total = n

    # --- Covariates ---
    # Age: right-skewed young tail, reflecting actual UK motor portfolio mix
    age = np.clip(
        rng.gamma(shape=6.0, scale=6.5, size=n_total) + 18,
        17, 85
    ).astype(float)

    # Vehicle value: log-normal, GBP 2,000 – 80,000
    vehicle_value = np.clip(
        np.exp(rng.normal(9.6, 0.6, size=n_total)),
        2_000, 80_000
    ).astype(float)

    # Region: 4 regions with realistic prevalence
    region_raw = rng.choice(4, size=n_total, p=[0.35, 0.30, 0.20, 0.15])
    # region effects on mean: London/SE, Midlands, North, Scotland/NI
    region_effects_mu = np.array([0.15, 0.05, -0.02, -0.08])
    region_effect = region_effects_mu[region_raw]

    # NCD level: 0-5
    ncd_level = rng.choice(6, size=n_total, p=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25]).astype(float)

    # --- Mean submodel ---
    log_mu = (
        7.5
        + 0.008 * (age - 30)
        + 0.000018 * vehicle_value
        + region_effect
        + 0.04 * ncd_level
    )
    mu_true = np.exp(log_mu)

    # --- Sigma (CV) submodel ---
    # Young driver penalty: decays linearly from age 17 to 28
    young_penalty = 0.035 * np.maximum(0.0, 28.0 - age)
    # High-value vehicle penalty: applies above £20k
    hv_penalty = 0.000040 * np.maximum(0.0, vehicle_value - 20_000)

    log_sigma = -1.4 + young_penalty + hv_penalty
    sigma_true = np.exp(log_sigma)

    # --- Draw claims from Gamma(mu, sigma) ---
    # sigma = CV, so shape k = 1/sigma^2, scale = mu * sigma^2
    shape = 1.0 / sigma_true**2
    scale = mu_true * sigma_true**2
    y = rng.gamma(shape=shape, scale=scale)

    df = pd.DataFrame({
        "age": age,
        "vehicle_value": vehicle_value,
        "region": region_raw.astype(float),  # treat as numeric — ordinal proxy
        "ncd_level": ncd_level,
    })

    print(f"Generated {n_total:,} policies")
    print(f"  Severity: mean £{y.mean():,.0f}, median £{np.median(y):,.0f}, "
          f"p95 £{np.percentile(y, 95):,.0f}")
    print(f"  True sigma: mean {sigma_true.mean():.3f}, "
          f"range [{sigma_true.min():.3f}, {sigma_true.max():.3f}]")
    print(f"  Young drivers (<28): {(age < 28).mean():.1%}, "
          f"mean sigma {sigma_true[age < 28].mean():.3f} vs "
          f"{sigma_true[age >= 28].mean():.3f} for 28+")

    return df, y, age, vehicle_value, region_raw, ncd_level, sigma_true, mu_true


df_full, y, age, vehicle_value, region_raw, ncd_level, sigma_true, mu_true = simulate_motor_severity(50_000)

# Train/test split: 80/20, chronological (simulate portfolio snapshot vs future period)
n_train = int(0.8 * len(df_full))
df_train = df_full.iloc[:n_train].reset_index(drop=True)
df_test  = df_full.iloc[n_train:].reset_index(drop=True)
y_train  = y[:n_train]
y_test   = y[n_train:]
sigma_true_test = sigma_true[n_train:]
mu_true_test    = mu_true[n_train:]

print(f"\nTrain: {len(df_train):,} | Test: {len(df_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Standard Gamma GLM (statsmodels) — baseline
# MAGIC
# MAGIC The standard approach: model only E[Y|X]. Dispersion is a single scalar
# MAGIC estimated from the Pearson chi-squared / degrees of freedom. It cannot vary
# MAGIC by risk segment.

# COMMAND ----------

# Standard Gamma GLM via statsmodels
X_train_glm = sm.add_constant(df_train[["age", "vehicle_value", "region", "ncd_level"]].values)
X_test_glm  = sm.add_constant(df_test[["age", "vehicle_value", "region", "ncd_level"]].values)

glm_gamma = sm.GLM(
    y_train,
    X_train_glm,
    family=sm.families.Gamma(link=sm.families.links.Log()),
)
glm_result = glm_gamma.fit()

print(glm_result.summary())

# COMMAND ----------

# GLM predictions
mu_glm_train = glm_result.predict(X_train_glm)
mu_glm_test  = glm_result.predict(X_test_glm)

# Constant dispersion: scale = 1/shape across all policies
# statsmodels Gamma scale = phi (dispersion), CV = sqrt(phi)
glm_scale = glm_result.scale  # estimated dispersion (1/shape)
glm_sigma_constant = np.sqrt(glm_scale)  # CV — same for every policy

print(f"Standard GLM:")
print(f"  Log-likelihood (train): {glm_result.llf:.1f}")
print(f"  Estimated dispersion (phi): {glm_scale:.4f}")
print(f"  Implied CV (sigma): {glm_sigma_constant:.4f}  [same for ALL policies]")
print(f"  True sigma range in test: [{sigma_true_test.min():.3f}, {sigma_true_test.max():.3f}]")
print()
print("The GLM uses a single sigma=%.3f for every policy." % glm_sigma_constant)
print("Young drivers in the test set have true sigma up to %.3f." % sigma_true_test[age[n_train:] < 28].max())
print("Their prediction intervals will be systematically too narrow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. GAMLSS Gamma — model both mu and sigma
# MAGIC
# MAGIC The key difference: sigma gets its own linear predictor. We use age and
# MAGIC vehicle_value as sigma covariates because the DGP tells us those drive CV.
# MAGIC In practice you'd select these via GAIC (see Section 5).
# MAGIC
# MAGIC Note on sigma formula choice: modelling sigma with too many covariates can
# MAGIC cause identifiability issues. Start with the 1-2 strongest drivers.

# COMMAND ----------

# Convert to polars for DistributionalGLM
df_train_pl = pl.from_pandas(df_train)
df_test_pl  = pl.from_pandas(df_test)

# GAMLSS: mu gets all four covariates, sigma gets age and vehicle_value
gamlss_model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age", "vehicle_value", "region", "ncd_level"],
        "sigma": ["age", "vehicle_value"],
    },
)

gamlss_model.fit(df_train_pl, y_train, verbose=True)
gamlss_model.summary()

# COMMAND ----------

# GAMLSS predictions
mu_gamlss_test    = gamlss_model.predict(df_test_pl, parameter="mu")
sigma_gamlss_test = gamlss_model.predict(df_test_pl, parameter="sigma")
cv_gamlss_test    = gamlss_model.volatility_score(df_test_pl)

print(f"GAMLSS predictions on test set:")
print(f"  mu:    mean £{mu_gamlss_test.mean():,.0f}, "
      f"range [£{mu_gamlss_test.min():,.0f}, £{mu_gamlss_test.max():,.0f}]")
print(f"  sigma: mean {sigma_gamlss_test.mean():.3f}, "
      f"range [{sigma_gamlss_test.min():.3f}, {sigma_gamlss_test.max():.3f}]")
print(f"  CV:    mean {cv_gamlss_test.mean():.3f}")
print()
print(f"True sigma range in test: [{sigma_true_test.min():.3f}, {sigma_true_test.max():.3f}]")
print(f"GAMLSS recovered sigma range: [{sigma_gamlss_test.min():.3f}, {sigma_gamlss_test.max():.3f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparison: log-likelihood and GAIC

# COMMAND ----------

# Compute log-likelihoods on held-out test set

# Standard GLM log-likelihood: Gamma with estimated shape and fitted means
glm_shape = 1.0 / glm_scale  # shape = 1/phi
ll_glm_test = np.sum(stats.gamma.logpdf(
    y_test,
    a=glm_shape,
    scale=mu_glm_test / glm_shape,
))

# GAMLSS log-likelihood: per-policy shape and scale
sigma_g = sigma_gamlss_test
mu_g    = mu_gamlss_test
shape_g = 1.0 / sigma_g**2
scale_g = mu_g * sigma_g**2
ll_gamlss_test = np.sum(stats.gamma.logpdf(y_test, a=shape_g, scale=scale_g))

# Number of parameters
n_test    = len(y_test)
n_params_glm    = 5 + 1   # 4 covariates + intercept + dispersion scalar
n_params_gamlss = 5 + 3   # mu: 5, sigma: 3 (intercept + age + vehicle_value)

aic_glm    = -2 * ll_glm_test    + 2 * n_params_glm
aic_gamlss = -2 * ll_gamlss_test + 2 * n_params_gamlss

print("=" * 55)
print(f"{'Metric':<35} {'GLM':>8} {'GAMLSS':>10}")
print("=" * 55)
print(f"{'Log-likelihood (test)':<35} {ll_glm_test:>8.1f} {ll_gamlss_test:>10.1f}")
print(f"{'AIC (test, approx)':<35} {aic_glm:>8.1f} {aic_gamlss:>10.1f}")
print(f"{'Mean NLL per policy (test)':<35} {-ll_glm_test/n_test:>8.3f} {-ll_gamlss_test/n_test:>10.3f}")
print(f"{'Number of parameters':<35} {n_params_glm:>8} {n_params_gamlss:>10}")
print("=" * 55)
print(f"\nGAMLSS LL improvement: {ll_gamlss_test - ll_glm_test:+.1f} nats on test set")
print(f"GAMLSS uses {n_params_gamlss - n_params_glm} extra parameters to achieve this.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prediction interval coverage by segment
# MAGIC
# MAGIC This is the key test. Standard GLMs produce prediction intervals by assuming
# MAGIC constant sigma. GAMLSS varies sigma by risk — which means its intervals should
# MAGIC be better calibrated within each segment.
# MAGIC
# MAGIC We compute 80% and 95% prediction intervals from the Gamma distribution using
# MAGIC fitted mu and sigma, then measure actual coverage by age group and vehicle value band.

# COMMAND ----------

def gamma_pi(mu, sigma, alpha):
    """
    Compute lower and upper quantiles of Gamma(mu, sigma) at probability alpha/2 and 1-alpha/2.
    sigma = CV => shape = 1/sigma^2, scale = mu * sigma^2
    """
    shape = 1.0 / sigma**2
    scale = mu * sigma**2
    lower = stats.gamma.ppf(alpha / 2, a=shape, scale=scale)
    upper = stats.gamma.ppf(1 - alpha / 2, a=shape, scale=scale)
    return lower, upper


age_test     = age[n_train:]
vehicle_test = vehicle_value[n_train:]

# Segment: age groups
segments = {
    "Age 17-27 (young)":    age_test < 28,
    "Age 28-44":             (age_test >= 28) & (age_test < 45),
    "Age 45-59":             (age_test >= 45) & (age_test < 60),
    "Age 60+ (mature)":      age_test >= 60,
}

# Vehicle value bands
vehicle_segs = {
    "Vehicle < £10k":         vehicle_test < 10_000,
    "Vehicle £10k-£20k":      (vehicle_test >= 10_000) & (vehicle_test < 20_000),
    "Vehicle £20k-£40k":      (vehicle_test >= 20_000) & (vehicle_test < 40_000),
    "Vehicle > £40k (HV)":    vehicle_test >= 40_000,
}

all_segs = {**segments, **vehicle_segs}

rows = []
for seg_name, mask in all_segs.items():
    n_seg = mask.sum()
    if n_seg < 50:
        continue

    y_seg = y_test[mask]

    # GLM intervals: constant sigma for all
    glm_lo80, glm_hi80 = gamma_pi(mu_glm_test[mask], np.full(n_seg, glm_sigma_constant), 0.20)
    glm_lo95, glm_hi95 = gamma_pi(mu_glm_test[mask], np.full(n_seg, glm_sigma_constant), 0.05)

    # GAMLSS intervals: per-policy sigma
    gam_lo80, gam_hi80 = gamma_pi(mu_gamlss_test[mask], sigma_gamlss_test[mask], 0.20)
    gam_lo95, gam_hi95 = gamma_pi(mu_gamlss_test[mask], sigma_gamlss_test[mask], 0.05)

    glm_cov80 = float(((y_seg >= glm_lo80) & (y_seg <= glm_hi80)).mean())
    glm_cov95 = float(((y_seg >= glm_lo95) & (y_seg <= glm_hi95)).mean())
    gam_cov80 = float(((y_seg >= gam_lo80) & (y_seg <= gam_hi80)).mean())
    gam_cov95 = float(((y_seg >= gam_lo95) & (y_seg <= gam_hi95)).mean())

    true_sigma_seg = sigma_true_test[mask].mean()
    gamlss_sigma_seg = sigma_gamlss_test[mask].mean()

    rows.append({
        "Segment": seg_name,
        "N": int(n_seg),
        "True sigma": round(true_sigma_seg, 3),
        "GLM sigma": round(glm_sigma_constant, 3),
        "GAMLSS sigma": round(gamlss_sigma_seg, 3),
        "GLM 80% cov": round(glm_cov80, 3),
        "GAMLSS 80% cov": round(gam_cov80, 3),
        "GLM 95% cov": round(glm_cov95, 3),
        "GAMLSS 95% cov": round(gam_cov95, 3),
    })

coverage_df = pd.DataFrame(rows)
print(coverage_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage table (HTML)

# COMMAND ----------

def _cell_style(actual, target, is_sigma=False):
    """Colour-code coverage cells: green = on target, orange = off."""
    if is_sigma:
        return ""
    diff = abs(actual - target)
    if diff < 0.02:
        bg = "#d4edda"  # green
    elif diff < 0.05:
        bg = "#fff3cd"  # amber
    else:
        bg = "#f8d7da"  # red
    return f"background:{bg}"


def render_coverage_table(df):
    hdr = """
    <style>
      .bc-table { border-collapse: collapse; font-family: monospace; font-size: 13px; }
      .bc-table th { background: #2c3e50; color: white; padding: 6px 12px; text-align: right; }
      .bc-table th:first-child { text-align: left; }
      .bc-table td { padding: 5px 12px; border-bottom: 1px solid #eee; text-align: right; }
      .bc-table td:first-child { text-align: left; font-weight: bold; }
    </style>
    <table class='bc-table'>
      <tr>
        <th>Segment</th><th>N</th>
        <th>True &sigma;</th><th>GLM &sigma;</th><th>GAMLSS &sigma;</th>
        <th>GLM 80%</th><th>GAMLSS 80%</th>
        <th>GLM 95%</th><th>GAMLSS 95%</th>
      </tr>
    """
    rows_html = ""
    for _, row in df.iterrows():
        seg = row["Segment"]
        n   = f"{row['N']:,}"
        ts  = f"{row['True sigma']:.3f}"
        gs  = f"{row['GLM sigma']:.3f}"
        gms = f"{row['GAMLSS sigma']:.3f}"
        g80 = row["GLM 80% cov"]
        m80 = row["GAMLSS 80% cov"]
        g95 = row["GLM 95% cov"]
        m95 = row["GAMLSS 95% cov"]

        rows_html += f"""
        <tr>
          <td>{seg}</td><td>{n}</td>
          <td>{ts}</td>
          <td style="color:#c0392b">{gs}</td>
          <td style="color:#27ae60">{gms}</td>
          <td style="{_cell_style(g80, 0.80)}">{g80:.1%}</td>
          <td style="{_cell_style(m80, 0.80)}">{m80:.1%}</td>
          <td style="{_cell_style(g95, 0.95)}">{g95:.1%}</td>
          <td style="{_cell_style(m95, 0.95)}">{m95:.1%}</td>
        </tr>"""

    footer = """
      <tr style="background:#f0f0f0;font-style:italic">
        <td colspan="5">Target coverage</td>
        <td>80.0%</td><td>80.0%</td>
        <td>95.0%</td><td>95.0%</td>
      </tr>
    </table>
    <p style="font-family:monospace;font-size:11px;color:#666">
      Green = within 2pp of target | Amber = within 5pp | Red = &gt;5pp off target
    </p>
    """
    displayHTML(hdr + rows_html + footer)


render_coverage_table(coverage_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Visualisations
# MAGIC
# MAGIC Three plots:
# MAGIC 1. **Sigma relativities by age** — how the fitted sigma varies with driver age
# MAGIC    vs the true DGP sigma, and the GLM's flat assumption
# MAGIC 2. **Calibration plot** — do the prediction intervals contain the right fraction
# MAGIC    of observations at each probability level?
# MAGIC 3. **Interval width vs observed spread** — does the model assign wider intervals
# MAGIC    to segments that actually have more spread?

# COMMAND ----------

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)
ax1 = fig.add_subplot(gs[0, :2])  # sigma by age — wide
ax2 = fig.add_subplot(gs[0, 2])   # calibration
ax3 = fig.add_subplot(gs[1, :])   # interval width vs spread

# ---------------------------------------------------------------
# Plot 1: Sigma relativities by age
# ---------------------------------------------------------------
age_bins = np.arange(17, 82, 2)
age_mids = (age_bins[:-1] + age_bins[1:]) / 2

age_test_arr = age[n_train:]

sigma_true_by_age  = []
sigma_gamlss_by_age = []

for lo, hi in zip(age_bins[:-1], age_bins[1:]):
    mask = (age_test_arr >= lo) & (age_test_arr < hi)
    if mask.sum() < 10:
        sigma_true_by_age.append(np.nan)
        sigma_gamlss_by_age.append(np.nan)
    else:
        sigma_true_by_age.append(sigma_true_test[mask].mean())
        sigma_gamlss_by_age.append(sigma_gamlss_test[mask].mean())

sigma_true_by_age   = np.array(sigma_true_by_age)
sigma_gamlss_by_age = np.array(sigma_gamlss_by_age)

ax1.plot(age_mids, sigma_true_by_age,   color="#e74c3c", lw=2.5, label="True sigma (DGP)")
ax1.plot(age_mids, sigma_gamlss_by_age, color="#2980b9", lw=2, ls="--", label="GAMLSS fitted sigma")
ax1.axhline(glm_sigma_constant, color="#7f8c8d", lw=1.5, ls=":", label=f"GLM constant sigma ({glm_sigma_constant:.3f})")
ax1.axvline(28, color="#e74c3c", alpha=0.25, lw=1)
ax1.text(28.5, sigma_gamlss_by_age[~np.isnan(sigma_gamlss_by_age)].max() * 0.97,
         "Age 28\nyoung driver\ncutoff", fontsize=8, color="#e74c3c", alpha=0.7)
ax1.set_xlabel("Driver age", fontsize=11)
ax1.set_ylabel("Sigma (coefficient of variation)", fontsize=11)
ax1.set_title("Fitted sigma vs true DGP sigma by driver age", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim(17, 80)
ax1.grid(alpha=0.3)

# ---------------------------------------------------------------
# Plot 2: Calibration curve
# ---------------------------------------------------------------
alphas = np.linspace(0.02, 0.98, 40)
glm_coverage   = []
gamlss_coverage = []

for a in alphas:
    lo_glm, hi_glm = gamma_pi(mu_glm_test, np.full(len(y_test), glm_sigma_constant), 1 - a)
    lo_gm,  hi_gm  = gamma_pi(mu_gamlss_test, sigma_gamlss_test, 1 - a)
    glm_coverage.append(float(((y_test >= lo_glm) & (y_test <= hi_glm)).mean()))
    gamlss_coverage.append(float(((y_test >= lo_gm) & (y_test <= hi_gm)).mean()))

ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax2.plot(alphas, glm_coverage,    color="#e67e22", lw=2, label="GLM")
ax2.plot(alphas, gamlss_coverage, color="#2980b9", lw=2, label="GAMLSS")
ax2.set_xlabel("Nominal coverage", fontsize=11)
ax2.set_ylabel("Actual coverage", fontsize=11)
ax2.set_title("Calibration curve\n(all test policies)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_aspect("equal")

# ---------------------------------------------------------------
# Plot 3: Interval width vs observed spread by segment
# ---------------------------------------------------------------
seg_labels = []
glm_widths  = []
gam_widths  = []
obs_stds    = []

for seg_name, mask in all_segs.items():
    if mask.sum() < 50:
        continue
    lo_glm, hi_glm = gamma_pi(mu_glm_test[mask],    np.full(mask.sum(), glm_sigma_constant), 0.05)
    lo_gam, hi_gam = gamma_pi(mu_gamlss_test[mask],  sigma_gamlss_test[mask], 0.05)

    # Width relative to segment mean (normalised)
    seg_mean = mu_gamlss_test[mask].mean()
    seg_labels.append(seg_name.replace("Vehicle ", "Veh "))
    glm_widths.append(float((hi_glm - lo_glm).mean() / seg_mean))
    gam_widths.append(float((hi_gam - lo_gam).mean() / seg_mean))
    obs_stds.append(float(y_test[mask].std() / seg_mean))

x_pos   = np.arange(len(seg_labels))
bar_w   = 0.25

ax3.bar(x_pos - bar_w, glm_widths,  bar_w, label="GLM 95% PI width / mean", color="#e67e22", alpha=0.85)
ax3.bar(x_pos,         gam_widths,  bar_w, label="GAMLSS 95% PI width / mean", color="#2980b9", alpha=0.85)
ax3.bar(x_pos + bar_w, obs_stds,    bar_w, label="Observed std / mean (actual spread)", color="#27ae60", alpha=0.85)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(seg_labels, rotation=30, ha="right", fontsize=9)
ax3.set_ylabel("Normalised width (relative to segment mean)", fontsize=10)
ax3.set_title("Prediction interval width vs actual spread by risk segment\n"
              "GAMLSS adapts to true heterogeneity — GLM cannot", fontsize=12, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(axis="y", alpha=0.3)

plt.suptitle(
    "GAMLSS vs Standard Gamma GLM — UK Motor Claim Severity\n"
    "insurance-distributional-glm benchmark",
    fontsize=14, fontweight="bold", y=1.01
)

plt.savefig("/tmp/distributional_glm_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sigma relativities: actuarial-style output
# MAGIC
# MAGIC The GAMLSS sigma submodel produces multiplicative relativities just like a
# MAGIC standard GLM mu submodel. These are the numbers a pricing analyst would use
# MAGIC to understand which risk segments need wider pricing bands.

# COMMAND ----------

# Sigma relativities — the key pricing output
sigma_rels = gamlss_model.relativities(parameter="sigma")
print("Sigma relativities (multiplicative effects on CV):")
print(sigma_rels)

# COMMAND ----------

mu_rels = gamlss_model.relativities(parameter="mu")
print("Mu relativities (multiplicative effects on expected severity):")
print(mu_rels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model selection: comparing Gamma, LogNormal, InverseGaussian
# MAGIC
# MAGIC `choose_distribution` fits each family and returns results ranked by GAIC.
# MAGIC Use this before committing to Gamma — on heavy-tailed portfolios (liability,
# MAGIC commercial motor) InverseGaussian sometimes wins.

# COMMAND ----------

selection_results = choose_distribution(
    df_train_pl,
    y_train,
    families=[Gamma(), LogNormal(), InverseGaussian()],
    formulas={
        "mu":    ["age", "vehicle_value", "region", "ncd_level"],
        "sigma": ["age", "vehicle_value"],
    },
    penalty=2.0,  # AIC; use np.log(len(y_train)) for BIC
)

print(f"\nFamily selection results (AIC, lower is better):")
print(f"{'Family':<20} {'GAIC(2)':>10} {'LogLik':>10} {'Converged':>10}")
print("-" * 55)
for r in selection_results:
    print(f"{r.family_name:<20} {r.gaic:>10.1f} {r.loglik:>10.1f} {str(r.converged):>10}")

best = selection_results[0]
print(f"\nBest family: {best.family_name} (GAIC = {best.gaic:.1f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quantile residuals diagnostic
# MAGIC
# MAGIC Randomised quantile residuals (Dunn & Smyth 1996) should be iid N(0,1)
# MAGIC for a correctly specified model. We check this on a subsample of 2,000
# MAGIC test observations to keep the call fast.

# COMMAND ----------

# Subsample for diagnostics
rng_diag = np.random.default_rng(999)
idx_sub  = rng_diag.choice(len(df_test), size=2000, replace=False)
df_test_sub = df_test_pl[idx_sub]
y_test_sub  = y_test[idx_sub]

resids_gamlss = quantile_residuals(gamlss_model, df_test_sub, y_test_sub, seed=42)
resids_gamlss = resids_gamlss[np.isfinite(resids_gamlss)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QQ plot
(osm, osr), (slope, intercept, r) = stats.probplot(resids_gamlss, dist="norm")
axes[0].scatter(osm, osr, s=6, alpha=0.4, color="steelblue")
axes[0].plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
axes[0].set_xlabel("Theoretical N(0,1) quantiles")
axes[0].set_ylabel("Quantile residuals")
axes[0].set_title("QQ plot — GAMLSS quantile residuals\n(should follow the diagonal)")
axes[0].grid(alpha=0.3)

# Histogram
axes[1].hist(resids_gamlss, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
x_norm = np.linspace(-4, 4, 200)
axes[1].plot(x_norm, stats.norm.pdf(x_norm), "r-", lw=2, label="N(0,1)")
axes[1].set_xlabel("Quantile residual")
axes[1].set_ylabel("Density")
axes[1].set_title("Residual distribution — GAMLSS\n(should match N(0,1) in red)")
axes[1].legend()
axes[1].grid(alpha=0.3)

resid_mean = resids_gamlss.mean()
resid_std  = resids_gamlss.std()
ks_stat, ks_p = stats.kstest(resids_gamlss, "norm")
print(f"Quantile residuals: mean={resid_mean:.3f}, std={resid_std:.3f}")
print(f"KS test vs N(0,1): stat={ks_stat:.4f}, p={ks_p:.4f}")
print("(p > 0.05 means we cannot reject normality — model is well specified)")

plt.tight_layout()
plt.savefig("/tmp/distributional_glm_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Business interpretation: volatility scoring for pricing
# MAGIC
# MAGIC The practical output: every policy gets a CV score. Underwriters and pricing
# MAGIC managers can use this to:
# MAGIC
# MAGIC - Apply larger contingency loadings to high-CV risks
# MAGIC - Set tighter reinsurance attachment points for volatile segments
# MAGIC - Flag policies that need manual referral (CV > threshold)
# MAGIC - Report on portfolio volatility concentration to risk committee

# COMMAND ----------

# Volatility scoring on the full test set
cv_test = gamlss_model.volatility_score(df_test_pl)

# Segment by CV quartile
cv_quartiles = np.quantile(cv_test, [0.25, 0.50, 0.75, 1.0])
cv_labels    = ["Q1 (low CV)", "Q2", "Q3", "Q4 (high CV)"]
cv_q_masks   = [
    cv_test <= cv_quartiles[0],
    (cv_test > cv_quartiles[0]) & (cv_test <= cv_quartiles[1]),
    (cv_test > cv_quartiles[1]) & (cv_test <= cv_quartiles[2]),
    cv_test > cv_quartiles[2],
]

print(f"{'CV Quartile':<18} {'CV range':>14} {'Mean age':>10} {'Mean £veh':>12} "
      f"{'% young':>10} {'Sigma':<8} {'Actual std/mean':>16}")
print("-" * 95)
for label, mask in zip(cv_labels, cv_q_masks):
    if mask.sum() == 0:
        continue
    cv_lo = cv_test[mask].min()
    cv_hi = cv_test[mask].max()
    mn_age = age[n_train:][mask].mean()
    mn_veh = vehicle_value[n_train:][mask].mean()
    pct_young = (age[n_train:][mask] < 28).mean()
    mn_sigma = sigma_gamlss_test[mask].mean()
    obs_cv   = y_test[mask].std() / y_test[mask].mean()
    print(f"{label:<18} [{cv_lo:.3f},{cv_hi:.3f}]  {mn_age:>8.1f}y  "
          f"£{mn_veh:>9,.0f}  {pct_young:>9.1%}  {mn_sigma:>6.3f}  {obs_cv:>14.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Final summary
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Dimension | Standard Gamma GLM | GAMLSS Gamma |
# MAGIC |-----------|-------------------|--------------|
# MAGIC | Sigma modelling | Single scalar for all policies | Per-policy, driven by age + vehicle value |
# MAGIC | Prediction interval calibration | Systematically miscalibrated by segment | Closer to nominal within each segment |
# MAGIC | Log-likelihood (test) | Lower | Higher |
# MAGIC | Parameters | 6 | 8 (2 extra: age and vehicle_value effects on sigma) |
# MAGIC | Actionable output | Point prediction only | CV score per policy for loading decisions |
# MAGIC
# MAGIC ### When to use GAMLSS
# MAGIC
# MAGIC Use it when your portfolio has **genuine heterogeneity of variance** — which
# MAGIC means almost all motor, household, and commercial lines books. The signal is
# MAGIC clear when:
# MAGIC
# MAGIC - Different age groups or vehicle classes have different empirical CVs
# MAGIC - Prediction interval coverage varies significantly across segments
# MAGIC - Reinsurance pricing or ILFs require reliable distributional assumptions
# MAGIC
# MAGIC ### When the standard GLM is adequate
# MAGIC
# MAGIC If your book is genuinely homogeneous (narrow vehicle class, single distribution
# MAGIC channel, stable NCD mix), the GAMLSS sigma submodel will converge to a near-
# MAGIC constant and you won't lose much by using the simpler model. Use GAIC to check.
# MAGIC
# MAGIC ### Important caveats
# MAGIC
# MAGIC The RS algorithm converges to a local optimum. On real data:
# MAGIC - Start sigma with 1-2 predictors maximum
# MAGIC - Check convergence (`model.converged`)
# MAGIC - Run quantile residuals to validate distributional assumption
# MAGIC - Use `choose_distribution` to confirm Gamma vs alternatives

# COMMAND ----------

# Print final summary numbers
print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"\nData: {len(df_train):,} train | {len(df_test):,} test policies")
print(f"\nModel log-likelihoods (test set):")
print(f"  Standard Gamma GLM:   {ll_glm_test:,.1f}")
print(f"  GAMLSS Gamma:         {ll_gamlss_test:,.1f}")
print(f"  Improvement:          {ll_gamlss_test - ll_glm_test:+,.1f} nats")
print(f"\nSigma estimation (test set):")
print(f"  True sigma range:     [{sigma_true_test.min():.3f}, {sigma_true_test.max():.3f}]")
print(f"  GLM constant sigma:   {glm_sigma_constant:.3f} (same for all)")
print(f"  GAMLSS sigma range:   [{sigma_gamlss_test.min():.3f}, {sigma_gamlss_test.max():.3f}]")
print(f"\nGAMLSS model converged: {gamlss_model.converged}")
print(f"GAMLSS GAIC(2): {gamlss_model.gaic(penalty=2):.1f}")
print(f"\nBest family by GAIC: {best.family_name}")
print("=" * 60)
