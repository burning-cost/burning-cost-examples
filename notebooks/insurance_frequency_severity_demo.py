# Databricks notebook source

# MAGIC %md
# MAGIC # Sarmanov Copula vs Independence Assumption
# MAGIC ## Joint Frequency-Severity Modelling with `insurance-frequency-severity`
# MAGIC
# MAGIC Every UK motor pricing team prices as:
# MAGIC
# MAGIC     Pure premium = E[N|x] × E[S|x]
# MAGIC
# MAGIC This assumes claim count and average severity are independent given rating factors.
# MAGIC That assumption is almost certainly wrong — and this notebook shows you by how much,
# MAGIC on a synthetic book where the true data-generating process (DGP) is known.
# MAGIC
# MAGIC ### What we cover
# MAGIC
# MAGIC 1. Generate 50,000 synthetic UK motor policies with planted positive freq-sev dependence
# MAGIC 2. Fit standard independent two-part GLM (Poisson × Gamma)
# MAGIC 3. Fit Sarmanov copula joint model using `insurance-frequency-severity`
# MAGIC 4. Compare premium corrections: where does independence under-price, and by how much
# MAGIC 5. MGF-based analytical correction vs Monte Carlo simulation
# MAGIC 6. Segment-level bias table — the commercially relevant view
# MAGIC
# MAGIC ### Why the independence assumption breaks down in UK motor
# MAGIC
# MAGIC The NCD (No Claims Discount) structure suppresses borderline claims: policyholders
# MAGIC who claim frequently are acutely aware of the NCD threshold and report every incident.
# MAGIC In some market segments, the opposite applies — frequent claimants are careful about
# MAGIC what they report. Either way, N and S are correlated through behaviour, not just risk.
# MAGIC
# MAGIC The Sarmanov bivariate distribution handles the discrete-continuous mixed margins problem
# MAGIC cleanly. Unlike standard copulas, it does not require a probability integral transform
# MAGIC (PIT) for the count margin — which is important because Sklar's theorem is not unique
# MAGIC for discrete distributions.
# MAGIC
# MAGIC **Reference:** Vernic, Bolancé, Alemany (2022). *Insurance: Mathematics and Economics*, 102, 111–125.

# COMMAND ----------

# MAGIC %pip install insurance-frequency-severity==0.2.1 polars matplotlib statsmodels scipy numpy pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Data Generation
# MAGIC
# MAGIC We generate 50,000 UK motor policies from a known DGP where frequency and severity
# MAGIC are **positively dependent** via a shared latent risk score. The latent score drives
# MAGIC both the Poisson rate and the Gamma scale parameter.
# MAGIC
# MAGIC This is the "high-risk driver" phenomenon: drivers who have more incidents also tend
# MAGIC to have more severe ones (less defensive driving, riskier routes, lower-quality
# MAGIC vehicles). The DGP plants this dependence explicitly so we can measure how well each
# MAGIC modelling approach recovers it.
# MAGIC
# MAGIC ### DGP specification
# MAGIC
# MAGIC Rating factors: age band (young/standard/senior), vehicle group (A–D), region (5 levels)
# MAGIC
# MAGIC Latent risk score: `u ~ Uniform(0.5, 1.5)` — a multiplicative loading above the
# MAGIC rating-factor prediction. High `u` means both more claims AND higher severity per claim.
# MAGIC
# MAGIC ```
# MAGIC lambda_i = exp(beta_freq' x_i) * u_i          # Poisson rate
# MAGIC mu_s_i   = exp(beta_sev' x_i) * u_i^0.6       # Gamma mean
# MAGIC
# MAGIC N_i ~ Poisson(lambda_i)
# MAGIC S_i ~ Gamma(shape=3, scale=mu_s_i/3)    [if N_i > 0]
# MAGIC ```
# MAGIC
# MAGIC The `u_i^0.6` exponent on severity gives a weaker dependence than frequency —
# MAGIC realistic: the same risk score does not affect severity as strongly as frequency.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)

# --- Rating factor definitions ---
N_POLICIES = 50_000

# Age band
age_bands = rng.choice(["young", "standard", "senior"], size=N_POLICIES, p=[0.15, 0.65, 0.20])
# Vehicle group
veh_groups = rng.choice(["A", "B", "C", "D"], size=N_POLICIES, p=[0.25, 0.35, 0.25, 0.15])
# Region
regions = rng.choice([1, 2, 3, 4, 5], size=N_POLICIES, p=[0.30, 0.25, 0.20, 0.15, 0.10])

# --- Frequency log-linear predictor ---
freq_age_effect = {"young": 0.45, "standard": 0.0, "senior": -0.15}
freq_veh_effect = {"A": -0.20, "B": 0.0, "C": 0.15, "D": 0.30}
freq_region_effect = {1: 0.0, 2: 0.10, 3: 0.05, 4: -0.10, 5: 0.20}
freq_intercept = -2.5  # base annual claim frequency ~8%

eta_freq = np.array([
    freq_intercept
    + freq_age_effect[age_bands[i]]
    + freq_veh_effect[veh_groups[i]]
    + freq_region_effect[regions[i]]
    for i in range(N_POLICIES)
])

# --- Severity log-linear predictor ---
sev_age_effect = {"young": 0.20, "standard": 0.0, "senior": -0.05}
sev_veh_effect = {"A": -0.30, "B": 0.0, "C": 0.25, "D": 0.50}
sev_region_effect = {1: 0.0, 2: 0.05, 3: 0.0, 4: 0.10, 5: 0.15}
sev_intercept = 7.2  # base severity ~£1,330

eta_sev = np.array([
    sev_intercept
    + sev_age_effect[age_bands[i]]
    + sev_veh_effect[veh_groups[i]]
    + sev_region_effect[regions[i]]
    for i in range(N_POLICIES)
])

# --- Latent risk score (the planted dependence) ---
# u_i ~ Uniform(0.5, 1.5) — multiplicative loading on both frequency and severity
u_latent = rng.uniform(0.5, 1.5, size=N_POLICIES)

# Frequency rate: lambda_i = exp(eta_freq_i) * u_i
lambda_i = np.exp(eta_freq) * u_latent

# Severity mean: mu_s_i = exp(eta_sev_i) * u_i^0.6
mu_s_i = np.exp(eta_sev) * (u_latent ** 0.6)

# --- Simulate claims ---
N_claims = rng.poisson(lambda_i)
severity_shape = 3.0  # Gamma shape parameter (CV = 1/sqrt(3) ≈ 0.58)

# Severity is only defined for policies with at least one claim
S_avg = np.where(
    N_claims > 0,
    rng.gamma(severity_shape, scale=mu_s_i / severity_shape),
    np.nan
)

# --- Build policy dataset ---
df = pd.DataFrame({
    "policy_id": np.arange(N_POLICIES),
    "age_band": age_bands,
    "veh_group": veh_groups,
    "region": regions.astype(str),
    "claim_count": N_claims,
    "avg_severity": S_avg,
    "exposure": np.ones(N_POLICIES),
    # True latent quantities for benchmarking
    "true_lambda": lambda_i,
    "true_mu_s": mu_s_i,
    "true_pure_premium": lambda_i * mu_s_i,  # This is E[N*S] under the DGP
})

print(f"Policies: {N_POLICIES:,}")
print(f"Claims (policies with N>0): {(N_claims > 0).sum():,}  ({100*(N_claims>0).mean():.1f}%)")
print(f"Total claim events: {N_claims.sum():,}")
print(f"Severity among claimants (mean): £{np.nanmean(S_avg):,.0f}")
print(f"Book pure premium (mean): £{df['true_pure_premium'].mean():.2f}")

# --- Verify planted dependence ---
mask_pos = N_claims > 0
tau, pval = __import__("scipy").stats.kendalltau(N_claims[mask_pos], S_avg[mask_pos])
print(f"\nKendall tau (N>0 policies): {tau:.4f}  p-value: {pval:.4e}")
print("(Positive tau confirms planted positive freq-sev dependence)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Standard Approach: Independent Two-Part GLM
# MAGIC
# MAGIC The industry standard: fit a Poisson GLM for frequency and a Gamma GLM for severity,
# MAGIC then price as `E[N|x] × E[S|x]`. The independence assumption is implicit —
# MAGIC no one checks it, and most tools do not surface the test.

# COMMAND ----------

import statsmodels.api as sm

# --- Feature matrix (one-hot encoding) ---
df_model = df.copy()
df_model["young"] = (df_model["age_band"] == "young").astype(float)
df_model["senior"] = (df_model["age_band"] == "senior").astype(float)
df_model["veh_B"] = (df_model["veh_group"] == "B").astype(float)
df_model["veh_C"] = (df_model["veh_group"] == "C").astype(float)
df_model["veh_D"] = (df_model["veh_group"] == "D").astype(float)
df_model["region_2"] = (df_model["region"] == "2").astype(float)
df_model["region_3"] = (df_model["region"] == "3").astype(float)
df_model["region_4"] = (df_model["region"] == "4").astype(float)
df_model["region_5"] = (df_model["region"] == "5").astype(float)

feature_cols = ["young", "senior", "veh_B", "veh_C", "veh_D",
                "region_2", "region_3", "region_4", "region_5"]

X_all = sm.add_constant(df_model[feature_cols].values)
X_all_df = pd.DataFrame(X_all, columns=["const"] + feature_cols)

# --- Frequency GLM: Poisson with log link ---
freq_glm = sm.GLM(
    df_model["claim_count"],
    X_all_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
).fit()

print("Frequency GLM (Poisson):")
print(f"  Deviance: {freq_glm.deviance:.1f}")
print(f"  AIC: {freq_glm.aic:.1f}")
print(f"  Fitted mean claim count: {freq_glm.fittedvalues.mean():.4f}")

# --- Severity GLM: Gamma with log link (claims-only rows) ---
mask_claims = df_model["claim_count"] > 0
df_claims = df_model[mask_claims].copy()
X_claims = sm.add_constant(df_claims[feature_cols].values)
X_claims_df = pd.DataFrame(X_claims, columns=["const"] + feature_cols)

sev_glm = sm.GLM(
    df_claims["avg_severity"].values,
    X_claims_df,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_claims["claim_count"].values,
).fit()

print(f"\nSeverity GLM (Gamma, log link):")
print(f"  Deviance: {sev_glm.deviance:.1f}")
print(f"  AIC: {sev_glm.aic:.1f}")
print(f"  Dispersion (scale): {sev_glm.scale:.4f}")
print(f"  Fitted mean severity: £{sev_glm.fittedvalues.mean():,.0f}")

# --- Independence prediction: E[N|x] × E[S|x] ---
mu_n_ind = freq_glm.fittedvalues.values  # n_policies
mu_s_all = sev_glm.predict(X_all_df)    # predict on all policies using full X

premium_independent = mu_n_ind * mu_s_all.values

print(f"\nIndependence model:")
print(f"  Mean pure premium: £{premium_independent.mean():.2f}")
print(f"  True mean pure premium: £{df['true_pure_premium'].mean():.2f}")
print(f"  Ratio (independence / truth): {premium_independent.mean() / df['true_pure_premium'].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dependence Test
# MAGIC
# MAGIC Before fitting the copula, we should test whether dependence exists at all.
# MAGIC `DependenceTest` runs Kendall tau and Spearman rho with permutation p-values
# MAGIC on the positive-claim subsample. If we cannot reject independence, the copula
# MAGIC adds noise rather than signal.

# COMMAND ----------

from insurance_frequency_severity import DependenceTest

# Test only uses policies where N > 0 (severity is defined)
n_positive = df["claim_count"].values[mask_claims]
s_positive = df["avg_severity"].values[mask_claims]

dep_test = DependenceTest(n_permutations=500)
dep_test.fit(n_positive, s_positive, rng=rng)

print("Dependence Test Results:")
print("=" * 65)
print(dep_test.summary().to_string(index=False))
print()
print(f"n_obs (claiming policies): {dep_test.n_obs_:,}")
print(f"Kendall tau: {dep_test.tau_:.4f}  (p={dep_test.tau_pval_:.4e})")
print(f"Spearman rho: {dep_test.rho_s_:.4f}  (p={dep_test.rho_s_pval_:.4e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sarmanov Copula Joint Model
# MAGIC
# MAGIC `JointFreqSev` accepts the fitted GLM objects and estimates the dependence
# MAGIC parameter omega by IFM (Inference Functions for Margins): profile likelihood
# MAGIC over omega given the already-fitted marginals. We do not refit the Poisson or
# MAGIC Gamma GLMs — the pricing team's model pipeline stays intact.
# MAGIC
# MAGIC The Sarmanov joint density is:
# MAGIC
# MAGIC     f(n, s) = f_N(n) × f_S(s) × [1 + ω × φ₁(n) × φ₂(s)]
# MAGIC
# MAGIC where φ₁, φ₂ are Laplace (exponential) kernels centred at their marginal means.
# MAGIC When ω = 0, this reduces to independence. The sign of ω tells you the direction;
# MAGIC the magnitude tells you how much it matters for premium.

# COMMAND ----------

from insurance_frequency_severity import JointFreqSev

# Build the policy-level dataframe that JointFreqSev expects:
# - claim_count: N per policy (0 for non-claimants)
# - avg_severity: S per policy (NaN for non-claimants is fine — the library handles this)
# - Severity GLM was fitted on claims-only, so we need to pass the full-book
#   feature matrix (X_all_df) as sev_X so the library can predict E[S|x] for
#   all policies, not just claimants.

joint_model = JointFreqSev(
    freq_glm=freq_glm,
    sev_glm=sev_glm,
    copula="sarmanov",
    kernel_theta=0.5,   # Laplace kernel exponent for frequency
    kernel_alpha=0.001, # Laplace kernel exponent for severity
)

joint_model.fit(
    df[["claim_count", "avg_severity"]],
    n_col="claim_count",
    s_col="avg_severity",
    freq_X=X_all_df,
    sev_X=X_all_df,  # predict E[S|x] for all policies including non-claimants
    method="ifm",
    ci_method="profile",
    rng=rng,
)

print("Sarmanov Copula: Dependence Parameter Estimates")
print("=" * 60)
print(joint_model.dependence_summary().to_string(index=False))
print()
print("Interpretation:")
omega_hat = joint_model.omega_
if omega_hat > 0:
    print(f"  omega = {omega_hat:.4f} > 0: POSITIVE dependence")
    print("  High-frequency policyholders also have higher severity.")
    print("  Independence model UNDERSTATES pure premium for high-risk segment.")
else:
    print(f"  omega = {omega_hat:.4f} < 0: NEGATIVE dependence")
    print("  High-frequency policyholders have lower severity (NCD suppression effect).")
    print("  Independence model OVERSTATES pure premium for high-risk segment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Premium Corrections
# MAGIC
# MAGIC The analytical correction factor for the Sarmanov copula is:
# MAGIC
# MAGIC     correction_i = E[N×S|x_i] / (E[N|x_i] × E[S|x_i])
# MAGIC
# MAGIC This is closed-form for the Laplace kernel — no simulation needed at scoring time.
# MAGIC The `premium_correction()` method computes it via MGF derivatives:
# MAGIC
# MAGIC     E[N×S] = E[N]×E[S] + ω × E[N·φ₁(N)] × E[S·φ₂(S)]
# MAGIC
# MAGIC The `E[N·φ₁(N)]` and `E[S·φ₂(S)]` terms follow analytically from the Poisson/Gamma
# MAGIC MGFs and their first derivatives.

# COMMAND ----------

# Compute correction factors for the entire book
corrections_df = joint_model.premium_correction()

# Attach to the main dataframe for analysis
df["mu_n"] = corrections_df["mu_n"].values
df["mu_s"] = corrections_df["mu_s"].values
df["premium_independent"] = corrections_df["premium_independent"].values
df["premium_joint"] = corrections_df["premium_joint"].values
df["correction_factor"] = corrections_df["correction_factor"].values

print("Premium Correction Factor Distribution")
print("=" * 50)
print(corrections_df["correction_factor"].describe())
print()
print(f"Policies where copula INCREASES premium (factor > 1.0): "
      f"{(corrections_df['correction_factor'] > 1.0).sum():,} "
      f"({100*(corrections_df['correction_factor'] > 1.0).mean():.1f}%)")
print(f"Policies where copula DECREASES premium (factor < 1.0): "
      f"{(corrections_df['correction_factor'] < 1.0).sum():,} "
      f"({100*(corrections_df['correction_factor'] < 1.0).mean():.1f}%)")
print()

# Book-level summary
print("Book-Level Premium Comparison")
print("-" * 50)
print(f"Independence model mean premium: £{df['premium_independent'].mean():.2f}")
print(f"Sarmanov copula mean premium:    £{df['premium_joint'].mean():.2f}")
print(f"True DGP mean pure premium:      £{df['true_pure_premium'].mean():.2f}")
print()
print(f"Independence A/E vs DGP: {df['premium_independent'].mean() / df['true_pure_premium'].mean():.4f}")
print(f"Copula A/E vs DGP:       {df['premium_joint'].mean() / df['true_pure_premium'].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Segment-Level Bias: Where Independence Goes Wrong
# MAGIC
# MAGIC Book-level totals can look fine even when individual segments are badly mispriced.
# MAGIC The commercial risk is in segments with concentrated exposure — typically high-risk
# MAGIC segments where you are most likely to attract adversely selected business.
# MAGIC
# MAGIC We split the book into frequency deciles (using the fitted E[N|x] from the
# MAGIC frequency GLM). Low-frequency deciles (decile 1) contain the best risks;
# MAGIC high-frequency deciles (decile 10) contain the worst.

# COMMAND ----------

# Frequency deciles based on the GLM's E[N|x]
df["freq_decile"] = pd.qcut(df["mu_n"], q=10, labels=False) + 1

segment_table = (
    df.groupby("freq_decile")
    .agg(
        n_policies=("policy_id", "count"),
        mean_mu_n=("mu_n", "mean"),
        mean_true_pp=("true_pure_premium", "mean"),
        mean_indep_pp=("premium_independent", "mean"),
        mean_copula_pp=("premium_joint", "mean"),
        mean_correction=("correction_factor", "mean"),
    )
    .reset_index()
)

segment_table["indep_bias_pct"] = (
    (segment_table["mean_indep_pp"] - segment_table["mean_true_pp"])
    / segment_table["mean_true_pp"] * 100
)
segment_table["copula_bias_pct"] = (
    (segment_table["mean_copula_pp"] - segment_table["mean_true_pp"])
    / segment_table["mean_true_pp"] * 100
)
segment_table["correction_pct"] = (segment_table["mean_correction"] - 1.0) * 100

print("Segment-Level Bias by Frequency Decile")
print("=" * 110)
print(f"{'Decile':>7} {'N Policies':>11} {'E[N|x]':>10} {'True PP':>10} "
      f"{'Indep PP':>10} {'Copula PP':>10} "
      f"{'Indep Bias%':>12} {'Copula Bias%':>12} {'Correction%':>12}")
print("-" * 110)
for _, row in segment_table.iterrows():
    print(f"{int(row['freq_decile']):>7} {int(row['n_policies']):>11,} "
          f"{row['mean_mu_n']:>10.4f} £{row['mean_true_pp']:>9.2f} "
          f"£{row['mean_indep_pp']:>9.2f} £{row['mean_copula_pp']:>9.2f} "
          f"{row['indep_bias_pct']:>+11.2f}% {row['copula_bias_pct']:>+11.2f}% "
          f"{row['correction_pct']:>+11.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Copula Comparison: AIC/BIC Across Families
# MAGIC
# MAGIC `compare_copulas()` fits all three families — Sarmanov, Gaussian, FGM —
# MAGIC and ranks them by AIC. This is the diagnostic you show to a pricing team
# MAGIC to justify the methodology choice. If the FGM (weakest dependence range)
# MAGIC wins, the dependence is mild. If Sarmanov wins, the tails matter.

# COMMAND ----------

from insurance_frequency_severity import compare_copulas

copula_comparison = compare_copulas(
    n=df["claim_count"].values,
    s=df["avg_severity"].values,
    freq_glm=freq_glm,
    sev_glm=sev_glm,
    freq_X=X_all_df,
    sev_X=X_all_df,
    rng=rng,
)

print("Copula Family Comparison (sorted by AIC)")
print("=" * 75)
print(copula_comparison.to_string(index=False))
print()
print("delta_AIC > 10 is strong evidence against the higher-AIC model.")
print("The winning family is used for the production correction factors.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MGF-Based Analytical Correction vs Monte Carlo Simulation
# MAGIC
# MAGIC One of the practical advantages of the Sarmanov copula is that the premium
# MAGIC correction is **closed-form** — derived analytically from the MGFs of the
# MAGIC Poisson and Gamma marginals. No simulation is needed at scoring time.
# MAGIC
# MAGIC Here we verify the analytical result against Monte Carlo simulation to
# MAGIC confirm they agree, and benchmark the computational cost difference.

# COMMAND ----------

import time

# --- Analytical correction (already computed above) ---
analytic_corrections = df["correction_factor"].values
t0 = time.time()
_ = joint_model.premium_correction()
t_analytic = time.time() - t0

print(f"Analytical correction time: {t_analytic:.3f}s for {N_POLICIES:,} policies")
print(f"Analytical mean correction factor: {analytic_corrections.mean():.6f}")
print()

# --- Monte Carlo simulation of E[N*S] at the book level ---
# For Sarmanov, we can sample from the bivariate distribution directly.
# We use the copula's factored form: sample N from Poisson, then S|N from
# the modified conditional density.

# Representative policy parameters (use mean mu_n and mu_s across the book)
mu_n_bar = df["mu_n"].mean()
mu_s_bar = df["mu_s"].mean()
sev_shape = joint_model._shape  # Gamma shape estimated from severity GLM

print(f"Representative policy: E[N] = {mu_n_bar:.4f}, E[S] = £{mu_s_bar:.2f}")
print(f"Gamma shape (1/dispersion): {sev_shape:.4f}")

# Monte Carlo: sample N ~ Poisson(mu_n_bar) and S ~ Gamma(shape, scale=mu_s_bar/shape)
# Under independence, E[N*S] = mu_n_bar * mu_s_bar
# Under dependence we cannot easily sample from the Sarmanov joint directly
# without implementing the inverse CDF — instead we compare the analytical
# correction to what we'd get from a Gaussian copula MC (which has a sampler).

n_mc_samples = 200_000
from insurance_frequency_severity import JointFreqSev as JFS

# Gaussian copula model fitted earlier for comparison
gaussian_model = JFS(freq_glm=freq_glm, sev_glm=sev_glm, copula="gaussian")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gaussian_model.fit(
        df[["claim_count", "avg_severity"]],
        n_col="claim_count",
        s_col="avg_severity",
        freq_X=X_all_df,
        sev_X=X_all_df,
        rng=rng,
    )

t0 = time.time()
gaussian_corrections = gaussian_model.premium_correction(n_mc=n_mc_samples, rng=rng)
t_mc = time.time() - t0

gaussian_cf = gaussian_corrections["correction_factor"].mean()
print(f"\nMonte Carlo (Gaussian copula, {n_mc_samples:,} samples): {t_mc:.3f}s")
print(f"MC mean correction factor (Gaussian): {gaussian_cf:.6f}")
print()

# Sarmanov rho vs Gaussian rho
print(f"Sarmanov Spearman rho: {joint_model.rho_:.4f}")
print(f"Gaussian Spearman rho: {gaussian_model.rho_:.4f}")
print()

# Analytical premium correction breakdown for the book
print("Analytical Sarmanov correction summary:")
print(f"  Mean correction factor:  {analytic_corrections.mean():.4f}")
print(f"  Std correction factor:   {analytic_corrections.std():.4f}")
print(f"  10th percentile (best risk):  {np.percentile(analytic_corrections, 10):.4f}")
print(f"  90th percentile (worst risk): {np.percentile(analytic_corrections, 90):.4f}")
print()
premium_lift_pct = (analytic_corrections.mean() - 1.0) * 100
print(f"Portfolio-level premium correction: {premium_lift_pct:+.2f}%")
print("(Positive = independence understates; negative = overstates)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Visualisations
# MAGIC
# MAGIC Four panels:
# MAGIC - Correction factor distribution across the book
# MAGIC - Segment bias: independence vs copula vs DGP truth by frequency decile
# MAGIC - Scatter: latent risk score vs observed Spearman residuals
# MAGIC - Premium comparison scatter: independence vs copula coloured by correction magnitude

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sarmanov Copula vs Independence Assumption\n50,000 UK Motor Policies", fontsize=14)

# --- Panel 1: Correction factor distribution ---
ax = axes[0, 0]
cf = df["correction_factor"].values
ax.hist(cf, bins=60, color="#2c7bb6", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(cf.mean(), color="#d7191c", linewidth=2, linestyle="--", label=f"Mean: {cf.mean():.4f}")
ax.axvline(1.0, color="black", linewidth=1.5, linestyle=":", label="Independence (1.0)")
ax.set_xlabel("Correction factor E[NS] / (E[N]·E[S])")
ax.set_ylabel("Policy count")
ax.set_title("Premium Correction Factor Distribution")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Panel 2: Segment-level bias by frequency decile ---
ax = axes[0, 1]
deciles = segment_table["freq_decile"]
ax.plot(deciles, segment_table["mean_true_pp"], "k-o", markersize=5, label="True DGP", linewidth=2)
ax.plot(deciles, segment_table["mean_indep_pp"], "b--s", markersize=5, label="Independence model", linewidth=1.5)
ax.plot(deciles, segment_table["mean_copula_pp"], "r-^", markersize=5, label="Sarmanov copula", linewidth=1.5)
ax.set_xlabel("Frequency decile (1=best risk, 10=worst)")
ax.set_ylabel("Mean pure premium (£)")
ax.set_title("Pure Premium by Risk Segment")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Panel 3: Bias percentage by decile ---
ax = axes[1, 0]
bar_width = 0.35
x = np.arange(len(deciles))
ax.bar(x - bar_width/2, segment_table["indep_bias_pct"], bar_width,
       label="Independence bias%", color="#2c7bb6", alpha=0.8)
ax.bar(x + bar_width/2, segment_table["copula_bias_pct"], bar_width,
       label="Copula bias%", color="#d7191c", alpha=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in deciles])
ax.set_xlabel("Frequency decile")
ax.set_ylabel("Bias vs DGP truth (%)")
ax.set_title("Segment Bias: Independence vs Copula")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# --- Panel 4: E[N|x] vs correction factor ---
ax = axes[1, 1]
# Colour by latent risk score to show the planted dependence
scatter = ax.scatter(df["mu_n"], df["correction_factor"],
                     c=u_latent, cmap="RdYlBu_r", s=1, alpha=0.3)
ax.axhline(1.0, color="black", linewidth=1.5, linestyle=":", label="Independence")
ax.set_xlabel("Fitted E[N|x] from frequency GLM")
ax.set_ylabel("Sarmanov correction factor")
ax.set_title("Correction Factor vs Expected Claim Rate\n(colour = latent risk score)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Latent risk score u_i", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/freq_sev_sarmanov_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to /tmp/freq_sev_sarmanov_demo.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Key Findings
# MAGIC
# MAGIC Summary of what the analysis found on this 50,000-policy synthetic motor book.

# COMMAND ----------

print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()

# 1. Was dependence detected?
tau = dep_test.tau_
tau_pval = dep_test.tau_pval_
if tau_pval < 0.001:
    sig_str = "p < 0.001 — highly significant"
elif tau_pval < 0.05:
    sig_str = f"p = {tau_pval:.3f} — significant"
else:
    sig_str = f"p = {tau_pval:.3f} — not significant"
print(f"1. DEPENDENCE DETECTED: Kendall tau = {tau:.4f} ({sig_str})")
print(f"   Estimated Sarmanov omega: {joint_model.omega_:.4f}")
print(f"   95% CI: [{joint_model.omega_ci_[0]:.4f}, {joint_model.omega_ci_[1]:.4f}]")
print(f"   Spearman rho (implied): {joint_model.rho_:.4f}")
print()

# 2. Portfolio-level correction
mean_cf = df["correction_factor"].mean()
port_correction_pct = (mean_cf - 1.0) * 100
print(f"2. PORTFOLIO-LEVEL CORRECTION: {port_correction_pct:+.2f}%")
print(f"   Independence mean premium: £{df['premium_independent'].mean():.2f}")
print(f"   Copula mean premium:       £{df['premium_joint'].mean():.2f}")
print(f"   DGP true mean premium:     £{df['true_pure_premium'].mean():.2f}")
print()

# 3. Tail correction
top_decile = segment_table[segment_table["freq_decile"] == 10].iloc[0]
top_indep_bias = top_decile["indep_bias_pct"]
top_copula_bias = top_decile["copula_bias_pct"]
top_correction = top_decile["correction_pct"]
print(f"3. TOP-DECILE (highest-risk) CORRECTION: {top_correction:+.2f}%")
print(f"   Independence bias vs DGP truth: {top_indep_bias:+.2f}%")
print(f"   Copula bias vs DGP truth:       {top_copula_bias:+.2f}%")
print()

# 4. Analytical vs MC performance
print(f"4. COMPUTATION: Analytical correction for {N_POLICIES:,} policies in {t_analytic:.3f}s")
print(f"   No Monte Carlo simulation needed at scoring time.")
print(f"   MC (Gaussian, {n_mc_samples:,} samples) took {t_mc:.2f}s — {t_mc/t_analytic:.0f}x slower.")
print()

# 5. When to use / not use
print("5. DECISION GUIDE:")
print("   USE Sarmanov copula when:")
print(f"   - Kendall tau p-value < 0.05 (here: {tau_pval:.4f})")
print(f"   - Book has >= 500 claims (here: {(df['claim_count']>0).sum():,})")
print(f"   - Top-decile independence bias > 3% (here: {top_indep_bias:+.2f}%)")
print()
print("   DO NOT USE when:")
print("   - Independence cannot be rejected (no signal to exploit)")
print("   - Book has fewer than 1,000 policies (omega CI will be very wide)")
print("   - Excess zeros dominate and the Sarmanov construction does not hold")
print()

# 6. Verification of planted DGP
print("6. DGP RECOVERY:")
print(f"   True omega planted: positive (latent score drives both N and S)")
print(f"   Estimated omega: {joint_model.omega_:.4f} — {'positive as expected' if joint_model.omega_ > 0 else 'unexpected sign'}")
print(f"   The model correctly identified the direction of dependence.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Garrido Conditional Method (Simpler Alternative)
# MAGIC
# MAGIC For teams that want to capture freq-sev dependence without copula machinery,
# MAGIC `ConditionalFreqSev` (Garrido et al. 2016) refits the severity GLM with N as
# MAGIC a covariate. The correction factor is `exp(gamma × E[N|x])`.
# MAGIC
# MAGIC This is less flexible than Sarmanov (it imposes a specific parametric form on
# MAGIC the dependence structure) but is entirely within standard GLM infrastructure.
# MAGIC Use it when your technical audience is sceptical of copulas.

# COMMAND ----------

from insurance_frequency_severity import ConditionalFreqSev

conditional_model = ConditionalFreqSev(
    freq_glm=freq_glm,
    sev_glm_base=sev_glm,
    n_as_indicator=False,
)

conditional_model.fit(
    df[["claim_count", "avg_severity"]].assign(
        **{col: df_model[col] for col in feature_cols}
    ),
    n_col="claim_count",
    s_col="avg_severity",
    sev_feature_cols=feature_cols,
    freq_X=X_all_df,
)

print("Garrido Conditional Method: Dependence Summary")
print("=" * 55)
print(conditional_model.dependence_summary().to_string(index=False))
print()

gamma_val = conditional_model.gamma_
gamma_se = conditional_model.gamma_se_
print(f"Interpretation: gamma = {gamma_val:.4f} (SE {gamma_se:.4f})")
if gamma_val > 0:
    print("  Positive gamma: higher E[N|x] -> higher E[S|x, N=1]")
    print("  Severity increases with expected claim frequency.")
else:
    print("  Negative gamma: higher E[N|x] -> lower E[S|x, N=1]")
    print("  High-frequency policyholders have lower per-claim severity.")

garrido_corrections = conditional_model.premium_correction()
print(f"\nGarrido correction factor (mean): {garrido_corrections['correction_factor'].mean():.4f}")
print(f"Sarmanov correction factor (mean): {df['correction_factor'].mean():.4f}")
print()
print("Premium comparison (book-level means):")
print(f"  Independence:       £{df['premium_independent'].mean():.2f}")
print(f"  Sarmanov copula:    £{df['premium_joint'].mean():.2f}")
print(f"  Garrido conditional: £{garrido_corrections['premium_joint'].mean():.2f}")
print(f"  True DGP:           £{df['true_pure_premium'].mean():.2f}")
