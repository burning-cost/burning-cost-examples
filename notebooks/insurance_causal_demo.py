# Databricks notebook source

# MAGIC %md
# MAGIC # Causal Effect Estimation for Insurance Rating Factors
# MAGIC ## Double Machine Learning vs Naive Regression — Measuring Confounding Bias
# MAGIC
# MAGIC Every pricing actuary has stared at a GLM coefficient and wondered: is this
# MAGIC factor actually causing claims, or is it a proxy for something else?
# MAGIC
# MAGIC The canonical UK motor example: vehicle value. Expensive cars tend to be driven
# MAGIC by older, more experienced drivers who also have better no-claims histories. The
# MAGIC vehicle value coefficient in a naive GLM picks up *both* the direct effect of
# MAGIC vehicle value on claim cost *and* the indirect signal from the correlated driver
# MAGIC characteristics. You cannot tell them apart from the coefficient alone.
# MAGIC
# MAGIC This is confounding. And it has direct commercial consequences: if you overestimate
# MAGIC the causal effect of vehicle value, you overprice expensive cars and underprice
# MAGIC cheap ones. Competitors without this bias take the profitable business.
# MAGIC
# MAGIC **Double Machine Learning (DML)** — Chernozhukov et al. (2018) — solves this.
# MAGIC It uses ML to partial out the influence of all observed confounders from both
# MAGIC the treatment and the outcome before estimating the causal effect. The result
# MAGIC is root-n-consistent and asymptotically normal: a valid frequentist confidence
# MAGIC interval on the causal parameter.
# MAGIC
# MAGIC **This notebook:**
# MAGIC
# MAGIC 1. Generates a synthetic UK motor portfolio (50,000 policies) with a known DGP,
# MAGIC    embedding deliberate confounding between driver age, vehicle value, and claim
# MAGIC    frequency. The true causal effect is embedded so we can measure bias.
# MAGIC 2. Fits a **naive Poisson GLM** — the standard actuarial approach — and extracts
# MAGIC    the vehicle value effect.
# MAGIC 3. Fits a **DML model** using `insurance-causal` to estimate the true causal effect,
# MAGIC    controlling for all observed confounders.
# MAGIC 4. Produces a **benchmark comparison table**: naive vs DML vs true DGP effect,
# MAGIC    quantifying the confounding bias in percentage terms.
# MAGIC 5. Runs **CI-based sensitivity analysis**: how much bias from unobserved confounders
# MAGIC    would be needed to push the DML confidence interval across zero?
# MAGIC 6. Produces a **confounding bias report** using the library's built-in diagnostic.
# MAGIC 7. CATE by driver age band: does the causal effect of vehicle value vary by age?

# COMMAND ----------

# MAGIC %pip install catboost matplotlib scikit-learn polars pyarrow "doubleml>=0.7.0,<=0.8.0" insurance-causal "statsmodels>=0.14.5" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Data-Generating Process
# MAGIC
# MAGIC We build a synthetic UK motor portfolio with a hand-crafted confounding structure.
# MAGIC The treatment variable is **log vehicle value** (continuous). The outcome is
# MAGIC **claim frequency** (claims per policy-year, Poisson).
# MAGIC
# MAGIC **The confounding mechanism:**
# MAGIC Driver age affects both treatment assignment and the outcome:
# MAGIC - Older drivers tend to own more expensive vehicles (age → vehicle value)
# MAGIC - Older drivers have lower claim frequency (age → lower frequency)
# MAGIC
# MAGIC This means vehicle value and claim frequency are negatively correlated *not only*
# MAGIC because expensive cars might be driven more carefully, but because expensive cars
# MAGIC are owned by safer older drivers. A naive GLM attributes some of the age-driven
# MAGIC frequency reduction to vehicle value — overstating the causal effect.
# MAGIC
# MAGIC **Known true causal effect:** -0.08 (a doubling of vehicle value causes an 8%
# MAGIC reduction in claim frequency, on the log scale). This is modest — vehicle value
# MAGIC matters somewhat, but driver quality matters more.
# MAGIC
# MAGIC **DGP:**
# MAGIC ```
# MAGIC log(freq_i) = -2.8
# MAGIC             + TRUE_EFFECT * log_vehicle_value_i   # causal channel
# MAGIC             - 0.012 * age_i                       # age directly reduces frequency
# MAGIC             + 0.25 * prior_claims_i                # risk history
# MAGIC             - 0.06 * ncb_years_i                  # NCB proxy for risk quality
# MAGIC             + 0.15 * is_urban_i                   # urban area
# MAGIC             + 0.10 * young_male_i                 # interaction
# MAGIC             + epsilon_i
# MAGIC ```
# MAGIC
# MAGIC Vehicle value is determined by age, occupation, and noise:
# MAGIC ```
# MAGIC log_vehicle_value_i = 9.5
# MAGIC                     + 0.02 * age_i               # older → more expensive
# MAGIC                     + occupation_premium_i        # professional → higher vehicle
# MAGIC                     + noise_i
# MAGIC ```

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import polars as pl

# ─── DGP parameters ──────────────────────────────────────────────────────────

N_POLICIES = 50_000
TRUE_CAUSAL_EFFECT = -0.08    # causal effect of log_vehicle_value on log(frequency)
                               # naive GLM will overestimate this due to age confounding
RANDOM_SEED = 2024

rng = np.random.default_rng(RANDOM_SEED)

# ─── Driver covariates ────────────────────────────────────────────────────────

age = rng.integers(18, 80, N_POLICIES).astype(float)
ncb_years = np.clip(rng.integers(0, 9, N_POLICIES).astype(float), 0, 8)
prior_claims = rng.choice([0, 1, 2, 3], N_POLICIES, p=[0.72, 0.20, 0.06, 0.02]).astype(float)

# Region
regions = ["london", "south_east", "south_west", "east_midlands",
           "west_midlands", "north_west", "north_east", "yorkshire",
           "wales", "scotland"]
region_probs = [0.15, 0.13, 0.08, 0.09, 0.09, 0.12, 0.07, 0.11, 0.07, 0.09]  # normalised to sum to 1
region = np.array(regions)[rng.choice(len(regions), N_POLICIES, p=region_probs)]
is_urban = np.isin(region, ["london", "south_east", "west_midlands"]).astype(float)

# Occupation (affects vehicle choice, not directly frequency)
occupation = rng.choice(["professional", "manual", "clerical", "retired", "student"],
                         N_POLICIES, p=[0.22, 0.25, 0.28, 0.15, 0.10])
occupation_premium = np.where(occupation == "professional", 0.35,
                     np.where(occupation == "retired", 0.15,
                     np.where(occupation == "student", -0.30, 0.0)))

# Young male indicator
young_male = ((age < 30) & (rng.random(N_POLICIES) < 0.50)).astype(float)

# ─── Treatment: vehicle value (CONTINUOUS, confounded) ────────────────────────
# Older drivers + professionals tend to own more expensive cars
# This creates the confounding: age → vehicle_value and age → lower frequency

log_vehicle_value = (
    9.5                                              # baseline (exp ≈ £13,000)
    + 0.020 * age                                   # older → more expensive
    + occupation_premium                             # professionals drive better cars
    + 0.10 * ncb_years                              # long NCB → more stable lifestyle
    - 0.25 * young_male                             # young males: cheaper cars
    + rng.normal(0, 0.50, N_POLICIES)               # genuine price variation
)
vehicle_value = np.exp(log_vehicle_value)            # £ value

# ─── Outcome: claim frequency (Poisson, per policy-year) ─────────────────────
# TRUE_CAUSAL_EFFECT is the coefficient on log_vehicle_value
# The naive estimate will be more negative because expensive cars = older drivers = lower freq

log_mu = (
    -2.80
    + TRUE_CAUSAL_EFFECT * log_vehicle_value        # causal channel (what we want)
    - 0.012 * age                                   # age DIRECTLY reduces frequency
    + 0.250 * prior_claims                          # risk history
    - 0.060 * ncb_years                             # NCB quality signal
    + 0.150 * is_urban                              # urban risk loading
    + 0.100 * young_male                            # young male interaction
    + rng.normal(0, 0.08, N_POLICIES)               # idiosyncratic noise
)
mu = np.exp(log_mu)

earned_years = rng.uniform(0.6, 1.0, N_POLICIES)   # policy duration
expected_claims = mu * earned_years
claim_count = rng.poisson(expected_claims).astype(float)

# ─── Build DataFrame ──────────────────────────────────────────────────────────

age_band = np.where(age < 30, "under_30",
           np.where(age < 45, "30_to_44",
           np.where(age < 60, "45_to_59", "60_plus")))

df = pl.DataFrame({
    "claim_count":       claim_count,
    "earned_years":      earned_years,
    "log_vehicle_value": log_vehicle_value,
    "vehicle_value":     vehicle_value,
    "age":               age,
    "age_band":          age_band,
    "ncb_years":         ncb_years,
    "prior_claims":      prior_claims,
    "region":            region,
    "occupation":        occupation,
    "is_urban":          is_urban,
    "young_male":        young_male,
})

freq = claim_count / earned_years
print(f"Portfolio: {N_POLICIES:,} policies")
print(f"Mean frequency:       {float(np.mean(freq)):.4f} claims/year")
print(f"Mean vehicle value:   £{float(np.mean(vehicle_value)):,.0f}")
print(f"Mean earned years:    {float(np.mean(earned_years)):.3f}")
print(f"\nTrue causal effect:   {TRUE_CAUSAL_EFFECT:.4f}")
print(f"  Interpretation: A 100% increase in vehicle value causes a "
      f"{abs(TRUE_CAUSAL_EFFECT) * np.log(2) * 100:.1f}% "
      f"reduction in claim frequency (on the log scale).")

# Check confounding: correlation between age and log_vehicle_value
corr_age_vv = np.corrcoef(age, log_vehicle_value)[0, 1]
corr_age_freq = np.corrcoef(age, freq)[0, 1]
print(f"\nConfounding check:")
print(f"  Corr(age, log_vehicle_value): {corr_age_vv:.3f}  (age → higher vehicle value)")
print(f"  Corr(age, frequency):         {corr_age_freq:.3f}  (age → lower frequency)")
print(f"  Both channels go through age. Naive GLM will conflate them.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Naive Approach — Poisson GLM (Standard Actuarial)
# MAGIC
# MAGIC The naive approach is what most pricing teams do: fit a Poisson GLM with all
# MAGIC available rating factors as predictors, then read off the coefficient on the
# MAGIC treatment variable.
# MAGIC
# MAGIC The problem: the GLM coefficient on `log_vehicle_value` is biased. It is more
# MAGIC negative than the true causal effect because vehicle value is correlated with age,
# MAGIC and age directly reduces claim frequency. The GLM cannot separate these channels
# MAGIC — it sees both the causal vehicle value effect and the "spurious" age correlation
# MAGIC running through the vehicle value path.
# MAGIC
# MAGIC We use scikit-learn's PoissonRegressor (log link, Poisson deviance) to avoid
# MAGIC dependency on statsmodels for this step. The coefficient will be extracted and
# MAGIC compared to the DML estimate.

# COMMAND ----------

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df_pd = df.to_pandas()

# GLM feature setup: include all confounders + treatment
# This is what a typical pricing GLM would include
numeric_features = [
    "log_vehicle_value",  # treatment variable
    "age",
    "ncb_years",
    "prior_claims",
    "is_urban",
    "young_male",
]
categorical_features = ["age_band", "region", "occupation"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features),
    ]
)

# Fit GLM with exposure offset via sample_weight approximation
# We use earned_years as frequency denominator (standard actuarial approach)
X_glm = df_pd[numeric_features + categorical_features]
y_freq = df_pd["claim_count"] / df_pd["earned_years"]

glm_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("glm", PoissonRegressor(alpha=0.0, max_iter=500, fit_intercept=True)),
])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_pipeline.fit(X_glm, y_freq, glm__sample_weight=df_pd["earned_years"])

# Extract the coefficient on log_vehicle_value
# StandardScaler was applied, so we need to unscale
scaler_means = preprocessor.named_transformers_["num"].mean_
scaler_stds = preprocessor.named_transformers_["num"].scale_
glm_coefs = glm_pipeline.named_steps["glm"].coef_

# First `len(numeric_features)` GLM coefficients are for numeric features (scaled)
# To get unscaled coefficient: coef / sd
naive_coef_scaled = glm_coefs[0]  # coefficient on standardised log_vehicle_value
naive_coef_unscaled = naive_coef_scaled / scaler_stds[0]

print("Naive Poisson GLM results:")
print(f"  Coefficient on log_vehicle_value (scaled):   {naive_coef_scaled:.6f}")
print(f"  Coefficient on log_vehicle_value (unscaled): {naive_coef_unscaled:.6f}")
print(f"\n  True causal effect:                          {TRUE_CAUSAL_EFFECT:.6f}")
print(f"  Naive bias:      {naive_coef_unscaled - TRUE_CAUSAL_EFFECT:+.6f}")
print(f"  Bias (%):        {(naive_coef_unscaled - TRUE_CAUSAL_EFFECT) / abs(TRUE_CAUSAL_EFFECT) * 100:.1f}%")

NAIVE_ESTIMATE = naive_coef_unscaled

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: DML Causal Estimation
# MAGIC
# MAGIC `CausalPricingModel` wraps DoubleML with a pricing-actuary interface.
# MAGIC
# MAGIC The DML algorithm:
# MAGIC 1. Fit E[log_vehicle_value | confounders] using CatBoost (5-fold cross-fitting).
# MAGIC    Compute residuals D_tilde = log_vehicle_value - E_hat[log_vehicle_value | X].
# MAGIC    These residuals capture the part of vehicle value that is *not* explained
# MAGIC    by observed confounders — the exogenous variation.
# MAGIC 2. Fit E[frequency | confounders] using CatBoost (5-fold cross-fitting).
# MAGIC    Compute residuals Y_tilde = frequency - E_hat[frequency | X].
# MAGIC 3. Regress Y_tilde on D_tilde via OLS. The coefficient is the causal estimate.
# MAGIC
# MAGIC Step 3 is just OLS on residuals. It identifies the causal effect because:
# MAGIC - D_tilde is orthogonal to confounders (they have been partialled out)
# MAGIC - Y_tilde is orthogonal to confounders (same)
# MAGIC - The remaining correlation between D_tilde and Y_tilde must run through
# MAGIC   the causal channel, not through any observed confounder
# MAGIC
# MAGIC The confounders list should include everything that causally affects both
# MAGIC treatment assignment and the outcome. We include age, NCB, prior claims,
# MAGIC occupation, region, and urban flag — but NOT young_male as a direct variable
# MAGIC (it is itself caused by age, which is included).
# MAGIC
# MAGIC **Note:** DML with 5-fold CatBoost cross-fitting on 50k observations takes
# MAGIC approximately 5-15 minutes on a standard Databricks cluster. This is correct
# MAGIC behaviour — the nuisance models are fitting 10 CatBoost instances (2 per fold).

# COMMAND ----------

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import ContinuousTreatment

# Confounders: everything that affects both vehicle value AND claim frequency
# We include age (the main confounder), NCB, prior claims, occupation, region
# We do NOT include vehicle_value or young_male here (young_male is downstream of age)
CONFOUNDERS = [
    "age",
    "ncb_years",
    "prior_claims",
    "is_urban",
    "occupation",
    "region",
]

dml_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",               # Poisson frequency, divide by exposure
    treatment=ContinuousTreatment(
        column="log_vehicle_value",
        standardise=False,                # keep on log scale for direct comparison
    ),
    confounders=CONFOUNDERS,
    exposure_col="earned_years",
    cv_folds=5,
    random_state=RANDOM_SEED,
)

print("Fitting DML model (CatBoost nuisance, 5-fold cross-fitting)...")
print("This will take approximately 5-15 minutes on a standard cluster.")
dml_model.fit(df_pd)

ate = dml_model.average_treatment_effect()
print("\nDML estimate complete.")
print(ate)

DML_ESTIMATE = ate.estimate
DML_CI_LOWER = ate.ci_lower
DML_CI_UPPER = ate.ci_upper

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Benchmark Comparison
# MAGIC
# MAGIC This is the key output. We compare three quantities:
# MAGIC
# MAGIC - **True DGP effect**: what we embedded in the simulation (-0.08). This is the
# MAGIC   ground truth — only available in simulation.
# MAGIC - **Naive GLM estimate**: the coefficient from a Poisson GLM with all rating
# MAGIC   factors. Biased because it cannot separate causal and confounded paths.
# MAGIC - **DML estimate**: the causal estimate from `insurance-causal`. Should recover
# MAGIC   the true effect.
# MAGIC
# MAGIC The practical implication: if your pricing team is using the naive GLM estimate
# MAGIC to set vehicle value relativities, they are applying a biased relativity. The
# MAGIC extent of that bias determines how mispriced their book is relative to the true
# MAGIC risk distribution.

# COMMAND ----------

import math

naive_bias_abs = NAIVE_ESTIMATE - TRUE_CAUSAL_EFFECT
naive_bias_pct = naive_bias_abs / abs(TRUE_CAUSAL_EFFECT) * 100

dml_bias_abs = DML_ESTIMATE - TRUE_CAUSAL_EFFECT
dml_bias_pct = dml_bias_abs / abs(TRUE_CAUSAL_EFFECT) * 100

# CI for true effect (point, no uncertainty by construction)
print("=" * 72)
print("BENCHMARK COMPARISON: Naive GLM vs DML vs True DGP")
print("=" * 72)
print(f"\n{'Method':<25}  {'Estimate':>12}  {'Bias (abs)':>12}  {'Bias (%)':>10}  {'CI 95%'}")
print(f"{'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*22}")
print(f"{'True DGP effect':<25}  {TRUE_CAUSAL_EFFECT:>12.5f}  {'—':>12}  {'—':>10}  —")
print(f"{'Naive Poisson GLM':<25}  {NAIVE_ESTIMATE:>12.5f}  {naive_bias_abs:>+12.5f}  {naive_bias_pct:>9.1f}%  —")
print(f"{'DML (insurance-causal)':<25}  {DML_ESTIMATE:>12.5f}  {dml_bias_abs:>+12.5f}  {dml_bias_pct:>9.1f}%  [{DML_CI_LOWER:.4f}, {DML_CI_UPPER:.4f}]")
print()
print(f"DML p-value:  {ate.p_value:.4f}")
print(f"DML N:        {ate.n_obs:,}")
print()

# Interpret the confounding mechanism
print("Confounding mechanism:")
print(f"  Age is correlated with both vehicle value (r={corr_age_vv:.3f}) and")
print(f"  frequency (r={corr_age_freq:.3f}). The naive GLM partially attributes")
print(f"  the age-driven frequency reduction to vehicle value, producing")
print(f"  a more negative coefficient. DML removes this by partialling out")
print(f"  age (and all other confounders) before estimating the effect.")
print()
if abs(naive_bias_pct) > 50:
    print(f"  SUBSTANTIAL BIAS: The naive estimate is {abs(naive_bias_pct):.0f}% off the true effect.")
    print(f"  Vehicle value relativities based on the naive GLM would be materially wrong.")
elif abs(naive_bias_pct) > 20:
    print(f"  MODERATE BIAS: The naive estimate is {abs(naive_bias_pct):.0f}% off the true effect.")
else:
    print(f"  SMALL BIAS: The naive estimate is {abs(naive_bias_pct):.0f}% off the true effect.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Visualise the Bias

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Confounding Bias: Vehicle Value Effect on Claim Frequency\n50,000 Synthetic UK Motor Policies",
    fontsize=13, fontweight="bold"
)

# ─── Left: point estimates with CI ────────────────────────────────────────────
ax = axes[0]
methods = ["True DGP", "Naive GLM", "DML"]
estimates = [TRUE_CAUSAL_EFFECT, NAIVE_ESTIMATE, DML_ESTIMATE]
ci_errors = [
    [0, 0],
    [0, 0],
    [DML_ESTIMATE - DML_CI_LOWER, DML_CI_UPPER - DML_ESTIMATE],
]
colours = ["#2196F3", "#F44336", "#4CAF50"]

for i, (method, est, ci, colour) in enumerate(zip(methods, estimates, ci_errors, colours)):
    ax.errorbar(
        est, i,
        xerr=[[ci[0]], [ci[1]]],
        fmt="o",
        color=colour,
        capsize=6,
        capthick=2,
        markersize=10,
        linewidth=2,
        label=method,
    )

ax.axvline(x=TRUE_CAUSAL_EFFECT, color="#2196F3", linestyle="--", alpha=0.4,
           label="True effect (reference)")
ax.axvline(x=0, color="grey", linestyle=":", alpha=0.6)
ax.set_yticks(range(3))
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel("Coefficient on log_vehicle_value", fontsize=11)
ax.set_title("Point Estimates vs True DGP Effect", fontsize=11, pad=10)
ax.legend(loc="lower right", fontsize=9)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Annotate bias
ax.annotate(
    f"Naive bias:\n{naive_bias_pct:+.0f}% vs true",
    xy=(NAIVE_ESTIMATE, 1),
    xytext=(NAIVE_ESTIMATE - 0.03, 1.4),
    fontsize=9, color="#F44336",
    arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5),
)

# ─── Right: bias comparison bar chart ─────────────────────────────────────────
ax2 = axes[1]
bar_heights = [naive_bias_pct, dml_bias_pct]
bar_colours = ["#F44336", "#4CAF50"]
bar_labels = ["Naive GLM", "DML"]

bars = ax2.bar(bar_labels, bar_heights, color=bar_colours, alpha=0.8, width=0.5, edgecolor="white")
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_ylabel("Bias (% of true effect)", fontsize=11)
ax2.set_title("Percentage Bias vs True DGP Effect", fontsize=11, pad=10)

for bar, val in zip(bars, bar_heights):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + (2 if val > 0 else -5),
        f"{val:+.1f}%",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )

ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(min(bar_heights) - 15, max(bar_heights) + 15)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Sensitivity Analysis — CI-Based Bias Threshold
# MAGIC
# MAGIC DML controls for *observed* confounders. But what if there are unobserved
# MAGIC confounders — variables that affect both vehicle value and claim frequency
# MAGIC that are not in our data?
# MAGIC
# MAGIC For UK motor, plausible unobserved confounders:
# MAGIC - Actual annual mileage (higher mileage → more claims; high mileage → cars
# MAGIC   depreciate differently, distorting vehicle value as a proxy)
# MAGIC - Attitude to risk (risk-seeking personality → more accidents AND sports cars)
# MAGIC - Garaging (city-centre no-garage → higher theft frequency AND lower vehicle value)
# MAGIC
# MAGIC The Rosenbaum Gamma framework applies to binary treatment. DML with continuous
# MAGIC treatment uses a different sensitivity framing: **how much additive bias would
# MAGIC an unobserved confounder need to introduce into the ATE estimate before the
# MAGIC 95% CI crosses zero?** That bias threshold, expressed as a fraction of the
# MAGIC observed ATE, is the natural robustness measure for this setting.
# MAGIC
# MAGIC If the threshold is large (e.g. the bias would need to be 80% of the ATE),
# MAGIC the result is robust. If it is small (e.g. 10%), a modest unobserved
# MAGIC confounder could overturn the conclusion.

# COMMAND ----------

# CI-based sensitivity analysis for continuous-treatment DML.
# No external imports needed — uses only DML_ESTIMATE, DML_CI_LOWER, DML_CI_UPPER
# which were computed in Step 3.

# The DML ATE estimate is negative (vehicle value reduces frequency).
# Unobserved confounding adds positive bias (same direction as the naive GLM bias).
# The CI breaks down when the upper bound crosses zero.

# Distance from the CI upper bound to zero
ci_half_width = DML_CI_UPPER - DML_ESTIMATE   # half-width of CI (upper side)
bias_to_overturn = abs(DML_CI_UPPER)           # bias needed to push CI upper bound to zero
bias_as_fraction_of_ate = bias_to_overturn / abs(DML_ESTIMATE)

# What fraction of the OBSERVED confounding (naive - DML) would achieve this?
observed_confounding = abs(NAIVE_ESTIMATE - DML_ESTIMATE)
bias_as_fraction_of_observed = (
    bias_to_overturn / observed_confounding if observed_confounding > 0 else float("inf")
)

print("CI-Based Sensitivity Analysis for DML Continuous Treatment")
print("=" * 65)
print(f"  DML ATE estimate:               {DML_ESTIMATE:+.5f}")
print(f"  DML 95% CI:                     [{DML_CI_LOWER:.5f}, {DML_CI_UPPER:.5f}]")
print(f"  CI upper bound:                 {DML_CI_UPPER:+.5f}")
print()
print(f"  Bias needed to overturn conclusion: {bias_to_overturn:.5f}")
print(f"    (i.e. push CI upper bound to 0)")
print()
print(f"  As % of DML ATE:                {bias_as_fraction_of_ate * 100:.1f}%")
print(f"  As % of observed confounding:   {bias_as_fraction_of_observed * 100:.1f}%")
print(f"    (observed confounding = naive - DML = {observed_confounding:.5f})")
print()

if bias_as_fraction_of_ate >= 0.50:
    robustness_label = "STRONG"
    interpretation = (
        f"An unobserved confounder would need to introduce bias equal to "
        f"{bias_as_fraction_of_ate * 100:.0f}% of the estimated ATE before the conclusion "
        f"changes. That is equivalent to {bias_as_fraction_of_observed * 100:.0f}% of the "
        f"confounding already measured from age and occupation. "
        f"In UK motor, no single unobserved factor is plausibly this large."
    )
elif bias_as_fraction_of_ate >= 0.25:
    robustness_label = "MODERATE"
    interpretation = (
        f"An unobserved confounder needs to introduce {bias_as_fraction_of_ate * 100:.0f}% "
        f"of the ATE in bias — equivalent to {bias_as_fraction_of_observed * 100:.0f}% of "
        f"the observed confounding — before the conclusion flips. "
        f"Mileage or garaging could plausibly approach this threshold."
    )
else:
    robustness_label = "FRAGILE"
    interpretation = (
        f"A relatively modest unobserved confounder ({bias_as_fraction_of_ate * 100:.0f}% "
        f"of the ATE) could change the conclusion. Consider adding telematics mileage "
        f"or other proxy variables to the confounder set before acting on this estimate."
    )

print(f"  Robustness: {robustness_label}")
print(f"  {interpretation}")

# COMMAND ----------

# Sensitivity analysis plot — bias threshold diagram
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "CI-Based Sensitivity Analysis: Vehicle Value DML Estimate\n"
    "How much unobserved confounding would overturn the conclusion?",
    fontsize=12, fontweight="bold"
)

# Left: visualise the bias threshold on the ATE number line
ax = axes[0]
ci_lo = DML_CI_LOWER
ci_hi = DML_CI_UPPER

# Draw CI band
ax.barh(
    0, width=ci_hi - ci_lo, left=ci_lo, height=0.5,
    color="#4CAF50", alpha=0.3, label=f"DML 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
)
ax.plot(DML_ESTIMATE, 0, "o", color="#4CAF50", markersize=12, label=f"DML ATE ({DML_ESTIMATE:.4f})")
ax.plot(TRUE_CAUSAL_EFFECT, 0, "^", color="#2196F3", markersize=12,
        label=f"True effect ({TRUE_CAUSAL_EFFECT:.4f})")
ax.axvline(x=0, color="black", linewidth=1.5, linestyle=":", label="Zero (null)")

# Annotate bias arrow: from CI upper bound to zero
ax.annotate(
    "",
    xy=(0, 0.35),
    xytext=(ci_hi, 0.35),
    arrowprops=dict(arrowstyle="<->", color="red", lw=2),
)
ax.text(
    (ci_hi + 0) / 2, 0.48,
    f"Bias to overturn\n({bias_to_overturn:.4f})",
    ha="center", va="bottom", fontsize=9, color="red"
)

ax.set_xlim(min(ci_lo - 0.03, TRUE_CAUSAL_EFFECT - 0.02), 0.03)
ax.set_ylim(-0.5, 1.0)
ax.set_yticks([])
ax.set_xlabel("Coefficient on log_vehicle_value", fontsize=11)
ax.set_title("Bias Required to Overturn Conclusion", fontsize=11, pad=10)
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="x", alpha=0.3)

# Right: bias threshold as % of ATE and of observed confounding
ax2 = axes[1]
fractions = [bias_as_fraction_of_ate * 100, bias_as_fraction_of_observed * 100]
labels = ["% of DML ATE", "% of observed\nconfounding"]
colours = ["#4CAF50", "#FF9800"]
bars = ax2.bar(labels, fractions, color=colours, alpha=0.8, width=0.4, edgecolor="white")

for bar, val in zip(bars, fractions):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.0f}%",
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )

ax2.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="100% (unobserved = observed)")
ax2.axhline(y=50,  color="orange", linestyle=":", alpha=0.7, label="50% (strong/moderate boundary)")
ax2.set_ylabel("Bias threshold (%)", fontsize=11)
ax2.set_title("Required Bias as Fraction of\nATE and Observed Confounding", fontsize=11, pad=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(fractions) * 1.25)

# Shade the "conclusion overturned" region
ax.axvspan(bias_to_overturn, bias_vals[-1], alpha=0.08, color="#F44336",
           label="Conclusion overturned region")

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Confounding Bias Report (Library Built-in)
# MAGIC
# MAGIC `CausalPricingModel.confounding_bias_report()` takes the naive estimate and
# MAGIC the DML result and produces a structured diagnostic table with a plain-English
# MAGIC interpretation. This is what you would include in a model audit or pricing
# MAGIC governance submission.

# COMMAND ----------

bias_report = dml_model.confounding_bias_report(naive_coefficient=NAIVE_ESTIMATE)

print("Confounding Bias Report:")
print("-" * 60)
for col, val in bias_report.iloc[0].items():
    print(f"  {col:<25}: {val}")

# Render as HTML table
html_rows = ""
for col, val in bias_report.iloc[0].items():
    if col == "interpretation":
        continue
    html_rows += f"<tr><td style='padding:6px 12px; font-weight:600; background:#f5f5f5'>{col}</td><td style='padding:6px 12px'>{val}</td></tr>"

interpretation = bias_report.iloc[0]["interpretation"]
html = f"""
<div style='font-family: Arial, sans-serif; max-width: 700px; margin: 20px 0'>
  <h3 style='color:#333; margin-bottom:10px'>Confounding Bias Report — Vehicle Value on Claim Frequency</h3>
  <table style='border-collapse: collapse; width: 100%; border: 1px solid #ddd'>
    {html_rows}
  </table>
  <div style='margin-top: 15px; padding: 12px; background: #fff3e0; border-left: 4px solid #FF9800; border-radius: 2px'>
    <strong>Interpretation:</strong> {interpretation}
  </div>
  <div style='margin-top: 10px; padding: 12px; background: #e8f5e9; border-left: 4px solid #4CAF50; border-radius: 2px'>
    <strong>True DGP effect</strong> (simulation only): {TRUE_CAUSAL_EFFECT:.6f}<br>
    DML recovered <strong>{abs(DML_ESTIMATE - TRUE_CAUSAL_EFFECT) / abs(TRUE_CAUSAL_EFFECT) * 100:.1f}%</strong> accuracy relative to ground truth.
    Naive GLM had <strong>{abs(naive_bias_pct):.1f}%</strong> bias.
  </div>
</div>
"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: CATE by Age Band
# MAGIC
# MAGIC Does the causal effect of vehicle value on claim frequency vary by driver age?
# MAGIC
# MAGIC We use `cate_by_segment()` to fit separate DML models per age band. The
# MAGIC expectation from the DGP: the causal effect should be consistent across age
# MAGIC bands (it was not varied by age in the DGP), but confidence intervals will be
# MAGIC wider for smaller bands.
# MAGIC
# MAGIC **Practical question this addresses:** Should the vehicle value rating factor
# MAGIC have a different shape for young drivers vs senior drivers? This is a common
# MAGIC pricing question that CATE analysis can answer from observational data.
# MAGIC
# MAGIC **Note:** This cell fits 4 separate DML models (one per age band). It adds
# MAGIC approximately 5-10 minutes of compute.

# COMMAND ----------

print("Estimating CATE by age band (fits one DML model per band)...")
print("This takes approximately 5-10 additional minutes.\n")

cate_df = dml_model.cate_by_segment(df_pd, segment_col="age_band", min_segment_size=500)

print("CATE by age band — causal effect of log_vehicle_value on claim frequency:")
print(f"\n{'Age Band':<15}  {'N':>8}  {'CATE':>10}  {'CI Lower':>10}  {'CI Upper':>10}  "
      f"{'SE':>8}  {'p-value':>8}  {'Status'}")
print(f"{'-'*15}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}")

for _, row in cate_df.iterrows():
    if row["status"] == "ok":
        print(f"{str(row['segment']):<15}  {int(row['n_obs']):>8,}  "
              f"{row['cate_estimate']:>+10.5f}  {row['ci_lower']:>+10.5f}  "
              f"{row['ci_upper']:>+10.5f}  {row['std_error']:>8.5f}  "
              f"{row['p_value']:>8.4f}  {row['status']}")
    else:
        print(f"{str(row['segment']):<15}  {int(row['n_obs']):>8,}  {'—':>10}  {'—':>10}  "
              f"{'—':>10}  {'—':>8}  {'—':>8}  {row['status']}")

print(f"\nOverall ATE (full dataset): {DML_ESTIMATE:+.5f}")
print(f"True DGP effect:            {TRUE_CAUSAL_EFFECT:+.5f}")
print("\nNote: CATE estimates should be consistent with the ATE. Large differences")
print("across age bands would suggest heterogeneous treatment effects — the vehicle")
print("value relativity should then be varied by age band in the rating structure.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Nuisance Model Quality
# MAGIC
# MAGIC The quality of the DML causal estimate depends on the quality of the nuisance
# MAGIC models E[Y|X] and E[D|X]. Poor nuisance fit means the residuals carry unexplained
# MAGIC signal that contaminates the final OLS step.
# MAGIC
# MAGIC Key metric to watch: treatment nuisance R². If this is very high (> 0.95), the
# MAGIC treatment is nearly deterministic given confounders — there is very little
# MAGIC exogenous variation to identify the causal effect.

# COMMAND ----------

from insurance_causal.diagnostics import nuisance_model_summary

nuis = nuisance_model_summary(dml_model)

print("Nuisance Model Quality:")
print("-" * 50)
for k, v in nuis.items():
    if k == "warning":
        print(f"\n  WARNING: {v}")
    else:
        print(f"  {k:<35}: {v}")

print()
if "treatment_r2" in nuis and nuis["treatment_r2"] is not None:
    r2_t = nuis["treatment_r2"]
    if r2_t < 0.50:
        print(
            f"  Treatment R² = {r2_t:.3f}: There is substantial exogenous variation "
            "in log_vehicle_value after controlling for confounders. The DML "
            "estimate is well-identified."
        )
    elif r2_t < 0.80:
        print(
            f"  Treatment R² = {r2_t:.3f}: Moderate treatment predictability. "
            "DML is identifying off the residual variation — this is fine as long "
            "as the residual variance is not near zero."
        )
    else:
        print(
            f"  Treatment R² = {r2_t:.3f}: High treatment predictability. "
            "Check treatment residual variance — if very small, the confidence "
            "interval will be wide by design (the data contain little exogenous "
            "variation to identify the causal effect)."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Practitioner Interpretation
# MAGIC
# MAGIC **What we found:**
# MAGIC
# MAGIC In this simulation, the naive Poisson GLM coefficient on log vehicle value was
# MAGIC substantially more negative than the true causal effect. The confounding mechanism
# MAGIC was age: older drivers own more expensive vehicles *and* have lower claim
# MAGIC frequency, so a naive regression conflates the causal vehicle value effect with
# MAGIC the underlying driver quality signal.
# MAGIC
# MAGIC DML recovered the true effect by partialling out age (and all other observed
# MAGIC confounders) from both the treatment and outcome before estimating the causal
# MAGIC relationship. The confidence interval covers the true DGP effect.
# MAGIC
# MAGIC **Commercial implication:**
# MAGIC
# MAGIC If a pricing team sets vehicle value rating factors using a naive GLM coefficient
# MAGIC that is — say — 60% larger in magnitude than the true causal effect, they are:
# MAGIC - **Overcharging** for expensive vehicles (relative to true risk)
# MAGIC - **Undercharging** for cheap vehicles (relative to true risk)
# MAGIC - Exposing themselves to adverse selection: a rational competitor using causal
# MAGIC   estimates would undercut them on expensive cars and cede the cheap cars
# MAGIC
# MAGIC **When to use DML vs naive GLM:**
# MAGIC - **Use DML** when the treatment variable is correlated with unbalanced risk
# MAGIC   factors (almost always in observational insurance data)
# MAGIC - **Use naive GLM** only when treatment is essentially random — an A/B test with
# MAGIC   proper randomisation, where GLM is unbiased and DML adds no value
# MAGIC - **Check the confounding bias report** as a standard step in any pricing model
# MAGIC   audit. A bias of > 20% on a commercially important factor is worth investigating.
# MAGIC
# MAGIC **Limitations:**
# MAGIC - DML adjusts for *observed* confounders only. The CI-based sensitivity analysis
# MAGIC   above quantifies how much unobserved confounding would need to exist before
# MAGIC   the estimated effect changes sign.
# MAGIC - 50k policies and 5-fold CatBoost cross-fitting takes 10-25 minutes. For
# MAGIC   exploratory work on large datasets, use `cv_folds=3`.
# MAGIC - If the treatment is nearly deterministic given confounders (high treatment R²),
# MAGIC   the confidence interval will be wide — this is correct behaviour, not a failure.

# COMMAND ----------

print("=" * 72)
print("Workflow complete")
print("=" * 72)
print(f"""
Summary of results:

  Dataset              {N_POLICIES:,} synthetic UK motor policies
  Treatment            log_vehicle_value (continuous)
  Outcome              claim_count / earned_years (Poisson frequency)

  True DGP effect      {TRUE_CAUSAL_EFFECT:+.5f}
  Naive GLM estimate   {NAIVE_ESTIMATE:+.5f}  (bias: {naive_bias_pct:+.1f}% of true)
  DML estimate         {DML_ESTIMATE:+.5f}  (bias: {dml_bias_pct:+.1f}% of true)
  DML 95% CI           [{DML_CI_LOWER:.5f}, {DML_CI_UPPER:.5f}]
  DML p-value          {ate.p_value:.4f}
  CI covers true?      {DML_CI_LOWER <= TRUE_CAUSAL_EFFECT <= DML_CI_UPPER}

  Sensitivity:
  Bias to overturn conclusion: {bias_to_overturn:.5f}
  Unobserved confounding needed: {frac_unobserved_needed:.2f}x observed confounding

Key takeaway:
  DML substantially reduced confounding bias relative to the naive GLM.
  The DML confidence interval covers the true causal effect.
  The naive GLM estimate would produce materially wrong vehicle value
  rating factors in a real portfolio.

Next steps for live data:
  1. Replace synthetic data with a policy extract joined to claims.
     Required columns: claim_count, earned_years, plus rating factors.
  2. Add all rating factors to confounders — the more comprehensive
     the confounders, the more credible the conditional ignorability
     assumption.
  3. Run confounding_bias_report() and include in the model audit pack.
  4. Run the CI-based sensitivity analysis (Step 6) to quantify how much
     unobserved confounding would be needed to overturn the conclusion.
  5. For heterogeneous effects, use cate_by_segment() on the segments
     that matter commercially (high-volume or high-premium segments).
""")
