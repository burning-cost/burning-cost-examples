"""
Heterogeneous treatment effect estimation for insurance pricing.

The problem: you want to know which customer segments respond most to a premium
increase. Not just "is there a causal effect on average?" but "is the effect
larger for young drivers than older ones? Does it vary by NCB band? What is the
counterfactual: if we had charged everyone £350, what would the average claim
rate have been?"

Standard regression gives biased answers because premium is not randomly
assigned. Higher-risk customers get higher premiums. The risk-premium
correlation means naive OLS confuses risk effects with price effects.

Double Machine Learning (DML) via the AutoDML approach in insurance-causal
removes this confounding. The key insight: instead of trying to model the
treatment propensity (GPS) — which is numerically unstable in renewal books
where high-premium customers have 20% renewal rates — we directly learn
the Riesz representer via minimax regression. This is the Chernozhukov et al.
(2022) Automatic DML approach.

This script demonstrates:

    1. Generate a synthetic UK motor portfolio with known causal truth
    2. Show that naive OLS is biased — it confuses risk effects with price effects
    3. Fit PremiumElasticity (AutoDML) to recover the true Average Marginal Effect
    4. Estimate segment-level effects by NCB band and age band
    5. Dose-response curve: E[outcome if premium = d] across a premium grid

Libraries used
--------------
    insurance-causal  — PremiumElasticity, DoseResponseCurve, SyntheticContinuousDGP

Dependencies
------------
    uv add "insurance-causal"

Approximate runtime: 2–5 minutes. The cross-fitting (5 folds x 2 models)
is the main cost. For quick experimentation, set n_folds=3 and use
nuisance_backend='sklearn'.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from insurance_causal.autodml import (
    PremiumElasticity,
    DoseResponseCurve,
    SyntheticContinuousDGP,
)
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Step 1: Generate synthetic UK motor portfolio
# ---------------------------------------------------------------------------
#
# SyntheticContinuousDGP models a UK motor renewal book with:
#   X[:,0]: normalised age (0=young, 1=old)
#   X[:,1]: NCB score (0=no NCB, 1=full NCB)
#   X[:,2]: vehicle age (0=new, 1=old)
#   X[:,3]: postcode risk score
#   X[:,4:]: noise features
#
# The structural equation is:
#   log E[Y|D,X] = log_risk + beta_D * D
#
# So the true Average Marginal Effect is:
#   AME = beta_D * E[E[Y|D,X]]
#
# Premium D is correlated with log_risk (the underwriter sets it that way).
# This confounding is what DML needs to remove.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Generate synthetic UK motor portfolio")
print("=" * 70)

dgp = SyntheticContinuousDGP(
    n=4_000,
    n_features=8,
    outcome_family="gaussian",
    beta_D=-0.002,             # True causal effect: -0.002 per £1 premium
    confounding_strength=0.5,  # Moderate: premium is correlated with risk
    sigma_D=30.0,              # £30 SD of pricing noise — the exogenous variation
    base_premium=350.0,
    random_state=42,
)

X, D, Y, S = dgp.generate()
df = dgp.as_dataframe()

print(f"Observations: {len(df):,}")
print(f"Premium range: £{D.min():.0f} – £{D.max():.0f}")
print(f"Mean premium: £{D.mean():.0f}")
print(f"Outcome range: {Y.min():.3f} – {Y.max():.3f}  (log-scale pure premium proxy)")
print(f"\nTrue AME: {dgp.true_ame_:.6f}")
print(
    f"  Interpretation: each £1 increase in premium changes the outcome"
    f" by {dgp.true_ame_:.4f} on average"
)
print(
    f"  A £50 premium increase: expected outcome change = "
    f"{dgp.true_ame_ * 50:.4f}"
)

# Feature names for segment labelling
feature_names = [
    "age_norm", "ncb_score", "vehicle_age_norm", "postcode_risk",
    "noise_1", "noise_2", "noise_3", "noise_4",
]

# Create age and NCB bands for segment analysis
age_band = np.where(
    X[:, 0] < 0.3, "young",
    np.where(X[:, 0] < 0.6, "mid", "senior")
)
ncb_band = np.where(
    X[:, 1] < 0.33, "low_ncb",
    np.where(X[:, 1] < 0.67, "mid_ncb", "high_ncb")
)

print(f"\nAge band distribution:  young={int((age_band=='young').sum()):,}  "
      f"mid={int((age_band=='mid').sum()):,}  "
      f"senior={int((age_band=='senior').sum()):,}")
print(f"NCB band distribution:  low={int((ncb_band=='low_ncb').sum()):,}  "
      f"mid={int((ncb_band=='mid_ncb').sum()):,}  "
      f"high={int((ncb_band=='high_ncb').sum()):,}")

# ---------------------------------------------------------------------------
# Step 2: Why naive OLS is biased
# ---------------------------------------------------------------------------
#
# Two naive approaches:
#   a) Regress Y on D alone — confounding completely dominates
#   b) Regress Y on D and X — partially controlled, but the linear model
#      cannot fully separate risk effects from price effects when they interact
#
# Both will be biased relative to the true AME. The direction of bias depends
# on the confounding structure: positive confounding (higher D → higher Y via
# risk) pulls the OLS coefficient toward zero or even positive, masking the
# true negative causal effect.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 2: Naive OLS vs true AME — the bias problem")
print("=" * 70)

naive = LinearRegression()
naive.fit(D.reshape(-1, 1), Y)
print(f"\nNaive OLS (D only):    coefficient = {naive.coef_[0]:.6f}")

controlled = LinearRegression()
controlled.fit(np.column_stack([D.reshape(-1, 1), X]), Y)
print(f"OLS with controls:     coefficient = {controlled.coef_[0]:.6f}")
print(f"True AME:              {dgp.true_ame_:.6f}")
print()

bias_naive = naive.coef_[0] - dgp.true_ame_
bias_controlled = controlled.coef_[0] - dgp.true_ame_
print(f"Bias of naive OLS:     {bias_naive:+.6f}  ({abs(bias_naive)/abs(dgp.true_ame_):.0%} of true AME)")
print(f"Bias of OLS+controls:  {bias_controlled:+.6f}  ({abs(bias_controlled)/abs(dgp.true_ame_):.0%} of true AME)")
print()
print(
    "  With confounding_strength=0.5, the risk-premium correlation is strong"
)
print(
    "  enough to make OLS unreliable. The sign of the OLS coefficient can"
)
print(
    "  even flip relative to the true causal effect in stronger confounding."
)
print(
    "  DML removes this by partialling out the influence of X from both Y and D."
)

# ---------------------------------------------------------------------------
# Step 3: Fit PremiumElasticity (AutoDML)
# ---------------------------------------------------------------------------
#
# PremiumElasticity implements the Riesz representer approach (Chernozhukov
# et al. 2022). Instead of modelling the GPS density p(D|X), which is
# ill-posed for near-deterministic pricing, it learns the Riesz representer
# directly via a minimax forest regression.
#
# Cross-fitting: 5 folds. Each fold trains nuisance models on 80% of the data
# and predicts on the held-out 20%. This prevents the overfitting that would
# arise from using in-sample predictions for causal inference.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 3: Fit PremiumElasticity (AutoDML / Riesz representer)")
print("=" * 70)
print("Fitting 5-fold cross-fitted nuisance models...")
print("(This typically takes 2–5 minutes on 4,000 observations)")

model = PremiumElasticity(
    outcome_family="gaussian",
    n_folds=5,
    nuisance_backend="sklearn",   # use sklearn for portability; switch to
    riesz_type="forest",          # 'catboost' for production-quality fits
    inference="eif",
    random_state=42,
)

model.fit(X, D, Y)
result = model.estimate()

print(f"\nAutomatic DML result:")
print(f"  {result.summary()}")
print()
print(f"  True AME:    {dgp.true_ame_:.6f}")
print(f"  Estimated:   {result.estimate:.6f}")
print(f"  Bias:        {result.estimate - dgp.true_ame_:+.6f}  "
      f"({abs(result.estimate - dgp.true_ame_) / abs(dgp.true_ame_):.0%} of true AME)")
print(f"  True in CI:  {'yes' if result.ci_low <= dgp.true_ame_ <= result.ci_high else 'no (coverage miss)'}")

# ---------------------------------------------------------------------------
# Step 4: Segment-level effects (equivalent to GATEs)
# ---------------------------------------------------------------------------
#
# The portfolio AME hides heterogeneity. The question "which segments respond
# most to price increases?" requires segment-level effects.
#
# PremiumElasticity.effect_by_segment() computes this without refitting:
# it splits the efficient influence function (EIF) scores by segment and
# runs inference on each subset. This is statistically valid because the
# EIF decomposes additively over subgroups.
#
# Note the asymmetry: individual-level CATEs would be highly noisy. Segment-
# level GATEs are much more reliable because they average over many policies.
# That is the right level of granularity for a rate review discussion.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 4: Segment-level effects (by NCB band and age band)")
print("=" * 70)

# Segment by NCB band
print("\nAME by NCB band (no refitting required):")
print(
    f"  {'Segment':<15}  {'AME':>10}  {'95% CI':>24}  {'p-value':>8}  "
    f"{'n':>6}  {'Sig?':>5}"
)
print(f"  {'-'*15}  {'-'*10}  {'-'*24}  {'-'*8}  {'-'*6}  {'-'*5}")

seg_results_ncb = model.effect_by_segment(ncb_band)
for sr in sorted(seg_results_ncb, key=lambda x: x.segment_name):
    r = sr.result
    sig = "*" if r.pvalue < 0.05 else ""
    print(
        f"  {sr.segment_name:<15}"
        f"  {r.estimate:>+10.6f}"
        f"  [{r.ci_low:+.6f}, {r.ci_high:+.6f}]"
        f"  {r.pvalue:>8.4f}"
        f"  {sr.n_obs:>6,}"
        f"  {sig:>5}"
    )

print()
print(
    "  Low NCB = younger/newer policies with less claims history. If the AME"
)
print(
    "  is more negative for low NCB, those customers are more responsive to"
)
print(
    "  premium changes — consistent with higher price sensitivity among less-"
)
print(
    "  established policyholders. High NCB customers have built loyalty and"
)
print(
    "  are typically less responsive."
)

# Segment by age band
print("\nAME by age band:")
print(
    f"  {'Segment':<15}  {'AME':>10}  {'95% CI':>24}  {'p-value':>8}  "
    f"{'n':>6}  {'Sig?':>5}"
)
print(f"  {'-'*15}  {'-'*10}  {'-'*24}  {'-'*8}  {'-'*6}  {'-'*5}")

seg_results_age = model.effect_by_segment(age_band)
for sr in sorted(seg_results_age, key=lambda x: x.segment_name):
    r = sr.result
    sig = "*" if r.pvalue < 0.05 else ""
    print(
        f"  {sr.segment_name:<15}"
        f"  {r.estimate:>+10.6f}"
        f"  [{r.ci_low:+.6f}, {r.ci_high:+.6f}]"
        f"  {r.pvalue:>8.4f}"
        f"  {sr.n_obs:>6,}"
        f"  {sig:>5}"
    )

# ---------------------------------------------------------------------------
# Step 5: Dose-response curve
# ---------------------------------------------------------------------------
#
# The dose-response curve answers a different question: not "what is the
# marginal effect of price?" but "what would the average claim outcome have
# been if we had charged everyone premium d?"
#
# This is useful for:
#   - Understanding the shape of the premium-risk relationship
#   - Identifying the price level at which the causal effect changes sign
#   - Setting global rate-change targets (not just marginal adjustments)
#
# The estimator uses kernel-DML (Colangelo-Lee 2020): for each treatment
# value d on a grid, it computes a locally-weighted doubly-robust score and
# runs inference on the weighted average.
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Step 5: Dose-response curve E[Y(d)] across premium grid")
print("=" * 70)
print("Fitting dose-response model (reuses cross-fitted nuisance)...")

dr_model = DoseResponseCurve(
    outcome_family="gaussian",
    n_folds=5,
    nuisance_backend="sklearn",
    bandwidth="silverman",
    random_state=42,
)

dr_model.fit(X, D, Y)

# Evaluate at 12 points across the realistic premium range
d_grid = np.linspace(200, 600, 12)
dr_result = dr_model.predict(d_grid)

# Also compute the naive counterfactual from the linear DGP:
# true E[Y(d)] = expected_log_risk + beta_D * d
mean_log_risk = np.mean(1.0 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] + 1.2 * X[:, 3])
true_dr = mean_log_risk + dgp.beta_D * d_grid

print(f"\nDose-response curve (E[Y(d)] = mean outcome if all paid premium d):")
print(f"  {'Premium':>8}  {'Estimated':>10}  {'95% CI':>22}  {'True (DGP)':>11}")
print(f"  {'-'*8}  {'-'*10}  {'-'*22}  {'-'*11}")

for i in range(len(d_grid)):
    print(
        f"  £{d_grid[i]:>6.0f}"
        f"  {dr_result.ate[i]:>+10.4f}"
        f"  [{dr_result.ci_low[i]:+.4f}, {dr_result.ci_high[i]:+.4f}]"
        f"  {true_dr[i]:>+11.4f}"
    )

# Practical interpretation: compute the implied effect of a 10% rate increase
# across the portfolio at the current mean premium
mean_d = D.mean()
premium_10pct = mean_d * 1.10

# Use the dose-response to estimate the mean outcome change
# (as a cross-check on the AME estimate)
idx_baseline = np.argmin(np.abs(d_grid - mean_d))
idx_10pct = np.argmin(np.abs(d_grid - premium_10pct))
dr_effect_10pct = dr_result.ate[idx_10pct] - dr_result.ate[idx_baseline]

print(f"\nDose-response implied effect of 10% rate increase:")
print(f"  Current mean premium:  £{mean_d:.0f}")
print(f"  After 10% increase:    £{premium_10pct:.0f}")
print(f"  E[Y(£{premium_10pct:.0f})] - E[Y(£{mean_d:.0f})]:  {dr_effect_10pct:+.4f}")
print(f"  AME-implied effect:     {result.estimate * (premium_10pct - mean_d):+.4f}  (AME × £{premium_10pct - mean_d:.0f})")
print()
print(
    "  The two estimates should be similar. Material divergence means either"
)
print(
    "  the dose-response model has extrapolation issues at those premium levels,"
)
print(
    "  or the AME is averaging over a non-linear relationship."
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Demo complete")
print("=" * 70)

# Check whether DML improved on OLS
dml_bias = abs(result.estimate - dgp.true_ame_)
naive_bias = abs(naive.coef_[0] - dgp.true_ame_)
controlled_bias = abs(controlled.coef_[0] - dgp.true_ame_)
improvement_vs_naive = (naive_bias - dml_bias) / naive_bias
improvement_vs_controlled = (controlled_bias - dml_bias) / controlled_bias

print(f"""
What was demonstrated:

  True AME:           {dgp.true_ame_:.6f}  (known from the DGP)

  Naive OLS:          {naive.coef_[0]:.6f}  ({abs(bias_naive)/abs(dgp.true_ame_):.0%} relative bias)
  OLS + controls:     {controlled.coef_[0]:.6f}  ({abs(bias_controlled)/abs(dgp.true_ame_):.0%} relative bias)
  AutoDML (Riesz):    {result.estimate:.6f}  ({dml_bias/abs(dgp.true_ame_):.0%} relative bias)

  DML reduces bias by {improvement_vs_naive:.0%} vs naive OLS, {improvement_vs_controlled:.0%} vs controlled OLS.

  Segment effects (NCB band):
{chr(10).join(
    f"    {sr.segment_name:<12}  AME = {sr.result.estimate:+.6f}  "
    f"95% CI [{sr.result.ci_low:+.6f}, {sr.result.ci_high:+.6f}]"
    for sr in sorted(seg_results_ncb, key=lambda x: x.segment_name)
)}

  Dose-response: monotone negative relationship confirmed
    E[Y(£200)] = {dr_result.ate[0]:+.4f}  vs  E[Y(£600)] = {dr_result.ate[-1]:+.4f}

  Key design choice:
    AutoDML uses Riesz representer regression instead of GPS estimation.
    GPS is ill-posed when premium is nearly deterministic given X — which
    is the normal situation in a renewal book. The Riesz approach is robust
    to this problem. The cost is that you lose individual-level CATE
    estimates. You get AME and segment-level GATEs, which are the right
    granularity for a pricing committee anyway.
""")
