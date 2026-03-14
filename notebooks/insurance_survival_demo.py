# Databricks notebook source

# MAGIC %md
# MAGIC # Survival Analysis for Insurance Retention — Cure Models, CLV, and Lapse Tables
# MAGIC ## Standard Kaplan-Meier vs Mixture Cure Model: Why the Gap Matters for Pricing
# MAGIC
# MAGIC Every renewal pricing model is implicitly a lapse model. If you think a customer
# MAGIC is going to leave, you price them higher. If you think they will stay regardless,
# MAGIC you price them at market. Getting this wrong has direct commercial consequences:
# MAGIC over-estimating lapse risk leads to over-discounting structural stayers, which
# MAGIC erodes margin on your best customers.
# MAGIC
# MAGIC The standard survival modelling assumption — that every customer *eventually* lapses,
# MAGIC the only question being when — is wrong for UK personal lines. Direct debit payers
# MAGIC with 5+ years NCB and a clean claims history behave differently from price-sensitive
# MAGIC customers who renew via aggregators. A meaningful fraction of your book has a genuine
# MAGIC never-lapse tendency: structural inertia, high switching costs, or simply that insurance
# MAGIC is not something they think about at renewal.
# MAGIC
# MAGIC **Mixture cure models** formalise this. The survival function becomes:
# MAGIC
# MAGIC ```
# MAGIC S(t|x) = pi(x) + (1 - pi(x)) * S_u(t|x)
# MAGIC ```
# MAGIC
# MAGIC where `pi(x)` is the cure fraction (structural non-lapse probability, a function of
# MAGIC covariates) and `S_u(t|x)` is the Weibull AFT survival function for the subgroup that
# MAGIC does eventually lapse. The Kaplan-Meier estimator and Cox PH both treat cured individuals
# MAGIC as *late censored observations* — they see the long-tailed survival and assume slow lapsers,
# MAGIC when actually these are never-lapsers. This biases the long-run survival estimate downward
# MAGIC and biases CLV downward for your best customers.
# MAGIC
# MAGIC **This notebook:**
# MAGIC
# MAGIC 1. Generates 50,000 synthetic UK motor policies with a known cure structure (35% structural
# MAGIC    non-lapsers) and covariate-dependent cure fraction. The true parameters are embedded so
# MAGIC    every benchmark comparison is against ground truth.
# MAGIC 2. Fits **Kaplan-Meier** and **Cox PH** as baselines and shows where they fail.
# MAGIC 3. Fits the **WeibullMixtureCureFitter** from `insurance-survival` and shows it correctly
# MAGIC    recovers the cure fraction and produces better 3-year retention predictions.
# MAGIC 4. Builds **actuarial lapse tables** (qx/px/lx) for two covariate profiles and shows
# MAGIC    how the cure model changes the long-tail behaviour.
# MAGIC 5. Runs **CLV estimation** using survival curves from both models and shows the monetary
# MAGIC    impact of getting the cure fraction wrong.
# MAGIC 6. Produces a **benchmark summary table** across all five metrics.

# COMMAND ----------

# MAGIC %pip install insurance-survival lifelines polars numpy scipy scikit-learn matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Data-Generating Process
# MAGIC
# MAGIC We simulate 50,000 motor policies with a known cure structure. The design choices
# MAGIC reflect what we observe in real UK motor data:
# MAGIC
# MAGIC - **NCD level** (0–8 years) is the strongest cure predictor. Customers with 5+
# MAGIC   years NCB are much more likely to be structural stayers. This makes intuitive
# MAGIC   sense: they have earned their discount, switching means losing it, and they have
# MAGIC   demonstrated loyalty already.
# MAGIC - **Payment method** (direct debit vs other) is the second strongest signal. Direct
# MAGIC   debit is a friction reducer for renewal — renewal is the path of least resistance.
# MAGIC - **Channel** (direct vs aggregator) enters the cure fraction. Aggregator customers
# MAGIC   explicitly shopped the market at inception; they are more likely to shop again.
# MAGIC
# MAGIC **DGP for cure fraction (logistic model):**
# MAGIC ```
# MAGIC logit(pi_i) = -1.2                   # baseline: ~23% cure at reference
# MAGIC             + 0.30 * ncd_level_i     # NCD strongly protective
# MAGIC             + 0.80 * direct_debit_i  # DD roughly doubles odds of being cured
# MAGIC             - 0.60 * aggregator_i    # aggregator halves odds
# MAGIC ```
# MAGIC
# MAGIC **DGP for time-to-lapse (Weibull AFT for uncured subgroup):**
# MAGIC ```
# MAGIC log(scale_i) = 0.40                  # median lapse ~1.5 years for uncured
# MAGIC              + 0.12 * ncd_level_i    # higher NCD → slower lapse even if uncured
# MAGIC              + 0.20 * direct_debit_i # DD → slower lapse
# MAGIC Weibull shape: rho = 2.0             # increasing hazard (renewal cliff effect)
# MAGIC ```
# MAGIC
# MAGIC The observation window is 5 years (policies up to 2020 inception, data through 2025).
# MAGIC Policies still active at the cutoff are right-censored. This matches the typical
# MAGIC modelling scenario: long-tenure customers are necessarily censored because your data
# MAGIC does not yet know whether they will eventually lapse.

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit

# ─── DGP parameters ──────────────────────────────────────────────────────────

N_POLICIES   = 50_000
TRUE_CURE    = 0.35           # marginal cure fraction (35% structural non-lapsers)
RANDOM_SEED  = 2025
OBS_YEARS    = 5.0            # observation window: 5 policy years

# Cure logistic coefficients
GAMMA_INTERCEPT   = -1.20
GAMMA_NCD         = +0.30
GAMMA_DD          = +0.80
GAMMA_AGGREGATOR  = -0.60

# Weibull AFT coefficients for uncured subgroup
BETA_INTERCEPT    = +0.40
BETA_NCD          = +0.12
BETA_DD           = +0.20
WEIBULL_SHAPE     = 2.0       # rho — increasing hazard at renewal

rng = np.random.default_rng(RANDOM_SEED)

# ─── Covariate simulation ─────────────────────────────────────────────────────

n = N_POLICIES

ncd_level     = rng.integers(0, 9, n).astype(float)         # 0–8 years
direct_debit  = rng.binomial(1, 0.60, n).astype(float)      # 60% pay by DD
aggregator    = rng.binomial(1, 0.45, n).astype(float)       # 45% came via aggregator
age           = rng.integers(18, 80, n).astype(float)
annual_premium = rng.uniform(280, 1400, n)                   # UK motor range

# Ensure aggregator and direct are not perfectly collinear
# Aggregator customers slightly less likely to be on DD
aggregator = np.where(
    direct_debit == 1,
    rng.binomial(1, 0.35, n).astype(float),
    rng.binomial(1, 0.60, n).astype(float),
)

# ─── Cure fraction per policy ─────────────────────────────────────────────────

logit_cure = (
    GAMMA_INTERCEPT
    + GAMMA_NCD * ncd_level
    + GAMMA_DD  * direct_debit
    + GAMMA_AGGREGATOR * aggregator
)
pi = expit(logit_cure)  # per-policy cure probability

# ─── Draw cured status ───────────────────────────────────────────────────────

is_cured = rng.binomial(1, pi).astype(bool)
empirical_cure = is_cured.mean()

# ─── Time-to-lapse for uncured (Weibull AFT) ─────────────────────────────────
# scale(x) = exp(beta_0 + beta_1*ncd + beta_2*dd)
# S_u(t) = exp(-(t/scale)^rho)
# Draw T from Weibull: T = scale * (-log(U))^(1/rho)

log_scale = BETA_INTERCEPT + BETA_NCD * ncd_level + BETA_DD * direct_debit
scale = np.exp(log_scale)
U = rng.uniform(0, 1, n)
t_lapse = scale * (-np.log(np.clip(U, 1e-10, 1.0))) ** (1.0 / WEIBULL_SHAPE)

# ─── Observed follow-up time and event indicator ─────────────────────────────
# Cured individuals are always censored (they never lapse within any window).
# Uncured individuals lapse at t_lapse if t_lapse < OBS_YEARS, else censored.

duration = np.where(
    is_cured,
    OBS_YEARS * rng.uniform(0.5, 1.0, n),   # cured: censored at some point in window
    np.minimum(t_lapse, OBS_YEARS),
)
event = np.where(
    is_cured,
    0,                                        # cured: no event
    (t_lapse <= OBS_YEARS).astype(int),
)

# ─── Build Polars DataFrame ───────────────────────────────────────────────────

survival_df = pl.DataFrame({
    "duration":       duration,
    "event":          event.astype(float),
    "ncd_level":      ncd_level,
    "direct_debit":   direct_debit,
    "aggregator":     aggregator,
    "age":            age,
    "annual_premium": annual_premium,
    "true_cured":     is_cured.astype(int),
    "true_pi":        pi,
})

policy_df = survival_df.clone()  # keep full copy for CLV section

# ─── Summary ─────────────────────────────────────────────────────────────────

total_lapses  = int(event.sum())
censor_rate   = 1.0 - total_lapses / n
median_tenure_uncured = float(np.median(t_lapse[~is_cured]))

print(f"Policies:              {n:,}")
print(f"True cure fraction:    {empirical_cure:.3f}  (target: {TRUE_CURE:.2f})")
print(f"Observed lapses:       {total_lapses:,}  ({total_lapses/n:.1%})")
print(f"Censoring rate:        {censor_rate:.1%}")
print(f"Median lapse (uncured): {median_tenure_uncured:.2f} policy years")
print(f"\nCure fraction by NCD level:")
for ncd in [0, 2, 4, 6, 8]:
    mask = (np.abs(ncd_level - ncd) < 0.5)
    if mask.sum() > 0:
        grp_cure = is_cured[mask].mean()
        print(f"  NCD {ncd}: {grp_cure:.2f} ({mask.sum():,} policies)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Kaplan-Meier — The Baseline
# MAGIC
# MAGIC Kaplan-Meier is non-parametric. It makes no distributional assumption and is the
# MAGIC correct first look at any survival dataset. But it has no concept of a cure fraction.
# MAGIC When the KM curve levels off and stops dropping, KM interprets this as censoring:
# MAGIC "we ran out of follow-up time, not that anyone has zero hazard." It cannot distinguish
# MAGIC between late lapsers (who will eventually go) and structural stayers (who will not).
# MAGIC
# MAGIC The consequence: if you read off `S(5)` from a KM curve that includes cured individuals,
# MAGIC you overestimate retention in the short term (the cured people inflate the curve) and
# MAGIC underestimate retention in the long term (KM will eventually pull the curve to zero even
# MAGIC though ~35% of your customers are never going to lapse). CLV calculated from a KM integral
# MAGIC is systematically wrong in ways that differ by customer segment.
# MAGIC
# MAGIC We also fit a **Cox PH model** on the full dataset. Cox PH is the workhorse of insurance
# MAGIC retention modelling in the UK. It correctly handles covariates, but like KM it assumes
# MAGIC the hazard is always positive — every customer has a non-zero instantaneous lapse rate.
# MAGIC Watch what happens to the 5-year retention estimate.

# COMMAND ----------

from lifelines import KaplanMeierFitter, CoxPHFitter

df_pd = survival_df.to_pandas()

# ─── Kaplan-Meier on full dataset ────────────────────────────────────────────

kmf = KaplanMeierFitter()
kmf.fit(df_pd["duration"], event_observed=df_pd["event"], label="Kaplan-Meier (all)")

km_s1  = float(kmf.predict(1.0))
km_s3  = float(kmf.predict(3.0))
km_s5  = float(kmf.predict(5.0))
km_s7  = float(kmf.predict(7.0))   # extrapolated beyond observation window

print("Kaplan-Meier survival estimates:")
print(f"  S(1 year) = {km_s1:.4f}")
print(f"  S(3 year) = {km_s3:.4f}")
print(f"  S(5 year) = {km_s5:.4f}")
print(f"  S(7 year) = {km_s7:.4f}  [extrapolated]")

# True S(t) from DGP: S(t) = pi + (1-pi) * S_u(t) averaged over covariate distribution
# Compute the portfolio-average true survival at each time point
def true_survival(t_val, pi_vec, log_scale_vec, rho):
    """Portfolio-average true survival at time t."""
    scale_vec = np.exp(log_scale_vec)
    sf_u = np.exp(-(t_val / scale_vec) ** rho)
    return float(np.mean(pi_vec + (1.0 - pi_vec) * sf_u))

true_s1 = true_survival(1.0, pi, log_scale, WEIBULL_SHAPE)
true_s3 = true_survival(3.0, pi, log_scale, WEIBULL_SHAPE)
true_s5 = true_survival(5.0, pi, log_scale, WEIBULL_SHAPE)
true_s7 = true_survival(7.0, pi, log_scale, WEIBULL_SHAPE)

print(f"\nTrue DGP survival (portfolio average):")
print(f"  S(1 year) = {true_s1:.4f}")
print(f"  S(3 year) = {true_s3:.4f}")
print(f"  S(5 year) = {true_s5:.4f}")
print(f"  S(7 year) = {true_s7:.4f}")

print(f"\nKM bias at 5 years: {km_s5 - true_s5:+.4f}")
print(f"  (positive = KM overstates retention at 5 years vs true DGP)")
print(f"  (at 7 years KM will under-predict as it trends to 0 rather than plateau)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Cox PH — Covariates Without Cure
# MAGIC
# MAGIC Cox PH correctly incorporates covariates (NCD, direct debit, aggregator) into the
# MAGIC hazard model. For short-horizon predictions (1–2 years) it performs well. The problem
# MAGIC emerges at longer horizons: without a cure component, the baseline hazard is estimated
# MAGIC under the assumption that the cumulative hazard tends to infinity (eventually everyone
# MAGIC lapses). This drags the long-run survival estimate towards zero.
# MAGIC
# MAGIC We will use the Cox PH survival curves to compute CLV in Step 5. Watch how the survival
# MAGIC integral differs from the cure model, and how that translates to CLV bias.

# COMMAND ----------

# ─── Cox PH model ─────────────────────────────────────────────────────────────

cox_features = ["ncd_level", "direct_debit", "aggregator"]

cph = CoxPHFitter(penalizer=0.01)
cph.fit(
    df_pd[["duration", "event"] + cox_features],
    duration_col="duration",
    event_col="event",
)

print("Cox PH coefficient summary:")
print(cph.summary[["coef", "exp(coef)", "p", "coef lower 95%", "coef upper 95%"]].to_string())

# Portfolio-average survival from Cox model
# Use the median covariate profile for a representative estimate
median_profile_pd = pd.DataFrame({
    "ncd_level":    [float(np.median(ncd_level))],
    "direct_debit": [float(np.median(direct_debit))],
    "aggregator":   [float(np.median(aggregator))],
})

cox_sf = cph.predict_survival_function(median_profile_pd, times=[1, 3, 5, 7])
cox_s1 = float(cox_sf.iloc[0, 0])
cox_s3 = float(cox_sf.iloc[1, 0])
cox_s5 = float(cox_sf.iloc[2, 0])
cox_s7 = float(cox_sf.iloc[3, 0])

# True survival at median profile
ncd_med   = float(np.median(ncd_level))
dd_med    = float(np.median(direct_debit))
agg_med   = float(np.median(aggregator))
logit_med = (GAMMA_INTERCEPT + GAMMA_NCD * ncd_med
             + GAMMA_DD * dd_med + GAMMA_AGGREGATOR * agg_med)
pi_med    = float(expit(logit_med))
ls_med    = BETA_INTERCEPT + BETA_NCD * ncd_med + BETA_DD * dd_med
scale_med = np.exp(ls_med)

def true_sf_profile(t_val, pi_p, scale_p, rho):
    return pi_p + (1.0 - pi_p) * np.exp(-(t_val / scale_p) ** rho)

true_s1_med = true_sf_profile(1.0, pi_med, scale_med, WEIBULL_SHAPE)
true_s3_med = true_sf_profile(3.0, pi_med, scale_med, WEIBULL_SHAPE)
true_s5_med = true_sf_profile(5.0, pi_med, scale_med, WEIBULL_SHAPE)
true_s7_med = true_sf_profile(7.0, pi_med, scale_med, WEIBULL_SHAPE)

print(f"\nMedian profile (NCD={ncd_med:.0f}, DD={dd_med:.0f}, Agg={agg_med:.0f}):")
print(f"  True cure fraction at median profile: {pi_med:.3f}")
print(f"\n{'Horizon':<10} {'Cox PH':>10} {'True DGP':>10} {'Error':>10}")
for t_label, cox_v, true_v in [
    ("1 year",  cox_s1, true_s1_med),
    ("3 years", cox_s3, true_s3_med),
    ("5 years", cox_s5, true_s5_med),
    ("7 years", cox_s7, true_s7_med),
]:
    print(f"  {t_label:<8} {cox_v:>10.4f} {true_v:>10.4f} {cox_v - true_v:>+10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Mixture Cure Model — Capturing the Never-Lapse Subgroup
# MAGIC
# MAGIC The `WeibullMixtureCureFitter` from `insurance-survival` extends the standard
# MAGIC Weibull AFT with a covariate-adjusted cure fraction. The model is:
# MAGIC
# MAGIC ```
# MAGIC S(t|x) = pi(x) + (1 - pi(x)) * exp(-(t/lambda(x))^rho)
# MAGIC
# MAGIC pi(x)      = logistic(gamma_0 + x_cure' gamma)
# MAGIC lambda(x)  = exp(beta_0 + x_uncured' beta)
# MAGIC ```
# MAGIC
# MAGIC Estimation proceeds by EM initialisation (15 iterations to get near the
# MAGIC maximum-likelihood region) followed by joint L-BFGS-B on the full log-likelihood.
# MAGIC Standard errors are computed via a finite-difference numerical Hessian.
# MAGIC
# MAGIC The key question after fitting is: does the model recover the true cure fraction
# MAGIC (35% at the population level)? And does it correctly attribute higher cure probability
# MAGIC to high-NCD, direct-debit customers?

# COMMAND ----------

from insurance_survival import WeibullMixtureCureFitter, LapseTable, SurvivalCLV

# ─── Fit mixture cure model ───────────────────────────────────────────────────

cure_fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_level", "direct_debit", "aggregator"],
    uncured_covariates=["ncd_level", "direct_debit"],
    penalizer=0.005,
    max_iter=400,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cure_fitter.fit(
        survival_df,
        duration_col="duration",
        event_col="event",
    )

print("Convergence:")
for k, v in cure_fitter.convergence_.items():
    print(f"  {k}: {v}")

print("\nCure fraction (logistic) parameters:")
print(cure_fitter.cure_params_.to_pandas().to_string(index=False))

print("\nUncured subgroup (Weibull AFT) parameters:")
print(cure_fitter.uncured_params_.to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cure fraction recovery
# MAGIC
# MAGIC The primary diagnostic is whether the model recovers the true cure fraction. We
# MAGIC check this at three levels:
# MAGIC
# MAGIC 1. **Portfolio average**: does `mean(predict_cure(df))` match `TRUE_CURE = 0.35`?
# MAGIC 2. **By NCD band**: does the covariate-dependence of cure fraction match the DGP
# MAGIC    coefficients (`GAMMA_NCD = 0.30`)?
# MAGIC 3. **Individual level**: for policies we *know* are cured (from the DGP), does the
# MAGIC    model assign higher predicted cure probability than for uncured policies?

# COMMAND ----------

# ─── Cure fraction recovery ───────────────────────────────────────────────────

predicted_cure = cure_fitter.predict_cure(survival_df)
mean_pred_cure = float(predicted_cure.mean())

print(f"True cure fraction (DGP): {empirical_cure:.4f}")
print(f"Predicted mean cure:       {mean_pred_cure:.4f}")
print(f"Absolute error:            {abs(mean_pred_cure - empirical_cure):.4f}")

# By NCD band
print(f"\nCure fraction by NCD level:")
print(f"{'NCD':>4} {'True':>8} {'Predicted':>10} {'Error':>8}")
for ncd in range(0, 9, 1):
    mask_pl = (survival_df["ncd_level"] == float(ncd))
    n_grp = mask_pl.sum()
    if n_grp < 50:
        continue
    true_grp = float(survival_df.filter(mask_pl)["true_cured"].mean())
    pred_grp = float(predicted_cure.filter(mask_pl).mean())
    print(f"  {ncd:>3}  {true_grp:>8.3f} {pred_grp:>10.3f} {pred_grp - true_grp:>+8.3f}")

# Separation: are cured policies assigned higher cure probability?
df_with_pred = survival_df.with_columns(predicted_cure.alias("pred_cure"))
mean_pred_cured   = float(df_with_pred.filter(pl.col("true_cured") == 1)["pred_cure"].mean())
mean_pred_uncured = float(df_with_pred.filter(pl.col("true_cured") == 0)["pred_cure"].mean())
print(f"\nSeparation test:")
print(f"  Mean predicted cure (true cured):    {mean_pred_cured:.4f}")
print(f"  Mean predicted cure (true uncured):  {mean_pred_uncured:.4f}")
print(f"  Separation:                          {mean_pred_cured - mean_pred_uncured:.4f}")
print(f"  (A well-specified model should show meaningful separation here)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Lapse Tables
# MAGIC
# MAGIC Actuarial lapse tables in qx/px/lx format are the deliverable that reserving and
# MAGIC finance teams recognise. `LapseTable` wraps any fitted survival model and produces
# MAGIC this output. We generate tables for two contrasting covariate profiles:
# MAGIC
# MAGIC - **Low-risk stayer**: NCD 8, direct debit, not aggregator. High cure probability.
# MAGIC   Expect a high plateau on the survival curve and low qx values throughout.
# MAGIC - **High-risk churner**: NCD 0, not direct debit, aggregator. Low cure probability.
# MAGIC   Expect rapid lapse in years 1–2 with low survival by year 5.
# MAGIC
# MAGIC The comparison between Cox PH and cure model tables is revealing: at year 5 and
# MAGIC beyond, Cox PH continues to predict positive qx for the stayer profile (it must —
# MAGIC it has no cure component) while the cure model correctly shows near-zero qx once
# MAGIC the uncured subgroup has largely lapsed.

# COMMAND ----------

# ─── Lapse tables: cure model ─────────────────────────────────────────────────

lapse_table_cure = LapseTable(
    survival_model=cure_fitter,
    radix=10_000,
    time_points=[1, 2, 3, 4, 5, 6, 7],
)

stayer_profile = {
    "ncd_level":    8.0,
    "direct_debit": 1.0,
    "aggregator":   0.0,
}

churner_profile = {
    "ncd_level":    0.0,
    "direct_debit": 0.0,
    "aggregator":   1.0,
}

stayer_table_cure   = lapse_table_cure.generate(stayer_profile)
churner_table_cure  = lapse_table_cure.generate(churner_profile)

# ─── Lapse tables: Cox PH ─────────────────────────────────────────────────────

lapse_table_cox = LapseTable(
    survival_model=cph,
    radix=10_000,
    time_points=[1, 2, 3, 4, 5, 6, 7],
)

stayer_table_cox   = lapse_table_cox.generate(stayer_profile)
churner_table_cox  = lapse_table_cox.generate(churner_profile)

# ─── Side-by-side comparison ──────────────────────────────────────────────────

print("STAYER PROFILE (NCD 8, Direct Debit, Non-aggregator)")
print(f"  True cure probability: {float(expit(GAMMA_INTERCEPT + GAMMA_NCD*8 + GAMMA_DD*1 + GAMMA_AGGREGATOR*0)):.3f}")
print()
print(f"{'Year':>4} | {'Cure-lx':>8} {'Cure-qx':>9} {'Cure-px':>9} | {'Cox-lx':>8} {'Cox-qx':>9} {'Cox-px':>9}")
print("-" * 65)
for row_c, row_x in zip(stayer_table_cure.iter_rows(named=True),
                         stayer_table_cox.iter_rows(named=True)):
    print(f"  {row_c['year']:>2}  |  {row_c['lx']:>6,}  {row_c['qx']:>8.4f}  {row_c['px']:>8.4f}  |"
          f"  {row_x['lx']:>6,}  {row_x['qx']:>8.4f}  {row_x['px']:>8.4f}")

print()
print("CHURNER PROFILE (NCD 0, Not DD, Aggregator)")
print(f"  True cure probability: {float(expit(GAMMA_INTERCEPT + GAMMA_NCD*0 + GAMMA_DD*0 + GAMMA_AGGREGATOR*1)):.3f}")
print()
print(f"{'Year':>4} | {'Cure-lx':>8} {'Cure-qx':>9} {'Cure-px':>9} | {'Cox-lx':>8} {'Cox-qx':>9} {'Cox-px':>9}")
print("-" * 65)
for row_c, row_x in zip(churner_table_cure.iter_rows(named=True),
                         churner_table_cox.iter_rows(named=True)):
    print(f"  {row_c['year']:>2}  |  {row_c['lx']:>6,}  {row_c['qx']:>8.4f}  {row_c['px']:>8.4f}  |"
          f"  {row_x['lx']:>6,}  {row_x['qx']:>8.4f}  {row_x['px']:>8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Survival Curves by Segment
# MAGIC
# MAGIC Before moving to CLV, we visualise survival curves for the two profiles under both
# MAGIC models. The key visual is the long-run plateau: the cure model levels off at `pi(x)`,
# MAGIC while Cox PH continues declining towards zero. The difference between the two curves
# MAGIC at years 5–7 is the "cure gap" — the region where Cox PH incorrectly predicts ongoing
# MAGIC lapse activity from what is actually a structurally stable cohort.

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

t_grid = np.linspace(0, 7, 300)

# True survival curves for each profile
def true_sf_grid(t_arr, ncd_v, dd_v, agg_v):
    logit_c = GAMMA_INTERCEPT + GAMMA_NCD*ncd_v + GAMMA_DD*dd_v + GAMMA_AGGREGATOR*agg_v
    pi_v = expit(logit_c)
    ls_v = BETA_INTERCEPT + BETA_NCD*ncd_v + BETA_DD*dd_v
    sc_v = np.exp(ls_v)
    return pi_v + (1.0 - pi_v) * np.exp(-(t_arr / sc_v)**WEIBULL_SHAPE)

true_stayer   = true_sf_grid(t_grid, 8.0, 1.0, 0.0)
true_churner  = true_sf_grid(t_grid, 0.0, 0.0, 1.0)

# Cure model predictions
stayer_df  = pl.DataFrame({k: [float(v)] for k, v in stayer_profile.items()})
churner_df = pl.DataFrame({k: [float(v)] for k, v in churner_profile.items()})

t_list = list(t_grid)
cure_sf_stayer  = cure_fitter.predict_survival_function(stayer_df,  times=t_list)
cure_sf_churner = cure_fitter.predict_survival_function(churner_df, times=t_list)

cure_stayer_arr  = np.array([float(cure_sf_stayer[f"S_t{k+1}"][0])  for k in range(len(t_list))])
cure_churner_arr = np.array([float(cure_sf_churner[f"S_t{k+1}"][0]) for k in range(len(t_list))])

# Cox PH predictions
stayer_pd  = pd.DataFrame(stayer_profile,  index=[0])
churner_pd = pd.DataFrame(churner_profile, index=[0])

cox_sf_stayer_full  = cph.predict_survival_function(stayer_pd,  times=t_list)
cox_sf_churner_full = cph.predict_survival_function(churner_pd, times=t_list)
cox_stayer_arr  = cox_sf_stayer_full.iloc[:, 0].values
cox_churner_arr = cox_sf_churner_full.iloc[:, 0].values

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, cure_arr, cox_arr, true_arr, title, pi_true in [
    (axes[0], cure_stayer_arr,  cox_stayer_arr,  true_stayer,
     "Stayer profile (NCD 8, DD, non-agg)",
     float(expit(GAMMA_INTERCEPT + GAMMA_NCD*8 + GAMMA_DD*1))),
    (axes[1], cure_churner_arr, cox_churner_arr, true_churner,
     "Churner profile (NCD 0, no DD, agg)",
     float(expit(GAMMA_INTERCEPT + GAMMA_AGGREGATOR*1))),
]:
    ax.plot(t_grid, true_arr,  "k-",  lw=2.5, label="True DGP")
    ax.plot(t_grid, cure_arr,  "b--", lw=2,   label="Cure model")
    ax.plot(t_grid, cox_arr,   "r:",  lw=2,   label="Cox PH")
    ax.axhline(pi_true, color="grey", linestyle="-.", lw=1.2, alpha=0.7,
               label=f"True cure fraction ({pi_true:.2f})")
    ax.set_xlabel("Policy year", fontsize=11)
    ax.set_ylabel("Survival probability S(t)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Survival curves: Cox PH vs Cure Model vs True DGP", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("/tmp/survival_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: /tmp/survival_curves.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: CLV Estimation
# MAGIC
# MAGIC Customer lifetime value using survival integration is the formula Consumer Duty
# MAGIC (PS21/11) requires for fair value analysis. The `SurvivalCLV` class computes:
# MAGIC
# MAGIC ```
# MAGIC CLV(x) = sum_{t=1}^{T} S(t|x(t)) * (P_t - L_t) * (1+r)^{-t}
# MAGIC ```
# MAGIC
# MAGIC where `P_t` is annual premium, `L_t` is expected loss cost, and `r = 0.05` is the
# MAGIC discount rate. `S(t|x(t))` is the survival probability at year `t`, with NCD level
# MAGIC projected forward using the UK motor standard transition rules (1 step up per claim-free
# MAGIC year, 2 steps back after a claim).
# MAGIC
# MAGIC We compare CLV estimates from:
# MAGIC 1. **Cure model** — the survival-correct estimate
# MAGIC 2. **Cox PH** — standard approach, no cure component
# MAGIC
# MAGIC We focus on a subsample of 2,000 policies to keep computation fast, stratified across
# MAGIC the NCD distribution. The key question: how much does CLV differ between the two models,
# MAGIC and for which customer segments is the difference largest?

# COMMAND ----------

# ─── Subsample for CLV analysis ───────────────────────────────────────────────

SUBSAMPLE_N = 2_000
rng_sub = np.random.default_rng(99)
sub_idx = rng_sub.choice(n, SUBSAMPLE_N, replace=False)
sub_idx_sorted = np.sort(sub_idx)

sub_df = survival_df[sub_idx_sorted].with_columns(
    pl.Series("policy_id", list(range(SUBSAMPLE_N))),
    pl.Series("expected_loss",
              (survival_df["annual_premium"][sub_idx_sorted].to_numpy() * 0.62)),
)

# ─── CLV from cure model ──────────────────────────────────────────────────────

clv_cure = SurvivalCLV(
    survival_model=cure_fitter,
    horizon=5,
    discount_rate=0.05,
)
cure_results = clv_cure.predict(
    sub_df,
    premium_col="annual_premium",
    loss_col="expected_loss",
)

# ─── CLV from Cox PH ──────────────────────────────────────────────────────────

clv_cox = SurvivalCLV(
    survival_model=cph,
    horizon=5,
    discount_rate=0.05,
)
cox_results = clv_cox.predict(
    sub_df,
    premium_col="annual_premium",
    loss_col="expected_loss",
)

# ─── True CLV from DGP ────────────────────────────────────────────────────────
# Compute true CLV using known pi and scale for each policy in subsample

sub_ncd  = sub_df["ncd_level"].to_numpy()
sub_dd   = sub_df["direct_debit"].to_numpy()
sub_agg  = sub_df["aggregator"].to_numpy()
sub_prem = sub_df["annual_premium"].to_numpy()
sub_loss = sub_prem * 0.62

logit_sub = (GAMMA_INTERCEPT + GAMMA_NCD * sub_ncd
             + GAMMA_DD * sub_dd + GAMMA_AGGREGATOR * sub_agg)
pi_sub    = expit(logit_sub)
ls_sub    = BETA_INTERCEPT + BETA_NCD * sub_ncd + BETA_DD * sub_dd
scale_sub = np.exp(ls_sub)

horizon   = 5
disc_rate = 0.05
true_clv  = np.zeros(SUBSAMPLE_N)

for yr in range(1, horizon + 1):
    s_t = pi_sub + (1.0 - pi_sub) * np.exp(-(yr / scale_sub) ** WEIBULL_SHAPE)
    disc = 1.0 / (1.0 + disc_rate) ** yr
    true_clv += s_t * (sub_prem - sub_loss) * disc

# ─── Comparison ───────────────────────────────────────────────────────────────

cure_clv_arr  = np.array(cure_results["clv"].to_list())
cox_clv_arr   = np.array(cox_results["clv"].to_list())
cure_prob_arr = np.array(cure_results["cure_prob"].to_list())

mae_cure = float(np.mean(np.abs(cure_clv_arr - true_clv)))
mae_cox  = float(np.mean(np.abs(cox_clv_arr  - true_clv)))
me_cure  = float(np.mean(cure_clv_arr - true_clv))
me_cox   = float(np.mean(cox_clv_arr  - true_clv))

print(f"CLV comparison (n = {SUBSAMPLE_N:,} policies):")
print(f"\n{'Metric':<35} {'Cure model':>12} {'Cox PH':>12} {'True DGP':>12}")
print("-" * 73)
print(f"  {'Mean CLV':.<33} {float(np.mean(cure_clv_arr)):>12.2f} {float(np.mean(cox_clv_arr)):>12.2f} {float(np.mean(true_clv)):>12.2f}")
print(f"  {'Mean absolute error':.<33} {mae_cure:>12.2f} {mae_cox:>12.2f} {'—':>12}")
print(f"  {'Mean error (bias)':.<33} {me_cure:>+12.2f} {me_cox:>+12.2f} {'—':>12}")
print(f"  {'Relative bias (% of true mean)':.<33} {100*me_cure/float(np.mean(true_clv)):>+11.1f}% {100*me_cox/float(np.mean(true_clv)):>+11.1f}% {'—':>12}")

# ─── CLV bias by predicted cure probability band ──────────────────────────────

print(f"\nCLV bias by cure probability band (cure model scores):")
print(f"{'Cure band':>14} {'n':>6} {'Cure bias':>12} {'Cox bias':>12}")
bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for lo, hi in bands:
    mask = (cure_prob_arr >= lo) & (cure_prob_arr < hi)
    n_b = mask.sum()
    if n_b < 10:
        continue
    cure_b = float(np.mean(cure_clv_arr[mask] - true_clv[mask]))
    cox_b  = float(np.mean(cox_clv_arr[mask]  - true_clv[mask]))
    print(f"  [{lo:.1f}, {hi:.1f})  {n_b:>6,} {cure_b:>+12.2f} {cox_b:>+12.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Benchmark Summary Table
# MAGIC
# MAGIC We consolidate the results into a single comparison table. The five metrics capture
# MAGIC different aspects of model quality:
# MAGIC
# MAGIC 1. **Cure fraction recovery**: does the model correctly estimate the never-lapse
# MAGIC    subgroup size? The cure model directly estimates this; Cox PH does not (we use
# MAGIC    the KM plateau as a proxy for Cox PH).
# MAGIC
# MAGIC 2. **3-year retention accuracy**: absolute error on `S(3)` at the median profile.
# MAGIC    This is the operational question for renewal pricing — who will still be with us
# MAGIC    in three years?
# MAGIC
# MAGIC 3. **5-year retention accuracy**: same but at 5 years. This is where the cure gap
# MAGIC    opens up most visibly.
# MAGIC
# MAGIC 4. **Long-run survival floor**: does the model predict a positive plateau (correct)
# MAGIC    or eventual extinction (incorrect for the stayer profile)?
# MAGIC
# MAGIC 5. **CLV mean absolute error**: monetary metric. This is what the commercial team
# MAGIC    cares about when making discount decisions post-Consumer Duty.

# COMMAND ----------

# ─── 3-year and 5-year retention at median profile ───────────────────────────

# Kaplan-Meier: read off from already-fitted KM
km_s3_med = float(kmf.predict(3.0))   # KM is marginal, not profile-specific
km_s5_med = float(kmf.predict(5.0))

# Cox PH: already computed above as cox_s3, cox_s5

# Cure model: predict at median profile
cure_sf_med = cure_fitter.predict_survival_function(
    pl.DataFrame({k: [v] for k, v in [
        ("ncd_level",    ncd_med),
        ("direct_debit", dd_med),
        ("aggregator",   agg_med),
    ]}),
    times=[3.0, 5.0],
)
cure_s3_med = float(cure_sf_med["S_t1"][0])
cure_s5_med = float(cure_sf_med["S_t2"][0])

# Cure fraction recovery
km_cure_est = float(kmf.predict(20.0))   # KM plateau at very long time ≈ cure fraction
cox_cure_est = float("nan")              # Cox PH has no cure component — N/A

cure_frac_pred = float(cure_fitter.predict_cure(
    pl.DataFrame({k: [v] for k, v in [
        ("ncd_level",    ncd_med),
        ("direct_debit", dd_med),
        ("aggregator",   agg_med),
    ]})
).mean())

# Build results table
results_data = {
    "Metric": [
        "3-yr retention S(3) — median profile",
        "5-yr retention S(5) — median profile",
        "Cure fraction — median profile",
        "CLV mean absolute error (£)",
        "CLV mean bias (£)",
        "Long-run survival (7yr) — stayer profile",
    ],
    "True DGP": [
        f"{true_s3_med:.4f}",
        f"{true_s5_med:.4f}",
        f"{pi_med:.4f}",
        "—",
        "—",
        f"{true_sf_profile(7.0, float(expit(GAMMA_INTERCEPT + GAMMA_NCD*8 + GAMMA_DD*1)), np.exp(BETA_INTERCEPT + BETA_NCD*8 + BETA_DD*1), WEIBULL_SHAPE):.4f}",
    ],
    "Kaplan-Meier": [
        f"{km_s3:.4f}  (err {km_s3 - true_s3_med:+.4f})",
        f"{km_s5:.4f}  (err {km_s5 - true_s5_med:+.4f})",
        f"~{km_cure_est:.3f} (plateau)",
        "n/a",
        "n/a",
        f"{float(kmf.predict(7.0)):.4f}",
    ],
    "Cox PH": [
        f"{cox_s3:.4f}  (err {cox_s3 - true_s3_med:+.4f})",
        f"{cox_s5:.4f}  (err {cox_s5 - true_s5_med:+.4f})",
        "n/a (no cure component)",
        f"£{mae_cox:.2f}",
        f"£{me_cox:+.2f}",
        f"{cox_stayer_arr[-1]:.4f}",
    ],
    "Cure Model": [
        f"{cure_s3_med:.4f}  (err {cure_s3_med - true_s3_med:+.4f})",
        f"{cure_s5_med:.4f}  (err {cure_s5_med - true_s5_med:+.4f})",
        f"{cure_frac_pred:.4f}  (err {cure_frac_pred - pi_med:+.4f})",
        f"£{mae_cure:.2f}",
        f"£{me_cure:+.2f}",
        f"{cure_stayer_arr[-1]:.4f}",
    ],
}

benchmark_df = pl.DataFrame(results_data)
print("\nBENCHMARK SUMMARY")
print("=" * 110)
print(benchmark_df.to_pandas().to_string(index=False))
print("=" * 110)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Discount Sensitivity Analysis (Consumer Duty)
# MAGIC
# MAGIC Under Consumer Duty (PS21/11), any renewal discount must be justified by CLV analysis.
# MAGIC The `SurvivalCLV.discount_sensitivity()` method computes CLV under a range of discount
# MAGIC amounts and returns `discount_justified = True` where the retention lift makes the
# MAGIC discount commercially viable.
# MAGIC
# MAGIC We run this for our two profiles. For the stayer profile (high cure fraction), the
# MAGIC customer is likely to stay regardless — a discount destroys margin without buying
# MAGIC retention. For the churner profile, a discount might be justified if it meaningfully
# MAGIC improves the renewal probability.
# MAGIC
# MAGIC Note: without a calibrated price elasticity model, we cannot model the retention lift
# MAGIC precisely. The `discount_sensitivity()` method applies a flat 5% elasticity as a
# MAGIC conservative default. A fully specified analysis would feed in elasticity estimates
# MAGIC from `insurance-demand`.

# COMMAND ----------

# ─── Discount sensitivity for stayer and churner ─────────────────────────────

discount_amounts = [0.0, 25.0, 50.0, 75.0, 100.0]

# Representative single-policy DataFrames with a mid-range premium
stayer_policy = pl.DataFrame({
    "policy_id":      [0],
    "ncd_level":      [8.0],
    "direct_debit":   [1.0],
    "aggregator":     [0.0],
    "annual_premium": [650.0],
    "expected_loss":  [403.0],    # 62% loss ratio
})

churner_policy = pl.DataFrame({
    "policy_id":      [0],
    "ncd_level":      [0.0],
    "direct_debit":   [0.0],
    "aggregator":     [1.0],
    "annual_premium": [650.0],
    "expected_loss":  [403.0],
})

stayer_disc  = clv_cure.discount_sensitivity(stayer_policy,  discount_amounts)
churner_disc = clv_cure.discount_sensitivity(churner_policy, discount_amounts)

print("Stayer profile (NCD 8, DD, non-aggregator):")
print(stayer_disc.to_pandas().to_string(index=False))
print()
print("Churner profile (NCD 0, no DD, aggregator):")
print(churner_disc.to_pandas().to_string(index=False))

print()
print("Interpretation:")
print("  'discount_justified' = True where CLV with discount >= CLV without discount.")
print("  For the stayer, almost no discount is justified because they would renew anyway.")
print("  For the churner, whether a discount pays depends on the elasticity assumption.")
print("  Pair this with insurance-demand elasticity models for calibrated answers.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | | Kaplan-Meier | Cox PH | Cure model |
# MAGIC |---|---|---|---|
# MAGIC | Cure fraction estimation | Indirect (KM plateau) | Not possible | Direct, covariate-adjusted |
# MAGIC | 3-year retention | Reasonably accurate | Reasonably accurate | Most accurate |
# MAGIC | 5-year retention | Deteriorating | Deteriorating for stayers | Correct plateau |
# MAGIC | Long-run survival floor | Tends to zero | Tends to zero | Correct `pi(x)` plateau |
# MAGIC | CLV calculation | No covariate adjustment | Yes, via Cox PH integral | Most accurate integral |
# MAGIC | Consumer Duty evidence | None | Partial | Full `S(t)` + cure prob output |
# MAGIC
# MAGIC ### The practical implication
# MAGIC
# MAGIC Pricing teams who use Cox PH for CLV will systematically undervalue structural stayers.
# MAGIC The cure model's higher CLV estimate for high-NCD, direct-debit customers changes the
# MAGIC renewal pricing decision: you can justify a larger loyalty discount and still show
# MAGIC positive expected CLV. For an aggregator-sourced, low-NCD customer, the cure model
# MAGIC produces lower CLV — signalling that a retention discount may not pay back.
# MAGIC
# MAGIC ### What to do next
# MAGIC
# MAGIC - **Competing risks**: if mid-term cancellation is a separate event type in your data,
# MAGIC   use `insurance_survival.competing_risks.FineGrayFitter` rather than treating all
# MAGIC   exits as identical. Mid-term cancellations and non-renewals have different economics
# MAGIC   and different covariate profiles.
# MAGIC - **Time-varying covariates**: `ExposureTransformer` produces start/stop format that
# MAGIC   feeds `lifelines.CoxTimeVaryingFitter` for policies with MTAs. Covariate changes
# MAGIC   within the year (vehicle upgrades, address moves) shift the hazard.
# MAGIC - **Price elasticity integration**: pair `SurvivalCLV.discount_sensitivity()` with
# MAGIC   estimated elasticities from `insurance-demand` for calibrated discount targeting.
# MAGIC - **MLflow deployment**: register the fitted cure model via `LifelinesMLflowWrapper`
# MAGIC   to serve renewal-time retention scores through Databricks Model Serving.

# COMMAND ----------

print("insurance-survival demo complete.")
print(f"\nLibrary version: {__import__('insurance_survival').__version__}")
print(f"\nKey results:")
print(f"  Empirical cure fraction (DGP):   {empirical_cure:.3f}")
print(f"  Cure model estimate (portfolio): {mean_pred_cure:.3f}  (error: {abs(mean_pred_cure - empirical_cure):.3f})")
print(f"  CLV MAE — cure model:            £{mae_cure:.2f}")
print(f"  CLV MAE — Cox PH:                £{mae_cox:.2f}")
print(f"  CLV improvement from cure model: £{mae_cox - mae_cure:.2f} MAE reduction")
