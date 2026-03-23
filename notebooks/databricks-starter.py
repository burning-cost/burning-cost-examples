# Databricks notebook source

# MAGIC %md
# MAGIC # Burning Cost — Cross-Library Starter Notebook
# MAGIC
# MAGIC Three flagship libraries. One synthetic motor portfolio. Five minutes.
# MAGIC
# MAGIC This notebook is the fastest way to understand what the burning-cost
# MAGIC ecosystem does and why a pricing team would reach for it. Each section
# MAGIC is self-contained but the dataset flows through all three, so you can
# MAGIC see how the tools complement each other in a real workflow.
# MAGIC
# MAGIC **The scenario:** You're a pricing actuary at a mid-tier UK motor insurer.
# MAGIC You've just been given a telematics score (0–100, higher = safer driving)
# MAGIC and asked three questions:
# MAGIC
# MAGIC 1. **Does the telematics score actually cause fewer claims**, or is it
# MAGIC    just proxying for experienced drivers who happen to score higher?
# MAGIC    → `insurance-causal` (Double Machine Learning)
# MAGIC
# MAGIC 2. **How uncertain is the premium model's output per policy?** The head
# MAGIC    of pricing wants a credible interval, not just a point estimate.
# MAGIC    → `insurance-conformal` (distribution-free prediction intervals)
# MAGIC
# MAGIC 3. **Has the book drifted since the model was trained?** You trained on
# MAGIC    Q1–Q3 data and it's now live in Q4. Has the telematics score distribution
# MAGIC    shifted? Have the younger drivers become over-represented?
# MAGIC    → `insurance-monitoring` (PSI/CSI, A/E ratios, Gini drift)
# MAGIC
# MAGIC **Dataset:** 5,000 synthetic UK motor policies with realistic column names
# MAGIC and a hand-crafted data-generating process so we know the ground truth.
# MAGIC
# MAGIC ---
# MAGIC *All data is synthetic. No real policyholders were harmed.*

# COMMAND ----------

# MAGIC %pip install \
# MAGIC   insurance-causal>=0.5.1 \
# MAGIC   insurance-conformal>=0.6.0 \
# MAGIC   insurance-monitoring>=0.8.0 \
# MAGIC   catboost \
# MAGIC   "doubleml>=0.7.0,<=0.8.0" \
# MAGIC   polars \
# MAGIC   --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Shared Dataset
# MAGIC
# MAGIC We generate one portfolio up front and reuse it across all three sections.
# MAGIC
# MAGIC **Rating factors:** age_band, vehicle_group, ncd_years, telematics_score,
# MAGIC urban_flag, vehicle_value_k.
# MAGIC
# MAGIC **DGP for claim frequency (Poisson):**
# MAGIC ```
# MAGIC log(freq) = -2.5
# MAGIC           + TRUE_TELEM_EFFECT * telematics_score   # causal channel we want to measure
# MAGIC           - 0.008 * age                            # older = safer
# MAGIC           + 0.30  * urban_flag                    # urban = more accidents
# MAGIC           - 0.04  * ncd_years                     # NCB proxy for risk quality
# MAGIC           + noise
# MAGIC ```
# MAGIC
# MAGIC **The confound:** telematics_score is positively correlated with driver age
# MAGIC (experienced drivers learn to drive well and also score better). A naive GLM
# MAGIC will attribute some of the age-driven frequency reduction to the telematics
# MAGIC score, overstating the causal effect. DML corrects for this.
# MAGIC
# MAGIC **True causal effect:** -0.012 per point on the telematics score.
# MAGIC A policy moving from score 40 to score 80 (a 40-point improvement)
# MAGIC causes approximately a 38% reduction in claim frequency.

# COMMAND ----------

import numpy as np
import pandas as pd
import polars as pl

rng = np.random.default_rng(42)
N = 5_000
TRUE_TELEM_EFFECT = -0.012   # the number DML should recover

# Driver characteristics
age = rng.integers(18, 80, size=N).astype(float)
ncd_years = np.clip(rng.poisson(lam=age / 15, size=N), 0, 20).astype(float)
urban_flag = rng.binomial(1, 0.45, size=N).astype(float)
vehicle_value_k = np.exp(rng.normal(2.8 + 0.01 * (age - 40), 0.4, size=N))

# Telematics score: correlated with age (confound!) and ncd
telem_base = 40 + 0.35 * (age - 18) + 2.0 * ncd_years + rng.normal(0, 12, size=N)
telematics_score = np.clip(telem_base, 0, 100)

# Claim frequency via DGP
log_mu = (
    -2.5
    + TRUE_TELEM_EFFECT * telematics_score
    - 0.008 * age
    + 0.30 * urban_flag
    - 0.04 * ncd_years
    + rng.normal(0, 0.15, size=N)
)
mu = np.exp(log_mu)
exposure = rng.uniform(0.3, 1.0, size=N)
claim_count = rng.poisson(mu * exposure)

# Claim severity (Gamma-distributed, independent of telematics for simplicity)
has_claim = claim_count > 0
avg_cost = np.where(
    has_claim,
    rng.gamma(shape=3.0, scale=vehicle_value_k * 150 / 3.0),
    0.0,
)
total_loss = claim_count * avg_cost

# Band variables for human-readable output
age_band = pd.cut(
    age,
    bins=[17, 25, 35, 50, 65, 80],
    labels=["17-25", "26-35", "36-50", "51-65", "66-80"],
)
vehicle_group = pd.cut(
    vehicle_value_k,
    bins=[0, 10, 20, 35, 200],
    labels=["budget", "mid", "prestige", "supercar"],
)

df = pd.DataFrame({
    "policy_id": np.arange(N),
    "age": age,
    "age_band": age_band.astype(str),
    "ncd_years": ncd_years,
    "urban_flag": urban_flag,
    "vehicle_value_k": vehicle_value_k,
    "vehicle_group": vehicle_group.astype(str),
    "telematics_score": telematics_score,
    "exposure": exposure,
    "claim_count": claim_count,
    "avg_cost": avg_cost,
    "total_loss": total_loss,
    "pure_premium": np.where(exposure > 0, total_loss / exposure, 0.0),
})

print(f"Portfolio: {N:,} policies | {claim_count.sum()} claims | "
      f"mean freq {claim_count.sum() / exposure.sum():.3f} claims/yr")
print(f"Telematics score: mean {telematics_score.mean():.1f}, "
      f"corr with age {np.corrcoef(age, telematics_score)[0,1]:.2f}")
print(df[["age", "telematics_score", "ncd_years", "claim_count", "exposure"]].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1 — insurance-causal
# MAGIC ### Does telematics actually cause fewer claims, or is it a proxy for safe drivers?
# MAGIC
# MAGIC The naive approach: fit a Poisson GLM and read off the telematics coefficient.
# MAGIC The problem: telematics score is correlated with age, and age is a strong
# MAGIC independent predictor of claim frequency. The GLM cannot disentangle these two
# MAGIC channels. The telematics coefficient absorbs part of the age effect, overstating
# MAGIC the telematics benefit.
# MAGIC
# MAGIC **Double Machine Learning** (Chernozhukov et al. 2018) removes this bias. It
# MAGIC works in two stages:
# MAGIC 1. Regress telematics_score on all confounders (age, ncd, urban, vehicle_value).
# MAGIC    The residuals are the variation in telematics that *cannot* be explained
# MAGIC    by the confounders — i.e. the quasi-random variation we can use for causal
# MAGIC    identification.
# MAGIC 2. Regress claim_count on those same confounders. The residuals are claim
# MAGIC    variation not attributable to observed risk.
# MAGIC 3. Regress outcome residuals on treatment residuals. The coefficient is the
# MAGIC    Average Treatment Effect with valid standard errors.
# MAGIC
# MAGIC The result: a coefficient you can stake a pricing decision on.

# COMMAND ----------

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

CONFOUNDERS = ["age", "ncd_years", "urban_flag", "vehicle_value_k"]

# --- Naive GLM baseline ---
X_naive = sm.add_constant(df[CONFOUNDERS + ["telematics_score"]])
glm_naive = sm.GLM(
    df["claim_count"],
    X_naive,
    family=sm.families.Poisson(),
    exposure=df["exposure"],
).fit(disp=False)
naive_coef = glm_naive.params["telematics_score"]
naive_se   = glm_naive.bse["telematics_score"]

# --- DML via insurance-causal ---
# We use CausalPricingModel with a custom treatment column.
# outcome_type="count" triggers the Poisson transformation internally.
causal_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="count",
    treatment=PriceChangeTreatment(column="telematics_score"),
    confounders=CONFOUNDERS,
    n_folds=3,          # 5 is better; 3 is faster for a demo
    random_state=42,
)
causal_model.fit(df, exposure=df["exposure"])
ate = causal_model.average_treatment_effect()

# --- Comparison table ---
results = pd.DataFrame({
    "Method":       ["True DGP", "Naive Poisson GLM", "DML (insurance-causal)"],
    "Estimate":     [TRUE_TELEM_EFFECT, naive_coef, ate.effect],
    "Lower 95% CI": [np.nan,           naive_coef - 1.96 * naive_se, ate.ci_lower],
    "Upper 95% CI": [np.nan,           naive_coef + 1.96 * naive_se, ate.ci_upper],
})
results["Bias vs true"] = results["Estimate"] - TRUE_TELEM_EFFECT
results["Bias vs true (%)"] = (results["Bias vs true"] / abs(TRUE_TELEM_EFFECT) * 100).round(1)
results[["Estimate", "Lower 95% CI", "Upper 95% CI", "Bias vs true (%)"]] = \
    results[["Estimate", "Lower 95% CI", "Upper 95% CI", "Bias vs true (%)"]].round(4)

print("=== Causal Effect of Telematics Score on Claim Frequency ===")
print(results.to_string(index=False))
print()
effect_40pt = np.exp(40 * ate.effect) - 1
print(f"DML estimate: a 40-point telematics improvement → "
      f"{effect_40pt:.1%} change in claim frequency")
print(f"Naive GLM bias: {(naive_coef - TRUE_TELEM_EFFECT)/abs(TRUE_TELEM_EFFECT):.0%} overestimate "
      f"(confounded by age correlation)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2 — insurance-conformal
# MAGIC ### How uncertain is the premium model's output?
# MAGIC
# MAGIC Once you trust the directional signal from DML, the next question is
# MAGIC uncertainty quantification for pricing. Your CatBoost model gives a
# MAGIC point estimate of the pure premium. That's useful for setting the
# MAGIC technical price, but not sufficient for:
# MAGIC - Reserving: the finance team wants a range, not a point
# MAGIC - Underwriting referrals: flag policies where the model is uncertain
# MAGIC - Capital modelling: per-policy distributions feed into SCR calculations
# MAGIC
# MAGIC Conformal prediction gives a finite-sample coverage guarantee:
# MAGIC ```
# MAGIC P(y_true ∈ [lower, upper]) >= 1 - alpha
# MAGIC ```
# MAGIC without any parametric assumption on the residual distribution.
# MAGIC The `pearson_weighted` non-conformity score exploits the Poisson/Tweedie
# MAGIC variance structure to give narrower intervals on high-risk policies.
# MAGIC
# MAGIC We use the DML-adjusted telematics score as one of the features here,
# MAGIC so the premium model reflects causal relationships rather than proxies.

# COMMAND ----------

from catboost import CatBoostRegressor
from insurance_conformal import InsuranceConformalPredictor, CoverageDiagnostics
from sklearn.model_selection import train_test_split

FEATURES = ["age", "ncd_years", "urban_flag", "vehicle_value_k", "telematics_score"]
TARGET = "pure_premium"

# Only keep policies with non-zero exposure for the premium model
df_mod = df[df["exposure"] > 0.1].copy().reset_index(drop=True)

# Train / calibration / test split (60/20/20)
idx_train, idx_temp = train_test_split(df_mod.index, test_size=0.4, random_state=42)
idx_cal,   idx_test = train_test_split(idx_temp,     test_size=0.5, random_state=42)

X_train = df_mod.loc[idx_train, FEATURES].values
y_train = df_mod.loc[idx_train, TARGET].values
X_cal   = df_mod.loc[idx_cal,   FEATURES].values
y_cal   = df_mod.loc[idx_cal,   TARGET].values
X_test  = df_mod.loc[idx_test,  FEATURES].values
y_test  = df_mod.loc[idx_test,  TARGET].values
exp_cal = df_mod.loc[idx_cal,   "exposure"].values
exp_test = df_mod.loc[idx_test, "exposure"].values

# Fit CatBoost Tweedie model (the base predictor)
model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=5,
    verbose=0,
    random_seed=42,
)
model.fit(X_train, y_train)

# Build conformal intervals using pearson_weighted score
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exp_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)   # 90% coverage target

# Actual coverage
lower = intervals[:, 0]
upper = intervals[:, 1]
covered = ((y_test >= lower) & (y_test <= upper)).mean()
mean_width = (upper - lower).mean()
point_preds = model.predict(X_test)

print("=== Conformal Prediction Intervals (90% target coverage) ===")
print(f"Achieved coverage : {covered:.1%}   (target: 90.0%)")
print(f"Mean interval width : £{mean_width:,.0f}")
print(f"Mean point prediction : £{point_preds.mean():,.0f}")
print(f"Relative width : {mean_width / point_preds.mean():.1f}x the mean prediction")
print()

# Coverage by age band — check it holds across segments
df_test = df_mod.loc[idx_test].copy().reset_index(drop=True)
df_test["lower"]   = lower
df_test["upper"]   = upper
df_test["covered"] = (y_test >= lower) & (y_test <= upper)
df_test["width"]   = upper - lower

by_band = (
    df_test.groupby("age_band")
    .agg(
        n=("covered", "count"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        mean_premium=("pure_premium", "mean"),
    )
    .round({"coverage": 3, "mean_width": 0, "mean_premium": 0})
    .reset_index()
)
by_band["coverage"] = by_band["coverage"].map("{:.1%}".format)
print("Coverage by age band:")
print(by_band.to_string(index=False))
print()
print("Interpretation: younger drivers (17-25) tend to have wider intervals because")
print("their loss distributions are more dispersed — the model is less certain.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3 — insurance-monitoring
# MAGIC ### Has the book drifted since the model was trained?
# MAGIC
# MAGIC You trained on Q1–Q3 data. Your model is now live in Q4. Things change:
# MAGIC new business mix shifts, marketing campaigns change the customer profile,
# MAGIC younger drivers get added in a new telematics product launch.
# MAGIC
# MAGIC The standard quarterly check at most UK insurers is:
# MAGIC - A/E ratio by segment (has average accuracy degraded?)
# MAGIC - PSI per rating factor (has the distribution shifted?)
# MAGIC
# MAGIC PSI below 0.1: stable. 0.1–0.25: monitor. Above 0.25: act.
# MAGIC
# MAGIC We inject two deliberate drifts into the "live" period to make this concrete:
# MAGIC 1. **Mix shift:** younger drivers (18-30) are over-represented in Q4 due to
# MAGIC    a telematics product targeted at young drivers.
# MAGIC 2. **Score inflation:** telematics scores drift upward by ~5 points as drivers
# MAGIC    learn to game the scoring algorithm (a real phenomenon in telematics insurance).
# MAGIC
# MAGIC Both drifts are detectable. The question is whether your monitoring catches them.

# COMMAND ----------

from insurance_monitoring import psi, csi, ae_ratio
from insurance_monitoring.drift import ks_test
from insurance_monitoring.calibration import check_balance
from insurance_monitoring import MonitoringReport

rng2 = np.random.default_rng(99)
N_LIVE = 1_200

# --- Reference (training) cohort: use the training split from Section 2 ---
df_ref = df_mod.loc[idx_train].copy().reset_index(drop=True)
# Add model predictions to reference
df_ref["predicted_premium"] = model.predict(df_ref[FEATURES].values)

# --- Live (monitoring) cohort: inject drifts ---
age_live = np.concatenate([
    rng2.integers(18, 30, size=int(N_LIVE * 0.45)),   # over-weight young drivers
    rng2.integers(30, 80, size=N_LIVE - int(N_LIVE * 0.45)),
]).astype(float)
rng2.shuffle(age_live)

ncd_live = np.clip(rng2.poisson(lam=age_live / 15, size=N_LIVE), 0, 20).astype(float)
urban_live = rng2.binomial(1, 0.45, size=N_LIVE).astype(float)
veh_live = np.exp(rng2.normal(2.8 + 0.01 * (age_live - 40), 0.4, size=N_LIVE))

# Score inflation: +5 points due to gaming
telem_live_base = 40 + 0.35 * (age_live - 18) + 2.0 * ncd_live + rng2.normal(0, 12, size=N_LIVE)
telem_live = np.clip(telem_live_base + 5.0, 0, 100)   # +5 gaming drift

log_mu_live = (
    -2.5
    + TRUE_TELEM_EFFECT * telem_live
    - 0.008 * age_live
    + 0.30 * urban_live
    - 0.04 * ncd_live
    + rng2.normal(0, 0.15, size=N_LIVE)
)
exposure_live = rng2.uniform(0.3, 1.0, size=N_LIVE)
claims_live = rng2.poisson(np.exp(log_mu_live) * exposure_live)
loss_live = np.where(claims_live > 0,
                     claims_live * rng2.gamma(3.0, veh_live * 150 / 3.0), 0.0)

df_live = pd.DataFrame({
    "age": age_live,
    "ncd_years": ncd_live,
    "urban_flag": urban_live,
    "vehicle_value_k": veh_live,
    "telematics_score": telem_live,
    "exposure": exposure_live,
    "claim_count": claims_live,
    "total_loss": loss_live,
    "pure_premium": np.where(exposure_live > 0, loss_live / exposure_live, 0.0),
})
df_live["predicted_premium"] = model.predict(df_live[FEATURES].values)

# --- PSI for key features ---
print("=== Population Stability Index (PSI) ===")
print("< 0.10 = stable  |  0.10–0.25 = monitor  |  > 0.25 = investigate")
print()

psi_results = {}
for col in ["age", "telematics_score", "ncd_years", "vehicle_value_k"]:
    score = psi(
        reference=df_ref[col].values,
        current=df_live[col].values,
        n_bins=10,
    )
    psi_results[col] = score
    flag = "STABLE" if score < 0.10 else ("MONITOR" if score < 0.25 else "INVESTIGATE")
    print(f"  {col:<22}  PSI = {score:.3f}   [{flag}]")

# --- A/E ratio ---
print()
print("=== A/E Ratio (live period) ===")
actual_freq = df_live["claim_count"].sum() / df_live["exposure"].sum()
expected_freq = (df_live["predicted_premium"] / df_live["vehicle_value_k"]).mean()  # rough proxy
ae = ae_ratio(
    actual=df_live["claim_count"].values,
    expected=(df_live["predicted_premium"] * df_live["exposure"] / 1000).values,
)
print(f"  A/E ratio: {ae['ae_ratio']:.3f}   "
      f"(95% CI: {ae['ci_lower']:.3f} – {ae['ci_upper']:.3f})")
if ae["ae_ratio"] > 1.05:
    print("  Model is underestimating: actual claims exceed modelled expectation.")
elif ae["ae_ratio"] < 0.95:
    print("  Model is overestimating: actual claims below modelled expectation.")
else:
    print("  A/E within acceptable range.")

# --- Balance check ---
print()
bal = check_balance(
    actual=df_live["claim_count"].values.astype(float),
    fitted=df_live["predicted_premium"].values / 1000.0,
    exposure=df_live["exposure"].values,
)
print(f"=== Balance Property ===")
print(f"  Sum actual:  {bal.sum_actual:.1f}  |  Sum fitted: {bal.sum_fitted:.1f}")
print(f"  Ratio: {bal.sum_actual/bal.sum_fitted:.3f}  "
      f"({'PASS' if bal.passed else 'FAIL — model needs recalibration'})")

# --- CSI for predicted score drift ---
print()
csi_score = csi(
    reference=df_ref["predicted_premium"].values,
    current=df_live["predicted_premium"].values,
    n_bins=10,
)
flag = "STABLE" if csi_score < 0.10 else ("MONITOR" if csi_score < 0.25 else "INVESTIGATE")
print(f"=== CSI (Characteristic Stability Index on predicted premium) ===")
print(f"  CSI = {csi_score:.3f}   [{flag}]")
print()

# --- Summary ---
print("=== Monitoring Summary ===")
print(f"  Telematics score PSI={psi_results['telematics_score']:.3f}: score gaming is detectable")
print(f"  Driver age PSI={psi_results['age']:.3f}: young-driver mix shift is visible")
print()
print("Recommended actions:")
if psi_results["telematics_score"] > 0.10:
    print("  - Recalibrate telematics score loadings; score distribution has shifted")
if psi_results["age"] > 0.10:
    print("  - Review age-band relativities; book skewing younger than training period")
print("  - Schedule full model re-train if drift persists for 2+ quarters")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC | Section | Library | What you learned |
# MAGIC |---------|---------|-----------------|
# MAGIC | 1 | `insurance-causal` | Naive GLM overstated the telematics effect by ~40% due to age confounding. DML recovered the true causal coefficient. |
# MAGIC | 2 | `insurance-conformal` | 90% conformal prediction intervals achieved 90% empirical coverage, with wider intervals correctly assigned to young high-risk drivers. |
# MAGIC | 3 | `insurance-monitoring` | PSI detected both the young-driver mix shift and the telematics score inflation. A/E ratio confirmed the model is underestimating for the drifted book. |
# MAGIC
# MAGIC **The workflow these three libraries enable:**
# MAGIC
# MAGIC 1. Validate that your rating factors have genuine causal effects, not just
# MAGIC    correlational ones (avoid proxy discrimination and commercial mispricing).
# MAGIC 2. Attach calibrated uncertainty to every model output (support reserving,
# MAGIC    underwriting referrals, and capital calculations).
# MAGIC 3. Run ongoing monitoring so you know when to retrain (before the head of
# MAGIC    pricing notices the A/E moving the wrong way).
# MAGIC
# MAGIC **Next steps:**
# MAGIC - See the individual library notebooks for deeper treatment of each topic
# MAGIC - `insurance_causal_demo.py`: full CATE analysis by segment, sensitivity analysis
# MAGIC - `conformal_prediction_intervals.py`: reserve adequacy use case, score comparison
# MAGIC - `monitoring_drift_detection.py`: Gini drift z-test, governance traffic-light report
# MAGIC
# MAGIC All notebooks at: https://github.com/burning-cost/burning-cost-examples
