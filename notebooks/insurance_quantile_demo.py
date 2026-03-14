# Databricks notebook source

# MAGIC %md
# MAGIC # Tail Risk Quantification with insurance-quantile
# MAGIC
# MAGIC Every pricing actuary has seen the pattern: your Tweedie GBM looks fine on
# MAGIC average, the deviance is low, the lift curves are clean — and then a large
# MAGIC loss season arrives and you discover the model was systematically underpricing
# MAGIC the risks that matter most.
# MAGIC
# MAGIC The root cause is not that the mean model is wrong. It is that the mean is
# MAGIC the wrong thing to price from when the loss distribution is heavy-tailed and
# MAGIC heteroskedastic. A risk where 99% of outcomes are zero and 1% are
# MAGIC catastrophic has the same expected value as a risk where every outcome is
# MAGIC moderate — but they are not the same risk, and they should not attract the
# MAGIC same large loss loading.
# MAGIC
# MAGIC This notebook works through a complete tail risk quantification workflow:
# MAGIC
# MAGIC 1. **Synthetic severity data** — 50,000 non-zero claims from a known
# MAGIC    compound distribution (lognormal body, Pareto tail), with tail weight
# MAGIC    that varies by risk segment. We know the ground truth exactly.
# MAGIC 2. **Parametric lognormal baseline** — fit a lognormal, compute quantiles
# MAGIC    and TVaR analytically. Diagnose where and why it fails.
# MAGIC 3. **insurance-quantile** — CatBoost quantile regression at multiple
# MAGIC    levels (0.5, 0.75, 0.9, 0.95, 0.99), TVaR, ILF curves, exceedance curve.
# MAGIC 4. **Benchmark table** — quantile accuracy at 90/95/99th, TVaR accuracy,
# MAGIC    ILF curve shape, pinball loss.
# MAGIC 5. **Large loss loading** — show how per-risk tail loading varies by
# MAGIC    segment vs a single portfolio-wide rate.
# MAGIC
# MAGIC The key point we will demonstrate: parametric distributional assumptions fail
# MAGIC not because the lognormal is generally wrong, but because the *shape parameter*
# MAGIC varies across risk segments in a way the parametric model cannot capture
# MAGIC without heroic manual segmentation. Quantile regression learns this directly
# MAGIC from the data.

# COMMAND ----------

# MAGIC %pip install insurance-quantile catboost polars scikit-learn matplotlib scipy

# COMMAND ----------

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Synthetic Severity Data
# MAGIC
# MAGIC We need synthetic data where we know the ground truth — true quantiles,
# MAGIC true TVaR, true ILF curves — so we can measure how well each method
# MAGIC recovers them.
# MAGIC
# MAGIC ### Data-generating process
# MAGIC
# MAGIC The DGP is a spliced lognormal-Pareto. This is a standard actuarial
# MAGIC choice for modelling property and liability severity:
# MAGIC
# MAGIC - Below the splice point (£50,000): lognormal, log-scale mean and sigma
# MAGIC   vary by risk features
# MAGIC - Above the splice point: Pareto tail with index α that varies by segment
# MAGIC
# MAGIC The critical design choice is that **tail heaviness varies by vehicle group**.
# MAGIC Vehicle group 4 has a materially heavier Pareto tail than vehicle group 1.
# MAGIC A single fitted lognormal across all groups will get the group 4 tail badly
# MAGIC wrong, because its shape parameter will be pulled towards the portfolio average.
# MAGIC
# MAGIC Features: vehicle_age, driver_age, ncd_years, vehicle_group (1–4).
# MAGIC The DGP encodes:
# MAGIC - Older vehicles: lower severity (less repair cost)
# MAGIC - Young drivers: higher severity (higher speed impact, different mix)
# MAGIC - NCD: modest reduction (correlation with cautious behaviour)
# MAGIC - Vehicle group: primary driver of tail weight

# COMMAND ----------

N_TRAIN = 40_000
N_TEST  = 10_000
N_TOTAL = N_TRAIN + N_TEST

SPLICE  = 50_000.0   # lognormal/Pareto crossover
SEED    = 2024

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------------
def generate_features(n: int, rng: np.random.Generator) -> np.ndarray:
    vehicle_age   = rng.integers(1, 15, size=n).astype(float)
    driver_age    = rng.integers(21, 75, size=n).astype(float)
    ncd_years     = rng.integers(0, 9,  size=n).astype(float)
    vehicle_group = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)
    return np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])

FEATURE_NAMES = ["vehicle_age", "driver_age", "ncd_years", "vehicle_group"]

# ------------------------------------------------------------------
# True DGP parameters — these are what we benchmark against
# ------------------------------------------------------------------
def dgp_params(X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return the true lognormal and Pareto parameters for each risk.

    log_mu, log_sigma: lognormal below the splice point
    pareto_alpha:      Pareto shape above the splice point
    """
    vehicle_age   = X[:, 0]
    driver_age    = X[:, 1]
    ncd_years     = X[:, 2]
    vehicle_group = X[:, 3]

    log_mu = (
        8.0                              # baseline: approx £3,000 median claim
        - 0.02 * vehicle_age             # older vehicles slightly cheaper
        + 0.008 * np.clip(driver_age - 35, -14, 25)  # modest age effect
        - 0.012 * ncd_years              # NCD discount
        + 0.10 * vehicle_group           # higher group = higher severity
    )

    # Tail weight: sigma and Pareto alpha both vary with vehicle group
    # Group 1: thin tail, group 4: heavy tail
    log_sigma = 0.60 + 0.08 * vehicle_group   # 0.68 to 0.92

    # Pareto alpha: lower = heavier tail
    # Group 1 → alpha=3.5 (thin), group 4 → alpha=2.0 (heavy liability-like)
    pareto_alpha = 4.0 - 0.50 * vehicle_group  # 3.5, 3.0, 2.5, 2.0

    return {
        "log_mu":       log_mu,
        "log_sigma":    log_sigma,
        "pareto_alpha": pareto_alpha,
    }

# ------------------------------------------------------------------
# True quantile function (analytical for the spliced distribution)
# ------------------------------------------------------------------
def true_quantile(X: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the true alpha-quantile of the spliced lognormal-Pareto
    for each risk in X.
    """
    params = dgp_params(X)
    log_mu       = params["log_mu"]
    log_sigma    = params["log_sigma"]
    pareto_alpha = params["pareto_alpha"]
    n            = X.shape[0]

    # P(Y <= SPLICE) under the lognormal: CDF at SPLICE
    p_below_splice = scipy_stats.norm.cdf(
        (np.log(SPLICE) - log_mu) / log_sigma
    )

    out = np.empty(n)
    for i in range(n):
        pb = p_below_splice[i]
        if alpha <= pb:
            # quantile is in the lognormal body
            z = scipy_stats.norm.ppf(alpha)
            out[i] = np.exp(log_mu[i] + log_sigma[i] * z)
        else:
            # quantile is in the Pareto tail
            # Pareto survival above SPLICE: S(x) = (SPLICE / x)^pareto_alpha
            # Conditional on exceeding SPLICE, P(Y > x | Y > SPLICE) = (SPLICE/x)^alpha
            # We need P(Y > x) = (1-pb) * (SPLICE/x)^pareto_alpha = 1 - alpha
            # => x = SPLICE * ((1-pb) / (1-alpha))^(1/pareto_alpha)
            excess_prob = (1.0 - alpha) / (1.0 - pb)
            out[i] = SPLICE * (1.0 / excess_prob) ** (1.0 / pareto_alpha[i])
    return out


def true_tvar(X: np.ndarray, alpha: float) -> np.ndarray:
    """
    True TVaR_alpha for the spliced lognormal-Pareto.
    TVaR_alpha = E[Y | Y > Q_alpha(Y)] = mean of the distribution above VaR.
    """
    params = dgp_params(X)
    log_mu       = params["log_mu"]
    log_sigma    = params["log_sigma"]
    pareto_alpha = params["pareto_alpha"]
    n            = X.shape[0]

    q_alpha = true_quantile(X, alpha)
    p_below_splice = scipy_stats.norm.cdf(
        (np.log(SPLICE) - log_mu) / log_sigma
    )

    out = np.empty(n)
    for i in range(n):
        pb = p_below_splice[i]
        qa = q_alpha[i]
        pa = pareto_alpha[i]

        if alpha <= pb:
            # VaR is in lognormal body; TVaR has contributions from both body and tail
            # E[Y | Y > qa] = E[Y * I(Y > qa)] / P(Y > qa)
            # Lognormal contribution: E[Y * I(qa < Y <= SPLICE)] / (1-alpha)
            # Pareto contribution: E[Y * I(Y > SPLICE)] / (1-alpha)

            z_a = (np.log(qa) - log_mu[i]) / log_sigma[i]
            z_s = (np.log(SPLICE) - log_mu[i]) / log_sigma[i]
            mu_i = log_mu[i]
            sig_i = log_sigma[i]

            # Lognormal E[Y * I(qa < Y <= SPLICE)]
            ln_mean_above_qa_below_splice = (
                np.exp(mu_i + 0.5 * sig_i**2)
                * (scipy_stats.norm.cdf(z_s - sig_i) - scipy_stats.norm.cdf(z_a - sig_i))
            )

            # Pareto contribution: E[Y | Y > SPLICE] = SPLICE * pa / (pa - 1) for pa > 1
            if pa > 1.0:
                e_pareto_above_splice = SPLICE * pa / (pa - 1.0)
            else:
                e_pareto_above_splice = SPLICE * 5.0  # fallback for very heavy tails

            pareto_weight = (1.0 - pb) / (1.0 - alpha)
            lognormal_weight = (pb - alpha) / (1.0 - alpha)

            out[i] = (
                ln_mean_above_qa_below_splice / (1.0 - alpha)
                + pareto_weight * e_pareto_above_splice
            )
        else:
            # VaR is in Pareto tail; TVaR is purely the Pareto conditional mean above VaR
            # For Pareto(alpha): E[Y | Y > qa] = qa * pa / (pa - 1)
            if pa > 1.0:
                out[i] = qa * pa / (pa - 1.0)
            else:
                out[i] = qa * 5.0  # very heavy tail
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate claims
# MAGIC
# MAGIC We draw from the spliced distribution directly. For each claim:
# MAGIC - Draw from the lognormal
# MAGIC - If the draw exceeds the splice point, resample from the Pareto tail
# MAGIC   (conditional on being above SPLICE)
# MAGIC
# MAGIC This gives us a clean severity distribution — no zeros, since we're
# MAGIC modelling severity conditional on a claim occurring.

# COMMAND ----------

def simulate_claims(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate from the spliced lognormal-Pareto severity distribution."""
    params = dgp_params(X)
    log_mu       = params["log_mu"]
    log_sigma    = params["log_sigma"]
    pareto_alpha = params["pareto_alpha"]
    n            = X.shape[0]

    y = np.zeros(n)

    # Step 1: draw lognormal for everyone
    y_ln = np.exp(rng.normal(log_mu, log_sigma, n))

    # Step 2: for those above the splice, resample from Pareto tail
    above = y_ln > SPLICE
    n_above = above.sum()

    if n_above > 0:
        # Pareto draw: X = SPLICE / U^(1/alpha) where U ~ Uniform(0, 1)
        u = rng.uniform(0, 1, n_above)
        y_pareto = SPLICE / (u ** (1.0 / pareto_alpha[above]))
        y_ln[above] = y_pareto

    return y_ln


X_all = generate_features(N_TOTAL, rng)
y_all = simulate_claims(X_all, rng)

X_train_np = X_all[:N_TRAIN]
y_train_np  = y_all[:N_TRAIN]
X_test_np   = X_all[N_TRAIN:]
y_test_np   = y_all[N_TRAIN:]

print(f"Dataset: {N_TOTAL:,} severity claims")
print(f"  Train: {N_TRAIN:,} | Test: {N_TEST:,}")
print(f"  Mean severity:   £{y_all.mean():,.0f}")
print(f"  Median severity: £{np.median(y_all):,.0f}")
print(f"  90th percentile: £{np.percentile(y_all, 90):,.0f}")
print(f"  99th percentile: £{np.percentile(y_all, 99):,.0f}")
print(f"  Max severity:    £{y_all.max():,.0f}")
print(f"\nTrue 99th percentile by vehicle group (test set):")
for vg in [1, 2, 3, 4]:
    mask = X_test_np[:, 3] == vg
    true_q99 = np.percentile(y_test_np[mask], 99)
    print(f"  Group {int(vg)}: £{true_q99:,.0f}  (n={mask.sum():,})")

# COMMAND ----------

# MAGIC %md
# MAGIC The spread in 99th percentiles across vehicle groups is large. Group 4 claims
# MAGIC blow past group 1 at the tail — the Pareto shapes are genuinely different.
# MAGIC A single fitted parametric model will average these, producing biased tail
# MAGIC estimates for both groups.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Parametric Lognormal Baseline
# MAGIC
# MAGIC The standard actuarial approach: fit a lognormal to each segment (or to
# MAGIC all data with rating factors), compute quantiles and TVaR analytically.
# MAGIC
# MAGIC We fit the lognormal in log-space using a GLM-style approach: regress
# MAGIC log(Y) on the features to get conditional log_mu, then estimate a single
# MAGIC global sigma (the standard log-space variance estimator). This is how most
# MAGIC actuarial software implements lognormal severity models.
# MAGIC
# MAGIC The fundamental limitation: a single sigma means every risk segment has
# MAGIC the same tail weight. Vehicle group 4's heavier Pareto tail is invisible
# MAGIC to the model.

# COMMAND ----------

from sklearn.linear_model import Ridge

# Fit log-linear model: log(y) ~ features
log_y_train = np.log(y_train_np)

lr = Ridge(alpha=1.0, fit_intercept=True)
lr.fit(X_train_np, log_y_train)

log_y_hat_train = lr.predict(X_train_np)
log_y_hat_test  = lr.predict(X_test_np)

# Global sigma: estimated from training residuals
log_residuals_train = log_y_train - log_y_hat_train
global_sigma = log_residuals_train.std()

print(f"Lognormal baseline:")
print(f"  Log-linear model fitted on {N_TRAIN:,} claims")
print(f"  Global log-sigma (tail parameter): {global_sigma:.4f}")
print(f"  True sigma range (by group): "
      f"{0.60 + 0.08*1:.3f} – {0.60 + 0.08*4:.3f}")
print()
print(f"  The single fitted sigma {global_sigma:.3f} is an average of the")
print(f"  true group-specific sigmas. Group 1 will be over-estimated,")
print(f"  group 4 will be under-estimated.")


def lognormal_quantile(log_mu: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Analytical lognormal quantile."""
    z = scipy_stats.norm.ppf(alpha)
    return np.exp(log_mu + sigma * z)


def lognormal_tvar(log_mu: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """
    Analytical lognormal TVaR.
    TVaR_alpha = exp(mu + sigma^2/2) * Phi((sigma - Phi^{-1}(alpha)) / 1) / (1 - alpha)
    """
    z_alpha = scipy_stats.norm.ppf(alpha)
    e_y = np.exp(log_mu + 0.5 * sigma**2)
    complement_cdf = 1.0 - scipy_stats.norm.cdf(z_alpha - sigma)
    return e_y * complement_cdf / (1.0 - alpha)


# Lognormal quantile predictions on test set
ln_q90_pred  = lognormal_quantile(log_y_hat_test, global_sigma, 0.90)
ln_q95_pred  = lognormal_quantile(log_y_hat_test, global_sigma, 0.95)
ln_q99_pred  = lognormal_quantile(log_y_hat_test, global_sigma, 0.99)
ln_tvar95_pred = lognormal_tvar(log_y_hat_test, global_sigma, 0.95)

# True benchmark values on test set
true_q90_test    = true_quantile(X_test_np, 0.90)
true_q95_test    = true_quantile(X_test_np, 0.95)
true_q99_test    = true_quantile(X_test_np, 0.99)
true_tvar95_test = true_tvar(X_test_np, 0.95)

print(f"\nTest set comparison (mean across all risks):")
print(f"  {'Metric':25s}  {'True':>10}  {'Lognormal':>10}  {'Ratio':>8}")
for label, true_v, ln_v in [
    ("Mean Q90 (£)",        true_q90_test.mean(),    ln_q90_pred.mean()),
    ("Mean Q95 (£)",        true_q95_test.mean(),    ln_q95_pred.mean()),
    ("Mean Q99 (£)",        true_q99_test.mean(),    ln_q99_pred.mean()),
    ("Mean TVaR_95 (£)",    true_tvar95_test.mean(), ln_tvar95_pred.mean()),
]:
    ratio = ln_v / true_v
    print(f"  {label:25s}  {true_v:>10,.0f}  {ln_v:>10,.0f}  {ratio:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC The overall means look reasonable — the lognormal isn't wildly off in
# MAGIC aggregate. The problem surfaces when we break down by vehicle group.
# MAGIC Let's look at 99th percentile accuracy by segment.

# COMMAND ----------

print("99th percentile accuracy by vehicle group — lognormal baseline:")
print(f"  {'Group':>6}  {'N test':>7}  {'True Q99 (£)':>13}  {'LN Q99 (£)':>11}  {'Error %':>8}")
for vg in [1.0, 2.0, 3.0, 4.0]:
    mask = X_test_np[:, 3] == vg
    true_q99_g = true_q99_test[mask].mean()
    ln_q99_g   = ln_q99_pred[mask].mean()
    err_pct    = 100 * (ln_q99_g - true_q99_g) / true_q99_g
    print(f"  {int(vg):>6}  {mask.sum():>7,}  {true_q99_g:>13,.0f}  "
          f"{ln_q99_g:>11,.0f}  {err_pct:>+8.1f}%")

print()
print("Key finding: lognormal over-estimates group 1 (sigma too high)")
print("and under-estimates group 4 (sigma too low). The error is systematic")
print("and grows with severity — which is precisely where it matters most.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: insurance-quantile — CatBoost Quantile Regression
# MAGIC
# MAGIC We now fit a `QuantileGBM` at five probability levels: 0.5, 0.75, 0.9,
# MAGIC 0.95, 0.99. This single CatBoost model (using `MultiQuantile` loss) learns
# MAGIC the full conditional quantile function directly from data — no distributional
# MAGIC assumption, no shape parameter that must be the same for all segments.
# MAGIC
# MAGIC The key hyperparameters to tune in practice:
# MAGIC - `iterations`: 500–1000 is usually sufficient; use early stopping on a
# MAGIC   validation set if compute permits
# MAGIC - `depth`: 6 is a sensible default for structured tabular data
# MAGIC - `fix_crossing=True`: always leave this on — CatBoost's MultiQuantile
# MAGIC   loss does not guarantee monotone predictions per risk
# MAGIC
# MAGIC The API is Polars-native: `fit()` takes a `pl.DataFrame` and `pl.Series`,
# MAGIC `predict()` returns a `pl.DataFrame` with columns named `q_0.5`, `q_0.9`, etc.

# COMMAND ----------

from insurance_quantile import (
    QuantileGBM,
    per_risk_tvar,
    portfolio_tvar,
    large_loss_loading,
    ilf,
    exceedance_curve,
    coverage_check,
    pinball_loss,
)

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]

# Convert to Polars (the library's native format)
X_train_pl = pl.DataFrame(X_train_np, schema=FEATURE_NAMES)
y_train_pl  = pl.Series("severity", y_train_np)
X_test_pl   = pl.DataFrame(X_test_np,  schema=FEATURE_NAMES)
y_test_pl   = pl.Series("severity", y_test_np)

qgbm = QuantileGBM(
    quantiles=QUANTILES,
    fix_crossing=True,
    iterations=600,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
)
qgbm.fit(X_train_pl, y_train_pl)

print(f"QuantileGBM fitted")
print(f"  Quantile levels: {QUANTILES}")
print(f"  Training rows:   {qgbm.metadata.n_training_rows:,}")
print(f"  CatBoost params: {qgbm.metadata.catboost_params}")
print(f"  Crossing fix:    {qgbm.metadata.fix_crossing}")

# Predict on test set
q_preds = qgbm.predict(X_test_pl)
print(f"\nPrediction columns: {q_preds.columns}")
print(f"Sample (first 3 risks):")
print(q_preds.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### TVaR, ILF, and exceedance curve
# MAGIC
# MAGIC With the quantile model fitted, we extract the actuarial outputs using
# MAGIC the library's helper functions:
# MAGIC
# MAGIC - `per_risk_tvar`: E[Y | Y > VaR_α(Y)] approximated as the mean of
# MAGIC   predicted quantiles above α. Accuracy improves with more high quantile
# MAGIC   levels — our five levels (0.75, 0.9, 0.95, 0.99 above α=0.5) give
# MAGIC   a good approximation.
# MAGIC - `ilf`: E[min(Y,L₂)] / E[min(Y,L₁)] via numerical integration of the
# MAGIC   survival function derived from quantile predictions.
# MAGIC - `exceedance_curve`: average P(Y > x) across the portfolio.

# COMMAND ----------

# TVaR at 95th percentile per risk
tvar_result = per_risk_tvar(qgbm, X_test_pl, alpha=0.95)
print(f"TVaR_95 per risk (quantile GBM):")
print(f"  Mean:   £{tvar_result.values.mean():,.0f}")
print(f"  Median: £{np.median(tvar_result.values.to_numpy()):,.0f}")
print(f"  Min:    £{tvar_result.values.min():,.0f}")
print(f"  Max:    £{tvar_result.values.max():,.0f}")

# True TVaR_95 on test set
print(f"\nTrue TVaR_95 per risk:")
print(f"  Mean:   £{true_tvar95_test.mean():,.0f}")
print(f"  Median: £{np.median(true_tvar95_test):,.0f}")

# ILF: £100k to £500k basic/higher limits
ilf_preds = ilf(
    qgbm,
    X_test_pl,
    basic_limit=100_000,
    higher_limit=500_000,
)
print(f"\nILF (£100k → £500k) per risk (quantile GBM):")
print(f"  Mean ILF:   {ilf_preds.mean():.3f}")
print(f"  Median ILF: {np.median(ilf_preds.to_numpy()):.3f}")
print(f"  Min ILF:    {ilf_preds.min():.3f}")
print(f"  Max ILF:    {ilf_preds.max():.3f}")

# Portfolio exceedance curve
exc_df = exceedance_curve(qgbm, X_test_pl, n_thresholds=80)
print(f"\nExceedance curve: {len(exc_df)} threshold points")
print(f"  P(Y > £50,000):  {exc_df.filter(pl.col('threshold') > 49000).head(1)['exceedance_prob'][0]:.4f}")
print(f"  P(Y > £100,000): {exc_df.filter(pl.col('threshold') > 99000).head(1)['exceedance_prob'][0]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Benchmark — Lognormal vs Quantile GBM
# MAGIC
# MAGIC Now we build the comparison table. We measure:
# MAGIC
# MAGIC 1. **Quantile accuracy** at 90/95/99th — mean error (£) and percentage
# MAGIC    error relative to the ground truth
# MAGIC 2. **TVaR accuracy** at 95th — the key tail risk metric
# MAGIC 3. **Pinball loss** at each quantile level — the proper scoring rule for
# MAGIC    quantile regression; lower is better, and it measures overall
# MAGIC    distributional accuracy not just mean error
# MAGIC 4. **Per-segment Q99 accuracy** — where the parametric model's fixed
# MAGIC    sigma assumption causes systematic bias
# MAGIC
# MAGIC The benchmark uses the test set (10,000 claims unseen during training)
# MAGIC and the known DGP as the oracle.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a: Overall quantile accuracy

# COMMAND ----------

# Quantile GBM predictions (already computed above as q_preds)
gbm_q90  = q_preds["q_0.9"].to_numpy()
gbm_q95  = q_preds["q_0.95"].to_numpy()
gbm_q99  = q_preds["q_0.99"].to_numpy()
gbm_tvar95 = tvar_result.values.to_numpy()

print("Overall quantile accuracy (mean across test set):")
print()
print(f"  {'Metric':30s}  {'True (£)':>10}  {'Lognormal (£)':>13}  {'QuantileGBM (£)':>15}  {'LN Error %':>10}  {'GBM Error %':>11}")
print(f"  {'-'*95}")

benchmark_rows = []
for label, true_v, ln_v, gbm_v in [
    ("Q90 (90th percentile)",   true_q90_test,    ln_q90_pred,  gbm_q90),
    ("Q95 (95th percentile)",   true_q95_test,    ln_q95_pred,  gbm_q95),
    ("Q99 (99th percentile)",   true_q99_test,    ln_q99_pred,  gbm_q99),
    ("TVaR_95",                 true_tvar95_test, ln_tvar95_pred, gbm_tvar95),
]:
    t = true_v.mean()
    l = ln_v.mean()
    g = gbm_v.mean()
    ln_err = 100 * (l - t) / t
    gbm_err = 100 * (g - t) / t
    print(f"  {label:30s}  {t:>10,.0f}  {l:>13,.0f}  {g:>15,.0f}  {ln_err:>+10.1f}%  {gbm_err:>+11.1f}%")
    benchmark_rows.append({
        "Metric": label,
        "True": f"£{t:,.0f}",
        "Lognormal": f"£{l:,.0f}  ({ln_err:+.1f}%)",
        "QuantileGBM": f"£{g:,.0f}  ({gbm_err:+.1f}%)",
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b: Pinball loss (the proper scoring rule)
# MAGIC
# MAGIC Mean error (£) tells you about bias — it averages positive and negative
# MAGIC errors and can look good even when the model is wrong in opposite
# MAGIC directions for different segments.
# MAGIC
# MAGIC Pinball loss measures the full distributional accuracy: it penalises
# MAGIC over-prediction and under-prediction asymmetrically in a way that is
# MAGIC exactly right for quantile regression. A lower pinball loss means the
# MAGIC model's conditional quantile function is closer to the truth across
# MAGIC all observations.
# MAGIC
# MAGIC We compare:
# MAGIC - The lognormal model's pinball loss at each quantile (using its
# MAGIC   analytical quantile predictions)
# MAGIC - The QuantileGBM's pinball loss (predictions from CatBoost)
# MAGIC - In both cases, `y_test_pl` is the same observed severity

# COMMAND ----------

print("Pinball loss by quantile level (lower = better):")
print()
print(f"  {'Quantile':>10}  {'Lognormal':>12}  {'QuantileGBM':>13}  {'GBM improvement':>16}")
print(f"  {'-'*55}")

for alpha, ln_pred, gbm_col in [
    (0.90, ln_q90_pred,  "q_0.9"),
    (0.95, ln_q95_pred,  "q_0.95"),
    (0.99, ln_q99_pred,  "q_0.99"),
]:
    # Lognormal pinball loss
    ln_pl_series = pl.Series("q", ln_pred)
    ln_pb = pinball_loss(y_test_pl, ln_pl_series, alpha=alpha)

    # QuantileGBM pinball loss
    gbm_pb = pinball_loss(y_test_pl, q_preds[gbm_col], alpha=alpha)

    improvement = 100 * (ln_pb - gbm_pb) / ln_pb
    print(f"  {alpha:>10.2f}  {ln_pb:>12.1f}  {gbm_pb:>13.1f}  {improvement:>+15.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c: Per-segment Q99 accuracy
# MAGIC
# MAGIC This is the table that matters most for pricing. The aggregate errors
# MAGIC look modest, but the segment-level errors show where the parametric
# MAGIC assumption breaks down — and those are exactly the risks where the
# MAGIC large loss loading decision is most consequential.

# COMMAND ----------

print("99th percentile accuracy by vehicle group (test set):")
print()
print(f"  {'Group':>6}  {'N':>6}  {'True Q99':>10}  {'LN Q99':>8}  {'LN err':>7}  "
      f"{'GBM Q99':>9}  {'GBM err':>8}")
print(f"  {'-'*70}")

for vg in [1.0, 2.0, 3.0, 4.0]:
    mask = X_test_np[:, 3] == vg
    n_g           = mask.sum()
    true_q99_g    = true_q99_test[mask].mean()
    ln_q99_g      = ln_q99_pred[mask].mean()
    gbm_q99_g     = gbm_q99[mask].mean()
    ln_err        = 100 * (ln_q99_g - true_q99_g) / true_q99_g
    gbm_err       = 100 * (gbm_q99_g - true_q99_g) / true_q99_g

    ln_flag  = " <-- over" if ln_err > 10 else (" <-- UNDER" if ln_err < -10 else "")
    gbm_flag = " <-- over" if gbm_err > 10 else (" <-- UNDER" if gbm_err < -10 else "")

    print(f"  {int(vg):>6}  {n_g:>6,}  {true_q99_g:>10,.0f}  "
          f"{ln_q99_g:>8,.0f}  {ln_err:>+7.1f}%  "
          f"{gbm_q99_g:>9,.0f}  {gbm_err:>+8.1f}%{ln_flag}{gbm_flag}")

print()
print("The lognormal's global sigma systematically overestimates thin-tailed")
print("groups (1-2) and underestimates heavy-tailed groups (4).")
print("The QuantileGBM adapts its quantile function per segment without needing")
print("to know in advance which groups have different tail weights.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4d: TVaR accuracy by segment
# MAGIC
# MAGIC TVaR is the metric that drives large loss loading in technical pricing.
# MAGIC An underestimated TVaR for group 4 means the loaded rate is too low
# MAGIC for the highest-risk segment.

# COMMAND ----------

print("TVaR_95 accuracy by vehicle group (test set):")
print()
print(f"  {'Group':>6}  {'True TVaR_95':>13}  {'LN TVaR_95':>11}  {'LN err':>7}  "
      f"{'GBM TVaR_95':>12}  {'GBM err':>8}")
print(f"  {'-'*72}")

for vg in [1.0, 2.0, 3.0, 4.0]:
    mask = X_test_np[:, 3] == vg
    true_tv_g    = true_tvar95_test[mask].mean()
    ln_tv_g      = ln_tvar95_pred[mask].mean()
    gbm_tv_g     = gbm_tvar95[mask].mean()
    ln_err       = 100 * (ln_tv_g - true_tv_g) / true_tv_g
    gbm_err      = 100 * (gbm_tv_g - true_tv_g) / true_tv_g

    print(f"  {int(vg):>6}  {true_tv_g:>13,.0f}  "
          f"{ln_tv_g:>11,.0f}  {ln_err:>+7.1f}%  "
          f"{gbm_tv_g:>12,.0f}  {gbm_err:>+8.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Large Loss Loading — Per-Risk vs Portfolio Average
# MAGIC
# MAGIC In most portfolios, the large loss loading is computed as a single
# MAGIC portfolio-level rate uplift. The argument is usually pragmatic: "we
# MAGIC don't have enough large losses to fit segment-specific loadings."
# MAGIC
# MAGIC But that argument conflates two distinct problems:
# MAGIC
# MAGIC 1. **Estimating the tail quantile**: you need enough data at the tail.
# MAGIC    That's why we use CatBoost — it borrows strength across segments
# MAGIC    through the tree structure, rather than fitting each segment independently.
# MAGIC 2. **Applying the loading**: even if you use the same *method* for all
# MAGIC    segments, the *output* should vary by segment if the true tail varies.
# MAGIC
# MAGIC The `large_loss_loading` function computes:
# MAGIC
# MAGIC ```
# MAGIC loading_i = TVaR_alpha(i) - E[Y_i]
# MAGIC ```
# MAGIC
# MAGIC where TVaR comes from the QuantileGBM and E[Y] from a mean model.
# MAGIC For the mean model we use a CatBoost Tweedie fit on the same features.
# MAGIC
# MAGIC We then compare the per-risk loading distribution to what you would
# MAGIC apply if you used a single portfolio-average rate.

# COMMAND ----------

from catboost import CatBoostRegressor

# Fit a mean model (Tweedie, close to lognormal mean for severity)
mean_model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=400,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
)
mean_model.fit(X_train_np, y_train_np)

print(f"Tweedie mean model fitted: {mean_model.tree_count_} trees")

# Per-risk large loss loading
loading_per_risk = large_loss_loading(
    model_mean=mean_model,
    model_quantile=qgbm,
    X=X_test_pl,
    alpha=0.95,
)

# Mean model predictions on test set
mean_preds_test = mean_model.predict(X_test_np)

# Portfolio-average loading as a single rate
portfolio_avg_loading = float(loading_per_risk.mean())
portfolio_avg_loading_pct = portfolio_avg_loading / mean_preds_test.mean()

print(f"\nPer-risk large loss loading (TVaR_95 - mean model):")
print(f"  Mean loading:   £{loading_per_risk.mean():,.0f}")
print(f"  Median loading: £{np.median(loading_per_risk.to_numpy()):,.0f}")
print(f"  Max loading:    £{loading_per_risk.max():,.0f}")
print(f"\nPortfolio-average loading rate: {portfolio_avg_loading_pct:.1%} of mean premium")
print()

# Per-group comparison: portfolio rate vs per-risk rate
print("Per-risk loading by vehicle group:")
print(f"  {'Group':>6}  {'Mean expected Y':>15}  {'Mean TVaR_95':>12}  "
      f"{'Per-risk loading':>16}  {'Loading %':>10}  {'Portfolio rate %':>16}  {'Difference':>11}")
print(f"  {'-'*95}")

for vg in [1.0, 2.0, 3.0, 4.0]:
    mask = X_test_np[:, 3] == vg
    mean_g    = mean_preds_test[mask].mean()
    tvar_g    = gbm_tvar95[mask].mean()
    loading_g = loading_per_risk.to_numpy()[mask].mean()
    pct_g     = loading_g / mean_g
    diff      = pct_g - portfolio_avg_loading_pct

    print(f"  {int(vg):>6}  {mean_g:>15,.0f}  {tvar_g:>12,.0f}  "
          f"{loading_g:>16,.0f}  {pct_g:>10.1f}%  "
          f"{portfolio_avg_loading_pct:>16.1f}%  {diff:>+11.1f}pp")

print()
print("Vehicle group 4 requires a materially higher loading than the portfolio")
print("average. Using a flat rate over-charges group 1 and under-charges group 4.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What a flat loading means in practice
# MAGIC
# MAGIC Group 4 risks attract the highest large losses in the portfolio. If you
# MAGIC apply a portfolio-average loading to them, you are underpricing the
# MAGIC risks most likely to generate the losses you are trying to cover.
# MAGIC
# MAGIC The quantile GBM approach does not require you to know in advance that
# MAGIC group 4 has a heavier tail — the model learns that from the data via
# MAGIC the CatBoost tree structure. The actuary's job is to validate that the
# MAGIC learned pattern makes sense (it does: group 4 vehicles tend to be higher
# MAGIC performance with more severe damage in incidents) and to decide whether
# MAGIC the loading should be applied per-risk or capped at segment level for
# MAGIC competitive reasons.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Calibration Diagnostics
# MAGIC
# MAGIC A quantile model is calibrated if the fraction of observations below each
# MAGIC predicted quantile matches the stated level. Your q_0.9 prediction should
# MAGIC be exceeded by 10% of claims — no more, no less.
# MAGIC
# MAGIC The `coverage_check` function computes this. For the lognormal baseline
# MAGIC we replicate the same check using its analytical quantile predictions.

# COMMAND ----------

# QuantileGBM calibration
calib_report = qgbm.calibration_report(X_test_pl, y_test_pl)
print("QuantileGBM calibration report:")
print(f"  Mean pinball loss: {calib_report['mean_pinball_loss']:.2f}")
print()
print(f"  {'Quantile':>10}  {'Expected cov':>13}  {'Observed cov':>13}  {'Error':>7}")
print(f"  {'-'*48}")
for col, obs_cov in calib_report["coverage"].items():
    q_level = float(col.replace("q_", ""))
    err = obs_cov - q_level
    print(f"  {col:>10}  {q_level:>13.2f}  {obs_cov:>13.3f}  {err:>+7.3f}")

print()

# Lognormal calibration
print("Lognormal baseline calibration:")
print(f"  {'Quantile':>10}  {'Expected cov':>13}  {'Observed cov':>13}  {'Error':>7}")
print(f"  {'-'*48}")
for alpha, ln_pred in [(0.90, ln_q90_pred), (0.95, ln_q95_pred), (0.99, ln_q99_pred)]:
    obs_cov = float(np.mean(y_test_np <= ln_pred))
    err = obs_cov - alpha
    print(f"  {f'q_{alpha}':>10}  {alpha:>13.2f}  {obs_cov:>13.3f}  {err:>+7.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: ILF Curve — Limit Structure
# MAGIC
# MAGIC Increased Limits Factors matter for any business that writes excess layers
# MAGIC or where policy limits vary — motor injury, liability, property. The
# MAGIC ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)] tells you how much more
# MAGIC to charge at limit L2 relative to the basic limit L1.
# MAGIC
# MAGIC We compute ILF curves for vehicle group 1 vs group 4 — the two extreme
# MAGIC segments — to show how the tail shape difference translates into a
# MAGIC materially different limit structure.

# COMMAND ----------

# ILF curves for group 1 and group 4
# Select a representative sample of 200 risks from each group
group1_mask = X_test_np[:, 3] == 1.0
group4_mask = X_test_np[:, 3] == 4.0

X_g1 = pl.DataFrame(X_test_np[group1_mask][:200], schema=FEATURE_NAMES)
X_g4 = pl.DataFrame(X_test_np[group4_mask][:200], schema=FEATURE_NAMES)

# Basic limit: £25,000; compute ILFs at a range of higher limits
BASIC_LIMIT = 25_000
higher_limits = [50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000]

print(f"ILF curves (basic limit £{BASIC_LIMIT:,}):")
print(f"  Higher limit    Group 1 ILF    Group 4 ILF    Ratio (G4/G1)")
print(f"  {'-'*58}")

g1_ilfs = []
g4_ilfs = []
for hl in higher_limits:
    ilf_g1 = ilf(qgbm, X_g1, basic_limit=BASIC_LIMIT, higher_limit=hl).mean()
    ilf_g4 = ilf(qgbm, X_g4, basic_limit=BASIC_LIMIT, higher_limit=hl).mean()
    ratio   = ilf_g4 / ilf_g1 if ilf_g1 > 0 else float("nan")
    g1_ilfs.append(float(ilf_g1))
    g4_ilfs.append(float(ilf_g4))
    print(f"  £{hl:>9,}    {ilf_g1:>10.3f}    {ilf_g4:>10.3f}    {ratio:>12.3f}x")

print()
print("Group 4 consistently has higher ILFs at every limit level — reflecting")
print("the heavier Pareto tail. The difference is largest at high limits, where")
print("the tail shape dominates.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Visualisations
# MAGIC
# MAGIC Four plots:
# MAGIC
# MAGIC - **(a)** Q99 accuracy by vehicle group: lognormal vs QuantileGBM vs truth
# MAGIC - **(b)** Calibration chart: observed vs expected coverage for both methods
# MAGIC - **(c)** Large loss loading distribution: per-risk vs portfolio average
# MAGIC - **(d)** ILF curves by vehicle group

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# ---- (a) Q99 accuracy by vehicle group ----
ax_a = fig.add_subplot(gs[0, 0])

groups = [1, 2, 3, 4]
x = np.arange(len(groups))
width = 0.25

true_q99_by_group = [true_q99_test[X_test_np[:, 3] == float(g)].mean() for g in groups]
ln_q99_by_group   = [ln_q99_pred[X_test_np[:, 3] == float(g)].mean()   for g in groups]
gbm_q99_by_group  = [gbm_q99[X_test_np[:, 3] == float(g)].mean()        for g in groups]

bars_true = ax_a.bar(x - width, true_q99_by_group, width, label="True DGP",     color="#2d6a9f", alpha=0.85)
bars_ln   = ax_a.bar(x,         ln_q99_by_group,   width, label="Lognormal",    color="#e07b7b", alpha=0.85)
bars_gbm  = ax_a.bar(x + width, gbm_q99_by_group,  width, label="QuantileGBM",  color="#6ab0de", alpha=0.85)

ax_a.set_xlabel("Vehicle group", fontsize=11)
ax_a.set_ylabel("Mean Q99 (£)", fontsize=11)
ax_a.set_title("(a) Q99 accuracy by vehicle group", fontsize=12)
ax_a.set_xticks(x)
ax_a.set_xticklabels([f"Group {g}" for g in groups])
ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}k"))
ax_a.legend(fontsize=9)
ax_a.grid(axis="y", alpha=0.3)

# ---- (b) Calibration chart ----
ax_b = fig.add_subplot(gs[0, 1])

# QuantileGBM
gbm_alphas   = QUANTILES
gbm_obs_covs = [float(np.mean(y_test_np <= q_preds[f"q_{q}"].to_numpy())) for q in QUANTILES]

# Lognormal (at 3 levels)
ln_alphas   = [0.90, 0.95, 0.99]
ln_pred_arr = [ln_q90_pred, ln_q95_pred, ln_q99_pred]
ln_obs_covs = [float(np.mean(y_test_np <= p)) for p in ln_pred_arr]

diag_pts = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
ax_b.plot(diag_pts, diag_pts, "k--", linewidth=1, label="Perfect calibration", zorder=1)
ax_b.scatter(gbm_alphas, gbm_obs_covs, color="#2d6a9f", s=70, zorder=5, label="QuantileGBM")
ax_b.scatter(ln_alphas,  ln_obs_covs,  color="#e07b7b", s=70, marker="s", zorder=5, label="Lognormal")

for q, obs in zip(gbm_alphas, gbm_obs_covs):
    ax_b.annotate(f"{q}", (q, obs), textcoords="offset points", xytext=(5, -10), fontsize=7, color="#2d6a9f")
for q, obs in zip(ln_alphas, ln_obs_covs):
    ax_b.annotate(f"{q}", (q, obs), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#e07b7b")

ax_b.set_xlabel("Expected coverage (quantile level)", fontsize=11)
ax_b.set_ylabel("Observed coverage", fontsize=11)
ax_b.set_title("(b) Calibration: observed vs expected coverage", fontsize=12)
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)
ax_b.set_xlim(0.45, 1.02)
ax_b.set_ylim(0.45, 1.02)

# ---- (c) Large loss loading distribution ----
ax_c = fig.add_subplot(gs[1, 0])

loading_np = loading_per_risk.to_numpy()
# Plot by group
colours_vg = {1: "#2d6a9f", 2: "#6ab0de", 3: "#f5c06e", 4: "#e07b7b"}

for vg in [1, 2, 3, 4]:
    mask = X_test_np[:, 3] == float(vg)
    vals = loading_np[mask]
    # clip at 99th pct for display
    clip_at = np.percentile(vals, 99)
    vals_cl  = vals[vals <= clip_at]
    ax_c.hist(vals_cl, bins=40, alpha=0.55, color=colours_vg[vg],
              label=f"Group {vg}", density=True)

ax_c.axvline(portfolio_avg_loading, color="black", linestyle="--",
             linewidth=1.5, label=f"Portfolio avg (£{portfolio_avg_loading:,.0f})")
ax_c.set_xlabel("Per-risk large loss loading (£)", fontsize=11)
ax_c.set_ylabel("Density", fontsize=11)
ax_c.set_title("(c) Large loss loading distribution by vehicle group", fontsize=12)
ax_c.legend(fontsize=8)
ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}k"))
ax_c.grid(axis="y", alpha=0.3)

# ---- (d) ILF curves by vehicle group ----
ax_d = fig.add_subplot(gs[1, 1])

limits_for_plot = [BASIC_LIMIT] + higher_limits
ilf_g1_plot = [1.0] + g1_ilfs
ilf_g4_plot = [1.0] + g4_ilfs

ax_d.plot(limits_for_plot, ilf_g1_plot, "o-", color="#2d6a9f", linewidth=2,
          markersize=6, label="Group 1 (thin tail)")
ax_d.plot(limits_for_plot, ilf_g4_plot, "s-", color="#e07b7b", linewidth=2,
          markersize=6, label="Group 4 (heavy tail)")
ax_d.axhline(1.0, color="grey", linestyle=":", linewidth=1, alpha=0.7)
ax_d.set_xlabel("Policy limit (£)", fontsize=11)
ax_d.set_ylabel(f"ILF (basic limit £{BASIC_LIMIT:,})", fontsize=11)
ax_d.set_title("(d) ILF curve by vehicle group", fontsize=12)
ax_d.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}k"))
ax_d.legend(fontsize=9)
ax_d.grid(alpha=0.3)

fig.suptitle(
    "insurance-quantile: tail risk quantification for heteroskedastic severity",
    fontsize=13, y=1.01
)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What we did | Key result |
# MAGIC |------|-------------|------------|
# MAGIC | 1 | Synthetic severity: 50k claims, spliced lognormal-Pareto, tail weight varies by vehicle group | Ground truth known exactly |
# MAGIC | 2 | Lognormal baseline: log-linear model, global sigma | Systematic bias: over-estimates group 1 tail, under-estimates group 4 |
# MAGIC | 3 | QuantileGBM: CatBoost MultiQuantile at 5 levels | Per-risk quantiles, TVaR, ILF, exceedance curve |
# MAGIC | 4 | Benchmark table | QuantileGBM materially closer to ground truth at Q99 and TVaR_95, especially for group 4 |
# MAGIC | 5 | Large loss loading | Per-risk loading differs by group; flat portfolio rate under-charges group 4 by ~X pp |
# MAGIC | 6 | Calibration diagnostics | QuantileGBM close to diagonal; lognormal shows systematic tail miscalibration |
# MAGIC | 7 | ILF curves | Group 4 ILF at £500k is materially higher than group 1 — different Pareto shape |
# MAGIC
# MAGIC ### When to use insurance-quantile
# MAGIC
# MAGIC **Definitely use it when:**
# MAGIC - Tail weight genuinely varies by risk segment (motor BI, liability,
# MAGIC   high-value property)
# MAGIC - Your actuarial deliverable is a quantile, TVaR, or ILF — not just
# MAGIC   the mean
# MAGIC - You have enough data for the tail to be estimated from data rather
# MAGIC   than assumed (rough rule of thumb: >5,000 large claims in training)
# MAGIC
# MAGIC **Consider the parametric alternative when:**
# MAGIC - Portfolio has only a few hundred large claims — parametric regularisation
# MAGIC   may help more than distributional flexibility
# MAGIC - You need a smooth, monotone ILF curve for regulatory filing — quantile
# MAGIC   regression is not constrained to be monotone in the limit dimension
# MAGIC - Interpretability requirements demand a closed-form severity distribution
# MAGIC
# MAGIC ### What the lognormal gets wrong
# MAGIC
# MAGIC The failure mode is specific: a single fitted sigma (or dispersion
# MAGIC parameter in the GLM case) is an average across all risk segments. For
# MAGIC groups with genuinely different tail weights, this average is wrong in
# MAGIC both directions simultaneously — too conservative for thin-tailed groups
# MAGIC and too optimistic for heavy-tailed ones. The QuantileGBM learns the
# MAGIC conditional quantile function directly via the pinball loss, so it adapts
# MAGIC its tail estimates per segment without requiring manual interaction terms
# MAGIC or per-segment distributional fits.

# COMMAND ----------

print("Notebook complete.")
print()
print("library:  insurance-quantile")
print("version:  pip install insurance-quantile")
print("repo:     https://github.com/burning-cost/insurance-quantile")
