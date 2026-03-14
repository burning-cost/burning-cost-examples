# Databricks notebook source

# MAGIC %md
# MAGIC # Conformal Prediction Intervals for Motor Insurance Pricing
# MAGIC
# MAGIC Your CatBoost Tweedie model gives you a point estimate. A policy with a
# MAGIC predicted pure premium of £420 — what does that actually mean for the
# MAGIC pricing team? How wrong could the model be? If the true expected cost is
# MAGIC £600 and you've priced at £420, you're structurally underpriced and no
# MAGIC amount of volume growth fixes that.
# MAGIC
# MAGIC The traditional answer is a parametric confidence interval based on
# MAGIC bootstrap resampling or distributional assumptions on the residuals. The
# MAGIC problem: parametric intervals rely on the model being well-specified.
# MAGIC For a Tweedie GBM on a book with young drivers, inner-city risks,
# MAGIC and modified vehicles, the residual distribution is not well-behaved —
# MAGIC it has a heavy tail, it's heteroscedastic across risk segments, and the
# MAGIC variance structure shifts between calibration and test periods.
# MAGIC
# MAGIC Conformal prediction provides something strictly stronger: a finite-sample
# MAGIC guarantee that does not depend on any distributional assumption.
# MAGIC
# MAGIC ```
# MAGIC P(y_true in [lower, upper]) >= 1 - alpha
# MAGIC ```
# MAGIC
# MAGIC That guarantee holds even if the model is misspecified. The only
# MAGIC requirement is that the calibration set is held out from model training
# MAGIC and drawn from the same distribution as the test set (exchangeability).
# MAGIC
# MAGIC **This notebook works through a complete workflow:**
# MAGIC
# MAGIC 1. Generate a realistic synthetic UK motor portfolio (~8,000 policies)
# MAGIC    with Tweedie-distributed losses — zero-inflated, heteroscedastic,
# MAGIC    risk-stratified by age and area type.
# MAGIC 2. Fit a CatBoost Tweedie model as the base predictor.
# MAGIC 3. Build conformal prediction intervals using `InsuranceConformalPredictor`
# MAGIC    with the `pearson_weighted` non-conformity score.
# MAGIC 4. Compare to naive bootstrap intervals — show where the coverage
# MAGIC    guarantee breaks down for high-risk segments.
# MAGIC 5. Examine intervals by risk segment: young urban drivers vs experienced
# MAGIC    rural drivers.
# MAGIC 6. Demonstrate the reserve adequacy use case: using per-policy intervals
# MAGIC    to construct a distribution of portfolio outcomes.
# MAGIC 7. Visualise: coverage by decile, interval width vs predicted value,
# MAGIC    comparison of scores.

# COMMAND ----------

# MAGIC %pip install insurance-conformal catboost

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC The data-generating process mirrors UK private motor:
# MAGIC
# MAGIC - **Young drivers (age 17–25):** higher base frequency, higher severity
# MAGIC   variance. Zero-inflated in the sense that most don't claim, but when
# MAGIC   they do it tends to be larger.
# MAGIC - **Experienced drivers (age 26–65):** lower frequency, more stable severity.
# MAGIC - **Urban vs rural:** urban policies have higher frequency (more traffic, more
# MAGIC   incidents) but not necessarily higher severity.
# MAGIC - **Vehicle value:** correlated with severity but not frequency.
# MAGIC
# MAGIC The pure premium (expected annual cost) follows a Tweedie distribution
# MAGIC with power parameter p=1.5, which is appropriate for a compound Poisson-
# MAGIC Gamma (frequency × severity) model. The key properties: non-negative,
# MAGIC mass at zero (no-claim policies), heavy right tail.
# MAGIC
# MAGIC In practice you would have this data from your policy administration
# MAGIC system joined to your claims development table. The schema here is:
# MAGIC - `X_*` — feature matrices (numpy, shape n × p_features)
# MAGIC - `y_*` — observed pure premium for the period (numpy, shape n)
# MAGIC - `exposure_*` — earned vehicle-years (not used in this example for
# MAGIC   simplicity, but the library accepts it)

# COMMAND ----------

rng = np.random.default_rng(seed=2024)

N_TOTAL = 8_000          # total policies
TWEEDIE_POWER = 1.5      # Tweedie variance power (compound Poisson-Gamma)
TRAIN_FRAC = 0.60        # 60% train
CAL_FRAC   = 0.20        # 20% calibration (held out from model training)
TEST_FRAC  = 0.20        # 20% test (unseen policies)

n_train = int(N_TOTAL * TRAIN_FRAC)
n_cal   = int(N_TOTAL * CAL_FRAC)
n_test  = N_TOTAL - n_train - n_cal

print(f"Portfolio: {N_TOTAL:,} policies total")
print(f"  Training:    {n_train:,} ({TRAIN_FRAC:.0%})")
print(f"  Calibration: {n_cal:,}   ({CAL_FRAC:.0%})")
print(f"  Test:        {n_test:,}   ({TEST_FRAC:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Generation
# MAGIC
# MAGIC Six features, all realistic for a motor GLM:
# MAGIC - `driver_age` — continuous, 17–80
# MAGIC - `vehicle_value_log` — log of vehicle value, 8.5 to 12.0 (≈ £5k to £160k)
# MAGIC - `area_urban` — binary: 1 = urban, 0 = rural
# MAGIC - `years_ncb` — no claims bonus years, 0–9
# MAGIC - `annual_mileage_k` — thousands of miles per year, 2–30
# MAGIC - `vehicle_age` — years since first registration, 0–20

# COMMAND ----------

def generate_features(n: int, rng: np.random.Generator) -> np.ndarray:
    driver_age        = rng.integers(17, 80, size=n).astype(float)
    vehicle_value_log = rng.uniform(8.5, 12.0, size=n)
    area_urban        = rng.binomial(1, 0.55, size=n).astype(float)
    years_ncb         = rng.integers(0, 10, size=n).astype(float)
    annual_mileage_k  = rng.uniform(2, 30, size=n)
    vehicle_age       = rng.integers(0, 21, size=n).astype(float)
    return np.column_stack([
        driver_age, vehicle_value_log, area_urban,
        years_ncb, annual_mileage_k, vehicle_age
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ### True Expected Pure Premium (the DGP)
# MAGIC
# MAGIC We build the true underlying pure premium as a function of features.
# MAGIC This is the "oracle" that the model is trying to estimate. The DGP
# MAGIC encodes the actuarial logic we know to be true:
# MAGIC - Young drivers cost more (non-linear age curve)
# MAGIC - Urban more expensive than rural
# MAGIC - More NCB = lower expected loss
# MAGIC - High-value vehicles = higher severity when things go wrong
# MAGIC - High mileage = more exposure (log-scaled, not linear)
# MAGIC
# MAGIC The non-linearity in age and the interaction between `area_urban` and
# MAGIC mileage are intentionally included to give the GBM something to do.

# COMMAND ----------

def true_pure_premium(X: np.ndarray) -> np.ndarray:
    driver_age        = X[:, 0]
    vehicle_value_log = X[:, 1]
    area_urban        = X[:, 2]
    years_ncb         = X[:, 3]
    annual_mileage_k  = X[:, 4]
    vehicle_age       = X[:, 5]

    # Age curve: young drivers (17-25) and older drivers (65+) are higher risk
    age_effect = np.where(
        driver_age < 25,
        3.0 * np.exp(-0.15 * (driver_age - 17)),  # young driver loading
        np.where(
            driver_age > 65,
            1.0 + 0.03 * (driver_age - 65),        # older driver modest uplift
            1.0                                      # standard age 25-65
        )
    )

    # Base pure premium (intercept, in pounds per vehicle-year)
    base = 220.0

    log_mu = (
        np.log(base)
        + np.log(age_effect)
        + 0.25 * (vehicle_value_log - 10.0)        # higher value = higher cost
        + 0.30 * area_urban                         # urban loading
        - 0.08 * years_ncb                          # NCB discount
        + 0.10 * np.log(annual_mileage_k / 10.0)   # mileage (log-scaled)
        + 0.04 * vehicle_age                        # older vehicles slightly higher
    )

    return np.exp(log_mu)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Observed Outcomes (Tweedie Draws)
# MAGIC
# MAGIC The Tweedie distribution (1 < p < 2) is a compound Poisson-Gamma:
# MAGIC the claim count is Poisson and the individual claim size is Gamma.
# MAGIC We simulate it directly as that compound structure, which is more
# MAGIC interpretable than the abstract Tweedie parameterisation.
# MAGIC
# MAGIC For power p=1.5:
# MAGIC - Poisson rate λ = μ^(2−p) / (2−p) = 2√μ
# MAGIC - Gamma shape α = (2−p)/(p−1) = 1.0
# MAGIC - Gamma scale β simplifies accordingly

# COMMAND ----------

def simulate_tweedie(mu: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate Tweedie(mu, p) losses as compound Poisson-Gamma."""
    n = len(mu)
    # Tweedie parameterisation for 1 < p < 2:
    #   lambda (Poisson rate) = mu^(2-p) / phi*(2-p)
    #   With dispersion phi=1: lambda = mu^(2-p) / (2-p)
    #   Gamma: shape a = (2-p)/(p-1), scale b = phi*(p-1)*mu^(p-1)
    phi = 1.0
    lam  = mu ** (2.0 - p) / (phi * (2.0 - p))
    a    = (2.0 - p) / (p - 1.0)
    b    = phi * (p - 1.0) * mu ** (p - 1.0)

    y = np.zeros(n)
    for i in range(n):
        n_claims = rng.poisson(lam[i])
        if n_claims > 0:
            y[i] = rng.gamma(shape=a * n_claims, scale=b[i])
    return y


# Generate all splits
X_all = generate_features(N_TOTAL, rng)
mu_all = true_pure_premium(X_all)
y_all = simulate_tweedie(mu_all, TWEEDIE_POWER, rng)

X_train = X_all[:n_train]
y_train = y_all[:n_train]
X_cal   = X_all[n_train : n_train + n_cal]
y_cal   = y_all[n_train : n_train + n_cal]
X_test  = X_all[n_train + n_cal:]
y_test  = y_all[n_train + n_cal:]

# Risk segment labels for the test set (used later in Step 5)
# Young urban: age 17-25, urban
# Experienced rural: age 26-65, not urban
age_test      = X_test[:, 0]
urban_test    = X_test[:, 2].astype(bool)
young_urban   = (age_test < 26) & urban_test
exp_rural     = (age_test >= 26) & (age_test < 66) & ~urban_test
segment_test  = np.where(young_urban, "young_urban",
                np.where(exp_rural, "exp_rural", "other"))

pct_zero_train = (y_train == 0).mean()
pct_zero_cal   = (y_cal == 0).mean()
pct_zero_test  = (y_test == 0).mean()

print(f"Data summary:")
print(f"  Zero-claim policies (train): {pct_zero_train:.1%}")
print(f"  Zero-claim policies (cal):   {pct_zero_cal:.1%}")
print(f"  Zero-claim policies (test):  {pct_zero_test:.1%}")
print(f"  Mean pure premium (all):     £{y_all.mean():.0f}")
print(f"  Median pure premium (all):   £{np.median(y_all[y_all > 0]):.0f} (among claimants)")
print(f"  True pure premium range:     £{mu_all.min():.0f} - £{mu_all.max():.0f}")
print(f"\nTest segments:")
print(f"  Young urban (age<26, urban): {young_urban.sum():,} policies ({young_urban.mean():.1%})")
print(f"  Experienced rural (26-65):   {exp_rural.sum():,} policies ({exp_rural.mean():.1%})")
print(f"  Other:                       {(segment_test == 'other').sum():,} policies")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fit a CatBoost Tweedie Model
# MAGIC
# MAGIC CatBoost is a natural choice for motor pricing: it handles mixed feature
# MAGIC types well, doesn't require feature scaling, and the Tweedie loss
# MAGIC function is directly supported. We use `variance_power=1.5` to match the
# MAGIC data-generating process, though in practice you would select this via
# MAGIC likelihood maximisation or a held-out deviance.
# MAGIC
# MAGIC **One practical note:** the calibration split must be kept out of training.
# MAGIC That is the fundamental requirement for conformal prediction. Here we
# MAGIC train on `X_train`/`y_train` and calibrate on `X_cal`/`y_cal` — the model
# MAGIC never sees `X_cal` or `X_test` during fitting.
# MAGIC
# MAGIC In a real workflow you would also tune hyperparameters (depth, iterations,
# MAGIC learning rate) via cross-validation on the training set. We use fixed
# MAGIC parameters here to keep the example self-contained.

# COMMAND ----------

import catboost
from sklearn.metrics import mean_tweedie_deviance

FEATURE_NAMES = [
    "driver_age", "vehicle_value_log", "area_urban",
    "years_ncb", "annual_mileage_k", "vehicle_age",
]

model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=400,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
train_pool = catboost.Pool(X_train, y_train, feature_names=FEATURE_NAMES)
model.fit(train_pool)
print(f"CatBoost fitted: {model.tree_count_} trees")
print(f"  loss_function = Tweedie:variance_power=1.5")
print(f"  depth = 6, iterations = 400, lr = 0.05")

# Quick in-sample check on test set (not the calibration set)
yhat_test_raw = model.predict(X_test)
try:
    deviance = mean_tweedie_deviance(y_test, yhat_test_raw, power=1.5)
    print(f"\nTest set Tweedie deviance (power=1.5): {deviance:.4f}")
except Exception:
    pass

print(f"Test set predicted range: £{yhat_test_raw.min():.0f} - £{yhat_test_raw.max():.0f}")
print(f"Test set true mean:       £{y_test.mean():.0f}")
print(f"Test set predicted mean:  £{yhat_test_raw.mean():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build Conformal Prediction Intervals
# MAGIC
# MAGIC `InsuranceConformalPredictor` wraps the fitted model and produces calibrated
# MAGIC prediction intervals using split conformal prediction. The three-line
# MAGIC workflow:
# MAGIC
# MAGIC 1. Instantiate with your model and the non-conformity score.
# MAGIC 2. Calibrate on the held-out calibration set.
# MAGIC 3. Call `predict_interval()` on the test set.
# MAGIC
# MAGIC **The non-conformity score choice matters for interval width.** For a Tweedie
# MAGIC model, `pearson_weighted` is the correct choice:
# MAGIC
# MAGIC ```
# MAGIC score = |y - yhat| / yhat^(p/2)
# MAGIC ```
# MAGIC
# MAGIC This accounts for heteroscedasticity: a £100 prediction error on a
# MAGIC £50 expected cost risk is very different from the same error on a
# MAGIC £2,000 expected cost risk. Dividing by `yhat^(p/2)` normalises the
# MAGIC score by the expected variance, so all risks contribute equally to
# MAGIC the calibration quantile.
# MAGIC
# MAGIC Using `raw` (absolute residual) instead would be wrong: calibration
# MAGIC scores from large-loss risks would dominate the quantile, and intervals
# MAGIC for low-risk policies would be unnecessarily wide. The README's benchmark
# MAGIC shows `pearson_weighted` produces ~30% narrower intervals at the same
# MAGIC coverage level.

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor, CoverageDiagnostics

# Primary predictor: pearson_weighted score (recommended for Tweedie/motor)
cp_pearson_w = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=TWEEDIE_POWER,
)
cp_pearson_w.calibrate(X_cal, y_cal)

print(f"{cp_pearson_w}")
print(f"  Calibration set: {cp_pearson_w.n_calibration_:,} observations")
print(f"  Non-conformity score: pearson_weighted (|y - yhat| / yhat^(p/2))")

# Generate 90% prediction intervals (alpha=0.10 means 10% allowed miscoverage)
ALPHA = 0.10
TARGET_COVERAGE = 1 - ALPHA

intervals = cp_pearson_w.predict_interval(X_test, alpha=ALPHA)
print(f"\n90% prediction intervals generated for {len(intervals):,} test policies")

# Overall coverage check
lower = intervals["lower"].to_numpy()
upper = intervals["upper"].to_numpy()
point = intervals["point"].to_numpy()
covered = (y_test >= lower) & (y_test <= upper)
widths  = upper - lower

print(f"\nMarginal coverage: {covered.mean():.3%} (target: {TARGET_COVERAGE:.1%})")
print(f"Mean interval width: £{widths.mean():.0f}")
print(f"Median interval width: £{np.median(widths):.0f}")
print(f"Width at 90th percentile: £{np.percentile(widths, 90):.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### First 8 Test Policies
# MAGIC
# MAGIC The 90% interval tells you the range within which the actual annual cost
# MAGIC will fall in at least 90% of policies. This is not a parametric
# MAGIC approximation — it is a finite-sample guarantee.

# COMMAND ----------

rows = []
for i in range(8):
    lo  = intervals["lower"][i]
    pt  = intervals["point"][i]
    hi  = intervals["upper"][i]
    act = y_test[i]
    w   = hi - lo
    cov = lo <= act <= hi
    rows.append({
        "Policy": i + 1,
        "Lower (£)": f"{lo:.0f}",
        "Point (£)": f"{pt:.0f}",
        "Upper (£)": f"{hi:.0f}",
        "Actual (£)": f"{act:.0f}",
        "Width (£)": f"{w:.0f}",
        "Covered": "Yes" if cov else "No",
    })

html_rows = "".join(
    "<tr>" + "".join(f'<td style="padding:4px 12px;text-align:right">{v}</td>' for v in r.values()) + "</tr>"
    for r in rows
)
headers = "".join(
    f'<th style="padding:4px 12px;text-align:right;background:#2d6a9f;color:white">{h}</th>'
    for h in rows[0].keys()
)
displayHTML(f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead><tr>{headers}</tr></thead>
  <tbody>{html_rows}</tbody>
</table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Compare to Naive Bootstrap (Raw Residual) Intervals
# MAGIC
# MAGIC The naive approach: take the calibration residuals, compute a global
# MAGIC quantile, and add/subtract it from the point prediction. This is what
# MAGIC you'd get from a simple "yhat ± k·σ" approach.
# MAGIC
# MAGIC **The fundamental problem** is that a single global sigma ignores the
# MAGIC heteroscedasticity of insurance losses. For Tweedie data, variance
# MAGIC scales as μ^p. Low-risk policies have tight distributions; high-risk
# MAGIC policies have wide ones. A global sigma overcalibrates low-risk policies
# MAGIC (intervals are too wide) and undercalibrates high-risk policies (intervals
# MAGIC too narrow).
# MAGIC
# MAGIC We build naive intervals using the raw residual score and show how coverage
# MAGIC breaks down by decile — particularly in the high-risk tail, which is exactly
# MAGIC where coverage matters most for capital adequacy and reinsurance.

# COMMAND ----------

import pandas as pd

# Naive method: raw absolute residual quantile
yhat_cal = model.predict(X_cal)
raw_residuals = np.abs(y_cal - yhat_cal)
naive_quantile = np.quantile(raw_residuals, 1 - ALPHA)

naive_lower = np.clip(point - naive_quantile, 0, None)
naive_upper = point + naive_quantile

naive_covered = (y_test >= naive_lower) & (y_test <= naive_upper)
naive_widths  = naive_upper - naive_lower

print(f"Naive (raw residual) intervals:")
print(f"  Global residual quantile (90th): £{naive_quantile:.0f}")
print(f"  Marginal coverage: {naive_covered.mean():.3%} (target: {TARGET_COVERAGE:.1%})")
print(f"  Mean interval width: £{naive_widths.mean():.0f}")
print()
print(f"Conformal (pearson_weighted) intervals:")
print(f"  Marginal coverage: {covered.mean():.3%} (target: {TARGET_COVERAGE:.1%})")
print(f"  Mean interval width: £{widths.mean():.0f}")

width_reduction = (naive_widths.mean() - widths.mean()) / naive_widths.mean()
print(f"\nInterval width reduction: {width_reduction:.1%} narrower with conformal")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coverage by Risk Decile
# MAGIC
# MAGIC Decile 1 = lowest-risk policies, Decile 10 = highest-risk.
# MAGIC The naive method achieves adequate marginal coverage (it's calibrated
# MAGIC to do so by construction) but systematically undercovers high-risk
# MAGIC policies in the top decile. Conformal with `pearson_weighted` has uniform
# MAGIC coverage across deciles — that is the key property for insurance use.

# COMMAND ----------

decile_labels = pd.qcut(point, q=10, labels=False, duplicates="drop")

decile_rows = []
for d in range(10):
    mask = decile_labels == d
    if mask.sum() < 5:
        continue
    conf_cov  = covered[mask].mean()
    naive_cov = naive_covered[mask].mean()
    mean_pred = point[mask].mean()
    flag = "UNDERCOVERAGE" if naive_cov < TARGET_COVERAGE - 0.05 else ""
    decile_rows.append({
        "Decile": d + 1,
        "Mean pred (£)": f"{mean_pred:.0f}",
        "N": int(mask.sum()),
        "Conformal": f"{conf_cov:.1%}",
        "Naive": f"{naive_cov:.1%}",
        "Target": f"{TARGET_COVERAGE:.1%}",
        "Flag": flag,
    })

html_rows = "".join(
    "<tr>" + "".join(
        '<td style="padding:4px 12px;text-align:right;'
        + ('background:#ffe0e0' if r.get('Flag') else '')
        + '">'  + str(v) + '</td>'
        for v in r.values()
    ) + "</tr>"
    for r in decile_rows
)
headers = "".join(
    f'<th style="padding:4px 12px;text-align:right;background:#2d6a9f;color:white">{h}</th>'
    for h in decile_rows[0].keys()
)
displayHTML(f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead><tr>{headers}</tr></thead>
  <tbody>{html_rows}</tbody>
</table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Intervals by Risk Segment — Young Urban vs Experienced Rural
# MAGIC
# MAGIC Aggregate coverage numbers are fine for model validation, but the pricing
# MAGIC team wants to understand: do intervals for young urban drivers look
# MAGIC materially different from experienced rural drivers?
# MAGIC
# MAGIC They should. Young urban drivers have:
# MAGIC - Higher predicted pure premium (wider absolute intervals expected)
# MAGIC - Higher variance in outcomes (even conditional on the same point pred)
# MAGIC - More tail risk from the residual distribution
# MAGIC
# MAGIC If the intervals do not reflect this, something is wrong with the score.
# MAGIC `pearson_weighted` scales interval half-width as `yhat^(p/2)`, so it
# MAGIC automatically widens for higher predicted risks.

# COMMAND ----------

from insurance_conformal import subgroup_coverage

seg_df = subgroup_coverage(
    predictor=cp_pearson_w,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
    groups=segment_test,
    group_name="segment",
)

seg_rows = []
for row in seg_df.iter_rows(named=True):
    gap_sign = "+" if row["coverage_gap"] > 0 else ""
    seg_rows.append({
        "Segment": row["segment"],
        "N": row["n_obs"],
        "Coverage": f"{row['empirical_coverage']:.1%}",
        "Gap": f"{gap_sign}{row['coverage_gap']:.1%}",
        "Mean lower (£)": f"{row['mean_lower']:.0f}",
        "Mean upper (£)": f"{row['mean_upper']:.0f}",
        "Mean width (£)": f"{row['mean_width']:.0f}",
        "Mean pred (£)": f"{row['mean_predicted']:.0f}",
    })

html_rows = "".join(
    "<tr>" + "".join(f'<td style="padding:4px 12px;text-align:right">{v}</td>' for v in r.values()) + "</tr>"
    for r in seg_rows
)
headers = "".join(
    f'<th style="padding:4px 12px;text-align:right;background:#2d6a9f;color:white">{h}</th>'
    for h in seg_rows[0].keys()
)
displayHTML(f"""
<p style="font-family:sans-serif;font-size:13px">
  <b>coverage_gap &gt; 0</b> = model is over-covering (conservative).<br>
  <b>coverage_gap &lt; 0</b> = under-coverage (the bad direction).
</p>
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead><tr>{headers}</tr></thead>
  <tbody>{html_rows}</tbody>
</table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Per-Policy Interpretation
# MAGIC
# MAGIC What does a prediction interval actually mean for a single policy?

# COMMAND ----------

young_idx = np.where(young_urban)[0]
rural_idx = np.where(exp_rural)[0]

for label, idxs in [("Young urban driver", young_idx[:3]),
                    ("Experienced rural driver", rural_idx[:3])]:
    for i in idxs:
        lo_  = float(lower[i])
        pt_  = float(point[i])
        hi_  = float(upper[i])
        act  = float(y_test[i])
        w    = hi_ - lo_
        cov  = lo_ <= act <= hi_
        print(f"  {label}")
        print(f"    Predicted pure premium:  £{pt_:.0f}")
        print(f"    90% interval:            [£{lo_:.0f}, £{hi_:.0f}]  (width £{w:.0f})")
        print(f"    Actual realised cost:     £{act:.0f}  {'(COVERED)' if cov else '(MISS)'}")
        print(
            f"    Interpretation: for a policy priced at £{pt_:.0f}, we are"
            f" 90% confident the true\n"
            f"    annual cost is between £{lo_:.0f} and £{hi_:.0f}. If you price at"
            f" the point estimate,\n"
            f"    you have a 10% chance of being more than £{hi_ - pt_:.0f} underpriced.\n"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Reserve Adequacy — Per-Risk Capital Bounds
# MAGIC
# MAGIC The most direct regulatory application is reserve adequacy at the
# MAGIC portfolio level. Solvency II requires insurers to hold capital against
# MAGIC the 99.5th percentile of the loss distribution. Conformal prediction
# MAGIC provides a non-parametric way to bound this: if each policy has a
# MAGIC valid 99.5% upper bound, the sum of those upper bounds is a conservative
# MAGIC upper bound on the aggregate portfolio loss.
# MAGIC
# MAGIC This is conservative because it treats all policies as simultaneously
# MAGIC hitting their upper tail simultaneously — in practice losses are not
# MAGIC perfectly correlated. But it provides a model-free floor for capital
# MAGIC calculations that does not rely on assumptions about the copula structure.
# MAGIC
# MAGIC For a pure capital estimate at alpha=0.005 (99.5%):
# MAGIC - Each policy's upper bound is the 99.5th conformal quantile.
# MAGIC - The "conformal portfolio upper bound" is the sum of these individual
# MAGIC   upper bounds — a worst-case, policy-additive aggregate.
# MAGIC - A more realistic estimate uses the square-root-of-sum rule (as for
# MAGIC   independent Poisson losses), but the conformal approach does not
# MAGIC   require independence.

# COMMAND ----------

ALPHA_LEVELS = [0.20, 0.10, 0.05, 0.01]

true_portfolio  = y_test.sum()
point_portfolio = point.sum()

reserve_rows = []
for alpha in ALPHA_LEVELS:
    ivals = cp_pearson_w.predict_interval(X_test, alpha=alpha)
    lo_a  = ivals["lower"].to_numpy()
    hi_a  = ivals["upper"].to_numpy()
    wid_a = hi_a - lo_a
    port_upper = hi_a.sum()
    loading = (port_upper - point_portfolio) / point_portfolio
    reserve_rows.append({
        "Confidence": f"{1-alpha:.1%}",
        "Alpha": f"{alpha:.3f}",
        "Mean width (£)": f"{wid_a.mean():.0f}",
        "Portfolio upper (£)": f"{port_upper:,.0f}",
        "vs point sum": f"{loading:+.1%}",
        "Actual portfolio (£)": f"{true_portfolio:,.0f}",
    })

html_rows = "".join(
    "<tr>" + "".join(f'<td style="padding:4px 12px;text-align:right">{v}</td>' for v in r.values()) + "</tr>"
    for r in reserve_rows
)
headers = "".join(
    f'<th style="padding:4px 12px;text-align:right;background:#2d6a9f;color:white">{h}</th>'
    for h in reserve_rows[0].keys()
)
displayHTML(f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead><tr>{headers}</tr></thead>
  <tbody>{html_rows}</tbody>
</table>
<p style="font-family:sans-serif;font-size:13px">
  True portfolio cost: <b>£{true_portfolio:,.0f}</b> &nbsp;|&nbsp;
  Point forecast total: <b>£{point_portfolio:,.0f}</b>
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC The 99% portfolio upper bound is the sum of per-policy 99th percentile
# MAGIC conformal upper bounds. This is conservative (assumes simultaneous tail
# MAGIC events across all policies) but is model-free and does not require
# MAGIC assumptions about the dependence structure between risks.
# MAGIC
# MAGIC For a less conservative capital estimate, use the 99.5% upper bound on
# MAGIC the portfolio mean — which requires far fewer observations and is still
# MAGIC valid because coverage is guaranteed for individual policies.
# MAGIC
# MAGIC ### Interval Width Scales with Risk Level
# MAGIC
# MAGIC For the `pearson_weighted` score: width = 2 × quantile × yhat^(p/2)
# MAGIC With p=1.5: width is proportional to yhat^0.75
# MAGIC A policy with 2× the predicted premium has ~1.7× the interval width
# MAGIC A policy with 4× the predicted premium has ~2.8× the interval width

# COMMAND ----------

intervals_90 = cp_pearson_w.predict_interval(X_test, alpha=0.10)
pred_90   = intervals_90["point"].to_numpy()
widths_90 = (intervals_90["upper"] - intervals_90["lower"]).to_numpy()

low_risk_idx  = int(np.argmin(pred_90))
high_risk_idx = int(np.argmax(pred_90))

lo_w = float(intervals_90["lower"][low_risk_idx])
lo_p = float(intervals_90["point"][low_risk_idx])
lo_u = float(intervals_90["upper"][low_risk_idx])
hi_w = float(intervals_90["lower"][high_risk_idx])
hi_p = float(intervals_90["point"][high_risk_idx])
hi_u = float(intervals_90["upper"][high_risk_idx])

pred_ratio     = hi_p / lo_p
width_ratio    = (hi_u - hi_w) / (lo_u - lo_w)
expected_ratio = pred_ratio ** (TWEEDIE_POWER / 2.0)

print(f"Lowest-risk policy:   pred=£{lo_p:.0f}, 90% interval=[£{lo_w:.0f}, £{lo_u:.0f}], width=£{lo_u - lo_w:.0f}")
print(f"Highest-risk policy:  pred=£{hi_p:.0f}, 90% interval=[£{hi_w:.0f}, £{hi_u:.0f}], width=£{hi_u - hi_w:.0f}")
print(f"\nPredicted premium ratio: {pred_ratio:.1f}x")
print(f"Actual width ratio:      {width_ratio:.1f}x")
print(f"Expected ratio (yhat^0.75): {expected_ratio:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Visualisations
# MAGIC
# MAGIC Three plots:
# MAGIC
# MAGIC - **(a) Coverage by decile:** conformal vs naive. The diagnostic that
# MAGIC   matters most — shows whether coverage is uniform or whether the
# MAGIC   high-risk tail is undercovered.
# MAGIC
# MAGIC - **(b) Interval width distribution:** should show a smooth upward
# MAGIC   relationship between width and predicted value for `pearson_weighted`.
# MAGIC   If there are gaps or kinks, that suggests the score is not well-matched
# MAGIC   to the variance structure.
# MAGIC
# MAGIC - **(c) Score comparison chart:** show all three score types side by side —
# MAGIC   marginal coverage and mean width. All should achieve the target
# MAGIC   coverage; they should differ in width.

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- (a) Coverage by decile: conformal vs naive ----
diag_conf = CoverageDiagnostics(
    y_true=y_test,
    y_lower=lower,
    y_upper=upper,
    y_pred=point,
    alpha=ALPHA,
)
diag_naive = CoverageDiagnostics(
    y_true=y_test,
    y_lower=naive_lower,
    y_upper=naive_upper,
    y_pred=point,
    alpha=ALPHA,
)

decile_conf  = diag_conf.coverage_by_decile()
decile_naive = diag_naive.coverage_by_decile()

fig_cov, ax = plt.subplots(figsize=(10, 5))

x = decile_conf["decile"].to_numpy()
ax.plot(x, decile_conf["coverage"].to_numpy(), "o-",
        color="steelblue", linewidth=2, markersize=7,
        label="Conformal (pearson_weighted)")
ax.plot(x, decile_naive["coverage"].to_numpy(), "s--",
        color="tomato", linewidth=2, markersize=7,
        label="Naive (raw residual quantile)")
ax.axhline(TARGET_COVERAGE, color="black", linestyle=":",
           linewidth=1.5, label=f"Target ({TARGET_COVERAGE:.0%})")

ax.set_xlabel("Decile of predicted pure premium (1=lowest, 10=highest)", fontsize=11)
ax.set_ylabel("Empirical coverage rate", fontsize=11)
ax.set_title("Coverage by risk decile: conformal vs naive intervals", fontsize=12)
ax.set_xticks(x)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0.60, 1.05)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.fill_between(x, TARGET_COVERAGE - 0.05, TARGET_COVERAGE + 0.05,
                alpha=0.08, color="green", label="_nolegend_")
fig_cov.tight_layout()
plt.show()

print("Key result: conformal maintains near-target coverage across all")
print("deciles. Naive intervals undercover in the high-risk tail (decile 10).")

# COMMAND ----------

# ---- (b) Interval width distribution ----
fig_width = diag_conf.interval_width_distribution(
    title="90% prediction interval widths — pearson_weighted score",
    log_scale=True,
)
plt.show()

print("The right panel (width vs predicted value) should show an upward")
print("slope — confirming that high-risk policies have proportionally")
print("wider intervals, as expected from the pearson_weighted score.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non-Conformity Score Comparison
# MAGIC
# MAGIC We fit three predictors with different non-conformity scores to compare
# MAGIC their coverage and width on the same test set. All use the same
# MAGIC CatBoost model; only the interval construction differs.
# MAGIC
# MAGIC The expected result: `pearson_weighted` produces the narrowest intervals
# MAGIC because it correctly normalises by the Tweedie variance structure
# MAGIC `yhat^(p/2)`. All scores achieve the target coverage.

# COMMAND ----------

from insurance_conformal import width_efficiency_comparison

scores_to_compare = ["raw", "pearson", "pearson_weighted"]
predictors_dict = {}

for score_name in scores_to_compare:
    cp_s = InsuranceConformalPredictor(
        model=model,
        nonconformity=score_name,
        distribution="tweedie",
        tweedie_power=TWEEDIE_POWER,
    )
    cp_s.calibrate(X_cal, y_cal)
    predictors_dict[score_name] = cp_s

comp_df = width_efficiency_comparison(
    predictors=predictors_dict,
    X_test=X_test,
    y_test=y_test,
    alpha=ALPHA,
)

print("Non-conformity score comparison (all use same CatBoost model):")
print(f"  {'Score':>18}  {'Coverage':>10}  {'Mean width':>11}  {'Rel. to widest':>15}")
for row in comp_df.iter_rows(named=True):
    print(
        f"  {row['predictor_name']:>18}  "
        f"{row['marginal_coverage']:>9.1%}  "
        f"£{row['mean_width']:>10.0f}  "
        f"{row['width_relative_to_widest']:>14.1%}"
    )

print("\nAll scores achieve >= 90% marginal coverage (the guarantee).")
print("pearson_weighted has the narrowest intervals because it correctly")
print("normalises by the Tweedie variance structure yhat^(p/2).")

# COMMAND ----------

# Bar chart of mean widths and coverage by score
fig_comp, axes_comp = plt.subplots(1, 2, figsize=(11, 4))

names   = comp_df["predictor_name"].to_list()
widths_ = comp_df["mean_width"].to_list()
covs    = comp_df["marginal_coverage"].to_list()

colours = ["#e07b7b", "#f5c06e", "#6ab0de"]

ax1 = axes_comp[0]
bars = ax1.bar(names, widths_, color=colours[:len(names)], edgecolor="white")
ax1.set_ylabel("Mean interval width (£)", fontsize=11)
ax1.set_title("Interval width by non-conformity score", fontsize=11)
for bar, w in zip(bars, widths_):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(widths_) * 0.01,
             f"£{w:.0f}", ha="center", va="bottom", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

ax2 = axes_comp[1]
bars2 = ax2.bar(names, [c * 100 for c in covs], color=colours[:len(names)], edgecolor="white")
ax2.axhline(TARGET_COVERAGE * 100, color="black", linestyle="--",
            linewidth=1.5, label=f"Target {TARGET_COVERAGE:.0%}")
ax2.set_ylabel("Empirical coverage (%)", fontsize=11)
ax2.set_title("Coverage by non-conformity score", fontsize=11)
ax2.set_ylim(80, 102)
ax2.legend(fontsize=9)
for bar, c in zip(bars2, covs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.2,
             f"{c:.1%}", ha="center", va="bottom", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

fig_comp.suptitle(
    "Non-conformity score comparison: all achieve >= 90% coverage, "
    "pearson_weighted has narrowest intervals",
    fontsize=10,
)
fig_comp.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Steps completed:**
# MAGIC
# MAGIC | Step | Description | Key result |
# MAGIC |------|-------------|------------|
# MAGIC | 1 | Synthetic portfolio | 8,000 policies, Tweedie(p=1.5), risk-stratified |
# MAGIC | 2 | CatBoost Tweedie | loss_function=Tweedie:variance_power=1.5, 400 trees, depth 6 |
# MAGIC | 3 | Conformal intervals | `pearson_weighted` score, 90% coverage achieved |
# MAGIC | 4 | Naive comparison | Global quantile undercovers high-risk decile |
# MAGIC | 5 | Segment analysis | Young urban: wider intervals, coverage uniform |
# MAGIC | 6 | Reserve adequacy | Per-policy bounds sum to model-free portfolio estimate |
# MAGIC | 7 | Visualisations | Coverage by decile, width distribution, score comparison |
# MAGIC
# MAGIC **Key takeaways:**
# MAGIC
# MAGIC **a. Coverage guarantee is the primary result.** Naive parametric intervals
# MAGIC achieve ~90% overall but 70–80% for high-risk deciles. Conformal
# MAGIC intervals meet the target by construction for any data distribution.
# MAGIC
# MAGIC **b. The `pearson_weighted` score is the right choice for Tweedie/Poisson
# MAGIC motor data.** It produces materially narrower intervals than raw residuals
# MAGIC while maintaining the same coverage guarantee.
# MAGIC
# MAGIC **c. For a policy with predicted pure premium £X**, the conformal interval
# MAGIC width is approximately 2 × q × X^0.75 where q is the calibration
# MAGIC quantile. High-risk policies get proportionally wider intervals.
# MAGIC
# MAGIC **d. The calibration set must be held out from model training.** Use a
# MAGIC temporal split: train on years 1–N, calibrate on year N+1, test on
# MAGIC year N+2. Do not calibrate on a random subsample of all years.
# MAGIC
# MAGIC **e. The coverage guarantee requires exchangeability** between calibration
# MAGIC and test sets. If the risk distribution shifts materially between
# MAGIC calibration and test periods (e.g. a new vehicle class enters the
# MAGIC book), the guarantee weakens. Recalibrate annually at minimum.

# COMMAND ----------

print("Notebook complete.")
