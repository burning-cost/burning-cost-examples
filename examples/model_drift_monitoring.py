"""
Model drift monitoring for a deployed motor pricing model.

The scenario: your motor frequency model was trained on 2021-2022 data and
deployed in January 2023. It is now mid-2023. Six months of live book have
accumulated in your inference table. The head of pricing wants to know:
is the model still fit for purpose?

This is the question that every UK insurer running a Solvency II internal
model or meeting SS1/23 model risk requirements needs to answer at least
quarterly. Most teams answer it with a spreadsheet showing aggregate A/E.
That is necessary but not sufficient — aggregate A/E can look fine while the
model is systematically mispricing a fast-growing segment. PSI per feature
catches this. The Gini z-test catches something even subtler: whether the
model's ability to rank risks has degraded, even if the average looks right.

This script works through the complete monitoring workflow:

    1. Generate a synthetic UK motor portfolio with a training period
       (the "reference window") and a live monitoring period where
       deliberate drift has been injected
    2. Compute exposure-weighted PSI and CSI to detect feature distribution
       drift — which rating factors have shifted since the model was trained?
    3. Compute A/E ratios by segment to find where the model is misfiring
    4. Run the Gini drift z-test (arXiv 2510.04556) to test whether
       discrimination power has degraded
    5. Build a MonitoringReport combining all checks into a traffic-light
       summary with a governance recommendation
    6. Interpret the output in terms the head of pricing and the model
       risk committee can act on

Three failure modes are deliberately injected in the live period:
    - Mix shift: younger drivers (<25) and older drivers (>70) enter the
      book in greater numbers, shifting the driver age distribution
    - Frequency uplift: the young driver segment has a genuine claims
      frequency increase (new risk, not captured in training data)
    - The model's age-band relativities are stale: it was trained before
      this mix shift, so its predictions for the drifted segments are wrong

These are the failure modes that occur in practice. They are independent
— you can get mix shift without frequency change, or frequency change
without mix shift. The monitoring checks detect each separately.

Libraries used
--------------
    insurance-monitoring — PSI, CSI, A/E, Gini, MonitoringReport

Dependencies
------------
    uv add insurance-monitoring polars numpy

The script runs without external data files. Expected runtime: 30-60 seconds
(Gini bootstrap dominates; reduce n_bootstrap if you want faster iteration).

Note on running: this script is intended to run on Databricks. On a
Raspberry Pi, it will likely time out or crash due to the bootstrap work.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _psi_band(v: float) -> str:
    """Map a PSI value to its traffic-light band label."""
    if v < 0.10:
        return 'green'
    elif v < 0.25:
        return 'amber'
    return 'red'


# ---------------------------------------------------------------------------
# Step 1: Generate a synthetic motor portfolio
# ---------------------------------------------------------------------------
#
# We build the data ourselves so the data-generating process is fully
# transparent. The key design choices:
#
#   - 60,000 policies in the reference (training) period. This is a
#     realistic size for a mid-tier UK motor insurer.
#
#   - 12,000 policies in the live monitoring period (6 months of a 24,000-
#     policy book, roughly). Monitoring periods are intentionally smaller
#     than training windows — you're monitoring one quarter against two
#     years of training data.
#
#   - Four rating factors: driver_age, vehicle_age, ncd_years, vehicle_value.
#     These are the most common continuous factors in UK motor pricing.
#
#   - The "true" model maps these factors to frequency. The deployed model
#     approximates this relationship — it was correct at training time but
#     the true relationship has since shifted.
#
# The frequency model (DGP):
#   freq = base_freq * age_factor(driver_age) * vehicle_factor(vehicle_age)
#          * ncd_factor(ncd_years)
#
# where the factors are calibrated to UK motor approximate relativities.
# ---------------------------------------------------------------------------

print("=" * 72)
print("Step 1: Generate synthetic motor portfolio")
print("=" * 72)

# Reference period: 2021-01 to 2022-12 (24 months)
N_REF = 60_000       # policies in training window
# Monitoring period: 2023-01 to 2023-06 (6 months)
N_MON = 12_000       # policies in live monitoring window

# The drift parameters — what changed in the live period
YOUNG_DRIVER_UPLIFT_FREQ = 1.35   # young drivers (<25) have 35% more claims than model expects
OLDER_DRIVER_MIX_INCREASE = 0.08  # 8pp more drivers >70 in the live book
YOUNG_DRIVER_MIX_INCREASE = 0.06  # 6pp more drivers <25 in the live book

rng = np.random.default_rng(seed=2023)


def _driver_age_ref(n: int) -> np.ndarray:
    """Reference period driver age distribution.

    UK motor book: mix of young, middle-aged, and older drivers.
    The distribution is roughly bimodal — young new drivers and
    experienced older drivers are both significant segments.
    """
    # Main population: middle-aged drivers (modal around 40)
    ages_main = rng.normal(loc=42, scale=14, size=int(n * 0.80)).clip(18, 70)
    # Young driver tail
    ages_young = rng.uniform(17, 25, size=int(n * 0.12))
    # Older driver segment
    ages_old = rng.normal(loc=73, scale=4, size=int(n * 0.08)).clip(70, 90)
    ages = np.concatenate([ages_main, ages_young, ages_old])[:n]
    rng.shuffle(ages)
    return ages.astype(np.float64)


def _driver_age_live(n: int) -> np.ndarray:
    """Live period driver age distribution — shifted relative to reference.

    Two things happen in the live period:
    1. Young drivers (<25) enter in greater numbers — perhaps via a new
       telematics product that attracted young drivers post-launch.
    2. Older drivers (>70) grow as the existing book ages and retains
       longer-tenure customers.

    The model was trained on the reference distribution. It has not seen
    this shift. Its age-band relativities are therefore stale.
    """
    # Main population shrinks to accommodate the shifted tails
    pct_main = 1.0 - OLDER_DRIVER_MIX_INCREASE - YOUNG_DRIVER_MIX_INCREASE - 0.06
    ages_main = rng.normal(loc=42, scale=14, size=int(n * pct_main)).clip(18, 70)
    ages_young = rng.uniform(17, 25, size=int(n * (0.12 + YOUNG_DRIVER_MIX_INCREASE)))
    ages_old = rng.normal(loc=73, scale=4, size=int(n * (0.08 + OLDER_DRIVER_MIX_INCREASE))).clip(70, 90)
    ages = np.concatenate([ages_main, ages_young, ages_old])[:n]
    rng.shuffle(ages)
    return ages.astype(np.float64)


def _age_factor(ages: np.ndarray) -> np.ndarray:
    """True frequency factor for driver age.

    Based on UK motor pricing actuarial research:
    - Under 25: very high frequency (~2.5x average)
    - 25-35: high (~1.4x)
    - 35-55: near-average (1.0x)
    - 55-70: slight reduction (~0.85x)
    - Over 70: rising again (~1.2x)
    """
    factors = np.where(ages < 25, 2.5,
              np.where(ages < 35, 1.4,
              np.where(ages < 55, 1.0,
              np.where(ages < 70, 0.85, 1.2))))
    return factors


def _model_age_factor(ages: np.ndarray) -> np.ndarray:
    """What the deployed model thinks the age factor is.

    This is the model's approximation, trained on 2021-2022 data.
    It underestimates young driver frequency (the young driver segment
    grew after training) and overestimates the older driver uplift
    (experience was limited at training time).

    This miscalibration is what the A/E and Gini monitoring should detect.
    """
    factors = np.where(ages < 25, 2.1,   # model says 2.1x, truth is now 2.5x × 1.35 = 3.4x
              np.where(ages < 35, 1.35,
              np.where(ages < 55, 1.0,
              np.where(ages < 70, 0.88, 1.15))))
    return factors


# Generate rating factor arrays
driver_age_ref = _driver_age_ref(N_REF)
driver_age_mon = _driver_age_live(N_MON)

vehicle_age_ref = rng.uniform(0, 15, N_REF)          # years since first registration
vehicle_age_mon = rng.uniform(0, 15, N_MON)          # no vehicle age drift in this example

ncd_years_ref = rng.choice(range(0, 10), N_REF, p=[
    0.12, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.12, 0.12
])
ncd_years_mon = rng.choice(range(0, 10), N_MON, p=[
    0.12, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.12, 0.12
])

vehicle_value_ref = rng.lognormal(mean=9.3, sigma=0.5, size=N_REF)   # approx GBP 5k-30k
vehicle_value_mon = rng.lognormal(mean=9.3, sigma=0.5, size=N_MON)   # no drift here

# Exposure: earned car-years. Not every policy is active for the full period.
# Reference period (24 months): many policies with >1 year exposure.
# Live period (6 months): all policies have at most 0.5 years exposure.
exposure_ref = rng.beta(a=8, b=2, size=N_REF) * 2.0   # 0.1 to 2.0 car-years
exposure_mon = rng.beta(a=6, b=2, size=N_MON) * 0.5   # 0.05 to 0.5 car-years

# True frequency for reference period (what actually happened)
base_freq = 0.075   # 7.5% annual claim frequency, typical UK motor
vehicle_age_factor_ref = 1.0 + 0.02 * vehicle_age_ref   # older vehicles slightly riskier
ncd_factor_ref = np.exp(-0.12 * ncd_years_ref)           # more NCD = lower frequency
true_freq_ref = base_freq * _age_factor(driver_age_ref) * vehicle_age_factor_ref * ncd_factor_ref
true_claims_ref = rng.poisson(true_freq_ref * exposure_ref).astype(np.float64)

# What the model predicted for the reference period
# (it was trained on the same data so predictions are close but not perfect)
model_freq_ref = base_freq * _model_age_factor(driver_age_ref) * vehicle_age_factor_ref * ncd_factor_ref
# Add a small noise term — in real life the model has variables not included
# in this simple DGP, and vice versa
model_freq_ref *= rng.lognormal(mean=0, sigma=0.08, size=N_REF)

# True frequency for the live period
# Here is where the drift is injected:
# 1. The true age factor has shifted (young drivers worse, model doesn't know)
# 2. The model's predictions are based on the stale training-period age factors
vehicle_age_factor_mon = 1.0 + 0.02 * vehicle_age_mon
ncd_factor_mon = np.exp(-0.12 * ncd_years_mon)

# True frequency in the live period: young drivers have YOUNG_DRIVER_UPLIFT_FREQ
# applied on top of the age factor, reflecting genuine claims environment change
age_factor_true_mon = _age_factor(driver_age_mon)
young_mask = driver_age_mon < 25
age_factor_true_mon = np.where(
    young_mask,
    age_factor_true_mon * YOUNG_DRIVER_UPLIFT_FREQ,
    age_factor_true_mon
)
true_freq_mon = base_freq * age_factor_true_mon * vehicle_age_factor_mon * ncd_factor_mon
true_claims_mon = rng.poisson(true_freq_mon * exposure_mon).astype(np.float64)

# Model predictions for the live period: still using stale age factors
# The model does not know about the frequency uplift in young drivers,
# nor does it know the mix has shifted. It applies the same relationship it
# learned at training time.
model_freq_mon = base_freq * _model_age_factor(driver_age_mon) * vehicle_age_factor_mon * ncd_factor_mon
model_freq_mon *= rng.lognormal(mean=0, sigma=0.08, size=N_MON)

# Segment labels for A/E breakdown (driver age bands)
def _age_band(ages: np.ndarray) -> np.ndarray:
    return np.where(ages < 25, "17-24",
           np.where(ages < 35, "25-34",
           np.where(ages < 55, "35-54",
           np.where(ages < 70, "55-69", "70+"))))

age_band_ref = _age_band(driver_age_ref)
age_band_mon = _age_band(driver_age_mon)

# Polars DataFrames for CSI (needed by the csi() function)
feat_ref = pl.DataFrame({
    "driver_age":    driver_age_ref.tolist(),
    "vehicle_age":   vehicle_age_ref.tolist(),
    "ncd_years":     ncd_years_ref.astype(float).tolist(),
    "vehicle_value": vehicle_value_ref.tolist(),
})
feat_mon = pl.DataFrame({
    "driver_age":    driver_age_mon.tolist(),
    "vehicle_age":   vehicle_age_mon.tolist(),
    "ncd_years":     ncd_years_mon.astype(float).tolist(),
    "vehicle_value": vehicle_value_mon.tolist(),
})

print(f"Reference period: {N_REF:,} policies, {exposure_ref.sum():.0f} car-years")
print(f"Live period:      {N_MON:,} policies, {exposure_mon.sum():.0f} car-years")
print(f"\nReference period — age distribution:")
for band in ["17-24", "25-34", "35-54", "55-69", "70+"]:
    n = (age_band_ref == band).sum()
    print(f"  {band}: {n:,} ({n/N_REF:.1%})")
print(f"\nLive period — age distribution (drift injected):")
for band in ["17-24", "25-34", "35-54", "55-69", "70+"]:
    n = (age_band_mon == band).sum()
    print(f"  {band}: {n:,} ({n/N_MON:.1%})")
print(f"\nReference period — overall A/E (should be ~1.0):")
ae_check = true_claims_ref.sum() / (model_freq_ref * exposure_ref).sum()
print(f"  {ae_check:.3f}")
print(f"Live period — overall A/E (drift will push this above 1.0):")
ae_check_mon = true_claims_mon.sum() / (model_freq_mon * exposure_mon).sum()
print(f"  {ae_check_mon:.3f}")


# ---------------------------------------------------------------------------
# Step 2: Feature distribution drift — PSI and CSI
# ---------------------------------------------------------------------------
#
# The first question when monitoring a deployed model: have the inputs changed?
# If the portfolio mix has shifted since training, the model is being applied
# to a different population than it was designed for. PSI quantifies this.
#
# PSI interpretation (FICO/industry standard):
#   < 0.10:  No significant shift. Model is likely still appropriate.
#   0.10-0.25: Moderate shift. Investigate. May need recalibration.
#   >= 0.25: Significant shift. Model is almost certainly stale. Refit.
#
# We deliberately injected a ~14pp shift in driver age distribution.
# This should show as PSI >> 0.25 for driver_age and near zero for the
# other features (vehicle_age, ncd_years, vehicle_value were not shifted).
#
# Insurance-correct approach: the `psi()` function accepts `exposure_weights`
# to weight bin proportions by earned car-years rather than policy count.
# A bin with 500 one-month policies should not be compared against a bin
# with 50 annual policies on equal terms. Always use exposure weighting
# for insurance monitoring.
#
# CSI (Characteristic Stability Index) is just PSI applied to every feature
# in a DataFrame. The result is a per-feature traffic-light table — the
# standard "CSI heat map" format used in monthly monitoring packs.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 2: Feature distribution drift (PSI and CSI)")
print("=" * 72)

from insurance_monitoring.drift import psi, csi, ks_test, wasserstein_distance

# PSI for the model score itself (log of predicted frequency)
# PSI > 0.25 on the model score means the model is being applied to a
# materially different distribution than it was scored on at training time.
# This is the most direct indicator that predictions may be unreliable.
score_ref = np.log(model_freq_ref)    # log-rate is the natural "score" for a Poisson model
score_mon = np.log(model_freq_mon)

score_psi = psi(
    reference=score_ref,
    current=score_mon,
    n_bins=10,
    exposure_weights=exposure_mon,   # weight by car-years, not policy count
)

print(f"Model score PSI: {score_psi:.4f}  ({_psi_band(score_psi)})")
print(
    "  The model score PSI measures whether the distribution of predicted"
    " rates has shifted. A shift here means the book the model is scoring"
    " today is different from the book it was trained on."
)

# PSI individually for driver_age — the deliberately shifted feature
driver_age_psi = psi(
    reference=driver_age_ref,
    current=driver_age_mon,
    n_bins=10,
    exposure_weights=exposure_mon,
)
print(f"\ndriver_age PSI: {driver_age_psi:.4f}  ({_psi_band(driver_age_psi)})")
print(
    f"  PSI > 0.25 means the driver age distribution has shifted enough"
    f" that the model's age-band relativities may no longer apply."
    f" The injected shift ({YOUNG_DRIVER_MIX_INCREASE:.0%} more young,"
    f" {OLDER_DRIVER_MIX_INCREASE:.0%} more old) should register here."
)

# vehicle_age PSI — should be near zero (not shifted)
vehicle_age_psi = psi(
    reference=vehicle_age_ref,
    current=vehicle_age_mon,
    n_bins=10,
)
print(f"\nvehicle_age PSI: {vehicle_age_psi:.4f}  ({_psi_band(vehicle_age_psi)})")
print("  Expected near zero — vehicle_age was not shifted in this scenario.")

# CSI heat map: all features in one call
# This is the format you produce for the monthly monitoring pack.
# One row per feature, traffic-light band, sortable by severity.
csi_table = csi(
    reference_df=feat_ref,
    current_df=feat_mon,
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_value"],
    n_bins=10,
)

print("\nCSI heat map (all rating factors):")
print(f"  {'Feature':<20} {'CSI':>8}  {'Band'}")
print(f"  {'-'*20}  {'-'*8}  {'-'*10}")
for row in csi_table.sort("csi", descending=True).iter_rows(named=True):
    print(f"  {row['feature']:<20} {row['csi']:>8.4f}  {row['band'].upper()}")

print(
    "\n  Reading: driver_age should be RED (injected shift)."
    " Other features should be GREEN. This tells you exactly which"
    " rating factor to investigate — you don't need to look at everything."
)

# Wasserstein distance: report the driver_age shift in original units (years)
# This is more interpretable for communicating to non-technical stakeholders.
age_shift = wasserstein_distance(driver_age_ref, driver_age_mon)
print(f"\nWasserstein distance (driver_age): {age_shift:.2f} years")
print(
    f"  Interpretation: the effective mean driver age has shifted by"
    f" approximately {age_shift:.1f} years (in the distributional sense)."
    f" This is a concrete, business-meaningful number."
)

# KS test: formal hypothesis test for driver_age shift
ks_result = ks_test(driver_age_ref, driver_age_mon)
print(f"\nKS test (driver_age): stat={ks_result['statistic']:.4f},"
      f" p={ks_result['p_value']:.4g},"
      f" significant={ks_result['significant']}")
print(
    "  Note: with 60k reference and 12k monitoring observations, the KS test"
    " will flag even economically trivial shifts. Use PSI for the dashboard;"
    " KS for formal quarterly sign-off."
)


# ---------------------------------------------------------------------------
# Step 3: Actual/Expected ratios by segment
# ---------------------------------------------------------------------------
#
# The second monitoring question: where is the model misfiring?
#
# Aggregate A/E tells you the model's global accuracy. It says nothing about
# who is inside the number. A/E = 1.05 on a portfolio could mean:
#   (a) The model is consistently 5% optimistic everywhere — cheap to fix
#   (b) The model is 30% pessimistic for young drivers and 8% optimistic
#       for middle-aged drivers, netting to 5% — a structural problem
#
# Segmented A/E tells you (a) vs (b). This is the diagnostic step that
# moves monitoring from "something is wrong" to "here is what and where."
#
# The Poisson confidence interval (Garwood exact intervals) is important:
# some segments are small. A/E = 1.25 on 40 claims is not statistically
# distinguishable from 1.00. The CI tells you which segment-level signals
# are real and which are sampling noise.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 3: Actual/Expected ratios by segment")
print("=" * 72)

from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Overall A/E with Poisson CI
ae_overall = ae_ratio_ci(
    actual=true_claims_mon,
    predicted=model_freq_mon,
    exposure=exposure_mon,
)
print("Overall A/E ratio (live period):")
print(f"  A/E = {ae_overall['ae']:.3f}")
print(f"  95% CI: [{ae_overall['lower']:.3f}, {ae_overall['upper']:.3f}]")
print(f"  Observed claims: {ae_overall['n_claims']:.0f}")
print(f"  Expected claims: {ae_overall['n_expected']:.1f}")
print(
    "\n  The overall A/E may look only slightly elevated because the young"
    " driver frequency uplift is partially offset by stable performance in"
    " the larger middle-aged segments. This is exactly the problem with"
    " stopping at aggregate A/E."
)

# Segmented A/E by driver age band
ae_by_age = ae_ratio(
    actual=true_claims_mon,
    predicted=model_freq_mon,
    exposure=exposure_mon,
    segments=age_band_mon,
)
ae_by_age = ae_by_age.sort("ae_ratio", descending=True)

print("\nA/E by driver age band (live period):")
print(f"  {'Band':<10} {'Actual':>8}  {'Expected':>8}  {'A/E':>6}  {'n_policies':>12}  Interpretation")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*30}")
for row in ae_by_age.iter_rows(named=True):
    ae_val = row["ae_ratio"]
    if ae_val > 1.10:
        interp = "RED — model is optimistic for this segment"
    elif ae_val > 1.05:
        interp = "AMBER — monitor closely"
    elif ae_val < 0.90:
        interp = "AMBER — model is pessimistic"
    else:
        interp = "GREEN"
    print(
        f"  {row['segment']:<10}  {row['actual']:>8.1f}  {row['expected']:>8.1f}"
        f"  {ae_val:>6.3f}  {row['n_policies']:>12,}  {interp}"
    )

print(
    "\n  The 17-24 band should show A/E significantly above 1.0 — this is"
    " the segment where frequency uplift was injected. The other bands"
    " should be near 1.0. This pattern is diagnostic: the model needs"
    " recalibration specifically for the young driver segment."
)

# A/E by NCD band — should be near 1.0 everywhere (NCD was not shifted)
ncd_band_mon = np.where(ncd_years_mon <= 1, "0-1 years",
               np.where(ncd_years_mon <= 3, "2-3 years",
               np.where(ncd_years_mon <= 6, "4-6 years", "7+ years")))

ae_by_ncd = ae_ratio(
    actual=true_claims_mon,
    predicted=model_freq_mon,
    exposure=exposure_mon,
    segments=ncd_band_mon,
)
print("\nA/E by NCD band (expected near 1.0 — NCD was not shifted):")
for row in ae_by_ncd.sort("segment").iter_rows(named=True):
    print(f"  {row['segment']:<12}  A/E={row['ae_ratio']:.3f}")


# ---------------------------------------------------------------------------
# Step 4: Gini drift z-test — has discrimination power degraded?
# ---------------------------------------------------------------------------
#
# The third monitoring question: has the model's ability to rank risks
# deteriorated?
#
# Calibration (A/E) and discrimination (Gini) are measuring different things.
# A model can maintain a good A/E ratio while its discrimination degrades:
# if all predictions shift uniformly (constant multiplier), A/E shifts too
# but the ranking is preserved. Conversely, if the mix shifts so that the
# model's predictions no longer correlate with actual risk, Gini falls
# even if A/E looks fine.
#
# The Gini drift z-test from arXiv 2510.04556 gives this a formal statistical
# test. The key insight: if Gini is falling, the model's risk stratification
# is breaking down. This is expensive to fix — it requires a full refit.
# If only A/E is off, you can fix it with a scalar recalibration. The
# monitoring framework distinguishes these cases explicitly.
#
# Default alpha = 0.32 (one-sigma rule, per arXiv 2510.04556 recommendation).
# The paper argues this is the right sensitivity for routine monitoring — you
# want early warning. Use alpha=0.05 for formal governance escalation.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 4: Gini drift z-test (arXiv 2510.04556)")
print("=" * 72)

from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(
    actual=true_claims_ref,
    predicted=model_freq_ref,
    exposure=exposure_ref,
)
gini_mon = gini_coefficient(
    actual=true_claims_mon,
    predicted=model_freq_mon,
    exposure=exposure_mon,
)

print(f"Reference period Gini: {gini_ref:.4f}")
print(f"Live period Gini:      {gini_mon:.4f}")
print(f"Change:                {gini_mon - gini_ref:+.4f}")
print(
    "\n  Gini in [0.35, 0.55] is typical for a UK motor frequency model."
    " A fall of more than 0.03-0.05 points over 6 months is significant."
    " The injected age-mix shift should reduce Gini because the model's"
    " ranking ability degrades when it misprices the fastest-growing segment."
)

# Two-sample z-test: test whether the Gini change is statistically significant.
# This version uses the raw reference and monitoring arrays to estimate
# bootstrap variance on both sides (Algorithm 2 from arXiv 2510.04556).
drift = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_mon,
    n_reference=N_REF,
    n_current=N_MON,
    reference_actual=true_claims_ref,
    reference_predicted=model_freq_ref,
    current_actual=true_claims_mon,
    current_predicted=model_freq_mon,
    reference_exposure=exposure_ref,
    current_exposure=exposure_mon,
    n_bootstrap=200,    # increase to 500 for production; 200 is fine for development
    alpha=0.32,         # one-sigma rule per arXiv 2510.04556 recommendation
)

print(f"\nGini drift test results:")
print(f"  z-statistic: {drift['z_statistic']:.3f}")
print(f"  p-value:     {drift['p_value']:.4f}")
print(f"  Significant: {drift['significant']}  (at alpha=0.32)")

if drift["p_value"] < 0.10:
    print(
        "\n  INTERPRETATION (RED): Gini has degraded significantly."
        " This is beyond recalibration. A full model refit on recent data"
        " is required. Escalate to model risk committee."
    )
elif drift["p_value"] < 0.32:
    print(
        "\n  INTERPRETATION (AMBER): Gini drift detected at monitoring"
        " sensitivity. Not yet at governance escalation threshold (p<0.10)."
        " Increase monitoring frequency. Review at next month-end."
    )
else:
    print(
        "\n  INTERPRETATION (GREEN): Gini is stable. The model's ranking"
        " ability has not materially degraded. A/E misfires may still exist"
        " in specific segments (check the A/E table above)."
    )

print(
    "\n  Technical note: the z-test uses bootstrap variance estimation for"
    " both reference and monitoring periods (Algorithm 2, arXiv 2510.04556)."
    " The default alpha=0.32 is the 'one-sigma rule' recommended by the"
    " paper for ongoing monitoring. For formal governance sign-off, re-run"
    " with alpha=0.05 and treat p<0.05 as the hard escalation trigger."
)


# ---------------------------------------------------------------------------
# Step 5: Full MonitoringReport — all checks in one call
# ---------------------------------------------------------------------------
#
# MonitoringReport orchestrates all the individual checks (PSI, CSI, A/E,
# Gini) in a single pass and applies the arXiv 2510.04556 decision tree to
# produce a governance recommendation.
#
# The decision tree:
#   Gini stable + A/E stable  -> NO_ACTION
#   Gini stable + A/E drifted -> RECALIBRATE (scalar offset correction, hours)
#   Gini drifted               -> REFIT (model rebuild, weeks)
#   Both drifted and conflicting -> INVESTIGATE
#   Any amber signal           -> MONITOR_CLOSELY
#
# Setting murphy_distribution="poisson" enables the Murphy decomposition,
# which sharpens the RECALIBRATE vs REFIT distinction. The Murphy MCB
# component measures miscalibration; the DSC component measures discrimination.
# When global miscalibration (GMCB) dominates, RECALIBRATE. When local
# miscalibration or DSC degradation dominates, REFIT.
#
# The to_polars() output is designed to be written directly to a Delta table
# or logged to MLflow as run metrics. This is the format for the monthly
# monitoring table in your Databricks workspace.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 5: MonitoringReport — combined traffic-light summary")
print("=" * 72)

from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=true_claims_ref,
    reference_predicted=model_freq_ref,
    current_actual=true_claims_mon,
    current_predicted=model_freq_mon,
    exposure=exposure_mon,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_mon,
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_value"],
    score_reference=score_ref,
    score_current=score_mon,
    murphy_distribution="poisson",
    n_bootstrap=200,
)

print(f"\nOverall recommendation: {report.recommendation}")
print(
    "\n  Possible values:"
    "\n    NO_ACTION      — no significant drift detected"
    "\n    MONITOR_CLOSELY — amber signals; increase review frequency"
    "\n    RECALIBRATE    — A/E drifted, Gini stable; cheap fix (hours)"
    "\n    REFIT          — Gini drifted; rebuild model on recent data (weeks)"
    "\n    INVESTIGATE    — conflicting signals; manual review first"
)

# Full traffic-light table
report_df = report.to_polars()
print("\nFull monitoring report (flat table):")
print(f"  {'Metric':<35} {'Value':>10}  {'Band'}")
print(f"  {'-'*35}  {'-'*10}  {'-'*15}")
for row in report_df.iter_rows(named=True):
    if row["metric"] == "recommendation":
        print(f"\n  {'RECOMMENDATION':<35} {'':>10}  {row['band']}")
    else:
        val_str = f"{row['value']:.4f}" if not (row["value"] != row["value"]) else "—"
        print(f"  {row['metric']:<35} {val_str:>10}  {row['band'].upper()}")


# ---------------------------------------------------------------------------
# Step 6: Governance report — the model was deployed 6 months ago
# ---------------------------------------------------------------------------
#
# The monitoring report above gives you the technical picture. What the
# model risk committee, the head of pricing, and — in an SS1/23 world —
# the PRA-facing model documentation need is a structured narrative that
# answers the governance question: is this model still fit for purpose?
#
# The section below produces that narrative from the monitoring results.
# It is deliberately in a format that can be copied directly into a
# model performance attestation or Confluence page.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 6: Model fitness-for-purpose governance report")
print("=" * 72)

# Extract key metrics from report
ae_value = report.results_["ae_ratio"]["value"]
ae_lower = report.results_["ae_ratio"]["lower_ci"]
ae_upper = report.results_["ae_ratio"]["upper_ci"]
ae_band = report.results_["ae_ratio"]["band"]

gini_ref_val = report.results_["gini"]["reference"]
gini_cur_val = report.results_["gini"]["current"]
gini_p = report.results_["gini"]["p_value"]
gini_band = report.results_["gini"]["band"]

max_csi_val = report.results_["max_csi"]["value"]
max_csi_band = report.results_["max_csi"]["band"]
worst_feature = report.results_["max_csi"]["worst_feature"]

murphy_available = report.murphy_available
if murphy_available:
    murphy_verdict = report.results_["murphy"]["verdict"]
    dsc = report.results_["murphy"]["discrimination"]
    mcb = report.results_["murphy"]["miscalibration"]
    gmcb = report.results_["murphy"]["global_mcb"]
    lmcb = report.results_["murphy"]["local_mcb"]


# Summarise each check with a clear verdict
def _verdict_line(metric: str, value: str, band: str, implication: str) -> str:
    band_symbol = {"green": "[OK]  ", "amber": "[WARN]", "red": "[FAIL]"}.get(band, "[    ]")
    return f"  {band_symbol}  {metric:<30} {value:<12}  {implication}"


governance_lines = [
    "",
    "MODEL PERFORMANCE ATTESTATION",
    "Motor Frequency Model — Q2 2023 Review",
    "=" * 60,
    "",
    "Model deployment date: 2023-01-01",
    "Review period: 2023-01 to 2023-06 (first 6 months post-deployment)",
    "Reference window: 2021-2022 training data",
    f"Monitoring policies: {N_MON:,}",
    f"Monitoring exposure: {exposure_mon.sum():.0f} car-years",
    "",
    "MONITORING RESULTS",
    "-" * 60,
    "",
    "Feature distribution stability (CSI):",
    _verdict_line(
        "driver_age",
        f"PSI={driver_age_psi:.3f}",
        "red" if driver_age_psi >= 0.25 else ("amber" if driver_age_psi >= 0.10 else "green"),
        "Book mix has shifted — investigate driver age segment"
    ),
    _verdict_line(
        "vehicle_age",
        f"PSI={vehicle_age_psi:.3f}",
        "green",
        "Stable"
    ),
    _verdict_line(
        "Worst feature overall",
        f"CSI={max_csi_val:.3f}",
        max_csi_band,
        f"Worst: {worst_feature}"
    ),
    "",
    "Calibration (A/E ratio):",
    _verdict_line(
        "Overall A/E",
        f"{ae_value:.3f}",
        ae_band,
        f"95% CI [{ae_lower:.3f}, {ae_upper:.3f}]"
    ),
]

# Add per-segment A/E lines
governance_lines.append("")
governance_lines.append("  Per-segment A/E (key segments):")
for row in ae_by_age.iter_rows(named=True):
    ae_val = row["ae_ratio"]
    seg_band = "red" if ae_val > 1.10 or ae_val < 0.90 else ("amber" if ae_val > 1.05 or ae_val < 0.95 else "green")
    governance_lines.append(_verdict_line(
        f"  Age {row['segment']}",
        f"A/E={ae_val:.3f}",
        seg_band,
        f"{row['n_policies']:,} policies"
    ))

governance_lines += [
    "",
    "Discrimination (Gini drift):",
    _verdict_line(
        "Gini (reference)",
        f"{gini_ref_val:.4f}",
        "green",
        "Baseline at model training"
    ),
    _verdict_line(
        "Gini (live period)",
        f"{gini_cur_val:.4f}",
        gini_band,
        f"p={gini_p:.3f} vs reference"
    ),
]

if murphy_available:
    governance_lines += [
        "",
        "Murphy decomposition (discrimination vs calibration):",
        _verdict_line(
            "Discrimination (DSC)",
            f"{dsc:.5f}",
            murphy_verdict,
            "Model ranking skill"
        ),
        _verdict_line(
            "Miscalibration (MCB)",
            f"{mcb:.5f}",
            murphy_verdict,
            f"Global={gmcb:.5f}, Local={lmcb:.5f}"
        ),
        "",
        f"  Murphy verdict: {murphy_verdict}",
        "  RECALIBRATE = global shift dominates (cheap fix)",
        "  REFIT = local structure is broken (full rebuild required)",
    ]

governance_lines += [
    "",
    "=" * 60,
    f"OVERALL RECOMMENDATION: {report.recommendation}",
    "=" * 60,
    "",
]

if report.recommendation == "REFIT":
    governance_lines += [
        "Action required: Initiate model refit on 2022-2023 data.",
        "The Gini drift test indicates discrimination power has degraded.",
        "A scalar recalibration will not resolve this — the model's risk",
        "ranking has broken down, most likely because the driver age mix",
        "has shifted beyond what the training data covered.",
        "",
        "Estimated effort: 3-6 weeks for refit, testing, and sign-off.",
        "Interim measure: apply segment-level load factors to the young",
        "driver (17-24) segment while the refit is in progress.",
    ]
elif report.recommendation == "RECALIBRATE":
    governance_lines += [
        "Action required: Recalibrate the model intercept.",
        "A/E is outside the green band but Gini is stable. The model's",
        "risk ranking is intact; only the overall level is off.",
        "A multiplicative offset correction will restore calibration.",
        "",
        "Estimated effort: 1-2 days. No regulatory notification required.",
    ]
elif report.recommendation == "MONITOR_CLOSELY":
    governance_lines += [
        "No immediate action required.",
        "Amber signals detected — increase monitoring frequency from",
        "quarterly to monthly. Schedule a re-review in 4 weeks.",
        "If amber signals persist or worsen, escalate to RECALIBRATE.",
    ]
else:
    governance_lines += [
        "No action required. Model is performing within expected parameters.",
        "Schedule next review: Q3 2023.",
    ]

governance_lines += [
    "",
    "Signed off by: Pricing Actuarial Team",
    "Review date: 2023-07-01",
    "Next scheduled review: 2023-10-01",
]

for line in governance_lines:
    print(line)


# ---------------------------------------------------------------------------
# Step 7: Practical trigger — when to escalate vs when to wait
# ---------------------------------------------------------------------------
#
# This section codifies the thresholds and their business rationale.
# The defaults are the FICO PSI convention (credit scoring, 1990s) and
# the arXiv 2510.04556 Gini test recommendations. You can override them
# with MonitoringThresholds for your specific portfolio.
# ---------------------------------------------------------------------------

print("=" * 72)
print("Step 7: Threshold reference and customisation")
print("=" * 72)

from insurance_monitoring import MonitoringThresholds, PSIThresholds, AERatioThresholds, GiniDriftThresholds

print("""
Industry-standard thresholds (built-in defaults):

  PSI / CSI per feature:
    GREEN  < 0.10   No significant population shift
    AMBER  0.10-0.25   Moderate shift — investigate the feature
    RED   >= 0.25   Significant shift — model is likely stale for this segment

  A/E ratio:
    GREEN  0.95-1.05   Within normal sampling variation
    AMBER  0.90-0.95 or 1.05-1.10   Monitor; consider recalibration
    RED   < 0.90 or > 1.10   Model is materially miscalibrated; escalate

  Gini drift (p-value from two-sample z-test):
    GREEN  p >= 0.32   No evidence of discrimination degradation
    AMBER  0.10 <= p < 0.32   Early warning signal; increase monitoring
    RED   p < 0.10   Statistically significant degradation; escalate to REFIT

  Note on the alpha=0.32 default for Gini: this is deliberate.
  arXiv 2510.04556 argues that in a monitoring context, catching genuine
  degradation early (sensitivity) matters more than avoiding false alarms
  (specificity). The one-sigma rule (alpha=0.32) gives earlier warning
  at the cost of more amber flags. Use alpha=0.05 for the formal governance
  escalation decision, not the routine monitoring dashboard.
""")

# Demonstrate threshold customisation
# A large motor book with monthly monitoring might want tighter PSI thresholds
# because small shifts are economically meaningful at high volume.
custom_thresholds = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),         # tighter for large book
    ae_ratio=AERatioThresholds(green_lower=0.97, green_upper=1.03),  # tighter A/E
    gini_drift=GiniDriftThresholds(amber_p_value=0.32, red_p_value=0.05),  # stricter escalation
)

report_tight = MonitoringReport(
    reference_actual=true_claims_ref,
    reference_predicted=model_freq_ref,
    current_actual=true_claims_mon,
    current_predicted=model_freq_mon,
    exposure=exposure_mon,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_mon,
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_value"],
    thresholds=custom_thresholds,
    murphy_distribution="poisson",
    n_bootstrap=200,
)

print(f"Default thresholds recommendation:   {report.recommendation}")
print(f"Tighter thresholds recommendation:   {report_tight.recommendation}")
print(
    "\n  Tighter PSI thresholds may elevate features from GREEN to AMBER"
    " even without genuine distributional change — use them when you have"
    " both the portfolio size and the operational capacity to investigate"
    " every amber flag every month."
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Workflow complete")
print("=" * 72)

print(f"""
Six-month post-deployment model review — summary:

  Reference (training) window:  {N_REF:,} policies, {exposure_ref.sum():.0f} car-years
  Monitoring window:            {N_MON:,} policies, {exposure_mon.sum():.0f} car-years

  Drift injected:
    driver_age mix: +{YOUNG_DRIVER_MIX_INCREASE:.0%} young (<25), +{OLDER_DRIVER_MIX_INCREASE:.0%} old (>70)
    young driver frequency: {YOUNG_DRIVER_UPLIFT_FREQ:.2f}x model expectation

  Monitoring results:
    driver_age PSI:     {driver_age_psi:.4f}   (threshold: 0.25 = RED)
    Overall A/E:        {ae_overall['ae']:.3f}
    Gini reference:     {gini_ref:.4f}
    Gini live:          {gini_mon:.4f}  (change: {gini_mon - gini_ref:+.4f})
    Gini drift p-value: {drift['p_value']:.4f}
    Recommendation:     {report.recommendation}

  The mix shift was detected by CSI (driver_age PSI > 0.25 = RED).
  The segment-level A/E table showed where the model is misfiring.
  The Gini test quantified whether the ranking has broken down.
  MonitoringReport assembled these into a single governance recommendation.

Next steps for a real portfolio:
  - Schedule MonitoringReport to run monthly against your inference table
  - Write the to_polars() output to a Delta table for trend tracking
  - Set up alerts when recommendation != 'NO_ACTION'
  - Apply chain-ladder development factors to actuals before computing A/E
    when monitoring periods are less than 12 months developed (motor: 12m,
    liability: 24m+)
  - For the Gini test, consider gini_drift_test_onesample() when you
    only have the stored training Gini (not raw training data)
""")

