"""
End-to-end UK motor pricing workflow.

This script walks through a complete pricing model development and deployment
pipeline using the Burning Cost open-source library stack:

    insurance-synthetic  — generate a realistic synthetic UK motor portfolio
    catboost             — train a Poisson GBM frequency model
    shap-relativities   — extract multiplicative rating factor relativities
    insurance-validation — PRA-compliant model validation report (if installed)
    insurance-deploy     — register models and run champion/challenger in shadow mode

The data generated here is entirely synthetic. In production you would replace
step 1 with your actual policy and claims extract. Everything from step 2 onwards
works identically on real data — that is the point.

Run this script top-to-bottom. Each section prints its key outputs so you can
follow what is happening without needing a notebook environment.

Dependencies
------------
    uv add insurance-synthetic shap-relativities insurance-deploy catboost shap scipy

Approximate runtime: 2–5 minutes, depending on the machine. The vine copula
fitting (step 1) and SHAP computation (step 3) dominate.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Section 1: Generate a synthetic UK motor portfolio
# ---------------------------------------------------------------------------
#
# We use insurance-synthetic to produce 20,000 synthetic policies. The library
# fits a vine copula to reproduce the dependency structure between rating factors
# — tail dependence between young driver age, high vehicle group, and zero NCD
# is the key actuarial feature that generic data generators miss.
#
# In production: replace this section with your actual portfolio extract. The
# column names below (driver_age, vehicle_group, ncd_years, etc.) match the
# uk_motor_schema() definition, so you can rename your columns to match and
# the rest of the script runs unchanged.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: Generate synthetic UK motor portfolio")
print("=" * 70)

from insurance_synthetic import InsuranceSynthesizer, uk_motor_schema

schema = uk_motor_schema()

# The schema tells us which columns are categorical, what the valid ranges are,
# and which column is the exposure and frequency target. Pull the constraints
# directly from it so they stay in sync.
schema_constraints = schema["constraints"]
schema_cols = {c.name: c for c in schema["columns"]}
categorical_cols = [
    c.name for c in schema["columns"] if c.dtype == "categorical"
]

# Build a small seed portfolio. InsuranceSynthesizer needs real data to fit
# marginals and the copula. We construct a minimal but plausible seed rather
# than using a completely hand-crafted DataFrame, so the synthetic output has
# realistic distributional properties.
#
# Real teams would pass their actual portfolio here.
rng = np.random.default_rng(42)
n_seed = 5_000

seed_driver_age = rng.integers(17, 80, size=n_seed)
seed_ncd_years = np.clip(
    rng.integers(0, 25, size=n_seed),
    0,
    seed_driver_age - 17,  # NCD can't exceed years eligible to drive
).astype(int)

seed_df = pl.DataFrame({
    "driver_age": seed_driver_age.tolist(),
    "vehicle_age": rng.integers(0, 20, size=n_seed).tolist(),
    "vehicle_group": rng.integers(1, 50, size=n_seed).tolist(),
    "region": rng.choice(
        schema_cols["region"].categories, size=n_seed
    ).tolist(),
    "ncd_years": seed_ncd_years.tolist(),
    "cover_type": rng.choice(
        schema_cols["cover_type"].categories, size=n_seed,
        p=[0.75, 0.15, 0.10],  # comprehensive is majority; realistic mix
    ).tolist(),
    "payment_method": rng.choice(
        schema_cols["payment_method"].categories, size=n_seed,
        p=[0.55, 0.30, 0.15],
    ).tolist(),
    "annual_mileage": rng.integers(3_000, 25_000, size=n_seed).tolist(),
    "exposure": np.clip(rng.beta(5, 2, size=n_seed), 0.01, 1.0).tolist(),
    # Frequency: ~7% of policies have a claim, rare to have more than one
    "claim_count": rng.choice([0, 1, 2], size=n_seed, p=[0.93, 0.065, 0.005]).tolist(),
    # Severity: zero for no-claim policies, gamma-distributed otherwise
    "claim_amount": [
        float(rng.gamma(shape=2.0, scale=2_500.0)) if c > 0 else 0.0
        for c in rng.choice([0, 1, 2], size=n_seed, p=[0.93, 0.065, 0.005])
    ],
})

print(f"Seed portfolio: {len(seed_df):,} policies")
print(f"Overall claim frequency: {seed_df['claim_count'].mean():.3f} claims/policy")

# Fit vine copula and generate a larger synthetic portfolio.
# trunc_lvl=3 is a practical choice for 11 variables: it captures the most
# important pairwise and triplet dependencies without the O(d^2) cost of a
# full vine. For a final production run, drop trunc_lvl to fit the full vine.
synth = InsuranceSynthesizer(
    method="vine",
    trunc_lvl=3,
    random_state=42,
)

synth.fit(
    seed_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
    categorical_cols=categorical_cols,
)

N_SYNTHETIC = 20_000
portfolio = synth.generate(N_SYNTHETIC, constraints=schema_constraints)

print(f"\nSynthetic portfolio: {len(portfolio):,} policies")
print(f"Claim frequency: {portfolio['claim_count'].mean():.3f} claims/policy")
print(f"Mean exposure: {portfolio['exposure'].mean():.3f} policy years")
print(f"Cover type mix:\n{portfolio['cover_type'].value_counts().sort('count', descending=True)}")


# ---------------------------------------------------------------------------
# Section 2: Train a CatBoost frequency model
# ---------------------------------------------------------------------------
#
# We model claim frequency as Poisson with a log-link. CatBoost handles the
# categorical rating factors (region, cover_type, payment_method) natively
# without one-hot encoding — this is the main practical advantage over XGBoost
# for insurance pricing work. Categorical encoding choices distort SHAP values
# for the encoded features; native CatBoost handling avoids that problem.
#
# The offset log(exposure) is passed via baseline_predictors. This is equivalent
# to the GLM offset — it tells the model that a policy with 0.5 years of exposure
# should expect half as many claims as an identical policy with 1.0 year.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: Train CatBoost frequency model")
print("=" * 70)

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    print("catboost not installed. Install with: pip install catboost")
    sys.exit(1)

# Split into train (70%) and holdout (30%). Holdout is used for validation
# and for the champion/challenger comparison.
n_train = int(0.7 * len(portfolio))
train_df = portfolio[:n_train]
holdout_df = portfolio[n_train:]

# Rating factors — exclude exposure, claim_count, claim_amount (the targets)
feature_cols = [
    "driver_age",
    "vehicle_age",
    "vehicle_group",
    "region",
    "ncd_years",
    "cover_type",
    "payment_method",
    "annual_mileage",
]

cat_feature_indices = [
    i for i, col in enumerate(feature_cols)
    if col in categorical_cols
]

# CatBoost expects pandas for the Pool API with categorical features; convert
# once here and work in pandas for model fitting only.
train_pd = train_df.select(feature_cols + ["claim_count", "exposure"]).to_pandas()
holdout_pd = holdout_df.select(feature_cols + ["claim_count", "exposure"]).to_pandas()

# log(exposure) offset: tells CatBoost this is a rate model
train_log_exposure = np.log(train_pd["exposure"].clip(lower=1e-9)).values
holdout_log_exposure = np.log(holdout_pd["exposure"].clip(lower=1e-9)).values

train_pool = Pool(
    data=train_pd[feature_cols],
    label=train_pd["claim_count"],
    cat_features=cat_feature_indices,
    baseline=train_log_exposure,  # offset = log(exposure)
)

holdout_pool = Pool(
    data=holdout_pd[feature_cols],
    label=holdout_pd["claim_count"],
    cat_features=cat_feature_indices,
    baseline=holdout_log_exposure,
)

# Challenger model: more trees, deeper. This is the model we will test against
# the champion in section 5. In practice you might also vary the feature set,
# the objective function, or add interaction terms.
model_v2 = CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)

model_v2.fit(train_pool, eval_set=holdout_pool)

# Champion model: fewer trees, shallower. This represents what was in production
# before the challenger was developed.
model_v1 = CatBoostRegressor(
    loss_function="Poisson",
    iterations=200,
    depth=3,
    learning_rate=0.08,
    l2_leaf_reg=5.0,
    random_seed=42,
    verbose=0,
)

model_v1.fit(train_pool)

# Quick lift check: RMSE on holdout
from catboost import metrics as cb_metrics

v2_preds = model_v2.predict(holdout_pd[feature_cols], ntree_end=500)
v1_preds = model_v1.predict(holdout_pd[feature_cols], ntree_end=200)
actuals = holdout_pd["claim_count"].values

v2_rmse = float(np.sqrt(np.mean((v2_preds - actuals) ** 2)))
v1_rmse = float(np.sqrt(np.mean((v1_preds - actuals) ** 2)))

print(f"Champion (v1) holdout RMSE: {v1_rmse:.4f}")
print(f"Challenger (v2) holdout RMSE: {v2_rmse:.4f}")
print(
    f"Challenger lift: {(v1_rmse - v2_rmse) / v1_rmse:.1%} RMSE improvement"
)


# ---------------------------------------------------------------------------
# Section 3: Extract SHAP relativities
# ---------------------------------------------------------------------------
#
# We convert the challenger model's SHAP values into multiplicative rating
# factor relativities. The output is directly comparable to GLM exp(beta)
# relativities — a table of (feature, level, relativity) triples where the
# base level has relativity = 1.0 and all other levels are relative to it.
#
# This format is what pricing actuaries are used to reviewing. It also makes
# the model interpretable to underwriters and regulators who are not familiar
# with tree models.
#
# We use the holdout set for SHAP computation. Using training data would give
# overly optimistic relativities due to overfitting.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: Extract SHAP relativities from challenger model")
print("=" * 70)

from shap_relativities import SHAPRelativities

# Pass the holdout feature matrix and exposure as Polars Series directly.
# SHAPRelativities handles the pandas bridge internally for the SHAP call.
holdout_features = holdout_df.select(feature_cols)
holdout_exposure = holdout_df["exposure"]

sr = SHAPRelativities(
    model=model_v2,
    X=holdout_features,
    exposure=holdout_exposure,
    categorical_features=["region", "cover_type", "payment_method"],
)

sr.fit()

# Base levels: the reference point for each categorical feature. Actuaries
# typically set these to the largest or most "standard" level. We use the
# Comprehensive cover type and London region as bases — these are the most
# common in the synthetic portfolio.
base_levels = {
    "region": "London",
    "cover_type": "Comprehensive",
    "payment_method": "Annual",
}

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels=base_levels,
    ci_method="clt",
)

print("Rating factor relativities (challenger model, Poisson GBM):\n")

# Print each feature's relativities in a clean table
for feature in feature_cols:
    feature_rels = relativities.filter(pl.col("feature") == feature)
    if feature_rels.is_empty():
        continue
    print(f"  {feature}:")
    for row in feature_rels.iter_rows(named=True):
        level_str = str(row["level"])[:20].ljust(22)
        rel = row["relativity"]
        if rel is not None and not np.isnan(rel):
            ci_lo = row.get("lower_ci", float("nan"))
            ci_hi = row.get("upper_ci", float("nan"))
            if ci_lo is not None and not np.isnan(ci_lo):
                print(f"    {level_str} {rel:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
            else:
                print(f"    {level_str} {rel:.3f}")
    print()

# Annualised base rate (exp(expected_value) after accounting for average exposure)
baseline = sr.baseline()
print(f"Annualised base claim rate: {baseline:.4f} claims per policy-year")

# Diagnostic: check SHAP reconstruction against model predictions
print("\nSHAP diagnostic checks:")
checks = sr.validate()
for check_name, result in checks.items():
    status = "PASS" if result.passed else "WARN"
    print(f"  [{status}] {check_name}: {result.message}")


# ---------------------------------------------------------------------------
# Section 4: Model validation report
# ---------------------------------------------------------------------------
#
# insurance-validation is a planned library in the Burning Cost stack that will
# produce PRA SS1/23-compliant model validation documentation. It is not yet
# published. This section shows where it will fit and runs a manual validation
# in the meantime.
#
# The manual checks below are the minimum a UK insurer should document for any
# frequency model submitted to actuarial function review:
#
#   - Lift: RMSE on holdout vs. baseline (claim mean rate)
#   - Calibration: ratio of predicted to observed claim frequency
#   - Gini coefficient on holdout (concentration of predictive power)
#
# When insurance-validation is released, replace the manual block with:
#
#   from insurance_validation import ValidationReport
#   report = ValidationReport(model_v2, holdout_df, target="claim_count",
#                             exposure="exposure", features=feature_cols)
#   report.run_all()
#   print(report.to_markdown())
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: Model validation")
print("=" * 70)

# Attempt to import insurance-validation if available
_validation_available = False
try:
    import insurance_validation  # noqa: F401
    _validation_available = True
    print("insurance-validation detected — using full validation suite")
except ImportError:
    print(
        "insurance-validation not installed (not yet released). "
        "Running manual validation checks.\n"
        "Install when available: uv add insurance-validation"
    )

if not _validation_available:
    # -- Calibration --
    # The ratio of total predicted claims to total observed claims should be
    # close to 1.0. Material deviation here means the model is systematically
    # over- or under-pricing.
    v2_predicted_total = float(
        np.sum(v2_preds * holdout_pd["exposure"].values)
    )
    v2_observed_total = float(holdout_pd["claim_count"].sum())
    calibration_ratio = v2_predicted_total / v2_observed_total if v2_observed_total > 0 else float("nan")
    print(f"Calibration ratio (predicted / observed claims): {calibration_ratio:.4f}")
    if abs(calibration_ratio - 1.0) > 0.05:
        print("  WARNING: calibration ratio outside 5% band — investigate systematic bias")
    else:
        print("  OK: within 5% tolerance")

    # -- Lift vs naive baseline --
    # Compare model RMSE to a naive model that predicts the overall mean rate.
    mean_rate = float(np.mean(actuals))
    naive_preds = np.full_like(actuals, fill_value=mean_rate, dtype=float)
    naive_rmse = float(np.sqrt(np.mean((naive_preds - actuals) ** 2)))
    lift_vs_naive = (naive_rmse - v2_rmse) / naive_rmse
    print(f"\nNaive (mean rate) holdout RMSE: {naive_rmse:.4f}")
    print(f"Challenger v2 holdout RMSE:     {v2_rmse:.4f}")
    print(f"Lift over naive:                {lift_vs_naive:.1%}")

    # -- Gini coefficient --
    # A standard ranking metric for frequency models. Computed on the holdout
    # set. Higher is better. UK motor frequency models typically achieve 0.15–0.35.
    def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalised Gini coefficient.

        Measures how well the model ranks risks. A score of 1.0 would mean
        the model perfectly identifies the highest-risk policies. A score of 0
        means it has no predictive power beyond the mean.
        """
        order = np.argsort(y_pred)
        y_sorted = y_true[order]
        n = len(y_sorted)
        cumsum = np.cumsum(y_sorted)
        gini_raw = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
        # Normalise by the theoretical maximum (perfect model)
        order_perfect = np.argsort(y_true)
        y_sorted_p = y_true[order_perfect]
        cumsum_p = np.cumsum(y_sorted_p)
        gini_max = (n + 1 - 2 * np.sum(cumsum_p) / cumsum_p[-1]) / n if cumsum_p[-1] > 0 else 1.0
        return gini_raw / gini_max if gini_max != 0 else 0.0

    gini = gini_coefficient(actuals, v2_preds)
    print(f"\nHoldout Gini coefficient: {gini:.3f}")
    print("  (UK motor frequency models typically achieve 0.15–0.35)")

    print("\nValidation summary:")
    print(f"  Calibration: {calibration_ratio:.4f} {'OK' if abs(calibration_ratio - 1.0) <= 0.05 else 'REVIEW'}")
    print(f"  RMSE lift vs naive: {lift_vs_naive:.1%}")
    print(f"  Gini coefficient: {gini:.3f}")
    print(
        "\n  Note: a full PRA SS1/23-compliant validation would also include "
        "out-of-time tests, stress tests, and sensitivity analysis. "
        "This is covered by the insurance-validation library when released."
    )


# ---------------------------------------------------------------------------
# Section 5: Champion/challenger deployment with insurance-deploy
# ---------------------------------------------------------------------------
#
# We register both models in a local ModelRegistry, set up a shadow-mode
# experiment, and simulate quoting 1,000 renewal policies. Shadow mode means
# the champion always prices the live quote; the challenger scores in parallel
# but its price is never shown to the customer. This is the correct starting
# point — it gives us model quality data with zero FCA Consumer Duty risk.
#
# After the experiment runs for 12–18 months and we have developed claims data,
# we would use ModelComparison.bootstrap_lr_test() to decide whether to promote
# the challenger to champion.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 5: Champion/challenger deployment (shadow mode)")
print("=" * 70)

from insurance_deploy import (
    ModelRegistry,
    Experiment,
    QuoteLogger,
    KPITracker,
    ModelComparison,
    ENBPAuditReport,
)

# Use a temporary directory so this demo does not leave state on the filesystem.
# In production you would use a persistent path: ModelRegistry("./model_registry")
_tmpdir = tempfile.mkdtemp(prefix="bc_examples_")
registry_path = Path(_tmpdir) / "registry"
log_path = Path(_tmpdir) / "quotes.db"

print(f"Registry path: {registry_path}")
print(f"Quote log path: {log_path}")

registry = ModelRegistry(str(registry_path))

# Register champion (v1) and challenger (v2).
# metadata should capture enough context to reconstruct the experiment in future.
mv1 = registry.register(
    model_v1,
    name="motor_frequency",
    version="1.0",
    metadata={
        "training_date": "2025-01-01",
        "training_rows": n_train,
        "features": feature_cols,
        "objective": "Poisson",
        "iterations": 200,
        "depth": 3,
        "holdout_rmse": round(v1_rmse, 6),
        "description": "Champion model: conservative hyperparameters",
    },
)

mv2 = registry.register(
    model_v2,
    name="motor_frequency",
    version="2.0",
    metadata={
        "training_date": "2025-07-01",
        "training_rows": n_train,
        "features": feature_cols,
        "objective": "Poisson",
        "iterations": 500,
        "depth": 4,
        "holdout_rmse": round(v2_rmse, 6),
        "description": "Challenger model: deeper trees, more iterations",
    },
)

print(f"\nRegistered: {mv1}")
print(f"Registered: {mv2}")

# Set v1 as the explicit champion. The registry is append-only — promoting v2
# later is done via set_champion(), not by overwriting.
registry.set_champion("motor_frequency", "1.0")
champion = registry.champion("motor_frequency")
print(f"Current champion: {champion}")

# Create shadow-mode experiment. 10% challenger split is typical for UK motor —
# large enough to accumulate meaningful volumes within 12 months.
experiment = Experiment(
    name="motor_v2_vs_v1_shadow",
    champion=mv1,
    challenger=mv2,
    challenger_pct=0.10,
    mode="shadow",  # shadow = zero regulatory risk
)
print(f"\nExperiment: {experiment}")

# Open the audit log
logger = QuoteLogger(str(log_path))

# ---------------------------------------------------------------------------
# Simulate quoting 1,000 renewal policies.
#
# In a real system this loop would be your quote API handler. For each policy:
#   1. Route it to champion or challenger (deterministic, by policy_id hash)
#   2. The champion always prices the live quote in shadow mode
#   3. The challenger scores in shadow (output is logged, not shown to customer)
#   4. Log both the quote and the ENBP for ICOBS 6B.2.51R audit
#
# The ENBP (Equivalent New Business Price) is what the same risk would cost
# as new business. FCA rules say you cannot charge renewals more than ENBP.
# The library records what you provide — calculating ENBP correctly is your
# pricing team's responsibility.
# ---------------------------------------------------------------------------

print("\nSimulating 1,000 renewal quotes...")

rng_quotes = np.random.default_rng(99)
n_quotes = 1_000

# Simulate over a 6-month period so KPI time estimates work correctly.
start_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

bound_policy_ids = []
bound_count = 0

for i in range(n_quotes):
    policy_id = f"POL-{i:05d}"
    arm = experiment.route(policy_id)

    # In shadow mode, live_model() always returns champion.
    # We also score the shadow model for comparison logging.
    live_mv = experiment.live_model(policy_id)

    # Pick a random holdout policy as the risk profile for this quote.
    row_idx = i % len(holdout_pd)
    risk_row = holdout_pd[feature_cols].iloc[[row_idx]]
    exposure = float(holdout_pd["exposure"].iloc[row_idx])

    # Champion prediction
    champion_rate = float(model_v1.predict(risk_row)[0])
    # Challenger prediction (shadow — not shown to customer)
    challenger_rate = float(model_v2.predict(risk_row)[0])

    # Simple premium = rate * exposure * average severity loading
    # Real systems would multiply frequency by expected severity and add loads.
    avg_severity = 4_500.0  # approximate average cost per claim
    champion_premium = max(150.0, champion_rate * exposure * avg_severity)
    challenger_premium = max(150.0, challenger_rate * exposure * avg_severity)

    # ENBP: new business equivalent. We use the challenger rate as a rough
    # proxy for what a new customer with this risk profile would pay.
    # In production this comes from your new business rater.
    enbp = max(150.0, challenger_rate * exposure * avg_severity * 1.02)

    # Simulate quote timestamp spread over 6 months
    quote_ts = start_ts + timedelta(days=float(rng_quotes.uniform(0, 180)))

    # Log the live quote (always champion in shadow mode)
    logger.log_quote(
        policy_id=policy_id,
        experiment_name=experiment.name,
        arm=arm,
        model_version=live_mv.version_id,
        quoted_price=champion_premium,
        enbp=enbp,
        renewal_flag=True,  # all simulated as renewals for ENBP demo
        exposure=exposure,
        timestamp=quote_ts,
    )

    # Simulate 35% bind rate (typical UK motor direct)
    if rng_quotes.random() < 0.35:
        bound_price = champion_premium * rng_quotes.uniform(0.97, 1.03)
        logger.log_bind(policy_id, bound_price=bound_price)
        bound_policy_ids.append(policy_id)
        bound_count += 1

        # Simulate some claims (about 8% of bound policies)
        if rng_quotes.random() < 0.08:
            claim_amount = float(rng_quotes.gamma(shape=2.0, scale=2_500.0))
            logger.log_claim(
                policy_id=policy_id,
                claim_date=date(2025, 3, 15),
                claim_amount=claim_amount,
                development_month=12,  # assume 12-month developed for demo
            )

print(f"Quoted: {n_quotes:,} policies")
print(f"Bound: {bound_count:,} policies ({bound_count/n_quotes:.1%} hit rate)")


# ---------------------------------------------------------------------------
# Section 6: KPI tracking and ENBP audit report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 6: KPI tracking and ENBP audit report")
print("=" * 70)

tracker = KPITracker(logger)

# Quote volume by arm
vol = tracker.quote_volume(experiment.name)
print("\nQuote volume by arm:")
for arm, stats in vol.items():
    print(
        f"  {arm.capitalize()}: {stats['n']:,} quotes, "
        f"mean price £{stats['mean_price']:.2f}"
    )

# Hit rate by arm
hr = tracker.hit_rate(experiment.name)
print("\nHit rate by arm:")
for arm, stats in hr.items():
    print(
        f"  {arm.capitalize()}: {stats['bound']:,} / {stats['quoted']:,} "
        f"= {stats['hit_rate']:.1%}"
    )

# ENBP compliance
enbp_stats = tracker.enbp_compliance(experiment.name)
print("\nENBP compliance (renewal quotes):")
for arm, stats in enbp_stats.items():
    rate = stats['compliance_rate']
    rate_str = f"{rate:.1%}" if not np.isnan(rate) else "N/A"
    print(
        f"  {arm.capitalize()}: {stats['compliant']:,} compliant, "
        f"{stats['breaches']:,} breaches — {rate_str} compliance rate"
    )

# GWP summary
gwp = tracker.gwp(experiment.name)
print("\nGWP on bound policies:")
for arm, stats in gwp.items():
    print(
        f"  {arm.capitalize()}: {stats['bound_policies']:,} policies, "
        f"£{stats['total_gwp']:,.0f} GWP, mean £{stats['mean_gwp']:.2f}"
    )

# Power analysis: how long before we can make a reliable comparison?
print("\nPower analysis (when can we promote?):")
power = tracker.power_analysis(experiment.name)
print(
    f"  Current volumes: champion={power['current_n_champion']:,}, "
    f"challenger={power['current_n_challenger']:,}"
)
if np.isfinite(power['lr_total_months_with_development']):
    print(
        f"  Est. months to LR significance: "
        f"{power['lr_total_months_with_development']:.1f} "
        f"(including {12}-month development period)"
    )
if np.isfinite(power['hr_months_to_significance']):
    print(
        f"  Est. months to hit rate significance: "
        f"{power['hr_months_to_significance']:.1f}"
    )
for note in power["notes"][:2]:
    print(f"  Note: {note}")

# Generate ENBP audit report
reporter = ENBPAuditReport(logger)
audit_md = reporter.generate(
    experiment_name=experiment.name,
    period_start="2025-01-01",
    period_end="2025-06-30",
    firm_name="Example Insurer Ltd",
    smf_holder="Jane Smith, Chief Actuary",
)

print("\n--- ENBP Audit Report (first 40 lines) ---")
for line in audit_md.split("\n")[:40]:
    print(line)
print("... (truncated — full report available via reporter.generate())")

# Save the full report
audit_path = Path(_tmpdir) / "enbp_audit_report.md"
audit_path.write_text(audit_md)
print(f"\nFull ENBP audit report saved to: {audit_path}")


# ---------------------------------------------------------------------------
# Section 7: Statistical comparison (with note on data maturity)
# ---------------------------------------------------------------------------
#
# ModelComparison.bootstrap_lr_test() requires 12+ months of developed claims.
# We have simulated 6 months and a small claim sample, so the test will
# correctly return INSUFFICIENT_EVIDENCE. This is the expected outcome at this
# stage — not a failure.
#
# In a real experiment you would re-run this section after 12–18 months of
# data and use the result to decide whether to promote the challenger.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 7: Statistical comparison (illustrative)")
print("=" * 70)

comparison = ModelComparison(tracker)

# Suppress the maturity warning — we know data is immature, that's the point
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr_result = comparison.bootstrap_lr_test(
        experiment_name=experiment.name,
        development_months=12,
        n_bootstrap=1_000,  # reduced for demo speed; use 10,000 in production
        seed=42,
    )

print("\nLoss ratio comparison (bootstrap test):")
print(lr_result.summary())
print(
    "\nNote: INSUFFICIENT_EVIDENCE is the correct result at 6 months. "
    "Re-run after 12–18 months of data with full claims development."
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("End-to-end workflow complete")
print("=" * 70)
print(f"""
Libraries used:
  insurance-synthetic  — synthetic UK motor portfolio (20,000 policies)
  catboost             — Poisson GBM frequency model (champion v1, challenger v2)
  shap-relativities   — multiplicative rating factor relativities
  insurance-deploy     — model registry, shadow experiment, ENBP audit

Outputs produced:
  Model relativities for {len(feature_cols)} rating factors
  Quote log with {n_quotes:,} renewal quotes and {bound_count:,} binds
  ENBP audit report (ICOBS 6B.2.51R)
  Champion/challenger KPIs (quote volume, hit rate, GWP, power analysis)

Next steps in a real deployment:
  1. Replace synthetic data with your actual portfolio extract
  2. Tune hyperparameters and feature set on your book
  3. Run the experiment for 12–18 months in shadow mode
  4. Re-run the bootstrap LR test once claims are developed
  5. Present ModelComparison results to actuarial function for promotion decision
  6. Call registry.set_champion("motor_frequency", "2.0") once approved

For PRA SS1/23 documentation, see the insurance-validation library (forthcoming).
For blog posts on methodology: https://burning-cost.github.io
""")
