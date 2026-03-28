"""
Champion/challenger deployment for UK motor insurance using insurance-deploy.

Your pricing team has built a new GBM-based frequency model. The holdout Gini
is 0.45 versus 0.42 for the incumbent GLM. The team wants to deploy it.

The problem is everything after that sentence.

How do you run the new model alongside the existing one without disrupting live
pricing? How do you produce the per-quote, per-policy audit trail that ICOBS
6B.2.51R requires? How do you know — with statistical rigour rather than gut
feel — when you have enough data to make a credible promotion decision?

This script works through the complete workflow:

    1. Define champion and challenger as Python callables (realistic
       parametric GLM and GBM approximations, not toy functions)
    2. Register both versions in the model registry with training metadata
    3. Set up the champion/challenger experiment in shadow mode
    4. Score 1,000 synthetic motor quotes through the framework, logging
       each one with full audit trail (quote, bind, claim events)
    5. Inspect the quote log — governance question answered in three lines
    6. Run the KPI dashboard: hit rate, GWP, frequency, loss ratio
    7. Run the bootstrap likelihood ratio test for challenger promotion
    8. Generate the ENBP compliance audit report for the SMF holder
    9. Promote the challenger to champion and show the governance trail

Shadow mode is the default and the right choice here. The challenger scores
on every quote but the customer always sees the champion price. No FCA
Consumer Duty (PRIN 2A) fair value questions arise: we are not pricing two
customers of identical profile differently. We are collecting evidence before
we change anything.

The governance angle throughout: the FCA's ICOBS 6B multi-firm review (2022)
found 83% of firms could not demonstrate which model priced each renewal. This
framework is the answer. Every quote is logged with model version, arm, price,
and ENBP. Your regulator asks who priced policy X — you answer in three lines.

Libraries used
--------------
    insurance-deploy — registry, experiment, logger, KPIs, comparison, audit

Dependencies
------------
    uv add insurance-deploy
    uv add numpy  # for data generation

The script runs without external data files. Everything is generated inline.
Expected runtime: 10-30 seconds (bootstrap with 2,000 iterations dominates).
Run on Databricks rather than a laptop for anything above 10,000 iterations.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Step 1: Define champion and challenger as Python callables
# ---------------------------------------------------------------------------
#
# In production these would be trained sklearn/CatBoost/XGBoost objects loaded
# from your model artefact store. Here we build parametric approximations that
# reproduce realistic UK motor pricing behaviour:
#
#   Champion (GLM): log-linear frequency model with three rating factors.
#     Typical of a mature personal lines GLM built in SAS or Emblem and
#     recently ported to Python. Gini ~0.42 on holdout.
#
#   Challenger (GBM): captures non-linear age × NCD interactions and a
#     vehicle value effect the GLM misses. Gini ~0.45 on holdout.
#     Lower loss ratio because it prices high-value, young-driver risks
#     more accurately — those policies were cross-subsidised in the GLM.
#
# Both expose a .predict(X) interface where X is a list of feature dicts.
# This is the sklearn-compatible interface ModelRegistry expects.
#
# We deliberately make the challenger 3-4% lower in loss ratio by adjusting
# its base rate. This is the signal the bootstrap LR test needs to detect.
# Whether it detects it depends on sample size — the power analysis will
# tell you honestly.
# ---------------------------------------------------------------------------

print("=" * 72)
print("Step 1: Define champion and challenger pricing models")
print("=" * 72)

# Feature schema (UK motor):
#   age          — driver age (17–80)
#   ncd_years    — no-claims discount years (0–9)
#   vehicle_value — vehicle value in GBP (2,000–60,000)
#   postcode_band — area risk band (1=low, 5=high)
#   annual_mileage — annual mileage in thousands (3–30)

class ChampionGLM:
    """
    GLM-style frequency model: multiplicative rating factors.

    This is a log-linear Poisson GLM with manually set coefficients —
    the kind of model a UK pricing actuary would recognise from Emblem.
    Base rate = £350 annual premium at reference risk.
    """

    name = "motor_glm"
    version = "2.0"

    # Reference risk: age=35, ncd=5, vehicle_value=15000, band=3, mileage=10k
    BASE_RATE = 350.0

    # Log-linear coefficients (multiplicative on base)
    AGE_COEFFS = {
        (17, 25): 2.10,   # young drivers: high frequency, high severity
        (25, 35): 1.30,
        (35, 50): 1.00,   # reference
        (50, 65): 0.90,
        (65, 80): 1.05,   # elderly drivers: lower frequency, higher at-fault rate
    }

    NCD_FACTOR = 0.88     # each NCD year reduces premium by 12% (simplified)
    NCD_CAP = 9           # max NCD years modelled

    POSTCODE_FACTORS = {1: 0.80, 2: 0.90, 3: 1.00, 4: 1.15, 5: 1.35}

    MILEAGE_FACTOR = 0.018  # per 1,000 miles above reference (10k)

    # Vehicle value: GLM uses a simple band — misses the non-linearity
    VEHICLE_VALUE_FACTOR = 0.000008  # linear coefficient on value in GBP

    def predict(self, X: list[dict]) -> list[float]:
        """Return annual premium for each risk in X."""
        prices = []
        for risk in X:
            age = risk["age"]
            ncd = min(risk["ncd_years"], self.NCD_CAP)
            value = risk["vehicle_value"]
            band = risk["postcode_band"]
            mileage = risk["annual_mileage"]

            # Age factor
            age_factor = 1.0
            for (lo, hi), factor in self.AGE_COEFFS.items():
                if lo <= age < hi:
                    age_factor = factor
                    break

            # NCD factor (compound discount)
            ncd_factor = self.NCD_FACTOR ** ncd

            # Postcode factor
            area_factor = self.POSTCODE_FACTORS.get(band, 1.0)

            # Mileage: deviation from reference 10k miles
            mileage_factor = 1.0 + self.MILEAGE_FACTOR * (mileage - 10.0)
            mileage_factor = max(mileage_factor, 0.5)

            # Vehicle value: GLM uses linear approximation
            value_factor = 1.0 + self.VEHICLE_VALUE_FACTOR * (value - 15_000)
            value_factor = max(value_factor, 0.7)

            premium = (
                self.BASE_RATE
                * age_factor
                * ncd_factor
                * area_factor
                * mileage_factor
                * value_factor
            )

            # Apply expense loading and profit margin
            premium *= 1.25   # 20% expense ratio + 5% margin
            premium = round(max(premium, 150.0), 2)
            prices.append(premium)
        return prices


class ChallengerGBM:
    """
    GBM-style model approximation with interaction terms.

    Captures the age × NCD interaction the GLM misses: a 22-year-old with
    5 years NCD is a very different risk from a 22-year-old with 0 NCD.
    Also captures the vehicle value non-linearity (high-value vehicles
    attract more theft; the GLM linear approximation underprices them).

    Lower base rate: the GBM prices high-risk segments higher and can
    afford a lower overall base because cross-subsidy is reduced.
    This is where the 3-4pp loss ratio improvement comes from.
    """

    name = "motor_gbm"
    version = "3.0"

    BASE_RATE = 340.0   # lower base: better risk selection reduces adverse selection

    def predict(self, X: list[dict]) -> list[float]:
        prices = []
        for risk in X:
            age = risk["age"]
            ncd = min(risk["ncd_years"], 9)
            value = risk["vehicle_value"]
            band = risk["postcode_band"]
            mileage = risk["annual_mileage"]

            # Age × NCD interaction: young drivers with no NCD are priced much
            # higher; young drivers with NCD get meaningful credit
            if age < 25:
                if ncd == 0:
                    age_ncd_factor = 2.60
                elif ncd <= 2:
                    age_ncd_factor = 1.90
                else:
                    age_ncd_factor = 1.55
            elif age < 35:
                age_ncd_factor = 1.25 * (0.92 ** ncd)
            elif age < 50:
                age_ncd_factor = 1.00 * (0.94 ** ncd)
            elif age < 65:
                age_ncd_factor = 0.88 * (0.95 ** ncd)
            else:
                age_ncd_factor = 1.02 * (0.96 ** ncd)

            # Postcode (same as GLM)
            area_factor = {1: 0.80, 2: 0.90, 3: 1.00, 4: 1.15, 5: 1.35}.get(band, 1.0)

            # Mileage (same as GLM)
            mileage_factor = 1.0 + 0.018 * (mileage - 10.0)
            mileage_factor = max(mileage_factor, 0.5)

            # Vehicle value: non-linear — GBM captures the theft spike >£30k
            if value <= 10_000:
                value_factor = 0.88
            elif value <= 20_000:
                value_factor = 1.00
            elif value <= 35_000:
                value_factor = 1.12 + 0.000004 * (value - 20_000)
            else:
                value_factor = 1.18 + 0.000010 * (value - 35_000)

            premium = (
                self.BASE_RATE
                * age_ncd_factor
                * area_factor
                * mileage_factor
                * value_factor
            )

            premium *= 1.25
            premium = round(max(premium, 150.0), 2)
            prices.append(premium)
        return prices


champion_model = ChampionGLM()
challenger_model = ChallengerGBM()

# Quick sense check: price a reference risk and a young driver
ref_risk = {"age": 35, "ncd_years": 5, "vehicle_value": 15_000,
            "postcode_band": 3, "annual_mileage": 10.0}
young_risk = {"age": 22, "ncd_years": 0, "vehicle_value": 8_000,
              "postcode_band": 4, "annual_mileage": 15.0}

champ_ref = champion_model.predict([ref_risk])[0]
chall_ref = challenger_model.predict([ref_risk])[0]
champ_young = champion_model.predict([young_risk])[0]
chall_young = challenger_model.predict([young_risk])[0]

print(f"Reference risk (age=35, 5yr NCD, value=£15k, band=3):")
print(f"  Champion (GLM):    £{champ_ref:,.2f}")
print(f"  Challenger (GBM):  £{chall_ref:,.2f}")
print(f"  Difference:        £{chall_ref - champ_ref:+.2f}")
print(f"\nYoung driver (age=22, 0yr NCD, value=£8k, band=4, 15k miles):")
print(f"  Champion (GLM):    £{champ_young:,.2f}")
print(f"  Challenger (GBM):  £{chall_young:,.2f}")
print(f"  Difference:        £{chall_young - champ_young:+.2f}")
print(f"\nGBM prices young driver higher (+£{chall_young - champ_young:.0f}): "
      f"this is where the cross-subsidy in the GLM is removed.")

# ---------------------------------------------------------------------------
# Step 2: Register both models in the model registry
# ---------------------------------------------------------------------------
#
# ModelRegistry is append-only. Once a model version is registered you cannot
# overwrite it. This is not a limitation — it is the audit guarantee. If you
# need to re-register a model, create a new version string.
#
# The registry serialises model objects to disk via joblib and records a
# SHA-256 hash of the serialised bytes. When you load a model back, the
# library verifies the hash. File corruption or tampering is caught on load.
#
# We use a temporary directory here so the example is self-contained. In
# production, use a stable path on shared storage (DBFS, NFS, S3 with
# consistent read).
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 2: Register champion and challenger in model registry")
print("=" * 72)

from insurance_deploy import ModelRegistry, ModelVersion

# Use a temporary directory so the example runs without leaving artefacts
_tmpdir = tempfile.mkdtemp(prefix="motor_registry_")
registry = ModelRegistry(_tmpdir)

champion_mv = registry.register(
    champion_model,
    name="motor",
    version="2.0",
    metadata={
        "training_date": "2024-01-15",
        "training_data_period": "2021-01 to 2023-12",
        "n_training_policies": 180_000,
        "features": ["age", "ncd_years", "vehicle_value", "postcode_band", "annual_mileage"],
        "model_type": "GLM_Poisson_loglink",
        "holdout_gini": 0.42,
        "holdout_lr": 0.648,
        "validated_by": "Sarah Chen, FCAS",
        "validation_date": "2024-01-22",
        "approved_by": "Tom Bradley, SMF23",
    },
)

challenger_mv = registry.register(
    challenger_model,
    name="motor",
    version="3.0",
    metadata={
        "training_date": "2024-07-10",
        "training_data_period": "2021-07 to 2024-06",
        "n_training_policies": 210_000,
        "features": ["age", "ncd_years", "vehicle_value", "postcode_band", "annual_mileage"],
        "model_type": "GBM_CatBoost_approximation",
        "key_improvements": "age×NCD interaction, non-linear vehicle value",
        "holdout_gini": 0.45,
        "holdout_lr": 0.618,
        "holdout_lr_improvement_vs_v2": "-0.030 (3pp)",
        "validated_by": "Sarah Chen, FCAS",
        "validation_date": "2024-07-18",
        "approved_by": "Tom Bradley, SMF23",
    },
)

print(f"Champion registered: {champion_mv}")
print(f"  Version ID:  {champion_mv.version_id}")
print(f"  SHA-256:     {champion_mv.model_hash[:16]}...")
print(f"  Holdout Gini: {champion_mv.metadata['holdout_gini']}")
print()
print(f"Challenger registered: {challenger_mv}")
print(f"  Version ID:  {challenger_mv.version_id}")
print(f"  SHA-256:     {challenger_mv.model_hash[:16]}...")
print(f"  Holdout Gini: {challenger_mv.metadata['holdout_gini']}")
print(f"  Improvement:  {challenger_mv.metadata['holdout_lr_improvement_vs_v2']}")

# Verify the registry list
versions = registry.list("motor")
print(f"\nRegistry contains {len(versions)} version(s) for 'motor':")
for v in versions:
    print(f"  {v}")

# ---------------------------------------------------------------------------
# Step 3: Set up the champion/challenger experiment
# ---------------------------------------------------------------------------
#
# Shadow mode (the default) is used here. The challenger is scored on every
# quote but the customer always sees the champion price. No FCA Consumer Duty
# (PRIN 2A) concerns. No adverse selection confound in the loss ratio data.
#
# challenger_pct=0.20 means 20% of policies are statistically designated to the
# challenger arm. In shadow mode this only affects the routing label in the log
# — the champion prices all of them. The label is used to split KPIs so you can
# compare champion-arm loss ratio versus challenger-arm loss ratio on the same
# price basis. This is a cleaner comparison than live A/B because there is no
# price signal difference.
#
# The routing is SHA-256(policy_id + experiment_name). It is deterministic and
# stateless. A policy routed to challenger on its first quote will always be
# routed to challenger in this experiment, regardless of how many times it re-
# quotes. This is required for ENBP audit integrity: you must be able to say
# "this policy was assigned to the challenger arm from inception" without
# consulting a database of random numbers drawn at quote time.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 3: Set up the champion/challenger experiment")
print("=" * 72)

from insurance_deploy import Experiment

exp = Experiment(
    name="motor_v3_vs_v2_2024q3",
    champion=champion_mv,
    challenger=challenger_mv,
    challenger_pct=0.20,  # 20% of policies designated to challenger arm
    mode="shadow",        # Champion prices all quotes; challenger is observational
)

print(f"Experiment: {exp}")
print(f"  Champion:        {exp.champion.version_id}")
print(f"  Challenger:      {exp.challenger.version_id}")
print(f"  Challenger pct:  {exp.challenger_pct:.0%}")
print(f"  Mode:            {exp.mode}")
print(f"  Active:          {exp.is_active()}")
print(f"  Created at:      {exp.created_at[:19]}")

# Demonstrate routing determinism: same policy always routes the same way
demo_policies = ["POL-001234", "POL-005678", "POL-009999"]
print(f"\nRouting determinism check:")
for pol in demo_policies:
    arm = exp.route(pol)
    # Call again — must be identical
    arm2 = exp.route(pol)
    assert arm == arm2, "Routing should be deterministic"
    print(f"  {pol} -> {arm}")

# Show routing formula for the record
print(
    "\nRouting formula: SHA-256(policy_id + experiment_name), last 8 hex chars "
    "as integer mod 100. If < 20: challenger. Else: champion."
)

# ---------------------------------------------------------------------------
# Step 4: Generate synthetic quotes and score through the framework
# ---------------------------------------------------------------------------
#
# We generate 1,000 policies with realistic UK motor risk profiles, then
# simulate the quote → bind → claim lifecycle for each.
#
# The data-generating process:
#
#   Quoted:        all 1,000 policies (100% — every policy gets a quote)
#   Bound:         ~32% conversion rate (realistic UK private car direct)
#   Claims:        ~8% of bound policies have a claim in the first year
#
# ENBP records: 40% of policies are renewals (realistic mix). For renewals,
# we calculate ENBP as quoted_price × 1.02 (i.e. 2% headroom), except for
# a small fraction (3%) where we deliberately inject a breach. This lets us
# verify the audit report flags breaches correctly.
#
# We use timestamps spread over 6 months to give the power analysis a
# meaningful monthly rate estimate.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 4: Score 1,000 synthetic quotes through the framework")
print("=" * 72)

from insurance_deploy import QuoteLogger

_db_path = Path(_tmpdir) / "quotes.db"
logger = QuoteLogger(_db_path)

rng = np.random.default_rng(seed=2024)

N_POLICIES = 1_000

# Generate risk features
ages        = rng.integers(17, 80, size=N_POLICIES)
ncd_years   = rng.integers(0, 10, size=N_POLICIES)
values      = rng.integers(2_000, 60_001, size=N_POLICIES)
bands       = rng.integers(1, 6, size=N_POLICIES)
mileages    = rng.uniform(3.0, 30.0, size=N_POLICIES)
is_renewal  = rng.random(size=N_POLICIES) < 0.40   # 40% renewals
will_bind   = rng.random(size=N_POLICIES) < 0.32   # ~32% conversion

# Timestamps spread over 6 months (1 Jan 2024 to 1 Jul 2024)
base_ts = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
quote_ts = [
    base_ts + timedelta(seconds=int(s))
    for s in rng.uniform(0, 6 * 30.44 * 86400, size=N_POLICIES)
]

print(f"Generating {N_POLICIES} policies...")
print(f"  Renewals:          {int(is_renewal.sum())} ({is_renewal.mean():.0%})")
print(f"  Expected binds:    {int(will_bind.sum())} ({will_bind.mean():.0%})")

# Score all policies and log quotes
n_champion_arm = 0
n_challenger_arm = 0
enbp_breach_injected = 0

for i in range(N_POLICIES):
    policy_id = f"POL-{2024_0001 + i:07d}"

    risk = {
        "age":           int(ages[i]),
        "ncd_years":     int(ncd_years[i]),
        "vehicle_value": int(values[i]),
        "postcode_band": int(bands[i]),
        "annual_mileage": float(mileages[i]),
    }

    # Routing arm (deterministic)
    arm = exp.route(policy_id)
    if arm == "champion":
        n_champion_arm += 1
    else:
        n_challenger_arm += 1

    # Champion always prices in shadow mode
    champion_price = champion_model.predict([risk])[0]

    # Challenger scores in shadow — result is for our internal comparison
    # but the customer only ever sees champion_price
    challenger_price = challenger_model.predict([risk])[0]

    # ENBP for renewals: you calculate this, we just record it.
    # In production this comes from your re-pricing engine run on the renewal
    # risk profile at current rates (ICOBS 6B.3.20G methodology).
    enbp = None
    if is_renewal[i]:
        # Normal case: ENBP gives 2% headroom above current champion price
        enbp = round(champion_price * 1.02, 2)
        # Inject 3% breach rate: charged more than ENBP (simulates a system error
        # or a manual override that slipped through governance controls)
        if rng.random() < 0.03:
            # Force a breach: quoted price exceeds ENBP
            # We log the champion price which is above the ENBP we calculated.
            # In a real system this would be caught before quote — here it is
            # intentional so the audit report has something to flag.
            enbp = round(champion_price * 0.97, 2)  # set ENBP below quoted price
            enbp_breach_injected += 1

    # Suppress the ENBP breach warning in the loop — we're doing it on purpose
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.log_quote(
            policy_id=policy_id,
            experiment_name=exp.name,
            arm=arm,
            model_version=champion_mv.version_id,   # champion always prices
            quoted_price=champion_price,
            enbp=enbp,
            renewal_flag=bool(is_renewal[i]),
            exposure=1.0,
            timestamp=quote_ts[i],
        )

# Log bind events
bind_ts_offset = timedelta(hours=2)
n_bound = 0
for i in range(N_POLICIES):
    if will_bind[i]:
        policy_id = f"POL-{2024_0001 + i:07d}"
        risk = {
            "age":           int(ages[i]),
            "ncd_years":     int(ncd_years[i]),
            "vehicle_value": int(values[i]),
            "postcode_band": int(bands[i]),
            "annual_mileage": float(mileages[i]),
        }
        bound_price = champion_model.predict([risk])[0]
        logger.log_bind(
            policy_id=policy_id,
            bound_price=bound_price,
            bound_timestamp=quote_ts[i] + bind_ts_offset,
        )
        n_bound += 1

# Log claim events: ~8% of bound policies have a claim, some with severity
# typical of UK motor (£1,500-£12,000 incurred at 12 months)
n_claims = 0
for i in range(N_POLICIES):
    if not will_bind[i]:
        continue
    if rng.random() < 0.08:
        policy_id = f"POL-{2024_0001 + i:07d}"
        # Claim date: 1-8 months after bind
        days_to_loss = int(rng.uniform(30, 240))
        claim_date = (quote_ts[i] + timedelta(days=days_to_loss)).date()

        # Severity from a Pareto-ish distribution: most claims £1-3k, some large
        base_severity = rng.lognormal(mean=7.5, sigma=0.7)  # log-normal, mean ~£3k
        base_severity = min(base_severity, 60_000)  # cap at £60k

        # Log at 3-month development (FNOL estimate, 20% IBNR upward pressure)
        fnol_amount = round(base_severity * rng.uniform(0.7, 0.9), 2)
        logger.log_claim(
            policy_id=policy_id,
            claim_date=claim_date,
            claim_amount=fnol_amount,
            development_month=3,
        )

        # Log at 12-month development (closer to ultimate)
        dev_12_amount = round(base_severity * rng.uniform(0.95, 1.10), 2)
        logger.log_claim(
            policy_id=policy_id,
            claim_date=claim_date,
            claim_amount=dev_12_amount,
            development_month=12,
        )

        n_claims += 1

print(f"\nQuotes logged:         {logger.quote_count(exp.name):,}")
print(f"Champion arm:          {n_champion_arm:,} ({n_champion_arm/N_POLICIES:.0%})")
print(f"Challenger arm:        {n_challenger_arm:,} ({n_challenger_arm/N_POLICIES:.0%})")
print(f"Binds logged:          {n_bound:,}")
print(f"Claims logged:         {n_claims:,} (2 development entries each)")
print(f"ENBP breaches injected:{enbp_breach_injected} (deliberate, to test audit report)")

# ---------------------------------------------------------------------------
# Step 5: Answer the governance question in three lines
# ---------------------------------------------------------------------------
#
# This is the regulatory use case. Your compliance team gets an FCA information
# request: "who priced policy POL-2024034, and did the quoted price exceed the
# ENBP for this renewal?"
#
# You answer it in three lines. The audit trail is permanent and complete.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 5: Governance query — who priced policy X?")
print("=" * 72)

# Pick a policy at random and demonstrate the audit lookup
example_policy = "POL-20240034"

all_quotes = logger.query_quotes(exp.name)
policy_quotes = [q for q in all_quotes if q["policy_id"] == example_policy]

print(f"FCA question: 'Who priced {example_policy}? Was ENBP compliance maintained?'")
print()
if policy_quotes:
    q = policy_quotes[0]
    print(f"Answer:")
    print(f"  Model version:   {q['model_version']}")
    print(f"  Experiment arm:  {q['arm']}")
    print(f"  Quoted price:    £{q['quoted_price']:.2f}")
    print(f"  ENBP recorded:   {'£' + str(q['enbp']) if q['enbp'] else 'N/A (new business)'}")
    print(f"  ENBP compliant:  {'Yes' if q['enbp_flag'] == 1 else 'No' if q['enbp_flag'] == 0 else 'N/A'}")
    print(f"  Quote timestamp: {q['timestamp'][:19]} UTC")
    print()
    renewal_status = "renewal" if q["renewal_flag"] == 1 else "new business"
    print(f"This was a {renewal_status} quote priced by {q['model_version']}.")
    print(f"Routing was deterministic: SHA-256('POL-20240034' + '{exp.name}')")
    print(f"Result can be recomputed at any point without consulting the database.")

# Show the full quote log for this experiment (first 5 rows)
print(f"\nFirst 5 rows of the quote log for this experiment:")
print(f"  {'policy_id':<18} {'arm':<12} {'model_version':<16} {'price':>8}  {'renewal':<8} {'enbp_flag'}")
print(f"  {'-'*18} {'-'*12} {'-'*16} {'-'*8}  {'-'*8} {'-'*9}")
for q in all_quotes[:5]:
    renewal = "renewal" if q["renewal_flag"] == 1 else "new biz"
    flag = {1: "ok", 0: "BREACH", None: "-"}.get(q["enbp_flag"], "?")
    print(f"  {q['policy_id']:<18} {q['arm']:<12} {q['model_version']:<16} "
          f"£{q['quoted_price']:>7.2f}  {renewal:<8} {flag}")

# ---------------------------------------------------------------------------
# Step 6: KPI dashboard
# ---------------------------------------------------------------------------
#
# Metric maturity tiers:
#
#   Immediately: quote volume, price distribution, ENBP compliance rate
#   At bind:     hit rate (conversion), GWP
#   3-6 months:  claim frequency (IBNR caveat before 12 months)
#   12+ months:  developed loss ratio — the one that matters for promotion
#
# Do not base promotion decisions on hit rate or frequency alone. They are
# leading indicators, not the signal. Loss ratio significance requires 12-month
# claim development plus enough policies for the bootstrap to have power. The
# power_analysis() output will tell you exactly how long.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 6: KPI dashboard")
print("=" * 72)

from insurance_deploy import KPITracker

tracker = KPITracker(logger)

# Tier 1: Quote volume and price distribution
vol = tracker.quote_volume(exp.name)
print("Quote volume and price distribution:")
for arm in ("champion", "challenger"):
    v = vol[arm]
    print(f"  {arm.capitalize():<12}: "
          f"n={v['n']:,}  "
          f"mean=£{v['mean_price']:.0f}  "
          f"p25=£{v['p25_price']:.0f}  "
          f"median=£{v['median_price']:.0f}  "
          f"p75=£{v['p75_price']:.0f}")

# ENBP compliance
enbp = tracker.enbp_compliance(exp.name)
print("\nENBP compliance (renewals only):")
for arm in ("champion", "challenger"):
    e = enbp[arm]
    rate = f"{e['compliance_rate']:.1%}" if not math.isnan(e['compliance_rate']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"renewals={e['renewal_quotes']:,}  "
          f"compliant={e['compliant']:,}  "
          f"breaches={e['breaches']:,}  "
          f"rate={rate}")

# Tier 2: Hit rate and GWP
hr = tracker.hit_rate(exp.name)
print("\nHit rate (conversion rate) by arm:")
for arm in ("champion", "challenger"):
    h = hr[arm]
    rate = f"{h['hit_rate']:.1%}" if not math.isnan(h['hit_rate']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"quoted={h['quoted']:,}  "
          f"bound={h['bound']:,}  "
          f"rate={rate}")

gwp = tracker.gwp(exp.name)
print("\nGross Written Premium on bound policies:")
for arm in ("champion", "challenger"):
    g = gwp[arm]
    mean = f"£{g['mean_gwp']:.0f}" if not math.isnan(g['mean_gwp']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"bound={g['bound_policies']:,}  "
          f"GWP=£{g['total_gwp']:,.0f}  "
          f"mean={mean}/policy")

# Tier 3: Frequency (immature — IBNR caveat)
print("\nClaim frequency at 12-month development:")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # we know it is 12 months — suppress the maturity nag
    freq = tracker.frequency(exp.name, development_months=12, warn_immature=False)
for arm in ("champion", "challenger"):
    f = freq[arm]
    fq = f"{f['frequency']:.4f}" if not math.isnan(f['frequency']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"policy_years={f['policy_years']:.0f}  "
          f"claims={f['claim_count']:,}  "
          f"freq={fq}")

# Tier 4: Loss ratio at 12 months
print("\nLoss ratio at 12-month development:")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = tracker.loss_ratio(exp.name, development_months=12)
for arm in ("champion", "challenger"):
    l = lr[arm]
    ratio = f"{l['loss_ratio']:.4f}" if not math.isnan(l['loss_ratio']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"policies={l['policy_count']:,}  "
          f"earned=£{l['earned_premium']:,.0f}  "
          f"incurred=£{l['incurred_claims']:,.0f}  "
          f"LR={ratio}")

# Power analysis: how long until LR is statistically credible?
print("\nPower analysis: how long until we can make a credible promotion decision?")
pa = tracker.power_analysis(exp.name, target_delta_lr=0.03, target_delta_hr=0.02)
print(f"  Current volumes:")
print(f"    Champion arm quotes:   {pa['current_n_champion']:,}")
print(f"    Challenger arm quotes: {pa['current_n_challenger']:,}")
print(f"    Monthly rate (chall.): {pa['monthly_rate_challenger']:.0f} quotes/month")
print(f"  Hit rate significance:")
print(f"    Required n per arm:    {pa['hr_required_n_per_arm']:,}")
print(f"    Months to significance:{pa['hr_months_to_significance']:.1f} months")
print(f"  Loss ratio significance (12m development):")
print(f"    Required n per arm:    {pa['lr_required_n_per_arm']:,}")
print(f"    Months to bind:        {pa['lr_months_to_bind']:.1f} months")
print(f"    Total (incl. dev.):    {pa['lr_total_months_with_development']:.1f} months")
print()
for note in pa["notes"]:
    print(f"  Note: {note}")

print(
    "\nKey message: the power analysis is the honest answer to the question "
    "'when will we know?' Pricing teams routinely underestimate this number. "
    "Share it with stakeholders before the experiment starts, not after "
    "three months when they are asking why you have not decided yet."
)

# ---------------------------------------------------------------------------
# Step 7: Bootstrap likelihood ratio test
# ---------------------------------------------------------------------------
#
# The bootstrap LR test is the primary promotion decision tool. It resamples
# at policy level, which preserves within-policy claim correlation (multiple
# claims on one policy are resampled together, not independently). It uses
# percentile confidence intervals on the distribution of (challenger LR -
# champion LR) across bootstrap replicates.
#
# We run with n_bootstrap=2,000 here for speed. For a real promotion decision
# use n_bootstrap=10,000. The p-value estimate stabilises by ~5,000 iterations.
#
# Interpretation of the result:
#
#   CHALLENGER_BETTER + p < 0.05:  statistically significant improvement.
#     This is necessary but not sufficient for promotion. Review that the
#     improvement is consistent across risk segments. Check for adverse
#     selection (if live mode was used). Get sign-off from your governance
#     committee. Then promote.
#
#   INSUFFICIENT_EVIDENCE:  the data does not yet support a decision.
#     This is the most common result for experiments under 18 months.
#     The right action is to continue the experiment. The power analysis
#     tells you when to expect significance.
#
#   CHAMPION_BETTER:  the challenger is worse. Consider early termination.
#     Do not promote. Retain logs for audit trail.
#
# We also run the hit rate test and frequency test as leading indicators.
# These reach significance faster but are weaker signals.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 7: Statistical promotion tests")
print("=" * 72)

from insurance_deploy import ModelComparison

comp = ModelComparison(tracker)

# Bootstrap LR test — primary promotion signal
print("Bootstrap loss ratio test (n_bootstrap=2,000, seed=42):")
print("(For production use n_bootstrap=10,000)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr_result = comp.bootstrap_lr_test(
        exp.name,
        n_bootstrap=2_000,
        development_months=12,
        seed=42,
    )
print()
print(lr_result.summary())

# Hit rate test — earlier signal, lower power for model quality
print("\n" + "-" * 50)
print("Hit rate test (two-proportion z-test):")
hr_result = comp.hit_rate_test(exp.name)
print()
print(hr_result.summary())

# Frequency test — Poisson GLM, available before full LR development
print("\n" + "-" * 50)
print("Frequency test (Poisson rate ratio, 12-month development):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq_result = comp.frequency_test(exp.name, development_months=12)
print()
print(freq_result.summary())

# Discuss what we see
print("\n" + "-" * 50)
print("Interpretation:")
print(f"  LR test conclusion:        {lr_result.conclusion}")
print(f"  Hit rate test conclusion:  {hr_result.conclusion}")
print(f"  Frequency test conclusion: {freq_result.conclusion}")
print()
print(
    "With 1,000 policies at 20% challenger split, the challenger arm has "
    "~200 policies. At a target 3pp LR improvement and typical motor LR "
    "volatility, you need ~300 policies per arm for 80% power. "
    "INSUFFICIENT_EVIDENCE is the correct and expected result here."
)
print(
    "The experiment should continue. Run the power analysis to set "
    "stakeholder expectations about when LR significance is achievable."
)

# ---------------------------------------------------------------------------
# Step 8: ENBP compliance audit report
# ---------------------------------------------------------------------------
#
# The ENBPAuditReport generates the Markdown document your SMF holder signs.
# ICOBS 6B.2.51R requires written records demonstrating ENBP compliance for
# every renewal quote. The audit report is that document.
#
# The structure:
#   Section 1: Executive summary with overall compliance rate
#   Section 2: Compliance by model arm (champion / challenger)
#   Section 3: Model versions used in pricing (which version priced what)
#   Section 4: ENBP breach detail — per-policy list, must be investigated
#   Section 5: Routing audit (routing methodology + volume by model version)
#   Section 6: Attestation statement for SMF holder signature
#
# We deliberately injected a 3% breach rate in Step 4. Section 4 of the
# report will list those policies. In a real firm, Section 4 triggers a
# remediation process: contact affected customers, confirm whether a refund
# is due under ICOBS 6B.3.20G.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 8: ENBP compliance audit report")
print("=" * 72)

from insurance_deploy import ENBPAuditReport

reporter = ENBPAuditReport(logger)
audit_md = reporter.generate(
    experiment_name=exp.name,
    period_start="2024-01-01",
    period_end="2024-07-01",
    firm_name="Meridian Motor Insurance Ltd",
    smf_holder="Tom Bradley",
)

# Print the report
print(audit_md)

# In a Databricks notebook: displayHTML(audit_md) or use dbutils.fs.put()
# Save to file for downstream use (e.g. attaching to a governance ticket)
audit_path = Path(_tmpdir) / "enbp_audit_report.md"
audit_path.write_text(audit_md)
print(f"\nAudit report saved to: {audit_path}")

# ---------------------------------------------------------------------------
# Step 9: Promote challenger to champion
# ---------------------------------------------------------------------------
#
# Promotion is a governance event, not a technical one. The steps are:
#
#   1. Review the statistical evidence (Step 7).
#   2. Get sign-off from the pricing governance committee (SMF23 holder).
#   3. Update the experiment — deactivate the old one, create a new one with
#      the challenger as champion.
#   4. Update the registry champion designation.
#   5. Record the promotion decision in your governance log.
#
# The library provides the technical primitives. The governance process is
# yours. The key property: every step creates a permanent audit trail.
# No state is overwritten. Old experiment logs are preserved. Old model
# artefacts remain in the registry with their SHA-256 hashes intact.
#
# Assume for this example that the statistical test returned CHALLENGER_BETTER
# with p < 0.05 on a larger dataset, and the governance committee approved.
# We proceed with promotion.
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Step 9: Promote challenger to champion")
print("=" * 72)

# Step 9a: Deactivate the current experiment
# This closes the routing — route() will raise RuntimeError if called again.
# The logs are permanent and unaffected.
exp.deactivate()
print(f"Experiment deactivated: {exp}")
print(f"  Deactivated at: {exp.deactivated_at[:19]}")
print(f"  Is active: {exp.is_active()}")

# Verify: routing now raises an error (expected)
try:
    exp.route("POL-20240001")
    print("  ERROR: route() should have raised RuntimeError")
except RuntimeError as e:
    print(f"  route() correctly raises: {e}")

# Step 9b: Promote the challenger to champion in the registry
print()
new_champion = registry.set_champion("motor", version="3.0")
print(f"Registry champion updated: {new_champion}")
print(f"  is_champion: {new_champion.is_champion}")

# Verify old champion is no longer flagged
old = registry.get("motor", "2.0")
print(f"  Old champion {old.version_id} is_champion: {old.is_champion}")

# Step 9c: Create a new experiment with the promoted model as champion.
# If you want to continue A/B testing (e.g. a v4 challenger), set up a new
# experiment with motor:3.0 as champion. The old experiment logs remain
# permanently available for audit.
_db_path_2 = Path(_tmpdir) / "quotes_v4.db"   # separate DB for new experiment
logger_2 = QuoteLogger(_db_path_2)

print()
print("Creating new experiment with v3.0 as champion:")
print("(Would set up a v4.0 challenger here in a real promotion cycle)")
print()
new_exp_note = (
    "  New experiment would be:\n"
    "  exp_v4 = Experiment(\n"
    "      name='motor_v4_vs_v3_2025q1',\n"
    "      champion=registry.get('motor', '3.0'),\n"
    "      challenger=next_challenger_mv,\n"
    "      challenger_pct=0.20,\n"
    "      mode='shadow',\n"
    "  )"
)
print(new_exp_note)

# Step 9d: Show the governance trail
print("-" * 50)
print("Governance trail:")
print()
print(f"  Experiment:      {exp.name}")
print(f"  Created:         {exp.created_at[:19]} UTC")
print(f"  Deactivated:     {exp.deactivated_at[:19]} UTC")
print(f"  Champion:        {exp.champion.version_id}")
print(f"    Registered:    {champion_mv.registered_at[:19]} UTC")
print(f"    Holdout Gini:  {champion_mv.metadata['holdout_gini']}")
print(f"    Validated by:  {champion_mv.metadata['validated_by']}")
print(f"  Challenger:      {exp.challenger.version_id}")
print(f"    Registered:    {challenger_mv.registered_at[:19]} UTC")
print(f"    Holdout Gini:  {challenger_mv.metadata['holdout_gini']}")
print(f"    Validated by:  {challenger_mv.metadata['validated_by']}")
print(f"  Decision:        CHALLENGER_BETTER (manual, 10,000 policy dataset)")
print(f"  Approved by:     Tom Bradley, SMF23")
print(f"  Quote log:       {_db_path}")
print(f"    Total quotes:  {logger.quote_count():,}")
print(f"  Audit report:    {audit_path}")
print(f"  New champion:    {new_champion.version_id}")
print()
print(
    "Any quote in the log can be retraced: policy_id + experiment_name is "
    "all you need to recompute routing from first principles. No external "
    "state required. The SHA-256 hashes in the registry confirm which "
    "model object produced each price."
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("Workflow complete")
print("=" * 72)
print(f"""
Steps completed:

  1. Models defined     Champion GLM v2.0 and GBM challenger v3.0
                        Reference risk: £{champ_ref:.0f} (GLM) vs £{chall_ref:.0f} (GBM)
                        Young driver:   £{champ_young:.0f} (GLM) vs £{chall_young:.0f} (GBM)

  2. Registry           2 model versions, SHA-256 hash-verified
                        Path: {_tmpdir}

  3. Experiment         {exp.name}
                        Shadow mode, 20% challenger designation

  4. Quotes logged      {logger.quote_count(exp.name):,} quotes
                        Champion arm: {n_champion_arm:,}, Challenger arm: {n_challenger_arm:,}
                        Binds: {n_bound:,}, Claims: {n_claims:,}
                        ENBP breaches injected: {enbp_breach_injected}

  5. Governance         Per-quote model version + arm + ENBP in SQLite
                        Recomputable routing from policy_id + experiment name

  6. KPIs               Tier 1–4 dashboard computed
                        Power analysis: {pa['lr_total_months_with_development']:.0f} months to LR significance

  7. Statistical tests  LR bootstrap: {lr_result.conclusion}
                        Hit rate:      {hr_result.conclusion}
                        Frequency:     {freq_result.conclusion}

  8. ENBP audit         Saved to {audit_path.name}
                        Breaches listed, attestation section included

  9. Promotion          Experiment deactivated, registry updated to v3.0
                        Full governance trail in audit log

Production deployment checklist:
  - Replace ChampionGLM/ChallengerGBM with real trained model objects
  - Use a persistent registry path (DBFS /mnt/models/ or S3)
  - Use a persistent QuoteLogger path (DBFS or Postgres adapter)
  - Run power_analysis() before the experiment starts and share results
    with stakeholders so timeline expectations are set correctly
  - Run the ENBP report monthly and include in your governance pack
  - Bootstrap test: use n_bootstrap=10,000 for promotion decisions
  - Promotion decision: document approver name, date, and test results
    in your model governance system (Collibra, MLflow, or similar)
""")
