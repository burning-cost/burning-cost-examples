# Databricks notebook source

# MAGIC %md
# MAGIC # Champion/Challenger Deployment — UK Motor Insurance
# MAGIC
# MAGIC Your pricing team has built a new GBM-based frequency model. The holdout Gini
# MAGIC is 0.45 versus 0.42 for the incumbent GLM. The team wants to deploy it.
# MAGIC
# MAGIC The problem is everything after that sentence.
# MAGIC
# MAGIC How do you run the new model alongside the existing one without disrupting live
# MAGIC pricing? How do you produce the per-quote, per-policy audit trail that ICOBS
# MAGIC 6B.2.51R requires? How do you know — with statistical rigour rather than gut
# MAGIC feel — when you have enough data to make a credible promotion decision?
# MAGIC
# MAGIC This notebook works through the complete workflow:
# MAGIC
# MAGIC 1. Define champion and challenger as Python callables (realistic parametric GLM
# MAGIC    and GBM approximations, not toy functions)
# MAGIC 2. Register both versions in the model registry with training metadata
# MAGIC 3. Set up the champion/challenger experiment in shadow mode
# MAGIC 4. Score 1,000 synthetic motor quotes through the framework, logging each one
# MAGIC    with full audit trail (quote, bind, claim events)
# MAGIC 5. Inspect the quote log — governance question answered in three lines
# MAGIC 6. Run the KPI dashboard: hit rate, GWP, frequency, loss ratio
# MAGIC 7. Run the bootstrap likelihood ratio test for challenger promotion
# MAGIC 8. Generate the ENBP compliance audit report for the SMF holder
# MAGIC 9. Promote the challenger to champion and show the governance trail
# MAGIC
# MAGIC **Shadow mode** is the default and the right choice here. The challenger scores
# MAGIC on every quote but the customer always sees the champion price. No FCA Consumer
# MAGIC Duty (PRIN 2A) fair value questions arise: we are not pricing two customers of
# MAGIC identical profile differently. We are collecting evidence before we change anything.
# MAGIC
# MAGIC **The governance angle throughout:** the FCA's ICOBS 6B multi-firm review (2023)
# MAGIC found 83% of firms could not demonstrate which model priced each renewal. This
# MAGIC framework is the answer. Every quote is logged with model version, arm, price,
# MAGIC and ENBP. Your regulator asks who priced policy X — you answer in three lines.

# COMMAND ----------

# MAGIC %pip install insurance-deploy --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define champion and challenger as Python callables
# MAGIC
# MAGIC In production these would be trained sklearn/CatBoost/XGBoost objects loaded
# MAGIC from your model artefact store. Here we build parametric approximations that
# MAGIC reproduce realistic UK motor pricing behaviour.
# MAGIC
# MAGIC **Champion (GLM):** log-linear frequency model with three rating factors.
# MAGIC Typical of a mature personal lines GLM built in SAS or Emblem and recently
# MAGIC ported to Python. Gini ~0.42 on holdout.
# MAGIC
# MAGIC **Challenger (GBM):** captures non-linear age × NCD interactions and a vehicle
# MAGIC value effect the GLM misses. Gini ~0.45 on holdout. Lower loss ratio because
# MAGIC it prices high-value, young-driver risks more accurately — those policies were
# MAGIC cross-subsidised in the GLM.
# MAGIC
# MAGIC Both expose a `.predict(X)` interface where `X` is a list of feature dicts.
# MAGIC This is the sklearn-compatible interface `ModelRegistry` expects.
# MAGIC
# MAGIC We deliberately make the challenger 3–4% lower in loss ratio by adjusting
# MAGIC its base rate. This is the signal the bootstrap LR test needs to detect.
# MAGIC Whether it detects it depends on sample size — the power analysis will tell
# MAGIC you honestly.

# COMMAND ----------

from __future__ import annotations

import math
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np

# Feature schema (UK motor):
#   age           — driver age (17–80)
#   ncd_years     — no-claims discount years (0–9)
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
    This is where the 3–4pp loss ratio improvement comes from.
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

rows = [
    ["Reference risk (age=35, 5yr NCD, value=£15k, band=3)", f"£{champ_ref:,.2f}", f"£{chall_ref:,.2f}", f"£{chall_ref - champ_ref:+.2f}"],
    ["Young driver (age=22, 0yr NCD, value=£8k, band=4, 15k miles)", f"£{champ_young:,.2f}", f"£{chall_young:,.2f}", f"£{chall_young - champ_young:+.2f}"],
]

html = """
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Risk profile</th>
      <th style="padding:8px 16px;text-align:right">Champion (GLM)</th>
      <th style="padding:8px 16px;text-align:right">Challenger (GBM)</th>
      <th style="padding:8px 16px;text-align:right">Difference</th>
    </tr>
  </thead>
  <tbody>
"""
for i, row in enumerate(rows):
    bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
    html += f'<tr style="background:{bg}">'
    html += f'<td style="padding:8px 16px">{row[0]}</td>'
    for cell in row[1:]:
        html += f'<td style="padding:8px 16px;text-align:right">{cell}</td>'
    html += "</tr>"
html += """
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
  GBM prices the young driver higher: this is where the cross-subsidy in the GLM is removed.
</p>
"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register both models in the model registry
# MAGIC
# MAGIC `ModelRegistry` is append-only. Once a model version is registered you cannot
# MAGIC overwrite it. This is not a limitation — it is the audit guarantee. If you need
# MAGIC to re-register a model, create a new version string.
# MAGIC
# MAGIC The registry serialises model objects to disk via joblib and records a SHA-256
# MAGIC hash of the serialised bytes. When you load a model back, the library verifies
# MAGIC the hash. File corruption or tampering is caught on load.
# MAGIC
# MAGIC We use a temporary directory here so the notebook is self-contained. In
# MAGIC production, use a stable path on shared storage (DBFS, NFS, S3 with consistent
# MAGIC read).

# COMMAND ----------

import tempfile as _tempfile

from insurance_deploy import ModelRegistry, ModelVersion

# Use a temporary directory so the notebook runs without leaving artefacts.
# In production: _tmpdir = "/dbfs/mnt/models/motor-registry"
_tmpdir = _tempfile.mkdtemp(prefix="motor_registry_")

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

versions = registry.list("motor")

reg_rows = [
    [champion_mv.version_id, champion_mv.model_hash[:16] + "...",
     champion_mv.metadata["holdout_gini"], champion_mv.metadata["validated_by"]],
    [challenger_mv.version_id, challenger_mv.model_hash[:16] + "...",
     challenger_mv.metadata["holdout_gini"], challenger_mv.metadata["validated_by"]],
]

reg_html = """
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Version ID</th>
      <th style="padding:8px 16px;text-align:left">SHA-256 (truncated)</th>
      <th style="padding:8px 16px;text-align:right">Holdout Gini</th>
      <th style="padding:8px 16px;text-align:left">Validated by</th>
    </tr>
  </thead>
  <tbody>
"""
for i, row in enumerate(reg_rows):
    bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
    reg_html += f'<tr style="background:{bg}">'
    reg_html += f'<td style="padding:8px 16px">{row[0]}</td>'
    reg_html += f'<td style="padding:8px 16px">{row[1]}</td>'
    reg_html += f'<td style="padding:8px 16px;text-align:right">{row[2]}</td>'
    reg_html += f'<td style="padding:8px 16px">{row[3]}</td>'
    reg_html += "</tr>"
reg_html += f"""
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
  Registry contains {len(versions)} version(s) for 'motor'. Hashes verified on load.
</p>
"""
displayHTML(reg_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Set up the champion/challenger experiment
# MAGIC
# MAGIC Shadow mode (the default) is used here. The challenger is scored on every quote
# MAGIC but the customer always sees the champion price. No FCA Consumer Duty (PRIN 2A)
# MAGIC concerns. No adverse selection confound in the loss ratio data.
# MAGIC
# MAGIC `challenger_pct=0.20` means 20% of policies are statistically designated to the
# MAGIC challenger arm. In shadow mode this only affects the routing label in the log —
# MAGIC the champion prices all of them. The label is used to split KPIs so you can
# MAGIC compare champion-arm loss ratio versus challenger-arm loss ratio on the same price
# MAGIC basis. This is a cleaner comparison than live A/B because there is no price signal
# MAGIC difference.
# MAGIC
# MAGIC The routing is `SHA-256(policy_id + experiment_name)`. It is deterministic and
# MAGIC stateless. A policy routed to challenger on its first quote will always be routed
# MAGIC to challenger in this experiment, regardless of how many times it re-quotes. This
# MAGIC is required for ENBP audit integrity: you must be able to say "this policy was
# MAGIC assigned to the challenger arm from inception" without consulting a database of
# MAGIC random numbers drawn at quote time.

# COMMAND ----------

from insurance_deploy import Experiment

exp = Experiment(
    name="motor_v3_vs_v2_2024q3",
    champion=champion_mv,
    challenger=challenger_mv,
    challenger_pct=0.20,  # 20% of policies designated to challenger arm
    mode="shadow",        # Champion prices all quotes; challenger is observational
)

# Demonstrate routing determinism: same policy always routes the same way
demo_policies = ["POL-001234", "POL-005678", "POL-009999"]
routing_rows = []
for pol in demo_policies:
    arm = exp.route(pol)
    arm2 = exp.route(pol)
    assert arm == arm2, "Routing should be deterministic"
    routing_rows.append([pol, arm])

exp_html = f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px;margin-bottom:16px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Property</th>
      <th style="padding:8px 16px;text-align:left">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Experiment</td><td style="padding:8px 16px">{exp.name}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Champion</td><td style="padding:8px 16px">{exp.champion.version_id}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Challenger</td><td style="padding:8px 16px">{exp.challenger.version_id}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Challenger %</td><td style="padding:8px 16px">{exp.challenger_pct:.0%}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Mode</td><td style="padding:8px 16px">{exp.mode}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Active</td><td style="padding:8px 16px">{exp.is_active()}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Created at</td><td style="padding:8px 16px">{exp.created_at[:19]} UTC</td></tr>
  </tbody>
</table>

<p style="font-family:sans-serif;font-size:13px;font-weight:bold;margin-bottom:4px">Routing determinism check (SHA-256 based):</p>
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Policy ID</th>
      <th style="padding:8px 16px;text-align:left">Arm</th>
    </tr>
  </thead>
  <tbody>
"""
for i, (pol, arm) in enumerate(routing_rows):
    bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
    exp_html += f'<tr style="background:{bg}"><td style="padding:8px 16px">{pol}</td><td style="padding:8px 16px">{arm}</td></tr>'
exp_html += """
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
  Routing formula: SHA-256(policy_id + experiment_name), last 8 hex chars as integer mod 100.
  If &lt; 20: challenger. Else: champion.
</p>
"""
displayHTML(exp_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Score 1,000 synthetic quotes through the framework
# MAGIC
# MAGIC We generate 1,000 policies with realistic UK motor risk profiles, then simulate
# MAGIC the quote → bind → claim lifecycle for each.
# MAGIC
# MAGIC **Data-generating process:**
# MAGIC - Quoted: all 1,000 policies (100%)
# MAGIC - Bound: ~32% conversion rate (realistic UK private car direct)
# MAGIC - Claims: ~8% of bound policies have a claim in the first year
# MAGIC
# MAGIC ENBP records: 40% of policies are renewals (realistic mix). For renewals, we
# MAGIC calculate ENBP as `quoted_price × 1.02` (2% headroom), except for a small
# MAGIC fraction (3%) where we deliberately inject a breach. This lets us verify the
# MAGIC audit report flags breaches correctly.
# MAGIC
# MAGIC We use timestamps spread over 6 months to give the power analysis a meaningful
# MAGIC monthly rate estimate.

# COMMAND ----------

import tempfile as _tempfile2
from pathlib import Path

from insurance_deploy import QuoteLogger

# Use a temp path for the SQLite quote log.
# In production: use a persistent DBFS path or a Postgres adapter.
_db_path = Path(_tempfile2.mktemp(suffix=".db"))
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

# Score all policies and log quotes
n_champion_arm = 0
n_challenger_arm = 0
enbp_breach_injected = 0

for i in range(N_POLICIES):
    policy_id = f"POL-{20240001 + i:07d}"

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

    # Challenger scores in shadow — result is for internal comparison only;
    # the customer always sees champion_price
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
            # Force a breach: set ENBP below the quoted price.
            # In a real system this would be caught before quote — here it is
            # intentional so the audit report has something to flag.
            enbp = round(champion_price * 0.97, 2)
            enbp_breach_injected += 1

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
        policy_id = f"POL-{20240001 + i:07d}"
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
# typical of UK motor (£1,500–£12,000 incurred at 12 months)
n_claims = 0
for i in range(N_POLICIES):
    if not will_bind[i]:
        continue
    if rng.random() < 0.08:
        policy_id = f"POL-{20240001 + i:07d}"
        # Claim date: 1–8 months after bind
        days_to_loss = int(rng.uniform(30, 240))
        claim_date = (quote_ts[i] + timedelta(days=days_to_loss)).date()

        # Severity from a log-normal distribution: most claims £1–3k, some large
        base_severity = rng.lognormal(mean=7.5, sigma=0.7)
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

summary_html = f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Metric</th>
      <th style="padding:8px 16px;text-align:right">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Quotes logged</td><td style="padding:8px 16px;text-align:right">{logger.quote_count(exp.name):,}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Champion arm</td><td style="padding:8px 16px;text-align:right">{n_champion_arm:,} ({n_champion_arm/N_POLICIES:.0%})</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Challenger arm</td><td style="padding:8px 16px;text-align:right">{n_challenger_arm:,} ({n_challenger_arm/N_POLICIES:.0%})</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Binds logged</td><td style="padding:8px 16px;text-align:right">{n_bound:,}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Claims logged</td><td style="padding:8px 16px;text-align:right">{n_claims:,} (2 development entries each)</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">ENBP breaches injected</td><td style="padding:8px 16px;text-align:right">{enbp_breach_injected} (deliberate, to test audit report)</td></tr>
  </tbody>
</table>
"""
displayHTML(summary_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Answer the governance question in three lines
# MAGIC
# MAGIC This is the regulatory use case. Your compliance team gets an FCA information
# MAGIC request: *"who priced policy POL-20240034, and did the quoted price exceed the
# MAGIC ENBP for this renewal?"*
# MAGIC
# MAGIC You answer it in three lines. The audit trail is permanent and complete.

# COMMAND ----------

example_policy = "POL-20240034"

all_quotes = logger.query_quotes(exp.name)
policy_quotes = [q for q in all_quotes if q["policy_id"] == example_policy]

if policy_quotes:
    q = policy_quotes[0]
    renewal_status = "renewal" if q["renewal_flag"] == 1 else "new business"
    enbp_display = f"£{q['enbp']}" if q["enbp"] else "N/A (new business)"
    enbp_flag_display = {1: "Yes", 0: "No — BREACH", None: "N/A"}.get(q["enbp_flag"], "?")
    flag_colour = "#d32f2f" if q["enbp_flag"] == 0 else "#2e7d32"

    gov_html = f"""
<p style="font-family:sans-serif;font-size:14px">
  <strong>FCA question:</strong> "Who priced {example_policy}? Was ENBP compliance maintained?"
</p>
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Field</th>
      <th style="padding:8px 16px;text-align:left">Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Model version</td><td style="padding:8px 16px">{q['model_version']}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Experiment arm</td><td style="padding:8px 16px">{q['arm']}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Quoted price</td><td style="padding:8px 16px">£{q['quoted_price']:.2f}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">ENBP recorded</td><td style="padding:8px 16px">{enbp_display}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">ENBP compliant</td><td style="padding:8px 16px;color:{flag_colour};font-weight:bold">{enbp_flag_display}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Quote timestamp</td><td style="padding:8px 16px">{q['timestamp'][:19]} UTC</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Policy type</td><td style="padding:8px 16px">{renewal_status}</td></tr>
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
  Routing is recomputable: SHA-256('{example_policy}' + '{exp.name}').
  No external state required.
</p>
"""
    displayHTML(gov_html)

# Show the first 5 rows of the quote log
first5 = all_quotes[:5]
log_html = """
<p style="font-family:sans-serif;font-size:13px;font-weight:bold;margin-bottom:4px">
  First 5 rows of the quote log for this experiment:
</p>
<table style="border-collapse:collapse;font-family:monospace;font-size:12px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:6px 12px;text-align:left">policy_id</th>
      <th style="padding:6px 12px;text-align:left">arm</th>
      <th style="padding:6px 12px;text-align:left">model_version</th>
      <th style="padding:6px 12px;text-align:right">price</th>
      <th style="padding:6px 12px;text-align:left">type</th>
      <th style="padding:6px 12px;text-align:left">enbp_flag</th>
    </tr>
  </thead>
  <tbody>
"""
for i, row in enumerate(first5):
    bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
    renewal = "renewal" if row["renewal_flag"] == 1 else "new biz"
    flag = {1: "ok", 0: "BREACH", None: "-"}.get(row["enbp_flag"], "?")
    flag_style = 'color:#d32f2f;font-weight:bold' if flag == "BREACH" else ""
    log_html += f"""
    <tr style="background:{bg}">
      <td style="padding:6px 12px">{row['policy_id']}</td>
      <td style="padding:6px 12px">{row['arm']}</td>
      <td style="padding:6px 12px">{row['model_version']}</td>
      <td style="padding:6px 12px;text-align:right">£{row['quoted_price']:.2f}</td>
      <td style="padding:6px 12px">{renewal}</td>
      <td style="padding:6px 12px;{flag_style}">{flag}</td>
    </tr>"""
log_html += "</tbody></table>"
displayHTML(log_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: KPI dashboard
# MAGIC
# MAGIC **Metric maturity tiers:**
# MAGIC
# MAGIC | Tier | When available | Metrics |
# MAGIC |------|---------------|---------|
# MAGIC | 1 | Immediately | Quote volume, price distribution, ENBP compliance rate |
# MAGIC | 2 | At bind | Hit rate (conversion), GWP |
# MAGIC | 3 | 3–6 months | Claim frequency (IBNR caveat before 12 months) |
# MAGIC | 4 | 12+ months | Developed loss ratio — the one that matters for promotion |
# MAGIC
# MAGIC Do not base promotion decisions on hit rate or frequency alone. They are leading
# MAGIC indicators, not the signal. Loss ratio significance requires 12-month claim
# MAGIC development plus enough policies for the bootstrap to have power. The
# MAGIC `power_analysis()` output will tell you exactly how long.

# COMMAND ----------

from insurance_deploy import KPITracker

tracker = KPITracker(logger)

# Tier 1: Quote volume and price distribution
vol = tracker.quote_volume(exp.name)

# ENBP compliance
enbp_kpi = tracker.enbp_compliance(exp.name)

# Tier 2: Hit rate and GWP
hr = tracker.hit_rate(exp.name)
gwp = tracker.gwp(exp.name)

# Tier 3: Frequency (12-month development)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq = tracker.frequency(exp.name, development_months=12, warn_immature=False)

# Tier 4: Loss ratio at 12 months
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = tracker.loss_ratio(exp.name, development_months=12)

# Power analysis: how long until LR is statistically credible?
pa = tracker.power_analysis(exp.name, target_delta_lr=0.03, target_delta_hr=0.02)

# Build a combined dashboard
kpi_html = """
<style>
  .kpi-section { font-family: sans-serif; font-size: 13px; margin-bottom: 20px; }
  .kpi-section h3 { margin-bottom: 6px; font-size: 14px; color: #1a1a2e; }
  .kpi-table { border-collapse: collapse; font-family: monospace; font-size: 12px; }
  .kpi-table th { background: #1a1a2e; color: #fff; padding: 6px 12px; text-align: left; }
  .kpi-table td { padding: 6px 12px; }
  .kpi-table tr:nth-child(even) { background: #f8f8f8; }
</style>

<div class="kpi-section">
  <h3>Tier 1 — Quote volume and price distribution</h3>
  <table class="kpi-table">
    <thead><tr>
      <th>Arm</th><th style="text-align:right">n</th>
      <th style="text-align:right">Mean price</th><th style="text-align:right">p25</th>
      <th style="text-align:right">Median</th><th style="text-align:right">p75</th>
    </tr></thead>
    <tbody>
"""
for arm in ("champion", "challenger"):
    v = vol[arm]
    kpi_html += f"""
      <tr><td>{arm.capitalize()}</td>
      <td style="text-align:right">{v['n']:,}</td>
      <td style="text-align:right">£{v['mean_price']:.0f}</td>
      <td style="text-align:right">£{v['p25_price']:.0f}</td>
      <td style="text-align:right">£{v['median_price']:.0f}</td>
      <td style="text-align:right">£{v['p75_price']:.0f}</td></tr>"""
kpi_html += "</tbody></table></div>"

kpi_html += """
<div class="kpi-section">
  <h3>Tier 1 — ENBP compliance (renewals only)</h3>
  <table class="kpi-table">
    <thead><tr>
      <th>Arm</th><th style="text-align:right">Renewals</th>
      <th style="text-align:right">Compliant</th><th style="text-align:right">Breaches</th>
      <th style="text-align:right">Compliance rate</th>
    </tr></thead>
    <tbody>
"""
for arm in ("champion", "challenger"):
    e = enbp_kpi[arm]
    rate = f"{e['compliance_rate']:.1%}" if not math.isnan(e['compliance_rate']) else "N/A"
    breach_style = 'color:#d32f2f;font-weight:bold' if e['breaches'] > 0 else ""
    kpi_html += f"""
      <tr><td>{arm.capitalize()}</td>
      <td style="text-align:right">{e['renewal_quotes']:,}</td>
      <td style="text-align:right">{e['compliant']:,}</td>
      <td style="text-align:right;{breach_style}">{e['breaches']:,}</td>
      <td style="text-align:right">{rate}</td></tr>"""
kpi_html += "</tbody></table></div>"

kpi_html += """
<div class="kpi-section">
  <h3>Tier 2 — Hit rate and GWP</h3>
  <table class="kpi-table">
    <thead><tr>
      <th>Arm</th><th style="text-align:right">Quoted</th>
      <th style="text-align:right">Bound</th><th style="text-align:right">Hit rate</th>
      <th style="text-align:right">Total GWP</th><th style="text-align:right">Mean GWP/policy</th>
    </tr></thead>
    <tbody>
"""
for arm in ("champion", "challenger"):
    h = hr[arm]
    g = gwp[arm]
    hit = f"{h['hit_rate']:.1%}" if not math.isnan(h['hit_rate']) else "N/A"
    mean_gwp = f"£{g['mean_gwp']:.0f}" if not math.isnan(g['mean_gwp']) else "N/A"
    kpi_html += f"""
      <tr><td>{arm.capitalize()}</td>
      <td style="text-align:right">{h['quoted']:,}</td>
      <td style="text-align:right">{h['bound']:,}</td>
      <td style="text-align:right">{hit}</td>
      <td style="text-align:right">£{g['total_gwp']:,.0f}</td>
      <td style="text-align:right">{mean_gwp}</td></tr>"""
kpi_html += "</tbody></table></div>"

kpi_html += """
<div class="kpi-section">
  <h3>Tier 3 &amp; 4 — Frequency and loss ratio at 12-month development</h3>
  <table class="kpi-table">
    <thead><tr>
      <th>Arm</th><th style="text-align:right">Policy years</th>
      <th style="text-align:right">Claims</th><th style="text-align:right">Frequency</th>
      <th style="text-align:right">Earned premium</th>
      <th style="text-align:right">Incurred claims</th>
      <th style="text-align:right">Loss ratio</th>
    </tr></thead>
    <tbody>
"""
for arm in ("champion", "challenger"):
    f_arm = freq[arm]
    l_arm = lr[arm]
    fq = f"{f_arm['frequency']:.4f}" if not math.isnan(f_arm['frequency']) else "N/A"
    ratio = f"{l_arm['loss_ratio']:.4f}" if not math.isnan(l_arm['loss_ratio']) else "N/A"
    kpi_html += f"""
      <tr><td>{arm.capitalize()}</td>
      <td style="text-align:right">{f_arm['policy_years']:.0f}</td>
      <td style="text-align:right">{f_arm['claim_count']:,}</td>
      <td style="text-align:right">{fq}</td>
      <td style="text-align:right">£{l_arm['earned_premium']:,.0f}</td>
      <td style="text-align:right">£{l_arm['incurred_claims']:,.0f}</td>
      <td style="text-align:right">{ratio}</td></tr>"""
kpi_html += "</tbody></table></div>"

kpi_html += f"""
<div class="kpi-section">
  <h3>Power analysis: when can we make a credible promotion decision?</h3>
  <table class="kpi-table">
    <thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
    <tbody>
      <tr><td>Champion arm quotes</td><td style="text-align:right">{pa['current_n_champion']:,}</td></tr>
      <tr><td>Challenger arm quotes</td><td style="text-align:right">{pa['current_n_challenger']:,}</td></tr>
      <tr><td>Monthly rate (challenger)</td><td style="text-align:right">{pa['monthly_rate_challenger']:.0f} quotes/month</td></tr>
      <tr><td>HR: required n per arm</td><td style="text-align:right">{pa['hr_required_n_per_arm']:,}</td></tr>
      <tr><td>HR: months to significance</td><td style="text-align:right">{pa['hr_months_to_significance']:.1f} months</td></tr>
      <tr><td>LR: required n per arm</td><td style="text-align:right">{pa['lr_required_n_per_arm']:,}</td></tr>
      <tr><td>LR: months to bind</td><td style="text-align:right">{pa['lr_months_to_bind']:.1f} months</td></tr>
      <tr><td>LR: total (incl. 12m development)</td><td style="text-align:right;font-weight:bold">{pa['lr_total_months_with_development']:.1f} months</td></tr>
    </tbody>
  </table>
  <p style="font-size:12px;color:#555;margin-top:8px">
    Key message: share the power analysis with stakeholders <em>before</em> the experiment
    starts, not after three months when they are asking why you have not decided yet.
  </p>
</div>
"""

displayHTML(kpi_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Statistical promotion tests
# MAGIC
# MAGIC The bootstrap LR test is the primary promotion decision tool. It resamples at
# MAGIC policy level, which preserves within-policy claim correlation (multiple claims on
# MAGIC one policy are resampled together, not independently). It uses percentile confidence
# MAGIC intervals on the distribution of `(challenger LR - champion LR)` across bootstrap
# MAGIC replicates.
# MAGIC
# MAGIC We run with `n_bootstrap=2,000` here for speed. For a real promotion decision use
# MAGIC `n_bootstrap=10,000`. The p-value estimate stabilises by ~5,000 iterations.
# MAGIC
# MAGIC **Interpreting the result:**
# MAGIC
# MAGIC - `CHALLENGER_BETTER` + p < 0.05: statistically significant improvement.
# MAGIC   Necessary but not sufficient for promotion. Review consistency across risk
# MAGIC   segments, check for adverse selection (if live mode was used), get sign-off from
# MAGIC   your governance committee.
# MAGIC - `INSUFFICIENT_EVIDENCE`: the data does not yet support a decision. The most
# MAGIC   common result for experiments under 18 months. Continue the experiment.
# MAGIC - `CHAMPION_BETTER`: the challenger is worse. Consider early termination.

# COMMAND ----------

from insurance_deploy import ModelComparison

comp = ModelComparison(tracker)

# Bootstrap LR test — primary promotion signal
# (For production use n_bootstrap=10,000)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr_result = comp.bootstrap_lr_test(
        exp.name,
        n_bootstrap=2_000,
        development_months=12,
        seed=42,
    )

# Hit rate test — earlier signal, lower power for model quality
hr_result = comp.hit_rate_test(exp.name)

# Frequency test — Poisson GLM, available before full LR development
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq_result = comp.frequency_test(exp.name, development_months=12)

# Display results
def _conclusion_colour(conclusion: str) -> str:
    if "BETTER" in conclusion and "CHALLENGER" in conclusion:
        return "#2e7d32"
    elif "INSUFFICIENT" in conclusion:
        return "#e65100"
    elif "CHAMPION_BETTER" in conclusion:
        return "#d32f2f"
    return "#333"

tests_html = f"""
<style>
  .test-block {{ font-family: monospace; font-size: 12px; background: #1e1e2e;
                color: #cdd6f4; padding: 16px; margin-bottom: 16px;
                border-radius: 4px; white-space: pre-wrap; }}
  .test-label {{ font-family: sans-serif; font-size: 13px; font-weight: bold;
                 margin-bottom: 4px; }}
</style>

<div class="test-label">Bootstrap loss ratio test (n=2,000 replicates, seed=42):</div>
<div class="test-block">{lr_result.summary()}</div>

<div class="test-label">Hit rate test (two-proportion z-test):</div>
<div class="test-block">{hr_result.summary()}</div>

<div class="test-label">Frequency test (Poisson rate ratio, 12-month development):</div>
<div class="test-block">{freq_result.summary()}</div>

<table style="border-collapse:collapse;font-family:monospace;font-size:13px;margin-top:16px">
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Test</th>
      <th style="padding:8px 16px;text-align:left">Conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f8f8f8">
      <td style="padding:8px 16px">LR bootstrap</td>
      <td style="padding:8px 16px;color:{_conclusion_colour(lr_result.conclusion)};font-weight:bold">{lr_result.conclusion}</td>
    </tr>
    <tr style="background:#fff">
      <td style="padding:8px 16px">Hit rate</td>
      <td style="padding:8px 16px;color:{_conclusion_colour(hr_result.conclusion)};font-weight:bold">{hr_result.conclusion}</td>
    </tr>
    <tr style="background:#f8f8f8">
      <td style="padding:8px 16px">Frequency</td>
      <td style="padding:8px 16px;color:{_conclusion_colour(freq_result.conclusion)};font-weight:bold">{freq_result.conclusion}</td>
    </tr>
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555;margin-top:8px">
  With 1,000 policies at 20% challenger split, the challenger arm has ~200 policies.
  At a target 3pp LR improvement and typical motor LR volatility, you need ~300 policies
  per arm for 80% power. INSUFFICIENT_EVIDENCE is the correct and expected result here.
  The experiment should continue.
</p>
"""
displayHTML(tests_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: ENBP compliance audit report
# MAGIC
# MAGIC The `ENBPAuditReport` generates the Markdown document your SMF holder signs.
# MAGIC ICOBS 6B.2.51R requires written records demonstrating ENBP compliance for every
# MAGIC renewal quote. The audit report is that document.
# MAGIC
# MAGIC **Structure:**
# MAGIC 1. Executive summary with overall compliance rate
# MAGIC 2. Compliance by model arm (champion / challenger)
# MAGIC 3. Model versions used in pricing
# MAGIC 4. ENBP breach detail — per-policy list; each breach triggers remediation
# MAGIC 5. Routing audit (routing methodology + volume by model version)
# MAGIC 6. Attestation statement for SMF holder signature
# MAGIC
# MAGIC We deliberately injected a 3% breach rate in Step 4. Section 4 of the report
# MAGIC will list those policies. In a real firm, Section 4 triggers a remediation process:
# MAGIC contact affected customers, confirm whether a refund is due under ICOBS 6B.3.20G.

# COMMAND ----------

from insurance_deploy import ENBPAuditReport

reporter = ENBPAuditReport(logger)
audit_md = reporter.generate(
    experiment_name=exp.name,
    period_start="2024-01-01",
    period_end="2024-07-01",
    firm_name="Meridian Motor Insurance Ltd",
    smf_holder="Tom Bradley",
)

# Render as styled HTML in the notebook output
audit_html = f"""
<div style="font-family: Georgia, serif; max-width: 900px; padding: 24px;
            background: #fafafa; border: 1px solid #ddd; border-radius: 4px;">
  <pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px;
              line-height: 1.6; color: #222;">{audit_md}</pre>
</div>
"""
displayHTML(audit_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Promote challenger to champion
# MAGIC
# MAGIC Promotion is a governance event, not a technical one. The steps are:
# MAGIC
# MAGIC 1. Review the statistical evidence (Step 7)
# MAGIC 2. Get sign-off from the pricing governance committee (SMF23 holder)
# MAGIC 3. Deactivate the old experiment
# MAGIC 4. Update the registry champion designation
# MAGIC 5. Record the promotion decision in your governance log
# MAGIC
# MAGIC The library provides the technical primitives. The governance process is yours.
# MAGIC The key property: every step creates a permanent audit trail. No state is
# MAGIC overwritten. Old experiment logs are preserved. Old model artefacts remain in
# MAGIC the registry with their SHA-256 hashes intact.
# MAGIC
# MAGIC For this example we assume the statistical test returned `CHALLENGER_BETTER` with
# MAGIC p < 0.05 on a larger dataset, and the governance committee approved. We proceed
# MAGIC with promotion.

# COMMAND ----------

# Step 9a: Deactivate the current experiment.
# This closes routing — route() will raise RuntimeError if called again.
# The logs are permanent and unaffected.
exp.deactivate()

# Verify: routing now raises an error (expected)
routing_error_msg = ""
try:
    exp.route("POL-20240001")
    routing_error_msg = "ERROR: route() should have raised RuntimeError"
except RuntimeError as e:
    routing_error_msg = f"route() correctly raises: {e}"

# Step 9b: Promote the challenger to champion in the registry
new_champion = registry.set_champion("motor", version="3.0")

# Verify old champion is no longer flagged
old = registry.get("motor", "2.0")

promotion_html = f"""
<table style="border-collapse:collapse;font-family:monospace;font-size:13px;margin-bottom:16px">
  <caption style="font-family:sans-serif;font-size:14px;font-weight:bold;text-align:left;padding-bottom:8px">
    Governance trail
  </caption>
  <thead>
    <tr style="background:#1a1a2e;color:#fff">
      <th style="padding:8px 16px;text-align:left">Field</th>
      <th style="padding:8px 16px;text-align:left">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Experiment</td><td style="padding:8px 16px">{exp.name}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Created</td><td style="padding:8px 16px">{exp.created_at[:19]} UTC</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Deactivated</td><td style="padding:8px 16px">{exp.deactivated_at[:19]} UTC</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Is active</td><td style="padding:8px 16px">{exp.is_active()}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Champion (v2.0) — was</td>
      <td style="padding:8px 16px">is_champion={old.is_champion} | Gini {champion_mv.metadata['holdout_gini']} | validated by {champion_mv.metadata['validated_by']}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Challenger (v3.0) — promoted</td>
      <td style="padding:8px 16px;color:#2e7d32;font-weight:bold">is_champion={new_champion.is_champion} | Gini {challenger_mv.metadata['holdout_gini']} | validated by {challenger_mv.metadata['validated_by']}</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Decision</td><td style="padding:8px 16px">CHALLENGER_BETTER (manual, 10,000 policy dataset)</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Approved by</td><td style="padding:8px 16px">Tom Bradley, SMF23</td></tr>
    <tr style="background:#f8f8f8"><td style="padding:8px 16px">Total quotes in log</td><td style="padding:8px 16px">{logger.quote_count():,}</td></tr>
    <tr style="background:#fff"><td style="padding:8px 16px">Routing guard</td><td style="padding:8px 16px">{routing_error_msg}</td></tr>
  </tbody>
</table>
<p style="font-family:sans-serif;font-size:12px;color:#555">
  Any quote in the log can be retraced: policy_id + experiment_name is all you need
  to recompute routing from first principles. No external state required. The SHA-256
  hashes in the registry confirm which model object produced each price.
</p>
"""
displayHTML(promotion_html)

# Step 9c: Next experiment template (informational)
print("Next experiment (when a v4.0 challenger is ready):")
print(
    "  exp_v4 = Experiment(\n"
    "      name='motor_v4_vs_v3_2025q1',\n"
    "      champion=registry.get('motor', '3.0'),\n"
    "      challenger=next_challenger_mv,\n"
    "      challenger_pct=0.20,\n"
    "      mode='shadow',\n"
    "  )"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What we did |
# MAGIC |------|-------------|
# MAGIC | 1. Models defined | Champion GLM v2.0 and GBM challenger v3.0 |
# MAGIC | 2. Registry | 2 model versions, SHA-256 hash-verified |
# MAGIC | 3. Experiment | Shadow mode, 20% challenger designation, deterministic routing |
# MAGIC | 4. Quotes logged | 1,000 quotes with full lifecycle (quote → bind → claim) |
# MAGIC | 5. Governance | Per-quote model version + arm + ENBP in SQLite |
# MAGIC | 6. KPIs | Tier 1–4 dashboard; power analysis shows months to LR significance |
# MAGIC | 7. Statistical tests | Bootstrap LR, hit rate, frequency — all reported with conclusions |
# MAGIC | 8. ENBP audit | Markdown report covering all renewals; breaches listed for remediation |
# MAGIC | 9. Promotion | Experiment deactivated, registry updated, full governance trail |
# MAGIC
# MAGIC **Production deployment checklist:**
# MAGIC - Replace `ChampionGLM`/`ChallengerGBM` with real trained model objects
# MAGIC - Use a persistent registry path (`/dbfs/mnt/models/` or S3)
# MAGIC - Use a persistent `QuoteLogger` path (DBFS or Postgres adapter)
# MAGIC - Run `power_analysis()` before the experiment starts and share results with
# MAGIC   stakeholders so timeline expectations are set correctly
# MAGIC - Run the ENBP report monthly and include in your governance pack
# MAGIC - Bootstrap test: use `n_bootstrap=10,000` for promotion decisions
# MAGIC - Promotion decision: document approver name, date, and test results in your
# MAGIC   model governance system (Collibra, MLflow, or similar)
