"""
AI/ML regulatory compliance pipeline for a UK insurance pricing model.

This script demonstrates a complete pre-deployment compliance workflow for a
CatBoost motor frequency model. It addresses four distinct regulatory obligations
that have crystallised in UK insurance regulation since 2023:

  PRA Dear CEO Letter (January 2026) — The PRA's letter to insurance CEOs
  explicitly names AI/ML pricing models as a supervisory priority for 2026.
  It requires firms to demonstrate that AI-driven decisions are explainable,
  that model risk management frameworks extend to ML models, and that ongoing
  monitoring is in place. This is not aspirational; PRA expects evidence.

  FCA Consumer Duty FG22/5 (live July 2023) — Firms must demonstrate good
  outcomes for all customer groups. For pricing, this means documenting that
  no rating factor is functioning as a proxy for a protected characteristic
  in a way that produces unjustifiable differential outcomes. The FCA's 2024
  multi-firm review of Consumer Duty implementation (2024) found this documentation is frequently absent.

  PRA SS1/23 — The supervisory statement on model risk management. All material
  models require formal validation documentation including: data quality
  assessment, performance metrics (Gini, A/E with confidence intervals),
  stability analysis, and sign-off from an independent validator.

  FRC TAS 100 (Actuarial Standards Board) — For models used in actuarial
  communications, TAS 100 requires that the actuary can demonstrate the model
  is fit for purpose, the assumptions are documented, and the output is
  reliable. The causal check step supports this by verifying that the key
  drivers are causally connected to the outcome, not spurious correlations.

The four compliance steps:

  Step 1: Bias audit (insurance-fairness)
    Proxy discrimination check: does any rating factor act as a proxy for a
    protected characteristic? Maps to FCA Consumer Duty Outcome 4 (Price and
    Value) and Equality Act 2010 s.19 (indirect discrimination).

  Step 2: Model validation report (insurance-governance)
    PRA SS1/23-style validation: Gini with CI, A/E with CI, Hosmer-Lemeshow,
    lift chart, risk tier classification. Maps to SS1/23 Principle 3 and the
    PRA Dear CEO Jan 2026 AI governance expectations.

  Step 3: Drift monitoring (insurance-monitoring)
    PSI per feature, Gini z-test, A/E monitoring with Murphy decomposition.
    Simulates a quarterly monitoring run. Maps to SS1/23 Principle 5
    (ongoing model risk management) and FCA requirement for continuous rather
    than point-in-time Consumer Duty monitoring.

  Step 4: Causal driver verification (insurance-causal)
    Double Machine Learning check: are the model's key drivers causally
    connected to claim frequency, or are they exploiting spurious correlations?
    Maps to FRC TAS 100 assumption documentation and the PRA Dear CEO Jan 2026
    requirement that AI decisions be explainable.

The data is entirely synthetic. In production, replace the data generation
section with your portfolio extract. Everything from Step 1 onwards works
identically on real data.

Libraries used
--------------
    insurance-datasets    — synthetic UK motor portfolio
    insurance-fairness    — proxy discrimination audit
    insurance-governance  — PRA SS1/23 validation report and MRM tier scoring
    insurance-monitoring  — drift detection and calibration monitoring
    insurance-causal      — causal treatment effect estimation (DML)

Dependencies
------------
    uv add insurance-datasets insurance-fairness insurance-governance \\
           insurance-monitoring insurance-causal catboost

Note on execution
-----------------
This script is designed to run on Databricks (serverless cluster or standard
compute). The DML cross-fitting in Step 4 and the Gini bootstrap in Step 3
each fit multiple CatBoost models and will run slowly on a laptop. On a
standard Databricks cluster with 8+ cores the full pipeline takes 5-10 minutes.
"""

from __future__ import annotations

import datetime
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data and model setup
# ---------------------------------------------------------------------------
#
# We generate 25,000 synthetic motor policies and train a CatBoost frequency
# model. In production this is your existing model under audit.
#
# The synthetic data includes two proxy risk factors that we will audit in
# Step 1:
#
#   occupation_class  — correlates with socioeconomic group and carries
#                       indirect discrimination risk under Equality Act 2010
#                       s.19 (indirect discrimination). The most plausible
#                       protected characteristic pathway is s.9 (race) via
#                       socioeconomic group correlation, and potentially s.10
#                       (religion/belief) in aggregate — but the primary
#                       mechanism is socioeconomic, not religious. The FCA
#                       has flagged occupation as a proxy concern in its
#                       Consumer Duty supervisory work.
#
#   postcode_band     — a synthetic area deprivation proxy. Real postcode-level
#                       data correlates with ethnicity. The FCA has said
#                       postcode-based pricing requires justification.
#
# We also add a synthetic price_change column for the causal check in Step 4.
# This represents the percentage price change at renewal — a common treatment
# variable when auditing whether a rating factor is causal or merely correlated.
# ---------------------------------------------------------------------------

print("=" * 70)
print("Setup: load data and train CatBoost frequency model")
print("=" * 70)

from insurance_datasets import load_motor

df_raw = load_motor(n_policies=25_000, seed=42)
rng = np.random.default_rng(42)

# Synthetic disadvantaged-area proxy: area bands E and F are inner-city bands
# that correlate with lower socioeconomic status and, in real UK data, with
# ethnicity. We add 8% noise so the proxy is imperfect (as in practice).
disadvantaged = df_raw["area"].isin(["E", "F"]).astype(int)
noise = rng.binomial(1, 0.08, len(df_raw))
df_raw = df_raw.assign(
    postcode_band=((disadvantaged.values + noise) % 2).astype(int)
)

# Synthetic price change at renewal: range -15% to +20%, correlated with
# driver_age (younger drivers face larger increases) as in real portfolios.
age_factor = (25.0 / df_raw["driver_age"].clip(lower=17)).values
price_change = rng.normal(0.05, 0.08, len(df_raw)) * age_factor
price_change = np.clip(price_change, -0.15, 0.25)
df_raw = df_raw.assign(price_change=price_change)

FEATURE_COLS = [
    "vehicle_age",
    "vehicle_group",
    "driver_age",
    "ncd_years",
    "area",
    "policy_type",
    "occupation_class",
]
CAT_COLS = ["area", "policy_type"]
cat_indices = [FEATURE_COLS.index(c) for c in CAT_COLS]

from catboost import CatBoostRegressor, Pool

n_train = int(0.75 * len(df_raw))
train_pd = df_raw.iloc[:n_train].copy()
holdout_pd = df_raw.iloc[n_train:].copy()

train_log_exp = np.log(train_pd["exposure"].clip(lower=1e-9).values)
holdout_log_exp = np.log(holdout_pd["exposure"].clip(lower=1e-9).values)

train_pool = Pool(
    data=train_pd[FEATURE_COLS],
    label=train_pd["claim_count"],
    cat_features=cat_indices,
    baseline=train_log_exp,
)
holdout_pool = Pool(
    data=holdout_pd[FEATURE_COLS],
    label=holdout_pd["claim_count"],
    cat_features=cat_indices,
    baseline=holdout_log_exp,
)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    depth=4,
    learning_rate=0.06,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)
model.fit(train_pool, eval_set=holdout_pool)

holdout_pd = holdout_pd.copy()
holdout_pd["predicted_rate"] = model.predict(holdout_pd[FEATURE_COLS])

train_pd_copy = train_pd.copy()
train_pd_copy["predicted_rate"] = model.predict(train_pd_copy[FEATURE_COLS])

holdout_actuals = holdout_pd["claim_count"].values
holdout_preds = holdout_pd["predicted_rate"].values
holdout_exp = holdout_pd["exposure"].values
train_actuals = train_pd_copy["claim_count"].values
train_preds = train_pd_copy["predicted_rate"].values
train_exp = train_pd_copy["exposure"].values

calibration = float(
    (holdout_preds * holdout_exp).sum() / max(holdout_actuals.sum(), 1e-9)
)
print(f"Portfolio: {len(df_raw):,} policies, holdout: {len(holdout_pd):,}")
print(f"Model trained. Calibration (holdout): {calibration:.4f}")

# Output directory for compliance artefacts
tmpdir = Path(tempfile.mkdtemp(prefix="bc_regulatory_"))
print(f"\nCompliance artefacts will be written to: {tmpdir}")

# ---------------------------------------------------------------------------
# Step 1: Bias audit (FCA Consumer Duty, Equality Act 2010)
# ---------------------------------------------------------------------------
#
# REGULATORY CONTEXT
# ------------------
# FCA Consumer Duty Outcome 4 (Price and Value) requires firms to demonstrate
# that customers are not receiving worse outcomes because of characteristics
# that correlate with protected groups. The FCA's multi-firm review of Consumer Duty implementation (2024)
# specifically identified that many firms cannot produce evidence of ongoing
# proxy discrimination monitoring — they do a one-off check at model launch
# and nothing thereafter.
#
# The legal standard is Equality Act 2010 s.19 (indirect discrimination):
# a provision, criterion or practice is indirectly discriminatory if it puts
# persons sharing a protected characteristic at a particular disadvantage, and
# it is not a proportionate means of achieving a legitimate aim.
#
# For insurance, "legitimate aim" = risk-based pricing. A factor that is both
# a proxy for a protected characteristic AND a genuine risk predictor can be
# justified if the actuarial evidence is documented. A factor that is purely
# a proxy with no independent risk signal cannot.
#
# WHAT insurance-fairness DOES
# ----------------------------
# 1. Proxy detection: for each rating factor, trains a small CatBoost model
#    to predict the protected characteristic from that factor alone. The R2
#    and mutual information quantify how much the factor reveals about the
#    protected group. High R2 = high proxy risk.
#
# 2. Demographic parity: exposure-weighted mean prediction ratio across groups.
#    If model predicts 1.25x higher rates for the disadvantaged area group,
#    that needs justification.
#
# 3. Calibration by group: does model accuracy differ between groups?
#    Different accuracy = the model is learning group membership implicitly.
#
# 4. Disparate impact ratio: the "4/5ths rule" is borrowed from US employment
#    discrimination law (EEOC, 29 CFR 1607) and adapted here as a practical
#    threshold. It has no formal UK statutory status; we use it as a
#    pragmatic benchmark to identify outcomes that warrant investigation.
#
# WHAT REGULATORS WANT TO SEE
# ---------------------------
# - A dated, signed-off proxy detection report for each protected characteristic
# - Factor-level RAG ratings with thresholds documented
# - For any amber/red factors: an actuarial justification file
# - Quarterly re-run evidence, not just a one-off at model launch
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 1: Bias audit — FCA Consumer Duty / Equality Act 2010")
print("=" * 70)
print("""
Regulatory basis: FCA Consumer Duty (PRIN 2A, FG22/5), Equality Act 2010 s.19.
Protected characteristics under audit: postcode_band (ethnicity proxy),
occupation_class (socioeconomic group proxy).
""")

from insurance_fairness import FairnessAudit

holdout_pl = pl.from_pandas(holdout_pd)

audit = FairnessAudit(
    model=model,
    data=holdout_pl,
    protected_cols=["postcode_band", "occupation_class"],
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=FEATURE_COLS,
    model_name="Motor Frequency Model v1.0",
    run_proxy_detection=True,
    run_counterfactual=False,   # requires protected col as model input
    n_bootstrap=0,              # skip bootstrapped CIs to reduce runtime
    proxy_catboost_iterations=100,
)

print("Running proxy discrimination audit...")
# Suppress third-party deprecation noise from CatBoost/NumPy during fitting.
# This does not suppress any compliance-relevant warnings — all fairness
# results are in fairness_report; exceptions still propagate normally.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fairness_report = audit.run()

fairness_report.summary()

# Print factor-level proxy scores for the regulator record
print("\nProxy detection scores by rating factor:")
print("  (R2 > 0.05 = amber; R2 > 0.10 = red; actuarial justification required for amber/red)")
print()
for pc, pc_result in fairness_report.results.items():
    if pc_result.proxy_detection is None:
        continue
    print(f"  Protected characteristic: {pc}")
    sorted_scores = sorted(
        pc_result.proxy_detection.scores,
        key=lambda s: s.proxy_r2,
        reverse=True,
    )
    for score in sorted_scores[:5]:
        flag = " <-- REQUIRES ACTUARIAL JUSTIFICATION" if score.rag in ("amber", "red") else ""
        print(
            f"    {score.factor:<25}"
            f"  proxy_R2={score.proxy_r2:.4f}"
            f"  MI={score.mutual_information:.4f}"
            f"  [{score.rag.upper()}]{flag}"
        )
    print()

audit_dict = fairness_report.to_dict()
overall_fairness_rag = audit_dict["overall_rag"].upper()

# Save the audit report — this is the document you attach to the model
# validation pack or Consumer Duty evidence folder
audit_md_path = tmpdir / "fairness_audit_report.md"
fairness_report.to_markdown(str(audit_md_path))

print(f"Consumer Duty audit report saved to: {audit_md_path}")
print(f"Overall RAG status: {overall_fairness_rag}")

if fairness_report.flagged_factors:
    flagged = ", ".join(fairness_report.flagged_factors)
    print(f"\nFlagged factors requiring actuarial justification: {flagged}")
    print(
        "Action required: for each flagged factor, produce a memo demonstrating"
        " either (a) the factor has independent actuarial justification beyond"
        " its proxy correlation, or (b) it is removed from the tariff."
        " This memo must be filed before model deployment. The FCA's multi-firm review of Consumer Duty"
        " implementation (2024) found most firms do not have this documentation."
    )
else:
    print("\nNo factors flagged. Audit evidence folder ready for Consumer Duty review.")

print(f"\nAudit metadata for model risk register:")
print(f"  Policies audited: {audit_dict['n_policies']:,}")
print(f"  Audit date:       {fairness_report.audit_date}")
print(f"  Protected characteristics: {', '.join(audit_dict['protected_cols'])}")

# ---------------------------------------------------------------------------
# Step 2: Model validation report (PRA SS1/23, PRA Dear CEO Jan 2026)
# ---------------------------------------------------------------------------
#
# REGULATORY CONTEXT
# ------------------
# PRA SS1/23 requires formal model validation for all material models. The
# January 2026 Dear CEO letter extended this expectation explicitly to
# AI/ML models used in insurance pricing. The key requirements are:
#
#   Section 3.1 (Model inventory): every material model must have a model
#   card recording: purpose, methodology, data sources, assumptions, owner.
#
#   Section 3.2 (Validation): independent validation by someone not involved
#   in model development, testing: discrimination, calibration, stability,
#   sensitivity.
#
#   Section 3.3 (Governance): documented approval process, risk tier, review
#   schedule. The PRA Dear CEO Jan 2026 letter specifically asks for evidence
#   that the MRC is actively reviewing AI model risk.
#
# WHAT insurance-governance DOES
# --------------------------------
# 1. ModelCard (MRM subpackage): structured metadata for model inventory.
#    Captures purpose, assumptions, limitations, owner, approval chain.
#
# 2. RiskTierScorer: objective tier assignment (1=Critical, 2=Significant,
#    3=Informational) using a weighted scorecard across six dimensions:
#    materiality, complexity, data quality, validation coverage, drift history,
#    and regulatory exposure. Produces a verbose rationale for the MRC.
#
# 3. GovernanceReport: executive summary format for the Model Risk Committee.
#    HTML output, print-to-PDF ready, combines all the above.
#
# 4. ModelValidationReport (validation subpackage): Gini with CI, A/E with CI,
#    Hosmer-Lemeshow, lift chart, PSI. HTML and JSON output for the full
#    technical validation package.
#
# WHAT REGULATORS WANT TO SEE
# ---------------------------
# - Model card with documented assumptions and limitations
# - Independent validation sign-off (not by the model developer)
# - Risk tier with scoring rationale
# - HTML/PDF validation report with Gini CI, A/E CI, H-L test
# - JSON output for ingestion into the model risk management system
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: Model validation report — PRA SS1/23 / PRA Dear CEO Jan 2026")
print("=" * 70)
print("""
Regulatory basis: PRA SS1/23 (Principles 1, 3, 5), PRA Dear CEO Letter Jan 2026.
FRC TAS 100: actuarial assumptions must be documented and assumptions tested.

Note on SS1/23 scope: SS1/23 is a PRA supervisory statement directed at banks,
not insurers. Insurers apply it here by analogy as a proxy framework because no
equivalent insurer-specific model risk management standard currently exists in
UK regulation. The PRA Dear CEO Jan 2026 letter implicitly endorses this approach
by applying SS1/23-style expectations to insurance AI/ML models.
""")

from insurance_governance import (
    MRMModelCard,
    RiskTierScorer,
    GovernanceReport,
    Assumption,
    Limitation,
    ModelValidationReport,
    ValidationModelCard,
    PerformanceReport,
)

# --- 2a: MRM Model Card (for the Model Risk Committee pack) ---
#
# The MRM card is the governance record. It answers the MRC's questions:
# what does this model do, who owns it, what are its risks, and who approved it.

mrm_card = MRMModelCard(
    model_id="motor-freq-v1",
    model_name="Motor Frequency Model",
    version="1.0.0",
    model_class="pricing",
    intended_use=(
        "Predict claim frequency for UK private motor portfolio. "
        "Used as the frequency component of pure premium pricing."
    ),
    not_intended_for=[
        "Claims reserving or IBNR estimation",
        "Commercial motor or fleet policies",
        "Solvency II internal model capital calculations",
    ],
    target_variable="claim_count",
    distribution_family="Poisson",
    model_type="GBM",
    rating_factors=FEATURE_COLS,
    training_data_period=("2021-01-01", "2023-12-31"),
    development_date="2024-03-01",
    developer="Pricing Analytics Team",
    champion_challenger_status="champion",
    portfolio_scope="UK private motor, all channels",
    geographic_scope="Great Britain",
    customer_facing=True,
    regulatory_use=False,
    gwp_impacted=85_000_000.0,
    assumptions=[
        Assumption(
            description="Claim frequency process is stationary over the training window",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring with Murphy decomposition",
            rationale="Frequency trended up 2021-2023 post-COVID; trend assumed stabilised",
        ),
        Assumption(
            description="Stated annual mileage is an unbiased proxy for driven mileage",
            risk="MEDIUM",
            mitigation="Annual audit of stated vs telematics mileage on opted-in customers",
            rationale="Known self-declaration bias; actuarial adjustment applied",
        ),
        Assumption(
            description="Occupation class captures socioeconomic risk differential",
            risk="LOW",
            mitigation="Proxy discrimination audit (see fairness_audit_report.md)",
            rationale="Confirmed actuarial justification; risk-based not proxy-based",
        ),
    ],
    limitations=[
        Limitation(
            description="No telematics data — mileage is self-declared only",
            impact="Possible systematic under-pricing of high-mileage customers",
            population_at_risk="Commuters and high-mileage professionals",
            monitoring_flag=True,
        ),
        Limitation(
            description="Trained on synthetic data; real portfolio recalibration required",
            impact="Calibration factor may differ from production claims experience",
            population_at_risk="All policies",
            monitoring_flag=True,
        ),
    ],
    outstanding_issues=[
        "Occupation class proxy audit amber for group 3 (manual labour) — justification memo in progress",
        "Area band resolution coarser than postcode district — improvement planned for v1.1",
    ],
    monitoring_owner="Head of Pricing Analytics",
    monitoring_frequency="Quarterly",
    monitoring_triggers={
        "ae_ratio_upper": 1.10,
        "ae_ratio_lower": 0.90,
        "psi_score": 0.20,
        "gini_change_abs": 0.05,
    },
    trigger_actions={
        "ae_ratio": "Immediate escalation to Chief Actuary; recalibration within 30 days",
        "psi_score": "Feature drift investigation; segment-level A/E within 5 days",
        "gini_change_abs": "Model refit assessment; MRC paper within 45 days",
    },
)

# --- 2b: Risk Tier Scoring ---
#
# RiskTierScorer maps six dimensions to a composite score that determines
# the validation and review schedule. This replaces subjective tier assignment
# with an auditable process — the MRC can see exactly why a model is Tier 1.
#
# For this model: £85M GWP, live champion model, customer-facing, no prior
# validation run on record. Expected result: Tier 1 (Critical) — requires
# annual independent validation and MRC sign-off.

scorer = RiskTierScorer()
tier_result = scorer.score(
    gwp_impacted=85_000_000,
    model_complexity="high",        # GBM with 7 features including interactions
    deployment_status="champion",   # live production model
    regulatory_use=False,           # not Solvency II capital model
    external_data=False,            # no external data feeds
    customer_facing=True,           # directly sets premiums
    validation_months_ago=None,     # no prior independent validation on record
    drift_triggers_last_year=0,     # no triggers fired yet (new model)
)

print(f"Risk Tier Assessment:")
print(f"  Tier: {tier_result.tier} ({tier_result.tier_label})")
print(f"  Composite score: {tier_result.score:.1f}/100")
print(f"  Review frequency: {tier_result.review_frequency}")
print(f"  Sign-off requirement: {tier_result.sign_off_requirement}")
print(f"\nDimension breakdown:")
for dim in tier_result.dimensions:
    print(f"  {dim.name:<25} {dim.score:>5.1f}/{dim.max_score:.1f}  —  {dim.rationale}")

# Update the MRM card with the tier result
mrm_card.materiality_tier = tier_result.tier
mrm_card.tier_rationale = tier_result.rationale

# --- 2c: Statistical validation tests ---
#
# The ValidationModelCard and ModelValidationReport implement the statistical
# tests required by SS1/23: Gini coefficient with confidence interval,
# A/E ratio with Poisson confidence interval, Hosmer-Lemeshow goodness-of-fit,
# and lift chart. These tests go in the technical validation document.

print("\nRunning statistical validation tests (SS1/23 Appendix requirements)...")

val_card = ValidationModelCard(
    name="Motor Frequency Model",
    version="1.0.0",
    purpose=(
        "Predict claim frequency for UK private motor portfolio. "
        "Output is the frequency component of the pure premium."
    ),
    methodology=(
        "CatBoost gradient boosting, Poisson objective, log(exposure) offset. "
        "300 trees, depth 4, L2 regularisation 3.0. Cross-validation on holdout "
        "25% of portfolio."
    ),
    target="claim_count",
    features=FEATURE_COLS,
    limitations=[
        "Trained on synthetic data — recalibrate before production deployment",
        "No telematics data — mileage is self-declared",
        "Area resolution at band level, not postcode district",
    ],
    owner="Pricing Analytics Team",
)

val_report = ModelValidationReport(
    model_card=val_card,
    y_val=holdout_actuals,
    y_pred_val=holdout_preds,
    exposure_val=holdout_exp,
    y_train=train_actuals,
    y_pred_train=train_preds,
)

html_path = tmpdir / "validation_report.html"
json_path = tmpdir / "validation_report.json"

# Suppress third-party deprecation noise (matplotlib, scikit-learn) during
# report generation. Exceptions propagate; compliance outputs are captured
# in the HTML/JSON files, not in warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    val_report.generate(str(html_path))
    val_report.to_json(str(json_path))

print(f"HTML validation report: {html_path}")
print(f"JSON validation report: {json_path}")

# Extract key metrics for the MRC paper
perf = PerformanceReport(holdout_actuals, holdout_preds, exposure=holdout_exp)

gini_result = perf.gini_with_ci()
ae_result = perf.ae_with_poisson_ci()
hl_result = perf.hosmer_lemeshow_test()

gini_val = gini_result.metric_value
gini_ci_lo = gini_result.extra.get("ci_lower")
gini_ci_hi = gini_result.extra.get("ci_upper")
ae_val = ae_result.metric_value
ae_ci_lo = ae_result.extra.get("ci_lower")
ae_ci_hi = ae_result.extra.get("ci_upper")
hl_stat = hl_result.extra.get("hl_statistic")
hl_pval = hl_result.extra.get("p_value")

print(f"\nKey metrics for SS1/23 validation document:")
print(f"  Gini coefficient:  {gini_val:.4f}  95% CI [{gini_ci_lo:.4f}, {gini_ci_hi:.4f}]")
print(f"  A/E ratio:         {ae_val:.4f}  95% CI [{ae_ci_lo:.4f}, {ae_ci_hi:.4f}]")
print(f"  Hosmer-Lemeshow:   chi2 = {hl_stat:.2f},  p = {hl_pval:.4f}  ({'PASS' if hl_result.passed else 'REVIEW'})")
print(f"  Calibration:       {calibration:.4f}")

# Lift chart — quantifies whether the model ranks risks correctly. A flat
# lift chart (actual/expected near 1.0 in every band) would indicate the
# model adds no value over the portfolio average. Regulators look for a
# monotone lift; non-monotone middle bands indicate feature interactions
# the model is not capturing.
print("\nLift chart (10 bands — ranked by predicted rate):")
print("  A flat A/E across bands = well-calibrated by risk decile")
lift_results = perf.lift_chart(n_bands=10)
lift_bands = lift_results[0].extra.get("bands", [])
print(f"  {'Band':<8} {'Predicted':>12} {'Actual':>10} {'A/E':>8}")
for b in lift_bands:
    ae_band = (
        b["actual_rate"] / b["predicted_rate"] if b["predicted_rate"] > 0 else float("nan")
    )
    print(
        f"  {b['band']:<8} {b['predicted_rate']:>12.5f} {b['actual_rate']:>10.5f} {ae_band:>8.4f}"
    )

# --- 2d: GovernanceReport for the MRC ---
#
# This is the 2-3 page executive pack for the Model Risk Committee. It
# combines the MRM card, tier result, and validation metrics into an HTML
# document that can be attached to an MRC agenda. The PRA Dear CEO Jan 2026
# letter asks for evidence that boards/MRCs are actively overseeing AI models —
# this is that evidence.

gov_report = GovernanceReport(
    card=mrm_card,
    tier=tier_result,
    validation_results={
        "overall_rag": "GREEN" if (gini_val or 0) > 0.15 and hl_result.passed else "AMBER",
        "run_date": str(datetime.date.today()),
        "gini": gini_val,
        "ae_ratio": ae_val,
        "hl_p_value": hl_pval,
    },
)

gov_html_path = tmpdir / "governance_report_mrc.html"
gov_html_path.write_text(gov_report.to_html())
print(f"\nGovernance report (MRC pack): {gov_html_path}")
print("This document is the PRA-facing evidence that MRC is overseeing the model.")

# ---------------------------------------------------------------------------
# Step 3: Drift monitoring (PRA SS1/23 Principle 5, FCA ongoing monitoring)
# ---------------------------------------------------------------------------
#
# REGULATORY CONTEXT
# ------------------
# PRA SS1/23 Principle 5 requires that model risk is managed on an ongoing
# basis, not just at launch. For AI/ML models specifically, the PRA Dear CEO
# Jan 2026 letter noted that many firms' monitoring frameworks were designed
# for GLMs and do not detect the subtler forms of drift that affect ML models.
#
# FCA Consumer Duty: the FCA's 2024 multi-firm review found that firms with
# good Consumer Duty frameworks run their proxy discrimination checks
# continuously, not once at model launch. Population mix shifts can turn a
# clean audit into a proxy discrimination issue within 6 months if the
# customer demographic changes (e.g. growth in aggregator channel).
#
# WHAT insurance-monitoring DOES
# --------------------------------
# 1. PSI (Population Stability Index): detects feature distribution drift.
#    If driver_age PSI > 0.20 it means the age mix of your new book differs
#    materially from the training data — your age-band relativities may be stale.
#
# 2. Gini drift z-test: tests whether the model's discrimination power has
#    degraded. Gini drift without A/E drift = model still calibrated but
#    ranking risks less well (classic feature staleness).
#
# 3. A/E ratio with CI: aggregate calibration check. Easy to communicate to
#    head of pricing: "we expected 1,000 claims, we got 1,080."
#
# 4. Murphy decomposition: decomposes the scoring rule into discrimination
#    (DSC, captured by Gini), miscalibration (MCB), and uncertainty. The
#    distinction matters: high MCB -> RECALIBRATE; low DSC -> REFIT.
#    Most firms conflate these two failure modes.
#
# SIMULATING THE MONITORING SCENARIO
# ------------------------------------
# We split the holdout into two quarters:
#   Q1 (earlier half): the reference quarter — baseline performance
#   Q2 (later half):   the monitoring quarter — we inject drift to simulate
#                      a model that is starting to stale
#
# Drift injected into Q2: young driver (<25) frequency is 30% higher than
# the training distribution. This is the kind of change that happens when
# a new comparison site campaign attracts a younger customer mix.
#
# WHAT REGULATORS WANT TO SEE
# ---------------------------
# - Quarterly evidence of each check being run, with dated outputs
# - A documented recommendation (NO_ACTION / RECALIBRATE / REFIT) with
#   the monitoring evidence that supports it
# - For amber/red signals: documented escalation to Chief Actuary / MRC
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: Drift monitoring — PRA SS1/23 Principle 5 / FCA ongoing monitoring")
print("=" * 70)
print("""
Regulatory basis: PRA SS1/23 Principle 5, PRA Dear CEO Jan 2026, FCA Consumer Duty (PRIN 2A).
Simulating a quarterly monitoring run: Q1 (reference) vs Q2 (with injected drift).
""")

from insurance_monitoring import MonitoringReport, psi
from insurance_monitoring.drift import csi

# Split holdout into Q1 (reference) and Q2 (monitoring)
n_half = len(holdout_pd) // 2
q1_pd = holdout_pd.iloc[:n_half].copy()
q2_pd = holdout_pd.iloc[n_half:].copy()

# Inject frequency drift into Q2: young drivers (<25) experience 30% more claims
# This simulates a real mix shift — the model's predictions do not change but
# actual claims do. A/E will drift; Gini may degrade if the model cannot rank
# young drivers relative to the new claim pattern.
young_mask = q2_pd["driver_age"] < 25
q2_pd.loc[young_mask, "claim_count"] = (
    q2_pd.loc[young_mask, "claim_count"] * 1.30
).round()

q1_actuals = q1_pd["claim_count"].values
q1_preds = q1_pd["predicted_rate"].values
q1_exp = q1_pd["exposure"].values

q2_actuals = q2_pd["claim_count"].values
q2_preds = q2_pd["predicted_rate"].values
q2_exp = q2_pd["exposure"].values

print(f"Q1 (reference): {len(q1_pd):,} policies, {q1_actuals.sum():.0f} claims")
print(f"Q2 (monitoring): {len(q2_pd):,} policies, {q2_actuals.sum():.0f} claims")
print(f"Young driver (<25) proportion Q2: {young_mask.mean():.2%}")
print(f"Drift injected: +30% frequency for young drivers in Q2")

# Feature DataFrames for CSI computation
feat_cols_numeric = ["vehicle_age", "driver_age", "ncd_years"]
q1_feat_pl = pl.from_pandas(q1_pd[feat_cols_numeric])
q2_feat_pl = pl.from_pandas(q2_pd[feat_cols_numeric])

print("\nRunning quarterly monitoring checks...")
mon_report = MonitoringReport(
    reference_actual=q1_actuals,
    reference_predicted=q1_preds,
    current_actual=q2_actuals,
    current_predicted=q2_preds,
    exposure=q2_exp,
    reference_exposure=q1_exp,
    feature_df_reference=q1_feat_pl,
    feature_df_current=q2_feat_pl,
    features=feat_cols_numeric,
    murphy_distribution="poisson",  # Murphy decomposition: RECALIBRATE vs REFIT
    gini_bootstrap=True,            # Percentile CI on Gini for the MRC report
    n_bootstrap=200,
)

results = mon_report.results_
recommendation = mon_report.recommendation

# A/E ratio
ae = results["ae_ratio"]
print(f"\nA/E ratio (Q2 vs Q1 reference):")
print(f"  A/E = {ae['value']:.4f}  95% CI [{ae['lower_ci']:.4f}, {ae['upper_ci']:.4f}]  [{ae['band'].upper()}]")
print(f"  SS1/23 trigger thresholds: <0.90 or >1.10")

# Gini drift
gini = results["gini"]
print(f"\nGini drift test (Q2 vs Q1 reference):")
print(f"  Reference Gini = {gini['reference']:.4f}")
print(f"  Current Gini   = {gini['current']:.4f}  95% CI [{gini.get('ci_lower', float('nan')):.4f}, {gini.get('ci_upper', float('nan')):.4f}]")
print(f"  Gini change    = {gini['change']:+.4f}  (p = {gini['p_value']:.4f})  [{gini['band'].upper()}]")
print(f"  SS1/23 trigger threshold: |change| > 0.05")

# Murphy decomposition
if "murphy" in results:
    murphy = results["murphy"]
    print(f"\nMurphy decomposition (sharpens RECALIBRATE vs REFIT decision):")
    print(f"  Discrimination (DSC): {murphy['discrimination']:.6f}  ({murphy['discrimination_pct']:.1f}% of total score)")
    print(f"  Miscalibration (MCB): {murphy['miscalibration']:.6f}  ({murphy['miscalibration_pct']:.1f}% of total score)")
    print(f"  Murphy verdict: {murphy['verdict']}")
    print(f"  Interpretation: {'MCB dominates -> calibration drift, cheaper to fix' if murphy['verdict'] == 'RECALIBRATE' else 'DSC degraded -> model needs refit on recent data'}")

# CSI per feature
if "csi" in results:
    print(f"\nCharacteristic Stability Index (CSI) per feature:")
    print(f"  (PSI/CSI > 0.10 = amber; > 0.20 = red; feature distribution has shifted)")
    for csi_row in results["csi"]:
        print(f"  {csi_row['feature']:<20}  CSI = {csi_row['csi']:.4f}  [{csi_row['band'].upper()}]")

print(f"\nMonitoring recommendation: {recommendation}")
print(f"  Decision logic (Gini ranking drift + calibration drift; see arXiv:2510.04556):")
print(f"  Gini {gini['band'].upper()} + A/E {ae['band'].upper()} => {recommendation}")

if recommendation in ("RECALIBRATE", "REFIT"):
    print(f"\n  ACTION REQUIRED before next quarterly monitoring cycle:")
    if recommendation == "RECALIBRATE":
        print(f"    RECALIBRATE: A/E has drifted but Gini stable. The model is ranking")
        print(f"    risks correctly but the overall level is wrong. Update the intercept")
        print(f"    or overall calibration factor. No need for a full model refit.")
    else:
        print(f"    REFIT: Gini has degraded. The model's ability to rank risks has")
        print(f"    declined. Recalibrating will not fix this. Retrain on recent data.")
        print(f"    Escalate to Chief Actuary within 5 business days per MRC trigger policy.")
else:
    print(f"\n  No immediate action required. Document and file as quarterly monitoring evidence.")

# Save monitoring output to file (goes in the model risk register)
mon_df = mon_report.to_polars()
mon_path = tmpdir / "monitoring_report_q2_2026.csv"
mon_df.write_csv(str(mon_path))
print(f"\nMonitoring report (Polars CSV): {mon_path}")

# ---------------------------------------------------------------------------
# Step 4: Causal driver verification (FRC TAS 100, PRA AI explainability)
# ---------------------------------------------------------------------------
#
# REGULATORY CONTEXT
# ------------------
# FRC TAS 100 (Technical Actuarial Standard) requires actuaries to be satisfied
# that the models they use are fit for purpose, that assumptions are documented,
# and that the results are reliable. For a ML pricing model, a key TAS 100 risk
# is that the model is exploiting a spurious correlation that will break down.
#
# PRA Dear CEO January 2026: the letter specifically requires that AI-driven
# pricing decisions be explainable — firms should be able to say not just
# "this factor predicts claims" but "this factor causally influences claims".
# A model that uses occupation_class because it correlates with claim frequency
# in historical data is not the same as a model that uses it because occupation
# causally affects driving behaviour. The former is fragile; the latter is not.
#
# WHAT DOUBLE MACHINE LEARNING DOES
# -----------------------------------
# DML (Chernozhukov et al. 2018) answers: what is the causal effect of a
# treatment D on outcome Y, controlling for confounders X?
#
# In our compliance context we use it differently: we treat occupation_class
# as the "treatment" and claim_count as the outcome, controlling for all
# other rating factors. If the DML estimate is significant and directionally
# consistent with actuarial priors (e.g. manual occupations have higher
# frequency), that is evidence the factor has causal content.
#
# If the DML estimate is near zero or reverses sign vs the naive correlation,
# it suggests the factor is acting as a proxy for something else (the confounders
# we controlled for). That is a warning sign for both TAS 100 (reliability)
# and Consumer Duty (proxy discrimination).
#
# We also run a price change causal check: what is the causal effect of a
# 1% price increase on claim frequency? Economically, it should be negative
# (higher price -> adverse selection -> higher remaining book frequency) or
# near zero. A positive causal effect would suggest price is confounded with
# an unobserved risk variable and the model needs an additional confounder.
#
# WHAT REGULATORS WANT TO SEE
# ---------------------------
# - For each key driver: is the direction and magnitude of the causal effect
#   consistent with actuarial understanding?
# - For prohibited factors (gender, ethnicity): the causal effect of proxies
#   should not be significant after controlling for risk factors
# - Sensitivity analysis: how robust are the causal estimates to unobserved
#   confounders?
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: Causal driver verification — FRC TAS 100 / PRA AI explainability")
print("=" * 70)
print("""
Regulatory basis: FRC TAS 100 (reliability of results), PRA Dear CEO Jan 2026
(AI explainability requirement).

Causal checks:
  A. Occupation class causal effect on claim frequency — is the signal causal
     or a proxy for confounders? If causal, justify inclusion (TAS 100). If
     proxy-only, consider removal (Consumer Duty).
  B. Price change causal effect on claim frequency — price-risk correlation check.
     This tests whether the model assigned higher prices to higher-risk policies
     (i.e. whether price_change correlates with risk after controlling for observed
     rating factors). It is NOT an adverse selection test: the dataset contains
     all policies regardless of whether they renewed, so we cannot observe lapse
     behaviour. A genuine adverse selection test requires renewal/lapse data and
     a lapse propensity model, which are not available here.
     Expected sign: positive (model priced riskier policies higher).
""")

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment, PriceChangeTreatment

# --- 4a: Is occupation_class a causal risk driver? ---
#
# We encode occupation_class as a binary: manual (high-risk) vs non-manual.
# Treatment: binary flag for manual occupation (1 = manual, 0 = non-manual).
# Outcome: claim_count (Poisson frequency with exposure offset).
# Confounders: all other rating factors.
#
# If the DML estimate is significantly positive, occupation genuinely causes
# higher claim frequency and its inclusion in the tariff is actuarially
# justified (TAS 100 passes, Consumer Duty passes with documentation).
#
# If the DML estimate is near zero, occupation's predictive power comes from
# correlation with confounders (area, vehicle type, etc.) and its inclusion
# requires stronger justification under Consumer Duty.

print("4a: Causal effect of occupation class on claim frequency")
print("    Treatment: manual occupation (binary), Outcome: claim_count")
print("    Confounders: vehicle_age, vehicle_group, driver_age, ncd_years, area, policy_type")
print()

# We need exposure as a column in the DataFrame for Poisson outcome type
train_causal = train_pd.copy()
train_causal["is_manual"] = (train_causal["occupation_class"] == 3).astype(int)

# The confounders exclude occupation_class itself (that is the treatment)
confounders_occ = [c for c in FEATURE_COLS if c != "occupation_class"]

occ_causal_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=BinaryTreatment(column="is_manual"),
    confounders=confounders_occ,
    exposure_col="exposure",
    nuisance_params={"iterations": 150, "depth": 4, "learning_rate": 0.1, "verbose": 0},
    cv_folds=3,     # 3-fold for demo speed; use 5 in production
    random_state=42,
)

print("Fitting DML model for occupation causal check (3-fold cross-fitting)...")
# Suppress third-party deprecation noise from CatBoost during DML cross-fitting.
# CausalPricingModel raises on convergence failures; suppression is safe here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    occ_causal_model.fit(train_causal)

occ_ate = occ_causal_model.average_treatment_effect()
print(occ_ate)

# Interpret the result
occ_sig = occ_ate.p_value < 0.05
occ_positive = occ_ate.estimate > 0
print(f"\nInterpretation (TAS 100 / Consumer Duty):")
if occ_sig and occ_positive:
    print(
        f"  CAUSAL SIGNAL CONFIRMED: occupation (manual) has a statistically significant"
        f" positive causal effect on claim frequency (ATE = {occ_ate.estimate:.4f},"
        f" p = {occ_ate.p_value:.4f}) after controlling for confounders."
        f" Inclusion in tariff is actuarially justified under TAS 100."
        f" File this output as the actuarial justification under Consumer Duty."
    )
elif occ_sig and not occ_positive:
    print(
        f"  DIRECTION REVERSAL: occupation has a significant NEGATIVE causal effect"
        f" after controlling for confounders (ATE = {occ_ate.estimate:.4f})."
        f" This suggests occupation is acting as an inverse proxy. Review urgently."
    )
else:
    print(
        f"  WEAK CAUSAL SIGNAL: occupation causal effect is not statistically significant"
        f" (ATE = {occ_ate.estimate:.4f}, p = {occ_ate.p_value:.4f}) after controlling"
        f" for confounders. The predictive power of occupation_class appears to be"
        f" mediated by correlated rating factors. Consider removing occupation from"
        f" the tariff — its Consumer Duty justification is weakened."
    )

# --- 4b: Price-risk correlation check (NOT an adverse selection test) ---
#
# IMPORTANT SCOPE NOTE: This is a price-risk correlation check, not an adverse
# selection check. The two tests are frequently confused.
#
# What this test does: estimates the causal effect of price_change on claim_count
# after controlling for observed rating factors. A positive estimate means the
# model assigned higher prices to policies that, conditional on rating factors,
# had higher claim frequency. This is a model fit / risk-price alignment check —
# it tests whether prices were set consistently with risk.
#
# What a true adverse selection test requires (and this dataset cannot provide):
#   1. Renewal/lapse data — which customers received higher prices and then did
#      NOT renew? Policies that lapse are absent from this dataset.
#   2. A lapse propensity model — to identify the price-sensitive segment.
#   3. Before/after claim experience — comparing retained vs lapsed cohorts.
# Without renewal conversion data, we cannot test whether price-sensitive
# low-risk customers exited the book, because we observe only those who stayed.
#
# Expected DML estimate: positive (model priced riskier policies higher, i.e.
# risk-price alignment is present in the data).
# Near-zero: model prices are uncorrelated with residual risk — review pricing.
# Negative: prices are inversely correlated with risk — indicates a mispricing
# pattern that warrants urgent investigation.

print("\n4b: Price-risk correlation check — causal effect of price change on claim frequency")
print("    Treatment: price_change (continuous %), Outcome: claim_count")
print("    Note: this tests risk-price alignment, NOT adverse selection (no lapse data available)")
print("    Expected direction: positive (higher prices assigned to higher-risk policies)")
print()

# We use the full training set including the synthetic price_change column
train_price = train_pd.copy()

price_causal_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=PriceChangeTreatment(
        column="price_change",
        scale="linear",                   # price_change already on percentage scale
        clip_percentiles=(0.02, 0.98),    # trim extreme re-rate outliers
    ),
    confounders=FEATURE_COLS,
    exposure_col="exposure",
    nuisance_params={"iterations": 150, "depth": 4, "learning_rate": 0.1, "verbose": 0},
    cv_folds=3,
    random_state=42,
)

print("Fitting DML model for price-risk correlation check (3-fold cross-fitting)...")
# Suppress third-party deprecation noise from CatBoost during DML cross-fitting.
# CausalPricingModel raises on convergence failures; suppression is safe here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    price_causal_model.fit(train_price)

price_ate = price_causal_model.average_treatment_effect()
print(price_ate)

price_sig = price_ate.p_value < 0.05
price_positive = price_ate.estimate > 0
print(f"\nInterpretation (TAS 100 economic sense check — price-risk alignment):")
if price_positive:
    print(
        f"  PRICE-RISK ALIGNMENT CONFIRMED: a 1-unit increase in price_change is"
        f" causally associated with a {price_ate.estimate:.4f} increase in claim rate"
        f" ({'' if not price_sig else 'statistically significant, '}p = {price_ate.p_value:.4f})"
        f" after controlling for observed rating factors. The model assigned higher"
        f" prices to policies with higher residual risk — consistent with sound"
        f" risk-based pricing. TAS 100 economic sense check: PASSED."
        f"\n  NOTE: This does not confirm adverse selection. To test adverse selection"
        f" you need renewal/lapse data showing which customers exited at higher prices."
    )
else:
    print(
        f"  PRICE-RISK ALIGNMENT NOT DETECTED (ATE = {price_ate.estimate:.4f},"
        f" p = {price_ate.p_value:.4f}). Prices do not correlate positively with"
        f" residual risk after controlling for rating factors. This warrants"
        f" investigation: either the rating factors fully capture the risk signal"
        f" (price_change adds nothing new), or there is a systematic mispricing"
        f" pattern. TAS 100 economic sense check: REVIEW REQUIRED."
        f"\n  NOTE: Absence of price-risk correlation is distinct from absence of"
        f" adverse selection — the two require different data to test."
    )

# ---------------------------------------------------------------------------
# Summary: mapping outputs to regulatory requirements
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Regulatory compliance summary")
print("=" * 70)

print(f"""
  Requirement                     Status    Evidence file
  -------------------------------- --------- ----------------------------------
  FCA Consumer Duty bias audit     {overall_fairness_rag:<9} fairness_audit_report.md
  PRA SS1/23 validation report     COMPLETE  validation_report.html/.json
  PRA MRM tier assessment          TIER {tier_result.tier}     governance_report_mrc.html
  PRA SS1/23 quarterly monitoring  {recommendation:<9} monitoring_report_q2_2026.csv
  FRC TAS 100 causal checks        COMPLETE  (printed above)
  PRA Dear CEO Jan 2026 AI gov.    COMPLETE  governance_report_mrc.html

Files written to: {tmpdir}

  fairness_audit_report.md         — Consumer Duty / Equality Act evidence
  validation_report.html           — SS1/23 technical validation document
  validation_report.json           — Machine-readable validation output
  governance_report_mrc.html       — Model Risk Committee executive pack
  monitoring_report_q2_2026.csv    — Quarterly monitoring evidence

Next steps before deployment
-----------------------------
  1. Independent validation sign-off on validation_report.html (SS1/23 Principle 3)
  2. MRC review of governance_report_mrc.html (Tier {tier_result.tier} = {tier_result.sign_off_requirement})
  3. File fairness_audit_report.md in Consumer Duty evidence folder (FCA FG22/5)
  4. For any flagged proxy factors: actuarial justification memo (Consumer Duty)
  5. Schedule quarterly monitoring run (next due: Q3 2026)
  6. If monitoring recommendation is RECALIBRATE/REFIT: escalate to Chief Actuary

Ongoing monitoring schedule (PRA SS1/23 Principle 5)
------------------------------------------------------
  - Quarterly: run this script on live portfolio data
  - Annually: independent revalidation (Tier {tier_result.tier} requirement: {tier_result.review_frequency})
  - Continuous: PITMonitor from insurance-monitoring for anytime-valid calibration
    change detection (avoids repeated-testing inflation in monthly H-L / A/E checks)
""")
