# Databricks notebook source

# MAGIC %md
# MAGIC # FCA Compliance Pipeline: Fairness Audit and Model Validation
# MAGIC
# MAGIC UK pricing actuaries operate under two overlapping regulatory frameworks that require documented evidence of compliance:
# MAGIC
# MAGIC - **FCA Consumer Duty (PRIN 2A):** firms must demonstrate good outcomes for all customer groups, including monitoring for differential outcomes by characteristics such as gender or socioeconomic group.
# MAGIC - **PRA SS1/23:** model risk management requirements mandating formal validation of all material risk models, including sensitivity analysis and out-of-time performance tests.
# MAGIC
# MAGIC This notebook runs a pricing model through both frameworks. The output is a structured audit record — the kind of thing you attach to a pricing committee paper or model validation submission.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Load synthetic motor data and train a CatBoost frequency model
# MAGIC 2. Proxy discrimination audit (`insurance-fairness`)
# MAGIC 3. PRA SS1/23 model validation report (`insurance-governance`)
# MAGIC 4. Interpret results and identify action items

# COMMAND ----------

# MAGIC %pip install insurance-datasets insurance-fairness insurance-governance catboost shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data and Train the Model Under Audit
# MAGIC
# MAGIC In a real compliance exercise the model already exists. Here we train a CatBoost frequency model so the rest of the notebook has something to audit.
# MAGIC
# MAGIC We add synthetic demographic proxies: `occupation_class` (proxy for socioeconomic group) and `disadvantaged_area` (binary proxy for area deprivation). Real motor portfolios don't contain gender or ethnicity columns (prohibited rating factors post-2013), but you might have aggregator channel flags, occupation classes, or ONS deprivation indices that correlate with protected characteristics.

# COMMAND ----------

from insurance_datasets import load_motor
from catboost import CatBoostRegressor, Pool

df_raw = load_motor(n_policies=20_000, seed=42)

rng = np.random.default_rng(42)
disadvantaged = (df_raw["area"].isin(["E", "F"])).astype(int)
noise = rng.binomial(1, 0.08, len(df_raw))
df_raw = df_raw.assign(disadvantaged_area=((disadvantaged + noise) % 2).astype(int))

print(f"Portfolio: {len(df_raw):,} policies")
print(f"Disadvantaged area proportion: {df_raw['disadvantaged_area'].mean():.2%}")

FEATURE_COLS = [
    "vehicle_age", "vehicle_group", "driver_age", "ncd_years",
    "area", "policy_type", "occupation_class",
]
CAT_COLS = ["area", "policy_type"]
cat_indices = [FEATURE_COLS.index(c) for c in CAT_COLS]

n_train = int(0.75 * len(df_raw))
train_pd = df_raw.iloc[:n_train].copy()
holdout_pd = df_raw.iloc[n_train:].copy()

train_log_exp = np.log(train_pd["exposure"].clip(lower=1e-9).values)
holdout_log_exp = np.log(holdout_pd["exposure"].clip(lower=1e-9).values)

train_pool = Pool(
    data=train_pd[FEATURE_COLS], label=train_pd["claim_count"],
    cat_features=cat_indices, baseline=train_log_exp,
)
holdout_pool = Pool(
    data=holdout_pd[FEATURE_COLS], label=holdout_pd["claim_count"],
    cat_features=cat_indices, baseline=holdout_log_exp,
)

model = CatBoostRegressor(
    loss_function="Poisson", iterations=300, depth=4,
    learning_rate=0.06, l2_leaf_reg=3.0, random_seed=42, verbose=0,
)
model.fit(train_pool, eval_set=holdout_pool)

holdout_pd = holdout_pd.copy()
holdout_pd["predicted_rate"] = model.predict(holdout_pd[FEATURE_COLS])

holdout_actuals = holdout_pd["claim_count"].values
holdout_preds = holdout_pd["predicted_rate"].values
holdout_exp = holdout_pd["exposure"].values

holdout_rmse = float(np.sqrt(np.mean((holdout_preds - holdout_actuals) ** 2)))
calibration = float((holdout_preds * holdout_exp).sum() / max(holdout_actuals.sum(), 1e-9))
print(f"Model trained. Holdout RMSE: {holdout_rmse:.5f}, calibration: {calibration:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: FCA Consumer Duty Proxy Discrimination Audit
# MAGIC
# MAGIC The FCA does not require pricing to produce equal outcomes across demographic groups — it requires you to demonstrate that:
# MAGIC - (a) protected characteristics are not rating factors (direct discrimination)
# MAGIC - (b) rating factors are not functioning as proxies for protected characteristics in a way that produces unjustifiable differential outcomes (indirect discrimination)
# MAGIC
# MAGIC `insurance-fairness` runs: proxy detection (R-squared and mutual information per factor), demographic parity, calibration by group, and the disparate impact ratio (4/5ths rule).

# COMMAND ----------

from insurance_fairness import FairnessAudit

holdout_pl = pl.from_pandas(holdout_pd)

audit = FairnessAudit(
    model=model,
    data=holdout_pl,
    protected_cols=["disadvantaged_area", "occupation_class"],
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=FEATURE_COLS,
    model_name="Motor Frequency Model v1.0",
    run_proxy_detection=True,
    run_counterfactual=False,
    n_bootstrap=0,  # skip bootstrapped CIs for demo speed
    proxy_catboost_iterations=100,
)

print("Running audit... (proxy detection fits CatBoost models per factor)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fairness_report = audit.run()

fairness_report.summary()

print("\nDetailed proxy scores by factor (top 5 per characteristic):")
for pc, pc_result in fairness_report.results.items():
    if pc_result.proxy_detection is None:
        continue
    print(f"\n  Protected characteristic: {pc}")
    for score in sorted(pc_result.proxy_detection.scores, key=lambda s: s.proxy_r2, reverse=True)[:5]:
        print(
            f"    {score.factor:<25}"
            f"  proxy_R2={score.proxy_r2:.4f}"
            f"  MI={score.mutual_information:.4f}"
            f"  [{score.rag.upper()}]"
        )

if fairness_report.flagged_factors:
    print(f"\nFactors requiring investigation: {', '.join(fairness_report.flagged_factors)}")
else:
    print("\nNo factors flagged as proxies at the default thresholds.")

tmpdir = Path(tempfile.mkdtemp(prefix="bc_compliance_"))
audit_md_path = tmpdir / "fairness_audit.md"
fairness_report.to_markdown(str(audit_md_path))
print(f"\nMarkdown audit report saved to: {audit_md_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: PRA SS1/23 Model Validation Report
# MAGIC
# MAGIC PRA SS1/23 requires insurers to produce formal model validation documentation for all material models. The minimum requirements are: data quality assessment, performance metrics (RMSE, Gini, A/E, lift chart), stability analysis, discrimination metrics with CIs, and Hosmer-Lemeshow calibration test.
# MAGIC
# MAGIC `insurance-governance` provides this via its validation subpackage (formerly the standalone `insurance-validation` package, now merged into `insurance-governance`).

# COMMAND ----------

from insurance_governance.validation import ModelCard, ModelValidationReport, PerformanceReport

card = ModelCard(
    name="Motor Frequency Model",
    version="1.0.0",
    purpose="Predict claim frequency for UK private motor portfolio.",
    methodology="CatBoost Poisson GBM, 300 trees, depth 4, L2 reg 3.0.",
    target="claim_count",
    features=FEATURE_COLS,
    limitations=[
        "No telematics data — mileage is self-declared annual_mileage only",
        "No claims history beyond NCD years",
        "Trained on synthetic data — must be retrained on actual portfolio",
    ],
    owner="Pricing Team",
)

train_pd_copy = train_pd.copy()
train_pd_copy["predicted_rate"] = model.predict(train_pd_copy[FEATURE_COLS])

print("Running validation tests...")
report = ModelValidationReport(
    model_card=card,
    y_val=holdout_actuals,
    y_pred_val=holdout_preds,
    exposure_val=holdout_exp,
    y_train=train_pd_copy["claim_count"].values,
    y_pred_train=train_pd_copy["predicted_rate"].values,
)

html_path = tmpdir / "validation_report.html"
json_path = tmpdir / "validation_report.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    report.generate(str(html_path))
    report.to_json(str(json_path))

print(f"HTML validation report: {html_path}")
print(f"JSON validation report: {json_path}")

perf = PerformanceReport(holdout_actuals, holdout_preds, exposure=holdout_exp)

gini_result = perf.gini_with_ci()
ae_result = perf.ae_with_poisson_ci()
hl_result = perf.hosmer_lemeshow_test()

gini_val = gini_result.metric_value
ae_val = ae_result.metric_value

print(f"\nKey validation metrics (holdout set, n={len(holdout_pd):,} policies):")
print(f"  Gini:  {gini_val:.4f}  (95% CI: [{gini_result.extra['ci_lower']:.4f}, {gini_result.extra['ci_upper']:.4f}])")
print(f"  A/E:   {ae_val:.4f}  (95% CI: [{ae_result.extra['ci_lower']:.4f}, {ae_result.extra['ci_upper']:.4f}])")
print(f"  H-L p: {hl_result.extra['p_value']:.4f}  ({'PASS' if hl_result.passed else 'REVIEW'})")
print(f"  Cal:   {calibration:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Interpret Results and Action Items
# MAGIC
# MAGIC Use this section to generate the management summary and identify required actions before model deployment.

# COMMAND ----------

overall_rag = fairness_report.overall_rag.upper()
hl_pval = hl_result.extra.get("p_value")

print(f"Compliance summary")
print(f"------------------")
print(f"Fairness audit:    {overall_rag}  (Consumer Duty / Equality Act 2010)")
print(f"Validation Gini:   {gini_val:.4f}")
print(f"A/E ratio:         {ae_val:.4f}")
print(f"H-L p-value:       {hl_pval:.4f}  ({'calibration OK' if hl_result.passed else 'calibration concerns'})")
print()
print("Required before deployment:")
print("1. Document actuarial justification for any flagged factors (FCA requirement).")
print("2. Independent validator must review and sign off the HTML validation report.")
print(f"3. Record in model risk register: date {fairness_report.audit_date}, RAG {overall_rag}.")
print()
print("Ongoing monitoring:")
print("Re-run this notebook quarterly with the live portfolio. The FCA's Consumer")
print("Duty multi-firm review (2024) found firms do not monitor differential outcomes")
print("continuously — regulators are looking for ongoing rather than point-in-time compliance.")
