# burning-cost-examples

Databricks notebooks demonstrating the burning-cost library ecosystem on realistic synthetic insurance data.

Each notebook is self-contained: it installs its dependencies, generates synthetic data, fits models, and produces benchmark comparisons against standard actuarial approaches. Run them on Databricks serverless compute.

## Start here: Burning Cost in 30 Minutes

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning_cost_30_minutes.ipynb)

A single Colab notebook that runs in-browser with zero setup. Covers `insurance-causal`, `insurance-conformal`, and `insurance-monitoring` in one end-to-end workflow on synthetic UK motor data. ~5 minutes runtime on Colab free tier.

[View notebook](notebooks/burning_cost_30_minutes.ipynb)

---

## Notebooks

| Notebook | Library | What it shows |
|----------|---------|---------------|
| [bayesian_pricing_demo.py](notebooks/bayesian_pricing_demo.py) | [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian vs raw experience on thin segments |
| [insurance_causal_demo.py](notebooks/insurance_causal_demo.py) | [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal effect vs naive Poisson GLM on confounded data |
| [causal_rate_change_evaluation.py](notebooks/causal_rate_change_evaluation.py) | [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID rate change evaluation with event study and HonestDiD |
| [conformal_prediction_intervals.py](notebooks/conformal_prediction_intervals.py) | [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Tweedie conformal intervals vs bootstrap on 50k motor |
| [insurance_conformal_ts_demo.py](notebooks/insurance_conformal_ts_demo.py) | [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | ACI/SPCI vs split conformal on non-exchangeable time series |
| [insurance_covariate_shift_demo.py](notebooks/insurance_covariate_shift_demo.py) | [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift) | Importance-weighted evaluation after distribution shift |
| [insurance_credibility_demo.py](notebooks/insurance_credibility_demo.py) | [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility vs raw experience on 30 segments |
| [temporal_cross_validation.py](notebooks/temporal_cross_validation.py) | [insurance-cv](https://github.com/burning-cost/insurance-cv) | Random CV vs temporal CV vs true OOT holdout |
| [champion_challenger_deployment.py](notebooks/champion_challenger_deployment.py) | [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Shadow mode, quote logging, bootstrap LR test, ENBP audit |
| [insurance_distill_demo.py](notebooks/insurance_distill_demo.py) | [insurance-distill](https://github.com/burning-cost/insurance-distill) | GBM-to-GLM distillation: CatBoost surrogate factor tables for Radar/Emblem |
| [insurance_dispersion_demo.py](notebooks/insurance_dispersion_demo.py) | [insurance-dispersion](https://github.com/burning-cost/insurance-dispersion) | DGLM vs constant-phi Gamma GLM, per-risk volatility scoring |
| [insurance_distributional_demo.py](notebooks/insurance_distributional_demo.py) | [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Distributional GBM (TweedieGBM) vs standard point predictions |
| [insurance_distributional_glm_demo.py](notebooks/insurance_distributional_glm_demo.py) | [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS vs standard Gamma GLM on heterogeneous-variance data |
| [insurance_dynamics_demo.py](notebooks/insurance_dynamics_demo.py) | [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | GAS Poisson filter vs static GLM, BOCPD changepoint detection |
| [fairness_audit_demo.py](notebooks/fairness_audit_demo.py) | [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination audit, bias metrics, Lindholm correction |
| [insurance_frequency_severity_demo.py](notebooks/insurance_frequency_severity_demo.py) | [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Sarmanov copula joint freq-sev vs independence assumption |
| [insurance_gam_demo.py](notebooks/insurance_gam_demo.py) | [insurance-gam](https://github.com/burning-cost/insurance-gam) | EBM/ANAM vs Poisson GLM with planted non-linear effects |
| [insurance_glm_tools_demo.py](notebooks/insurance_glm_tools_demo.py) | [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | Nested GLM embeddings for 500 vehicle makes vs dummy-coded GLM |
| [insurance_governance_demo.py](notebooks/insurance_governance_demo.py) | [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 validation workflow, MRM risk tiering, HTML report |
| [insurance_interactions_demo.py](notebooks/insurance_interactions_demo.py) | [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | CANN/NID interaction detection vs exhaustive pairwise GLM search |
| [monitoring_drift_detection.py](notebooks/monitoring_drift_detection.py) | [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Exposure-weighted PSI/CSI, A/E ratios, Gini drift z-test |
| [insurance_multilevel_demo.py](notebooks/insurance_multilevel_demo.py) | [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | CatBoost + REML random effects vs one-hot encoding |
| [insurance_optimise_demo.py](notebooks/insurance_optimise_demo.py) | [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | SLSQP constrained optimisation, efficient frontier, FCA audit |
| [insurance_quantile_demo.py](notebooks/insurance_quantile_demo.py) | [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | CatBoost quantile regression vs lognormal, TVaR, ILF curves |
| [insurance_severity_demo.py](notebooks/insurance_severity_demo.py) | [insurance-severity](https://github.com/burning-cost/insurance-severity) | Spliced Lognormal-GPD + DRN vs Gamma GLM, tail quantiles |
| [spatial_territory_ratemaking.py](notebooks/spatial_territory_ratemaking.py) | [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 territory factors vs postcode grouping, Moran's I |
| [insurance_survival_demo.py](notebooks/insurance_survival_demo.py) | [insurance-survival](https://github.com/burning-cost/insurance-survival) | Cure models vs KM/Cox PH, CLV bias by cure band |
| [synthetic_portfolio_generation.py](notebooks/synthetic_portfolio_generation.py) | [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Vine copula generation, fidelity report, TSTR benchmarks |
| [insurance_telematics_demo.py](notebooks/insurance_telematics_demo.py) | [insurance-telematics](https://github.com/burning-cost/insurance-telematics) | HMM latent-state features vs raw trip aggregates |
| [insurance_thin_data_demo.py](notebooks/insurance_thin_data_demo.py) | [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) | GLMTransfer + TabPFN vs raw GLM on thin segments |
| [insurance_trend_demo.py](notebooks/insurance_trend_demo.py) | [insurance-trend](https://github.com/burning-cost/insurance-trend) | Automated trend selection vs naive OLS, structural breaks |
| [insurance_whittaker_demo.py](notebooks/insurance_whittaker_demo.py) | [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | W-H smoothing with REML lambda vs manual step smoothing |
| [shap_relativities_demo.py](notebooks/shap_relativities_demo.py) | [shap-relativities](https://github.com/burning-cost/shap-relativities) | CatBoost relativities vs GLM vs true DGP on synthetic motor |

## How to run

### Colab (no setup required)

Click the badge at the top of this README. The 30-minute notebook runs entirely in-browser.

### Databricks

Import a notebook into your Databricks workspace:

```bash
databricks workspace import notebooks/insurance_distributional_demo.py \
  /Workspace/Users/you@example.com/insurance_distributional_demo \
  --language PYTHON --overwrite
```

Or drag-and-drop the `.py` file in the Databricks UI (File > Import).

All notebooks use `%pip install` cells and `dbutils.library.restartPython()` — no pre-installed libraries required beyond what Databricks Runtime provides.

## Format

Databricks notebooks use the `.py` format:
- `# COMMAND ----------` separates cells
- `# MAGIC %md` lines are markdown cells
- Compatible with Databricks Runtime 13.x and above

The `burning_cost_30_minutes.ipynb` file is standard Jupyter format, compatible with Colab and local Jupyter.

## Note on synthetic data

All portfolios generated in these examples are entirely synthetic. They have realistic statistical properties but do not represent any real book of business. In production, replace the synthetic data generation step with your actual policy and claims extract.
