# Burning Cost Examples

End-to-end insurance pricing workflows using the Burning Cost open-source libraries.

---

## The problem

UK pricing teams often work in silos. The data science team builds a GBM. The actuarial team validates it in a separate spreadsheet. The IT team deploys it without an audit trail. The compliance team scrambles when the FCA asks for ENBP records.

These examples show how the Burning Cost stack fits together into coherent, end-to-end workflows — from synthetic data through to ICOBS 6B.2.51R audit reporting and FCA Consumer Duty evidence packs.

---

## Pipeline examples

Full workflows that span multiple libraries and stages of the pricing lifecycle. Start here if you want to understand how the pieces fit together.

### `end_to_end_motor_pricing.py`

A complete UK motor pricing pipeline: synthetic portfolio generation via vine copula, CatBoost frequency model training, SHAP relativity extraction, holdout validation, and champion/challenger deployment with ENBP audit logging.

```bash
pip install insurance-synthetic shap-relativities insurance-deploy catboost shap scipy polars
uv run python examples/end_to_end_motor_pricing.py
```

### `gbm_to_glm_pipeline.py`

Takes a fitted CatBoost frequency model and converts it to a GLM-compatible relativity table: SHAP decomposition into additive contributions, monotonicity smoothing, and GLM re-fit with SHAP-derived offsets. The output is a rating table an underwriting committee can read and a regulator can audit.

```bash
pip install insurance-synthetic shap-relativities catboost shap polars scikit-learn
uv run python examples/gbm_to_glm_pipeline.py
```

### `fca_compliance_pipeline.py`

Runs a full Consumer Duty compliance workflow: proxy discrimination audit using insurance-fairness, ENBP breach detection via insurance-deploy, and causal rate change attribution with insurance-causal-policy. Output is a structured evidence pack in Markdown.

```bash
pip install insurance-fairness insurance-deploy insurance-causal-policy polars scipy
uv run python examples/fca_compliance_pipeline.py
```

### `thin_data_pipeline.py`

Demonstrates credibility-weighted pricing for low-volume segments: Bühlmann-Straub credibility blending of segment-level experience with portfolio priors, hierarchical Bayesian posterior estimation, and uncertainty quantification for underwriting sign-off.

```bash
pip install insurance-credibility bayesian-pricing polars scipy numpy
uv run python examples/thin_data_pipeline.py
```

---

## Library deep dives

Focused examples that explore one library in depth. Use these when you are integrating a specific library and want to understand its full API.

### `causal_rate_change_evaluation.py`

Full SDID workflow using [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy): synthetic motor portfolio with staggered rate changes across segments, event study pre-treatment validation, HonestDiD sensitivity analysis, and structured FCA Consumer Duty evidence pack output. The example shows why a before-and-after comparison gives a biased estimate and how SDID eliminates that bias.

```bash
pip install insurance-causal-policy polars scipy cvxpy matplotlib
uv run python examples/causal_rate_change_evaluation.py
```

### `price_elasticity_optimisation.py`

Complete DML elasticity workflow using [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity): estimation on a synthetic 50,000-policy motor book, heterogeneous CATE by NCD band, channel, and age, ENBP-constrained profit-maximising optimiser, and an efficient frontier showing the renewal rate versus expected profit trade-off. Covers the near-deterministic price problem and how to diagnose it before fitting.

```bash
pip install "insurance-elasticity[all]" polars
uv run python examples/price_elasticity_optimisation.py
```

### `conformal_prediction_intervals.py`

Tweedie conformal prediction intervals versus bootstrap using [insurance-conformal](https://github.com/burning-cost/insurance-conformal): side-by-side comparison on a synthetic motor book, per-segment coverage analysis across risk deciles and vehicle groups, and interval width distribution. Shows exactly where the bootstrap fails its stated coverage target and confirms conformal holds by construction.

```bash
pip install "insurance-conformal[all]" catboost polars
uv run python examples/conformal_prediction_intervals.py
```

### `champion_challenger_deployment.py`

Full deployment lifecycle using [insurance-deploy](https://github.com/burning-cost/insurance-deploy): shadow mode experiment setup, per-quote logging with ENBP compliance flags, bootstrap likelihood-ratio test for model promotion, ENBP audit report generation, and power analysis showing realistic timelines to loss ratio significance. A practical reference for building a first FCA-compliant champion/challenger setup.

```bash
pip install insurance-deploy polars scipy numpy
uv run python examples/champion_challenger_deployment.py
```

### `model_drift_monitoring.py`

Full monitoring stack using [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) with three deliberately induced failure modes: covariate shift (older driver mix), calibration deterioration (segment-level A/E drift), and discriminatory power loss (Gini decay). Covers exposure-weighted PSI and CSI, segment A/E ratios with Poisson confidence intervals, Gini drift z-test, and governance reporting suitable for a PRA SS1/23 model risk log.

```bash
pip install insurance-monitoring polars numpy scipy
uv run python examples/model_drift_monitoring.py
```

---

## Installation

Requires Python 3.10 or later. We recommend [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/burning-cost/burning-cost-examples
cd burning-cost-examples
uv sync
```

Individual dependency commands are listed under each example above. The `uv sync` command installs everything needed for all examples at once.

---

## Note on synthetic data

All portfolios generated in these examples are entirely synthetic. They have realistic statistical properties but do not represent any real book of business. In production, replace the synthetic data generation step with your actual policy and claims extract. Column names match the `uk_motor_schema()` definition in [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) — rename your columns to match and the rest of each pipeline runs unchanged.

---

## Further reading

Methodology posts and detailed walkthroughs for each library are at [burning-cost.github.io](https://burning-cost.github.io).

---

## Licence

MIT
