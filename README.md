# Burning Cost Examples

End-to-end insurance pricing workflows using the Burning Cost open-source libraries.

---

## The problem

UK pricing teams often work in silos. The data science team builds a GBM. The actuarial team validates it in a separate spreadsheet. The IT team deploys it without an audit trail. The compliance team scrambles when the FCA asks for ENBP records.

These examples show how the Burning Cost stack fits together into a single, coherent workflow — from synthetic data through to ICOBS 6B.2.51R audit reporting.

---

## What the example does

`examples/end_to_end_motor_pricing.py` runs a full UK motor pricing pipeline:

1. **Generate a synthetic portfolio** using [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic). A vine copula captures the tail dependence between young driver age, high vehicle group, and zero NCD — the interactions that matter most for UK motor frequency.

2. **Train a CatBoost frequency model** — Poisson GBM with log(exposure) offset. Two versions: a conservative champion and a more complex challenger. CatBoost handles categorical rating factors natively, which avoids the SHAP distortion you get from one-hot encoding.

3. **Extract SHAP relativities** using [shap-relativities](https://github.com/burning-cost/shap-relativities). Output is a table of `(feature, level, relativity)` triples in the same format as GLM `exp(beta)` relativities — readable by any pricing actuary, presentable to an underwriting committee.

4. **Run a model validation** using manual checks (calibration ratio, RMSE lift, Gini coefficient). The [insurance-validation](https://github.com/burning-cost/insurance-validation) library will slot in here when released to produce full PRA SS1/23-compliant documentation.

5. **Deploy with champion/challenger** using [insurance-deploy](https://github.com/burning-cost/insurance-deploy). Shadow mode by default — the challenger scores every quote without affecting customer prices. Hash-based deterministic routing means every decision can be audited. ENBP compliance is tracked per ICOBS 6B.2.51R.

The script prints outputs at each stage. Read it top-to-bottom as a workflow guide.

---

## Installation

Requires Python 3.10 or later. We recommend [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/burning-cost/burning-cost-examples
cd burning-cost-examples
uv sync
```

With pip:

```bash
pip install insurance-synthetic shap-relativities insurance-deploy catboost shap scipy polars
```

---

## Run the example

```bash
uv run python examples/end_to_end_motor_pricing.py
```

Expected runtime: 2–5 minutes. The vine copula fitting and SHAP computation are the bottlenecks.

---

## Libraries used

| Library | What it does |
|---------|-------------|
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Vine copula synthetic portfolio generator with actuarially valid exposure handling |
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Multiplicative rating factor relativities from GBM SHAP values |
| [insurance-validation](https://github.com/burning-cost/insurance-validation) | PRA SS1/23 model validation documentation (forthcoming) |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger framework with ENBP audit trail (ICOBS 6B.2.51R) |

---

## Further reading

Methodology posts and worked examples are at [burning-cost.github.io](https://burning-cost.github.io).

The Burning Cost Pricing Course covers the actuarial background behind each step — vine copulas for insurance data, SHAP attribution for correlated rating factors, and champion/challenger design under FCA Consumer Duty.

---

## Note on synthetic data

The portfolio generated in step 1 is entirely synthetic. It has realistic statistical properties but does not represent any real book of business. In production, replace the seed portfolio construction with your actual policy and claims extract. The column names match the `uk_motor_schema()` definition in insurance-synthetic — rename your columns to match and the rest of the pipeline runs unchanged.

---

## Licence

MIT
