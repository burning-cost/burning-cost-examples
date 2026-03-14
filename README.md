# burning-cost-examples

Databricks notebooks demonstrating the burning-cost library ecosystem on realistic synthetic insurance data.

Each notebook is self-contained: it installs its dependencies, generates synthetic data, fits models, and produces benchmark comparisons against standard actuarial approaches. Run them on Databricks serverless compute.

## Notebooks

| Notebook | Library | What it shows |
|----------|---------|---------------|
| [insurance_trend_demo.py](notebooks/insurance_trend_demo.py) | [insurance-trend](https://github.com/burning-cost/insurance-trend) | Automated freq/sev trend selection vs naive OLS on UK motor data with a structural break |
| [insurance_distributional_demo.py](notebooks/insurance_distributional_demo.py) | [insurance-dispersion](https://github.com/burning-cost/insurance-dispersion) | Double GLM (DGLM) vs constant-phi Gamma GLM on 50k motor severity records with heteroscedastic DGP |

## How to run

Import a notebook into your Databricks workspace:

```bash
databricks workspace import notebooks/insurance_distributional_demo.py \
  /Workspace/Users/you@example.com/insurance_distributional_demo \
  --language PYTHON --overwrite
```

Or drag-and-drop the `.py` file in the Databricks UI (File > Import).

All notebooks use `%pip install` cells and `dbutils.library.restartPython()` — no pre-installed libraries required beyond what Databricks Runtime provides.

## Format

Notebooks use the Databricks `.py` format:
- `# COMMAND ----------` separates cells
- `# MAGIC %md` lines are markdown cells
- Compatible with Databricks Runtime 13.x and above

## Note on synthetic data

All portfolios generated in these examples are entirely synthetic. They have realistic statistical properties but do not represent any real book of business. In production, replace the synthetic data generation step with your actual policy and claims extract.
