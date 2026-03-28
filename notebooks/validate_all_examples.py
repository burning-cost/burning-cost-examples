# Databricks notebook source
# MAGIC %md
# MAGIC # Validate all pipeline example scripts
# MAGIC
# MAGIC Runs each script in `/examples/` as a subprocess and reports pass/fail.
# MAGIC
# MAGIC This is the CI check notebook for the burning-cost-examples repo.
# MAGIC Submitted via the Jobs API with dependencies in the environment spec.

# COMMAND ----------

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

EXAMPLES_DIR = Path("/Workspace/Shared/burning-cost-examples/examples")
RESULTS = {}


def run_script(script_path: str, timeout: int = 600) -> dict:
    """Run a Python script as a subprocess, capture stdout/stderr."""
    start = datetime.utcnow()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        elapsed = (datetime.utcnow() - start).total_seconds()
        return {
            "script": Path(script_path).name,
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
            "elapsed_s": elapsed,
            "status": (
                "SKIP" if result.returncode == 0 and result.stdout.startswith("SKIP:")
                else "PASS" if result.returncode == 0
                else "FAIL"
            ),
        }
    except subprocess.TimeoutExpired:
        elapsed = (datetime.utcnow() - start).total_seconds()
        return {
            "script": Path(script_path).name,
            "returncode": -1,
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
            "elapsed_s": elapsed,
            "status": "TIMEOUT",
        }
    except Exception as e:
        return {
            "script": Path(script_path).name,
            "returncode": -2,
            "stdout": "",
            "stderr": str(e),
            "elapsed_s": 0,
            "status": "ERROR",
        }


scripts = sorted(EXAMPLES_DIR.glob("*.py"))
print(f"Found {len(scripts)} example scripts:")
for s in scripts:
    print(f"  {s.name}")

# COMMAND ----------

# MAGIC %md ## Run: thin_data_pipeline.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "thin_data_pipeline.py"), timeout=300)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: gbm_to_glm_pipeline.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "gbm_to_glm_pipeline.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: gbm_to_tariff_pipeline.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "gbm_to_tariff_pipeline.py"), timeout=900)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: fca_compliance_pipeline.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "fca_compliance_pipeline.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: regulatory_compliance_pipeline.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "regulatory_compliance_pipeline.py"), timeout=900)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: conformal_prediction_intervals.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "conformal_prediction_intervals.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: champion_challenger_deployment.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "champion_challenger_deployment.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: causal_rate_change_evaluation.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "causal_rate_change_evaluation.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: causal_forest_demo.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "causal_forest_demo.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: model_drift_monitoring.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "model_drift_monitoring.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: sequential_testing_demo.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "sequential_testing_demo.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: price_elasticity_optimisation.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "price_elasticity_optimisation.py"), timeout=900)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: pareto_optimisation_demo.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "pareto_optimisation_demo.py"), timeout=600)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Run: end_to_end_motor_pricing.py

# COMMAND ----------

r = run_script(str(EXAMPLES_DIR / "end_to_end_motor_pricing.py"), timeout=900)
RESULTS[r["script"]] = r
print(f"[{r['status']}] {r['script']}  ({r['elapsed_s']:.1f}s)")
if r["status"] != "PASS":
    print("STDERR:", r["stderr"][-3000:])
    print("STDOUT:", r["stdout"][-3000:])
else:
    lines = r["stdout"].strip().split("\n")
    for line in lines[-40:]:
        print(line)

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("\n" + "=" * 72)
print("VALIDATION SUMMARY")
print("=" * 72)
print(f"{'Script':<45} {'Status':<8} {'Time':>7}")
print("-" * 62)

all_pass = True
for name, r in sorted(RESULTS.items()):
    icon = "OK" if r["status"] == "PASS" else r["status"]
    print(f"  {name:<43} [{icon}]  {r['elapsed_s']:>6.1f}s")
    if r["status"] not in ("PASS", "SKIP"):
        all_pass = False

print()
n_pass = sum(1 for r in RESULTS.values() if r["status"] == "PASS")
n_skip = sum(1 for r in RESULTS.values() if r["status"] == "SKIP")
n_total = len(RESULTS)
print(f"Result: {n_pass}/{n_total} scripts passed, {n_skip} skipped")

if all_pass:
    print("\nAll examples validated successfully.")
else:
    print("\nSome scripts failed:")
    for name, r in sorted(RESULTS.items()):
        if r["status"] != "PASS":
            print(f"\n  FAILED: {name}")
            if r["stderr"]:
                lines = r["stderr"].strip().split("\n")
                print(f"  Last 10 lines of stderr:")
                for line in lines[-10:]:
                    print(f"    {line}")

# Export results via dbutils.notebook.exit() for API capture
import json as _json
_summary = {
    name: {
        "status": r["status"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "stderr_tail": r["stderr"][-300:] if r.get("stderr") else "",
        "stdout_tail": r["stdout"][-300:] if r.get("stdout") else "",
    }
    for name, r in RESULTS.items()
}
_summary["__meta__"] = {
    "n_pass": n_pass,
    "n_skip": n_skip,
    "n_total": n_total,
    "all_pass": all_pass,
}
dbutils.notebook.exit(_json.dumps(_summary))
