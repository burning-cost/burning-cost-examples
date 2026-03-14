# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-datasets: Synthetic UK Insurance Data with Known DGPs
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC When you are building and testing a pricing model, you need data where you
# MAGIC know what the right answer is. Real policyholder data has unknown true
# MAGIC parameters, access restrictions, and structural quirks that make it hard to
# MAGIC tell whether a failing model is wrong or the data is weird.
# MAGIC
# MAGIC `insurance-datasets` gives you clean, realistic synthetic UK motor and home
# MAGIC insurance data where the true Poisson-Gamma DGP parameters are published.
# MAGIC You fit a GLM, compare the coefficients to the known truth, and you know
# MAGIC immediately whether your implementation is correct.
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC - Load and inspect the motor and home datasets
# MAGIC - Understand the column schema and rating factor distributions
# MAGIC - Inspect the published true DGP parameters (frequency and severity)
# MAGIC - Fit a Poisson frequency GLM and verify coefficients recover the true values
# MAGIC - Fit a Gamma severity GLM on claims-only data
# MAGIC - Demonstrate the full burning cost calculation: frequency x severity
# MAGIC - Compare model relativities to the known DGP relativities

# COMMAND ----------

# MAGIC %pip install insurance-datasets statsmodels matplotlib pandas numpy

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the motor dataset

# COMMAND ----------

from insurance_datasets import (
    load_motor,
    load_home,
    MOTOR_TRUE_FREQ_PARAMS,
    MOTOR_TRUE_SEV_PARAMS,
    HOME_TRUE_FREQ_PARAMS,
    HOME_TRUE_SEV_PARAMS,
)

motor = load_motor(n_policies=50_000, seed=42)
home  = load_home(n_policies=50_000, seed=42)

print("Motor shape:", motor.shape)
print("Home shape: ", home.shape)
print()
print("Motor columns:")
print(motor.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Schema overview and rating factor distributions

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== Motor dataset: first 5 rows ===")
print(motor.head())
print()
print("=== Key statistics ===")
print(f"Policies:          {len(motor):,}")
print(f"Total exposure:    {motor['exposure'].sum():,.0f} earned years")
print(f"Claim count:       {motor['claim_count'].sum():,}")
print(f"Portfolio freq:    {motor['claim_count'].sum() / motor['exposure'].sum():.3%}")
claimants = motor[motor['claim_count'] > 0]
print(f"Mean severity:     £{claimants['incurred'].mean():,.0f}")
print()
print("Area band distribution:")
print(motor['area'].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The known true DGP parameters

# COMMAND ----------

print("=== Motor frequency DGP (Poisson, log-link) ===")
for k, v in MOTOR_TRUE_FREQ_PARAMS.items():
    print(f"  {k:<30} {v:+.4f}  (relativity: {np.exp(v):.3f})")

print()
print("=== Motor severity DGP (Gamma, log-link, shape=2) ===")
for k, v in MOTOR_TRUE_SEV_PARAMS.items():
    print(f"  {k:<30} {v:+.4f}  (relativity: {np.exp(v):.3f})")

print()
print("True baseline frequency: ~{:.1%}".format(np.exp(MOTOR_TRUE_FREQ_PARAMS['intercept'])))
print("True baseline severity:  £{:,.0f}".format(np.exp(MOTOR_TRUE_SEV_PARAMS['intercept'])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit a Poisson frequency GLM and recover the true coefficients

# COMMAND ----------

import statsmodels.api as sm
import statsmodels.formula.api as smf

df = motor.copy()

# Engineer features to match DGP parameterisation
df["log_exposure"]      = np.log(df["exposure"].clip(lower=1e-6))
df["has_convictions"]   = (df["conviction_points"] > 0).astype(int)

# Young driver blend: full load under 25, linear taper 25-30, zero from 30
df["young_driver"] = np.where(
    df["driver_age"] < 25, 1.0,
    np.where(df["driver_age"] < 30, (30 - df["driver_age"]) / 5.0, 0.0)
)
df["old_driver"] = (df["driver_age"] >= 70).astype(int)

for band in ["B", "C", "D", "E", "F"]:
    df[f"area_{band}"] = (df["area"] == band).astype(int)

features = [
    "vehicle_group", "ncd_years", "young_driver", "old_driver",
    "has_convictions",
    "area_B", "area_C", "area_D", "area_E", "area_F",
]

X = sm.add_constant(df[features])
freq_model = sm.GLM(
    df["claim_count"],
    X,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df["log_exposure"],
).fit()

print(freq_model.summary2())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare fitted coefficients to the true DGP

# COMMAND ----------

# Map statsmodels param names to DGP names
name_map = {
    "const":           "intercept",
    "vehicle_group":   "vehicle_group",
    "ncd_years":       "ncd_years",
    "young_driver":    "driver_age_young",
    "old_driver":      "driver_age_old",
    "has_convictions": "has_convictions",
    "area_B":          "area_B",
    "area_C":          "area_C",
    "area_D":          "area_D",
    "area_E":          "area_E",
    "area_F":          "area_F",
}

print(f"{'Parameter':<30} {'True':>8} {'Fitted':>8} {'Error':>8}")
print("-" * 58)
for sm_name, dgp_name in name_map.items():
    if dgp_name in MOTOR_TRUE_FREQ_PARAMS:
        true_val   = MOTOR_TRUE_FREQ_PARAMS[dgp_name]
        fitted_val = freq_model.params[sm_name]
        error      = fitted_val - true_val
        print(f"  {dgp_name:<28} {true_val:>8.4f} {fitted_val:>8.4f} {error:>+8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Fit a Gamma severity GLM on claims-only data

# COMMAND ----------

claims = df[df["claim_count"] > 0].copy()
claims["avg_severity"] = claims["incurred"] / claims["claim_count"]

sev_features = ["vehicle_group", "young_driver"]
X_sev = sm.add_constant(claims[sev_features])

sev_model = sm.GLM(
    claims["avg_severity"],
    X_sev,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print(sev_model.summary2())
print()
print(f"{'Parameter':<30} {'True':>8} {'Fitted':>8} {'Error':>8}")
print("-" * 58)
sev_name_map = {"const": "intercept", "vehicle_group": "vehicle_group", "young_driver": "driver_age_young"}
for sm_name, dgp_name in sev_name_map.items():
    if dgp_name in MOTOR_TRUE_SEV_PARAMS:
        true_val   = MOTOR_TRUE_SEV_PARAMS[dgp_name]
        fitted_val = sev_model.params[sm_name]
        error      = fitted_val - true_val
        print(f"  {dgp_name:<28} {true_val:>8.4f} {fitted_val:>8.4f} {error:>+8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Burning cost: frequency x severity relativities

# COMMAND ----------

# Build a rating grid to show pure premium relativities
rating_grid = pd.DataFrame({
    "vehicle_group":   np.arange(1, 51),
    "ncd_years":       [2] * 50,
    "young_driver":    [0.0] * 50,
    "old_driver":      [0.0] * 50,
    "has_convictions": [0] * 50,
    "area_B":          [0] * 50,
    "area_C":          [0] * 50,
    "area_D":          [0] * 50,
    "area_E":          [0] * 50,
    "area_F":          [0] * 50,
})

X_grid      = sm.add_constant(rating_grid[features])
X_grid_sev  = sm.add_constant(rating_grid[sev_features])

freq_pred = freq_model.predict(X_grid)   # per year, no offset needed (offset=0 -> exp(0)=1)
sev_pred  = sev_model.predict(X_grid_sev)
pure_prem = freq_pred * sev_pred

# Relativities indexed to vehicle_group=25 (mid-point)
base_pp    = pure_prem.iloc[24]
relativity = pure_prem / base_pp

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(np.arange(1, 51), freq_pred, color="steelblue")
axes[0].set_title("Fitted Frequency by Vehicle Group\n(NCD=2, Area A, no convictions, age 35)")
axes[0].set_xlabel("ABI Vehicle Group")
axes[0].set_ylabel("Claim frequency (per year)")

axes[1].plot(np.arange(1, 51), sev_pred, color="coral")
axes[1].set_title("Fitted Severity by Vehicle Group")
axes[1].set_xlabel("ABI Vehicle Group")
axes[1].set_ylabel("Mean claim cost (£)")

axes[2].plot(np.arange(1, 51), relativity, color="seagreen")
axes[2].axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
axes[2].set_title("Pure Premium Relativity\n(indexed to group 25)")
axes[2].set_xlabel("ABI Vehicle Group")
axes[2].set_ylabel("Relativity")

plt.tight_layout()
plt.savefig("/tmp/insurance_datasets_relativities.png", dpi=120)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Home dataset: inspect and verify

# COMMAND ----------

print("=== Home dataset: first 5 rows ===")
print(home.head())
print()
print("=== Home key statistics ===")
print(f"Policies:          {len(home):,}")
print(f"Total exposure:    {home['exposure'].sum():,.0f} earned years")
print(f"Claim count:       {home['claim_count'].sum():,}")
print(f"Portfolio freq:    {home['claim_count'].sum() / home['exposure'].sum():.3%}")
home_claims = home[home['claim_count'] > 0]
print(f"Mean severity:     £{home_claims['incurred'].mean():,.0f}")
print()
print("=== Home frequency DGP ===")
for k, v in HOME_TRUE_FREQ_PARAMS.items():
    print(f"  {k:<40} {v:+.4f}  (relativity: {np.exp(v):.3f})")
print()
print("=== Home severity DGP ===")
for k, v in HOME_TRUE_SEV_PARAMS.items():
    print(f"  {k:<40} {v:+.4f}  (relativity: {np.exp(v):.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC `insurance-datasets` gives you:
# MAGIC
# MAGIC - Two realistic UK insurance datasets (motor and home) with sensible column schemas
# MAGIC - Published true DGP parameters — you can check your model recovers the right coefficients
# MAGIC - Poisson-Gamma structure that mirrors what UK pricing teams actually fit
# MAGIC
# MAGIC The GLM recovers the true frequency coefficients to within a few hundredths
# MAGIC of a log-point on 50k policies — exactly what you would expect from a correctly
# MAGIC specified model on data generated from that DGP.
# MAGIC
# MAGIC Use this package to:
# MAGIC - **Benchmark new modelling approaches** — if your model can't recover
# MAGIC   known coefficients on clean data, it won't work on real data
# MAGIC - **Test preprocessing pipelines** — does your feature engineering produce
# MAGIC   the features the DGP needs?
# MAGIC - **Teach pricing concepts** — show trainees what a well-specified GLM looks like
# MAGIC
# MAGIC Next steps: try fitting the model on a subset (10k, 5k, 1k policies) and
# MAGIC watch the coefficient variance grow as the data thins out.
