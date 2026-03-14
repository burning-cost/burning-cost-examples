# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-datasets: Synthetic UK Insurance Data with a Known DGP
# MAGIC
# MAGIC ## The problem this library solves
# MAGIC
# MAGIC When you build or evaluate a pricing model, you need to know whether it is correct.
# MAGIC With real policyholder data, correctness is permanently unknowable — the true
# MAGIC parameters are never observed. You can check consistency, bias, lift — but you
# MAGIC cannot ask "does my GLM recover the true NCD coefficient?" because the true
# MAGIC NCD coefficient does not exist in any file you have access to.
# MAGIC
# MAGIC `insurance-datasets` solves this by providing synthetic UK insurance data where
# MAGIC the data generating process is published. The true Poisson frequency coefficients
# MAGIC and Gamma severity coefficients are available as Python dicts. You fit your model,
# MAGIC compare it to the ground truth, and see exactly how close you got.
# MAGIC
# MAGIC This is useful for:
# MAGIC
# MAGIC - **Library testing**: verify a new GLM implementation recovers the right answers
# MAGIC - **Benchmarking**: compare two modelling approaches on a problem with a known answer
# MAGIC - **Training**: show juniors what a "correct" model output looks like
# MAGIC - **Cross-validation research**: test whether a CV scheme gives unbiased estimates
# MAGIC
# MAGIC ## What this notebook covers
# MAGIC
# MAGIC 1. Load and inspect the motor and home datasets
# MAGIC 2. Inspect the published true DGP parameters
# MAGIC 3. Fit a Poisson frequency GLM and compare coefficients to ground truth
# MAGIC 4. Visualise coefficient recovery
# MAGIC 5. Fit a Gamma severity GLM
# MAGIC 6. Burning cost: frequency x severity relativities by vehicle group
# MAGIC 7. Home dataset: flood zone validation and GLM coefficient recovery
# MAGIC 8. Sample size stability: how much data do you need?
# MAGIC 9. Reproducibility guarantee

# COMMAND ----------

# MAGIC %pip install insurance-datasets statsmodels matplotlib pandas numpy --quiet

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_datasets import (
    load_motor,
    load_home,
    MOTOR_TRUE_FREQ_PARAMS,
    MOTOR_TRUE_SEV_PARAMS,
    HOME_TRUE_FREQ_PARAMS,
    HOME_TRUE_SEV_PARAMS,
)

print("insurance-datasets loaded OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the motor dataset
# MAGIC
# MAGIC One row per policy. 18 columns. Default 50,000 policies, accident years 2019–2023.
# MAGIC At 50k the coefficient estimates are stable — within a few percent of the true values.

# COMMAND ----------

motor = load_motor(n_policies=50_000, seed=42)

print(f"Shape: {motor.shape}")
print(f"\nColumns:\n{list(motor.columns)}")
print(f"\nClaim frequency: {motor['claim_count'].sum() / motor['exposure'].sum():.4f} per earned year")
print(f"Mean severity (claims only): £{motor.loc[motor['claim_count'] > 0, 'incurred'].mean():,.0f}")
print(f"\nAccident year distribution:\n{motor['accident_year'].value_counts().sort_index()}")
print(f"\nArea band distribution:\n{motor['area'].value_counts().sort_index()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The published true DGP parameters
# MAGIC
# MAGIC These are the parameters that generated the data. A correctly specified GLM fitted
# MAGIC on enough data should recover values close to these.

# COMMAND ----------

print("Motor frequency DGP (Poisson, log link):")
print(f"  {'Parameter':<30} {'Log coef':>10} {'Relativity':>12}")
print("  " + "-" * 55)
for k, v in MOTOR_TRUE_FREQ_PARAMS.items():
    print(f"  {k:<30} {v:>+10.4f} {np.exp(v):>12.3f}")

print()
print("Motor severity DGP (Gamma, log link, shape=2):")
print(f"  {'Parameter':<30} {'Log coef':>10} {'Relativity':>12}")
print("  " + "-" * 55)
for k, v in MOTOR_TRUE_SEV_PARAMS.items():
    print(f"  {k:<30} {v:>+10.4f} {np.exp(v):>12.3f}")

print()
print(f"Baseline frequency:  {np.exp(MOTOR_TRUE_FREQ_PARAMS['intercept']):.3%}")
print(f"Baseline severity:   £{np.exp(MOTOR_TRUE_SEV_PARAMS['intercept']):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rating factor distributions
# MAGIC
# MAGIC A visual check that the generated data has realistic marginal distributions.

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Motor dataset: rating factor distributions (50,000 policies)", fontsize=13)

axes[0, 0].hist(motor["driver_age"], bins=30, color="#2ecc71", edgecolor="white", linewidth=0.5)
axes[0, 0].set_title("Driver age")
axes[0, 0].set_xlabel("Years")

axes[0, 1].hist(motor["ncd_years"], bins=6, color="#3498db", edgecolor="white", linewidth=0.5)
axes[0, 1].set_title("NCD years (UK 0-5 scale)")
axes[0, 1].set_xlabel("Years")

axes[0, 2].hist(motor["vehicle_group"], bins=25, color="#9b59b6", edgecolor="white", linewidth=0.5)
axes[0, 2].set_title("Vehicle group (ABI 1-50)")
axes[0, 2].set_xlabel("Group")

area_counts = motor["area"].value_counts().sort_index()
axes[1, 0].bar(area_counts.index, area_counts.values, color="#e74c3c", edgecolor="white", linewidth=0.5)
axes[1, 0].set_title("Area band (A=rural, F=inner city)")
axes[1, 0].set_xlabel("Band")

axes[1, 1].hist(motor["conviction_points"], bins=10, color="#f39c12", edgecolor="white", linewidth=0.5)
axes[1, 1].set_title("Conviction points (0=clean)")
axes[1, 1].set_xlabel("Points")

axes[1, 2].hist(motor["exposure"], bins=30, color="#1abc9c", edgecolor="white", linewidth=0.5)
axes[1, 2].set_title("Exposure (earned years)")
axes[1, 2].set_xlabel("Years")

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit a Poisson frequency GLM and compare to the true parameters
# MAGIC
# MAGIC This is the main use case. Fit the correctly specified model and verify that
# MAGIC the fitted coefficients match the published true values.
# MAGIC
# MAGIC The DGP has a young driver load that tapers linearly from full strength at age 25
# MAGIC to zero at age 30 (blended entry into the rating factor). We replicate that here.

# COMMAND ----------

df = motor.copy()

# Feature engineering to match the DGP spec
df["log_exposure"]    = np.log(df["exposure"].clip(lower=1e-6))
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)

# Young driver: full load under 25, linear taper 25-30, zero from 30
df["young_driver"] = np.where(
    df["driver_age"] < 25, 1.0,
    np.where(df["driver_age"] < 30, (30 - df["driver_age"]) / 5.0, 0.0)
)
df["old_driver"] = (df["driver_age"] >= 70).astype(int)

for band in ["B", "C", "D", "E", "F"]:
    df[f"area_{band}"] = (df["area"] == band).astype(int)

freq_features = [
    "vehicle_group", "ncd_years", "young_driver", "old_driver",
    "has_convictions",
    "area_B", "area_C", "area_D", "area_E", "area_F",
]

X_freq = sm.add_constant(df[freq_features])
freq_model = sm.GLM(
    df["claim_count"],
    X_freq,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df["log_exposure"],
).fit(disp=False)

# Compare fitted to true
name_map = {
    "const":           "intercept",
    "vehicle_group":   "vehicle_group",
    "ncd_years":       "ncd_years",
    "young_driver":    "driver_age_young",
    "old_driver":      "driver_age_old",
    "has_convictions": "has_convictions",
    "area_B": "area_B", "area_C": "area_C", "area_D": "area_D",
    "area_E": "area_E", "area_F": "area_F",
}

print("Frequency GLM: fitted vs true parameters")
print(f"{'Parameter':<28} {'True':>10} {'Fitted':>10} {'Error':>10}")
print("-" * 60)
for sm_name, dgp_name in name_map.items():
    if dgp_name in MOTOR_TRUE_FREQ_PARAMS:
        true_val   = MOTOR_TRUE_FREQ_PARAMS[dgp_name]
        fitted_val = freq_model.params[sm_name]
        error      = fitted_val - true_val
        flag = "  *" if abs(error) > 0.05 else ""
        print(f"  {dgp_name:<26} {true_val:>10.4f} {fitted_val:>10.4f} {error:>+10.4f}{flag}")

print("\n* = error > 0.05 (would be worth investigating at this sample size)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Coefficient recovery plot
# MAGIC
# MAGIC Each dot is one parameter. The closer to the 45-degree line, the better the
# MAGIC recovery. At 50k policies the main effects cluster tightly around the diagonal.

# COMMAND ----------

fitted_vals, true_vals, labels = [], [], []
for sm_name, dgp_name in name_map.items():
    if dgp_name in MOTOR_TRUE_FREQ_PARAMS:
        f = freq_model.params[sm_name]
        t = MOTOR_TRUE_FREQ_PARAMS[dgp_name]
        fitted_vals.append(f)
        true_vals.append(t)
        labels.append(dgp_name)

fig, ax = plt.subplots(figsize=(7, 6))
mn = min(min(true_vals), min(fitted_vals)) - 0.15
mx = max(max(true_vals), max(fitted_vals)) + 0.15
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.4, label="Perfect recovery")
ax.scatter(true_vals, fitted_vals, s=80, color="#e74c3c", zorder=5)
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (true_vals[i], fitted_vals[i]), fontsize=7.5,
                xytext=(5, 4), textcoords="offset points")
ax.set_xlabel("True parameter (DGP)", fontsize=11)
ax.set_ylabel("Fitted parameter (GLM)", fontsize=11)
ax.set_title("Poisson GLM coefficient recovery\n50,000 synthetic motor policies", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Fit a Gamma severity GLM on claims-only data
# MAGIC
# MAGIC Severity is modelled on policies that had at least one claim. The DGP has vehicle
# MAGIC group and a young driver uplift as the main severity drivers.

# COMMAND ----------

claims = df[df["claim_count"] > 0].copy()
claims["avg_severity"] = claims["incurred"] / claims["claim_count"]

print(f"Policies with claims: {len(claims):,} ({100 * len(claims) / len(df):.1f}%)")
print(f"Mean severity: £{claims['avg_severity'].mean():,.0f}")
print(f"Median severity: £{claims['avg_severity'].median():,.0f}")
print()

sev_features = ["vehicle_group", "young_driver"]
X_sev = sm.add_constant(claims[sev_features])

sev_model = sm.GLM(
    claims["avg_severity"],
    X_sev,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit(disp=False)

sev_name_map = {
    "const":          "intercept",
    "vehicle_group":  "vehicle_group",
    "young_driver":   "driver_age_young",
}

print("Severity GLM: fitted vs true parameters")
print(f"{'Parameter':<28} {'True':>10} {'Fitted':>10} {'Error':>10}")
print("-" * 60)
for sm_name, dgp_name in sev_name_map.items():
    if dgp_name in MOTOR_TRUE_SEV_PARAMS:
        true_val   = MOTOR_TRUE_SEV_PARAMS[dgp_name]
        fitted_val = sev_model.params[sm_name]
        error      = fitted_val - true_val
        print(f"  {dgp_name:<26} {true_val:>10.4f} {fitted_val:>10.4f} {error:>+10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Burning cost: frequency x severity relativities by vehicle group
# MAGIC
# MAGIC Score a rating grid across all 50 vehicle groups, holding other factors at base
# MAGIC values, to show how frequency, severity, and pure premium vary with vehicle group.

# COMMAND ----------

rating_grid = pd.DataFrame({
    "vehicle_group":   np.arange(1, 51),
    "ncd_years":       [2] * 50,
    "young_driver":    [0.0] * 50,
    "old_driver":      [0.0] * 50,
    "has_convictions": [0] * 50,
    "area_B": [0]*50, "area_C": [0]*50, "area_D": [0]*50,
    "area_E": [0]*50, "area_F": [0]*50,
})

# Offset = 1 year of exposure for these grid predictions
X_grid     = sm.add_constant(rating_grid[freq_features].values, has_constant='add')
X_grid_sev = sm.add_constant(rating_grid[sev_features].values, has_constant='add')

freq_pred = freq_model.predict(X_grid, offset=np.zeros(50))
sev_pred  = sev_model.predict(X_grid_sev)
pure_prem = freq_pred * sev_pred

# True DGP pure premium on same grid (for comparison)
true_freq  = np.exp(
    MOTOR_TRUE_FREQ_PARAMS["intercept"]
    + MOTOR_TRUE_FREQ_PARAMS["vehicle_group"] * np.arange(1, 51)
    + MOTOR_TRUE_FREQ_PARAMS["ncd_years"] * 2
)
true_sev  = np.exp(
    MOTOR_TRUE_SEV_PARAMS["intercept"]
    + MOTOR_TRUE_SEV_PARAMS["vehicle_group"] * np.arange(1, 51)
)
true_pp = true_freq * true_sev

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(np.arange(1, 51), freq_pred, color="steelblue", label="Fitted")
axes[0].plot(np.arange(1, 51), true_freq, "--", color="tomato", linewidth=1.5, label="True DGP")
axes[0].set_title("Claim frequency by vehicle group\n(NCD=2, Area A, age 35)")
axes[0].set_xlabel("ABI Vehicle Group")
axes[0].set_ylabel("Frequency per year")
axes[0].legend(fontsize=9)

axes[1].plot(np.arange(1, 51), sev_pred, color="coral", label="Fitted")
axes[1].plot(np.arange(1, 51), true_sev, "--", color="steelblue", linewidth=1.5, label="True DGP")
axes[1].set_title("Mean severity by vehicle group")
axes[1].set_xlabel("ABI Vehicle Group")
axes[1].set_ylabel("Mean claim cost (£)")
axes[1].legend(fontsize=9)

base_idx = 24  # group 25 as base
axes[2].plot(np.arange(1, 51), pure_prem / pure_prem[base_idx], color="seagreen", label="Fitted")
axes[2].plot(np.arange(1, 51), true_pp / true_pp[base_idx], "--", color="orange",
             linewidth=1.5, label="True DGP")
axes[2].axhline(1.0, linestyle=":", color="grey", linewidth=0.8)
axes[2].set_title("Pure premium relativity\n(indexed to group 25)")
axes[2].set_xlabel("ABI Vehicle Group")
axes[2].set_ylabel("Relativity")
axes[2].legend(fontsize=9)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Home dataset: flood zone validation
# MAGIC
# MAGIC The home DGP has a large flood zone effect. Zone 3 vs Zone 1 should give a raw
# MAGIC frequency ratio close to exp(0.85) = 2.34. This is an unadjusted check so it
# MAGIC includes the effect of any correlation between flood zone and other factors.

# COMMAND ----------

home = load_home(n_policies=50_000, seed=42)

print(f"Home shape: {home.shape}")
print(f"Claim frequency: {home['claim_count'].sum() / home['exposure'].sum():.4f}")
print()

zone_freq = (
    home.groupby("flood_zone")
    .apply(lambda g: g["claim_count"].sum() / g["exposure"].sum())
    .reset_index()
    .rename(columns={0: "obs_freq"})
    .sort_values("flood_zone")
)
z1_freq = zone_freq.loc[zone_freq["flood_zone"] == "Zone 1", "obs_freq"].values[0]
zone_freq["obs_relativity"] = zone_freq["obs_freq"] / z1_freq
zone_freq["true_relativity"] = zone_freq["flood_zone"].map({
    "Zone 1": 1.0,
    "Zone 2": np.exp(HOME_TRUE_FREQ_PARAMS.get("flood_zone_2", 0.30)),
    "Zone 3": np.exp(HOME_TRUE_FREQ_PARAMS.get("flood_zone_3", 0.85)),
})

print("Flood zone frequency relativities (unadjusted, Zone 1 = base)")
print(zone_freq[["flood_zone", "obs_freq", "obs_relativity", "true_relativity"]].to_string(index=False))

z3_obs  = zone_freq.loc[zone_freq["flood_zone"] == "Zone 3", "obs_relativity"].values[0]
z3_true = np.exp(HOME_TRUE_FREQ_PARAMS.get("flood_zone_3", 0.85))
print(f"\nZone 3 observed: {z3_obs:.2f}x  |  Zone 3 true: {z3_true:.2f}x  |  Diff: {abs(z3_obs - z3_true):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Home frequency GLM: coefficient recovery

# COMMAND ----------

h = home.copy()
h["log_exposure"]       = np.log(h["exposure"].clip(lower=1e-6))
h["prop_value_log"]     = np.log(h["property_value"] / 250_000)
h["contents_val_log"]   = np.log(h["contents_value"] / 30_000)
h["is_non_standard"]    = (h["construction_type"] == "Non-Standard").astype(int)
h["is_listed"]          = (h["construction_type"] == "Listed").astype(int)
h["flood_zone_2"]       = (h["flood_zone"] == "Zone 2").astype(int)
h["flood_zone_3"]       = (h["flood_zone"] == "Zone 3").astype(int)
h["subsidence"]         = h["is_subsidence_risk"].astype(int)
h["security_std"]       = (h["security_level"] == "Standard").astype(int)
h["security_enh"]       = (h["security_level"] == "Enhanced").astype(int)

home_freq_features = [
    "prop_value_log", "is_non_standard", "is_listed",
    "flood_zone_2", "flood_zone_3", "subsidence", "security_std", "security_enh",
]
X_home = sm.add_constant(h[home_freq_features])
home_freq_model = sm.GLM(
    h["claim_count"], X_home,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=h["log_exposure"],
).fit(disp=False)

home_name_map = {
    "prop_value_log":  "property_value_log",
    "is_non_standard": "construction_non_standard",
    "is_listed":       "construction_listed",
    "flood_zone_2":    "flood_zone_2",
    "flood_zone_3":    "flood_zone_3",
    "subsidence":      "subsidence_risk",
    "security_std":    "security_standard",
    "security_enh":    "security_enhanced",
}

print("Home frequency GLM: fitted vs true parameters")
print(f"{'Parameter':<30} {'True':>10} {'Fitted':>10} {'Error':>10}")
print("-" * 62)
for sm_name, dgp_name in home_name_map.items():
    if dgp_name in HOME_TRUE_FREQ_PARAMS:
        true_val   = HOME_TRUE_FREQ_PARAMS[dgp_name]
        fitted_val = home_freq_model.params[sm_name]
        error      = fitted_val - true_val
        print(f"  {dgp_name:<28} {true_val:>10.4f} {fitted_val:>10.4f} {error:>+10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Sample size stability
# MAGIC
# MAGIC How much data do you need before the coefficient estimates are reliable?
# MAGIC This cell measures RMSE of coefficient recovery at several sample sizes.

# COMMAND ----------

sample_sizes = [1_000, 5_000, 10_000, 25_000, 50_000]
rmse_by_n = []

for n in sample_sizes:
    df_n = load_motor(n_policies=n, seed=42)
    df_n["log_exposure"]    = np.log(df_n["exposure"].clip(lower=1e-6))
    df_n["has_convictions"] = (df_n["conviction_points"] > 0).astype(int)
    df_n["young_driver"] = np.where(
        df_n["driver_age"] < 25, 1.0,
        np.where(df_n["driver_age"] < 30, (30 - df_n["driver_age"]) / 5.0, 0.0)
    )
    df_n["old_driver"] = (df_n["driver_age"] >= 70).astype(int)
    for band in ["B", "C", "D", "E", "F"]:
        df_n[f"area_{band}"] = (df_n["area"] == band).astype(int)

    X_n = sm.add_constant(df_n[freq_features])
    try:
        res_n = sm.GLM(
            df_n["claim_count"], X_n,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=df_n["log_exposure"],
        ).fit(disp=False)
        errors_sq = []
        for sm_name, dgp_name in name_map.items():
            if dgp_name in MOTOR_TRUE_FREQ_PARAMS:
                f = res_n.params.get(sm_name, np.nan)
                t = MOTOR_TRUE_FREQ_PARAMS[dgp_name]
                if not np.isnan(f):
                    errors_sq.append((f - t) ** 2)
        rmse_by_n.append(np.sqrt(np.mean(errors_sq)))
    except Exception:
        rmse_by_n.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 4))
valid = [(n, e) for n, e in zip(sample_sizes, rmse_by_n) if not np.isnan(e)]
ns, es = zip(*valid)
ax.plot(ns, es, "o-", color="#3498db", linewidth=2, markersize=8)
for n, e in valid:
    ax.annotate(f"{e:.3f}", (n, e), fontsize=8.5, xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("Number of policies")
ax.set_ylabel("RMSE (fitted vs true log-coefficients)")
ax.set_title("Coefficient recovery accuracy vs training size\nMotor frequency Poisson GLM", fontsize=11)
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)
plt.close()

print("RMSE by sample size:")
for n, e in zip(sample_sizes, rmse_by_n):
    print(f"  n={n:>6,}  RMSE={e:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Reproducibility check
# MAGIC
# MAGIC The same seed always produces identical data. Useful for reproducible benchmarks
# MAGIC and for sharing test cases across a team.

# COMMAND ----------

m1 = load_motor(n_policies=5_000, seed=99)
m2 = load_motor(n_policies=5_000, seed=99)
m3 = load_motor(n_policies=5_000, seed=100)

freq1 = m1["claim_count"].sum() / m1["exposure"].sum()
freq2 = m2["claim_count"].sum() / m2["exposure"].sum()
freq3 = m3["claim_count"].sum() / m3["exposure"].sum()

print(f"seed=99,  run 1:  frequency = {freq1:.6f}")
print(f"seed=99,  run 2:  frequency = {freq2:.6f}  (identical: {np.isclose(freq1, freq2)})")
print(f"seed=100, run 1:  frequency = {freq3:.6f}  (different: {not np.isclose(freq1, freq3)})")
print()
print(f"Row hash (seed=99 run 1 vs run 2): {(m1['incurred'] == m2['incurred']).all()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Home severity GLM

# COMMAND ----------

home_claims = h[h["claim_count"] > 0].copy()
home_claims["avg_severity"] = home_claims["incurred"] / home_claims["claim_count"]

print(f"Home claims: {len(home_claims):,} ({100 * len(home_claims) / len(h):.1f}%)")

home_sev_features = ["prop_value_log", "flood_zone_3"]
X_home_sev = sm.add_constant(home_claims[home_sev_features])

home_sev_model = sm.GLM(
    home_claims["avg_severity"],
    X_home_sev,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit(disp=False)

home_sev_name_map = {
    "const":          "intercept",
    "prop_value_log": "property_value_log",
    "flood_zone_3":   "flood_zone_3",
}

print("\nHome severity GLM: fitted vs true")
print(f"{'Parameter':<28} {'True':>10} {'Fitted':>10} {'Error':>10}")
print("-" * 60)
for sm_name, dgp_name in home_sev_name_map.items():
    if dgp_name in HOME_TRUE_SEV_PARAMS:
        true_val   = HOME_TRUE_SEV_PARAMS[dgp_name]
        fitted_val = home_sev_model.params[sm_name]
        error      = fitted_val - true_val
        print(f"  {dgp_name:<26} {true_val:>10.4f} {fitted_val:>10.4f} {error:>+10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Task | Result |
# MAGIC |------|--------|
# MAGIC | Motor Poisson frequency GLM | All main parameters within 0.05 of true values at 50k policies |
# MAGIC | Motor Gamma severity GLM | Vehicle group and young driver uplift recovered accurately |
# MAGIC | Home frequency GLM | Flood zone 3, subsidence, and construction type recovered |
# MAGIC | Home severity GLM | Property value log coefficient and flood zone 3 uplift recovered |
# MAGIC | Flood zone unadjusted check | Zone 3 / Zone 1 ratio ≈ exp(0.85) = 2.34 as expected |
# MAGIC | Sample size stability | RMSE halves roughly each time sample size quadruples |
# MAGIC | Reproducibility | Same seed gives identical datasets across runs |
# MAGIC
# MAGIC ### When to use this library
# MAGIC
# MAGIC **Testing GLM implementations**: fit the correctly specified model and verify
# MAGIC coefficient recovery. If your implementation is wrong, the deviation from
# MAGIC `MOTOR_TRUE_FREQ_PARAMS` will tell you exactly which parameter is off.
# MAGIC
# MAGIC **Benchmarking modelling approaches**: compare a GAM, GBM, and GLM on a problem
# MAGIC where you know the true answer. The metric is not just holdout deviance — you can
# MAGIC also measure how close each model gets to the true DGP parameters.
# MAGIC
# MAGIC **Training and education**: junior actuaries can see what a "correct" model output
# MAGIC looks like before working on real data where correctness is unknowable.
# MAGIC
# MAGIC ### Related libraries
# MAGIC
# MAGIC - `insurance-gam`: use this dataset to benchmark EBM/ANAM against a Poisson GLM on a known DGP
# MAGIC - `insurance-interactions`: test interaction detection recall when ground truth is known
# MAGIC - `insurance-cv`: validate walk-forward CV strategies on a controlled problem
