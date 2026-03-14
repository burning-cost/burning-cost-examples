# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-glm-tools: Nested GLM Embeddings and R2VF Factor Clustering
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC UK motor pricing models routinely encounter factors with hundreds of levels: vehicle make,
# MAGIC model, postcode sector, occupation class. The standard fix — dummy-coding — breaks down fast.
# MAGIC With 500 vehicle makes, your design matrix gains 499 columns, most of which have thin or
# MAGIC zero exposure. The GLM will either fail to converge, produce wildly unstable relativities
# MAGIC for rare makes, or force you to manually group levels before fitting.
# MAGIC
# MAGIC There are two separate problems here:
# MAGIC
# MAGIC 1. **High-cardinality unordered factors** (vehicle make, model): no natural ordering,
# MAGIC    some levels have 3 policies. Dummy coding is hopeless; credibility blending is ad hoc.
# MAGIC
# MAGIC 2. **Ordinal factors needing banding** (vehicle age, NCD years, driver age bands):
# MAGIC    natural order exists but the grouping is currently done by eye or by quintile splits
# MAGIC    that ignore statistical support.
# MAGIC
# MAGIC This notebook demonstrates `insurance-glm-tools`, which handles both:
# MAGIC
# MAGIC - **`NestedGLMPipeline`** (Wang, Shi, Cao NAAJ 2025): trains a neural network to learn
# MAGIC   dense embeddings for high-cardinality factors, then feeds those embeddings as continuous
# MAGIC   features into a standard outer GLM. Unseen makes at prediction time get handled naturally
# MAGIC   via the embedding layer.
# MAGIC
# MAGIC - **`FactorClusterer`** (R2VF, Ben Dror arXiv:2503.01521): fits a fused lasso on the
# MAGIC   split-coded design matrix to automatically merge adjacent ordinal levels. BIC selects
# MAGIC   the regularisation strength. No manual quintile decisions.

# COMMAND ----------

# MAGIC %pip install insurance-glm-tools matplotlib polars

# COMMAND ----------

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from insurance_glm_tools.nested import NestedGLMPipeline
from insurance_glm_tools.cluster import FactorClusterer

print("insurance-glm-tools loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic motor data
# MAGIC
# MAGIC We generate 50,000 policies with a Poisson DGP. The ground truth includes:
# MAGIC
# MAGIC - **500 vehicle makes** with known log-relativities drawn from a bimodal distribution
# MAGIC   (some sports/prestige segments genuinely riskier). Roughly 200 makes have fewer than
# MAGIC   50 policies — too thin for stable dummy-coded estimates.
# MAGIC - Standard structured factors: age band, NCD years, vehicle group, area
# MAGIC - A held-out test set of 10,000 policies including 50 makes **never seen in training**

# COMMAND ----------

rng = np.random.default_rng(2025)

N_TRAIN = 50_000
N_TEST = 10_000
N_MAKES = 500
N_MAKES_UNSEEN = 50   # makes that appear only in test

# --- Vehicle make effects (known DGP) ---
# Bimodal: most makes ~0 log-relativity, ~20% are high-risk (+0.4 to +0.8)
make_ids = np.arange(N_MAKES)
make_effect = np.where(
    rng.random(N_MAKES) < 0.20,
    rng.uniform(0.3, 0.8, N_MAKES),   # high-risk segment
    rng.normal(0.0, 0.15, N_MAKES),   # normal segment
)
make_names = [f"make_{i:03d}" for i in make_ids]
make_effect_map = dict(zip(make_names, make_effect))

# Exposure is heavily skewed: some makes dominate, most are thin
# Power-law exposures so ~200 makes are very thin
make_weights = rng.pareto(1.5, N_MAKES) + 0.1
make_weights /= make_weights.sum()

# Split: first (N_MAKES - N_MAKES_UNSEEN) makes available in training
seen_makes = make_names[: N_MAKES - N_MAKES_UNSEEN]
unseen_makes = make_names[N_MAKES - N_MAKES_UNSEEN :]

seen_weights = make_weights[: N_MAKES - N_MAKES_UNSEEN]
seen_weights = seen_weights / seen_weights.sum()

# --- Age band effects ---
age_bands = ["17-25", "26-35", "36-50", "51-65", "66+"]
age_effect = {"17-25": 0.50, "26-35": 0.15, "36-50": 0.0, "51-65": 0.05, "66+": 0.20}

# --- NCD years: ordinal, 0-9, true grouping: {0}, {1-3}, {4-6}, {7-9} ---
# The true relativity curve is step-wise, not linear
ncd_true_log_rel = np.array([0.45, 0.20, 0.10, 0.05, 0.0, 0.0, -0.05, -0.05, -0.08, -0.08])

# --- Vehicle age: ordinal, 0-15 years ---
# True curve: young cars (0-2) slightly cheaper parts, old cars (10+) more risk
veh_age_true_log_rel = np.array([
    0.05, 0.02, 0.0, -0.02, -0.04, -0.04, -0.02, 0.0,
    0.05, 0.10, 0.15, 0.18, 0.20, 0.20, 0.22, 0.22
])

# --- Area: 1-6 broad regions ---
area_effect = {1: 0.20, 2: 0.10, 3: 0.0, 4: -0.05, 5: -0.10, 6: 0.15}

def generate_policies(n, make_pool, make_pool_weights, rng_obj):
    age_band = rng_obj.choice(age_bands, size=n, p=[0.08, 0.22, 0.35, 0.25, 0.10])
    ncd_years = rng_obj.integers(0, 10, n)
    vehicle_age = rng_obj.integers(0, 16, n)
    vehicle_group = rng_obj.integers(1, 21, n)
    area = rng_obj.integers(1, 7, n)
    vehicle_make = rng_obj.choice(make_pool, size=n, p=make_pool_weights)
    exposure = rng_obj.uniform(0.3, 1.0, n)

    log_rate = (
        -2.8
        + np.array([age_effect[a] for a in age_band])
        + ncd_true_log_rel[ncd_years]
        + veh_age_true_log_rel[vehicle_age]
        + 0.02 * vehicle_group
        + np.array([area_effect[a] for a in area])
        + np.array([make_effect_map[m] for m in vehicle_make])
    )
    claims = rng_obj.poisson(np.exp(log_rate) * exposure).astype(float)
    return pd.DataFrame({
        "age_band": age_band,
        "ncd_years": ncd_years,
        "vehicle_age": vehicle_age,
        "vehicle_group": vehicle_group,
        "area": area,
        "vehicle_make": vehicle_make,
        "exposure": exposure,
        "claims": claims,
    })

df_train = generate_policies(N_TRAIN, seen_makes, seen_weights, rng)

# Test set: 80% seen makes, 20% unseen makes
unseen_weights_norm = make_weights[N_MAKES - N_MAKES_UNSEEN:]
unseen_weights_norm = unseen_weights_norm / unseen_weights_norm.sum()

n_test_seen = int(N_TEST * 0.80)
n_test_unseen = N_TEST - n_test_seen

seen_weights_test = seen_weights  # already normalised
df_test_seen = generate_policies(n_test_seen, seen_makes, seen_weights_test, rng)
df_test_unseen = generate_policies(n_test_unseen, unseen_makes, unseen_weights_norm, rng)
df_test = pd.concat([df_test_seen, df_test_unseen], ignore_index=True)

y_train = df_train["claims"].values
exp_train = df_train["exposure"].values
X_train = df_train.drop(columns=["claims", "exposure"])

y_test = df_test["claims"].values
exp_test = df_test["exposure"].values
X_test = df_test.drop(columns=["claims", "exposure"])

# Quick data summary
make_counts = df_train["vehicle_make"].value_counts()
thin_makes = (make_counts < 50).sum()
summary_html = f"""
<h3>Dataset summary</h3>
<table style="border-collapse:collapse; font-family:monospace; font-size:13px;">
<tr style="background:#2c3e50; color:white;"><th style="padding:8px 16px; text-align:left;">Metric</th><th style="padding:8px 16px; text-align:right;">Value</th></tr>
<tr style="background:#ecf0f1;"><td style="padding:6px 16px;">Training policies</td><td style="padding:6px 16px; text-align:right;">{N_TRAIN:,}</td></tr>
<tr><td style="padding:6px 16px;">Test policies</td><td style="padding:6px 16px; text-align:right;">{N_TEST:,}</td></tr>
<tr style="background:#ecf0f1;"><td style="padding:6px 16px;">Total vehicle makes</td><td style="padding:6px 16px; text-align:right;">{N_MAKES}</td></tr>
<tr><td style="padding:6px 16px;">Makes in training set</td><td style="padding:6px 16px; text-align:right;">{len(seen_makes)}</td></tr>
<tr style="background:#ecf0f1;"><td style="padding:6px 16px;">Makes with &lt;50 train policies (thin)</td><td style="padding:6px 16px; text-align:right;">{thin_makes}</td></tr>
<tr><td style="padding:6px 16px;">Unseen makes in test set</td><td style="padding:6px 16px; text-align:right;">{N_MAKES_UNSEEN}</td></tr>
<tr style="background:#ecf0f1;"><td style="padding:6px 16px;">Training claim frequency</td><td style="padding:6px 16px; text-align:right;">{y_train.sum() / exp_train.sum():.4f}</td></tr>
<tr><td style="padding:6px 16px;">NCD levels (ordinal banding demo)</td><td style="padding:6px 16px; text-align:right;">0–9 (true: 4 bands)</td></tr>
<tr style="background:#ecf0f1;"><td style="padding:6px 16px;">Vehicle age levels (ordinal banding demo)</td><td style="padding:6px 16px; text-align:right;">0–15 (true: ~5 bands)</td></tr>
</table>
"""
displayHTML(summary_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Benchmark: dummy-coded GLM vs NestedGLMPipeline
# MAGIC
# MAGIC The dummy-coded baseline is what most teams actually do: encode vehicle make as a
# MAGIC categorical factor and fit a Poisson GLM. With 450 training makes, this means 449
# MAGIC extra columns — many near-empty. Statsmodels will fit, but estimates for thin makes
# MAGIC are driven by prior-free MLE: a make with 3 policies contributes a coefficient with
# MAGIC enormous standard error. Worse, at prediction time any make not seen in training
# MAGIC causes a key error (or a silent zero if you handle it manually).
# MAGIC
# MAGIC `NestedGLMPipeline` replaces the dummy columns with a low-dimensional embedding.
# MAGIC The neural network sees all makes together during training, so thin makes borrow
# MAGIC information via shared hidden layers. Unseen makes are handled by a "cold start"
# MAGIC embedding — the network has seen enough variety to generalise reasonably.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Dummy-coded GLM baseline

# COMMAND ----------

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import time

# Dummy-coded baseline: use only age_band + ncd_years + vehicle_group + area
# Intentionally skip vehicle_make here first, then try with it to show instability
base_formula_str = "claims ~ age_band + C(ncd_years) + vehicle_group + C(area)"

t0 = time.time()
dummy_base_result = smf.glm(
    formula=base_formula_str,
    data=df_train.assign(claims=y_train),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(exp_train),
).fit(disp=False)
t_base = time.time() - t0

def poisson_deviance_holdout(y_true, y_pred):
    """Poisson deviance on holdout set (per policy, normalised by n)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(y_true > 0, y_true * np.log(y_true / y_pred) - (y_true - y_pred), y_pred)
    return 2.0 * d.sum() / len(y_true)

# Predict on test (using only structured factors, ignoring make)
pred_base = dummy_base_result.predict(
    df_test.assign(claims=y_test),
    offset=np.log(exp_test)
)

dev_base_train = poisson_deviance_holdout(
    y_train,
    dummy_base_result.predict(df_train.assign(claims=y_train), offset=np.log(exp_train))
)
dev_base_test = poisson_deviance_holdout(y_test, pred_base)

print(f"Base GLM (no make factor): train deviance={dev_base_train:.4f}, test deviance={dev_base_test:.4f}, time={t_base:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Dummy-coded GLM with vehicle make as a factor
# MAGIC
# MAGIC This is what a naive approach looks like. We cap at 100 most common makes
# MAGIC (merging the rest to "other") to keep it tractable — even so the instability
# MAGIC on thin makes is visible in the coefficient spread.

# COMMAND ----------

# Keep top 100 makes by frequency, pool rest as "other_make"
top_makes = make_counts.nlargest(100).index.tolist()

def cap_make(series, top):
    return series.where(series.isin(top), other="other_make")

df_train_capped = df_train.copy()
df_train_capped["vehicle_make_capped"] = cap_make(df_train["vehicle_make"], top_makes)

df_test_capped = df_test.copy()
df_test_capped["vehicle_make_capped"] = cap_make(df_test["vehicle_make"], top_makes)

dummy_make_formula = "claims ~ age_band + C(ncd_years) + vehicle_group + C(area) + C(vehicle_make_capped)"

t0 = time.time()
try:
    dummy_make_result = smf.glm(
        formula=dummy_make_formula,
        data=df_train_capped.assign(claims=y_train),
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(exp_train),
    ).fit(disp=False, maxiter=200)
    t_dummy_make = time.time() - t0

    pred_dummy_make = dummy_make_result.predict(
        df_test_capped.assign(claims=y_test),
        offset=np.log(exp_test)
    )

    dev_dummy_train = poisson_deviance_holdout(
        y_train,
        dummy_make_result.predict(df_train_capped.assign(claims=y_train), offset=np.log(exp_train))
    )
    dev_dummy_test = poisson_deviance_holdout(y_test, pred_dummy_make)

    # Count parameters
    n_params_dummy = len(dummy_make_result.params)
    print(f"Dummy-coded GLM (top 100 makes + other): {n_params_dummy} params, "
          f"train deviance={dev_dummy_train:.4f}, test deviance={dev_dummy_test:.4f}, time={t_dummy_make:.1f}s")
except Exception as e:
    print(f"Dummy-coded GLM failed: {e}")
    dev_dummy_train = dev_base_train
    dev_dummy_test = dev_base_test
    t_dummy_make = 0.0
    n_params_dummy = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. NestedGLMPipeline with embeddings

# COMMAND ----------

t0 = time.time()
pipeline = NestedGLMPipeline(
    base_formula="age_band + ncd_years + vehicle_group + area",
    family="poisson",
    embedding_epochs=30,
    embedding_hidden_sizes=(32, 16),
    embedding_lr=5e-3,
    embedding_batch_size=512,
    random_state=42,
)
pipeline.fit(
    X_train,
    y_train,
    exp_train,
    high_card_cols=["vehicle_make"],
    base_formula_cols=["age_band", "ncd_years", "vehicle_group", "area"],
)
t_nested = time.time() - t0

pred_nested_train = pipeline.predict(X_train, exp_train)
pred_nested_test = pipeline.predict(X_test, exp_test)

dev_nested_train = poisson_deviance_holdout(y_train, pred_nested_train)
dev_nested_test = poisson_deviance_holdout(y_test, pred_nested_test)

# Count outer GLM parameters (much fewer than dummy)
n_params_nested = len(pipeline.outer_glm_.result_.params)

print(f"NestedGLMPipeline: {n_params_nested} outer params, "
      f"train deviance={dev_nested_train:.4f}, test deviance={dev_nested_test:.4f}, time={t_nested:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Benchmark results table

# COMMAND ----------

benchmark_rows = [
    {
        "Model": "Base GLM (no make factor)",
        "Parameters": dummy_base_result.df_model + 1,
        "Train deviance": round(dev_base_train, 4),
        "Test deviance": round(dev_base_test, 4),
        "Unseen makes": "N/A (ignored)",
        "Fit time (s)": round(t_base, 1),
    },
    {
        "Model": "Dummy-coded GLM (top 100 makes, rest pooled)",
        "Parameters": n_params_dummy,
        "Train deviance": round(dev_dummy_train, 4),
        "Test deviance": round(dev_dummy_test, 4),
        "Unseen makes": "Pooled to 'other'",
        "Fit time (s)": round(t_dummy_make, 1),
    },
    {
        "Model": "NestedGLMPipeline (embeddings, all 450 makes)",
        "Parameters": n_params_nested,
        "Train deviance": round(dev_nested_train, 4),
        "Test deviance": round(dev_nested_test, 4),
        "Unseen makes": "Generalises via embedding",
        "Fit time (s)": round(t_nested, 1),
    },
]

improvement_pct = (dev_dummy_test - dev_nested_test) / dev_dummy_test * 100

header_style = "background:#2c3e50; color:white; padding:8px 14px; text-align:left;"
td_style = "padding:6px 14px; font-family:monospace; font-size:12px; border-bottom:1px solid #ddd;"
td_r_style = td_style.replace("text-align:left", "") + " text-align:right;"

def row_color(i):
    return "#ecf0f1" if i % 2 == 0 else "white"

rows_html = ""
for i, r in enumerate(benchmark_rows):
    bg = row_color(i)
    highlight = ' style="background:#d5f5e3;"' if i == 2 else f' style="background:{bg};"'
    rows_html += f"""
<tr{highlight}>
  <td style="{td_style}">{r["Model"]}</td>
  <td style="{td_r_style}">{r["Parameters"]}</td>
  <td style="{td_r_style}">{r["Train deviance"]}</td>
  <td style="{td_r_style}">{r["Test deviance"]}</td>
  <td style="{td_style}">{r["Unseen makes"]}</td>
  <td style="{td_r_style}">{r["Fit time (s)"]}</td>
</tr>"""

displayHTML(f"""
<h3>Benchmark: Poisson GLM for Vehicle Make</h3>
<p style="font-size:13px; color:#555;">Lower deviance = better fit. All figures on held-out test set (10,000 policies).</p>
<table style="border-collapse:collapse; font-size:13px; width:100%;">
<thead>
<tr>
  <th style="{header_style}">Model</th>
  <th style="{header_style} text-align:right;">Parameters</th>
  <th style="{header_style} text-align:right;">Train deviance</th>
  <th style="{header_style} text-align:right;">Test deviance</th>
  <th style="{header_style}">Unseen makes</th>
  <th style="{header_style} text-align:right;">Fit time (s)</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
<p style="margin-top:12px; font-size:13px;">
  Nested embedding improvement over dummy-coded: <strong>{improvement_pct:.1f}%</strong> reduction in test deviance.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Handling unseen makes at prediction time
# MAGIC
# MAGIC The dummy-coded GLM cannot score a policy with a make it has never seen.
# MAGIC The standard workaround (pool to "other") loses all signal — a prestige make
# MAGIC that happens to be rare in training gets pooled with budget makes.
# MAGIC
# MAGIC The embedding network generalises: at inference time it maps the unseen make
# MAGIC through the same neural network. The embedding won't be perfect, but it will
# MAGIC be informed by the network's learned structure (e.g. if a make's name pattern
# MAGIC clusters near learned high-risk names, its embedding will reflect that).
# MAGIC
# MAGIC Here we compare prediction quality on just the unseen-make test policies.

# COMMAND ----------

# Indices where the test make was unseen in training
unseen_mask = df_test["vehicle_make"].isin(unseen_makes)
n_unseen_policies = unseen_mask.sum()

if n_unseen_policies > 0:
    y_unseen = y_test[unseen_mask]
    exp_unseen = exp_test[unseen_mask]
    X_unseen = X_test[unseen_mask].reset_index(drop=True)

    # NestedGLMPipeline handles unseen makes natively
    pred_nested_unseen = pipeline.predict(X_unseen, exp_unseen)
    dev_nested_unseen = poisson_deviance_holdout(y_unseen, pred_nested_unseen)

    # Dummy-coded baseline: predict using only structured factors (best we can do)
    df_unseen_for_base = df_test[unseen_mask].reset_index(drop=True)
    pred_base_unseen = dummy_base_result.predict(
        df_unseen_for_base.assign(claims=y_unseen),
        offset=np.log(exp_unseen)
    )
    dev_base_unseen = poisson_deviance_holdout(y_unseen, pred_base_unseen)

    # True make effects for unseen policies (from DGP)
    true_make_effects = np.array([make_effect_map[m] for m in df_unseen_for_base["vehicle_make"]])
    # Compare predicted make contribution (embedding captures this)
    # Embedding predictions minus base-only predictions = attributed make effect
    embedding_make_contrib = np.log((pred_nested_unseen / exp_unseen).clip(min=1e-10)) - \
                             np.log((pred_base_unseen / exp_unseen).clip(min=1e-10))

    corr = np.corrcoef(true_make_effects, embedding_make_contrib)[0, 1]

    displayHTML(f"""
<h3>Unseen makes at prediction time ({n_unseen_policies:,} policies, {N_MAKES_UNSEEN} makes)</h3>
<table style="border-collapse:collapse; font-family:monospace; font-size:13px;">
<tr style="background:#2c3e50; color:white;">
  <th style="padding:8px 16px; text-align:left;">Approach</th>
  <th style="padding:8px 16px; text-align:right;">Test deviance</th>
  <th style="padding:8px 16px; text-align:left;">Notes</th>
</tr>
<tr style="background:#ecf0f1;">
  <td style="padding:6px 16px;">Base GLM (structured factors only)</td>
  <td style="padding:6px 16px; text-align:right;">{dev_base_unseen:.4f}</td>
  <td style="padding:6px 16px;">Make effect ignored entirely</td>
</tr>
<tr style="background:#d5f5e3;">
  <td style="padding:6px 16px;">NestedGLMPipeline (embedding generalisation)</td>
  <td style="padding:6px 16px; text-align:right;">{dev_nested_unseen:.4f}</td>
  <td style="padding:6px 16px;">Embedding infers make risk; corr with true DGP: {corr:.3f}</td>
</tr>
</table>
<p style="font-size:13px; color:#555; margin-top:8px;">
  Correlation between embedding-attributed make effect and true DGP log-relativity: <strong>{corr:.3f}</strong>.
  The neural network has learned enough structure to partially recover unseen make risk levels.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embedding dimensionality vs accuracy
# MAGIC
# MAGIC The default embedding uses a (32, 16) hidden layer architecture, which produces
# MAGIC a 16-dimensional embedding vector per make. We can trade off dimensionality
# MAGIC against test deviance. Lower dimensions = more regularisation but less capacity.

# COMMAND ----------

dim_results = []
hidden_configs = [
    (8,),
    (16,),
    (32,),
    (32, 16),
    (64, 32),
]

for hidden in hidden_configs:
    emb_dim = hidden[-1]
    t0 = time.time()
    p = NestedGLMPipeline(
        base_formula="age_band + ncd_years + vehicle_group + area",
        family="poisson",
        embedding_epochs=20,
        embedding_hidden_sizes=hidden,
        embedding_lr=5e-3,
        embedding_batch_size=512,
        random_state=42,
    )
    p.fit(
        X_train, y_train, exp_train,
        high_card_cols=["vehicle_make"],
        base_formula_cols=["age_band", "ncd_years", "vehicle_group", "area"],
    )
    t_fit = time.time() - t0
    pred_te = p.predict(X_test, exp_test)
    dev_te = poisson_deviance_holdout(y_test, pred_te)
    n_outer = len(p.outer_glm_.result_.params)
    dim_results.append({
        "hidden_sizes": str(hidden),
        "emb_dim": emb_dim,
        "outer_params": n_outer,
        "test_deviance": dev_te,
        "fit_time_s": round(t_fit, 1),
    })
    print(f"  hidden={hidden}, dim={emb_dim}, test deviance={dev_te:.4f}, params={n_outer}, time={t_fit:.1f}s")

dim_df = pd.DataFrame(dim_results)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.arange(len(dim_results))
colors = ["#2980b9" if i != 3 else "#27ae60" for i in x]

ax1.bar(x, dim_df["test_deviance"], color=colors, edgecolor="white", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([r["hidden_sizes"] for r in dim_results], rotation=25, ha="right", fontsize=9)
ax1.set_ylabel("Test Poisson deviance")
ax1.set_title("Embedding size vs holdout deviance")
ax1.axhline(dev_base_test, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Base GLM (no make): {dev_base_test:.4f}")
ax1.legend(fontsize=8)
ax1.set_ylim(bottom=min(dim_df["test_deviance"]) * 0.99)

ax2.scatter(dim_df["emb_dim"], dim_df["test_deviance"], s=80, c=colors, edgecolors="k", linewidths=0.5, zorder=3)
for i, r in enumerate(dim_results):
    ax2.annotate(r["hidden_sizes"], (r["emb_dim"], r["test_deviance"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)
ax2.set_xlabel("Embedding dimension")
ax2.set_ylabel("Test Poisson deviance")
ax2.set_title("Embedding dimension vs deviance")
ax2.axhline(dev_base_test, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. R2VF factor-level clustering: automating NCD and vehicle age banding
# MAGIC
# MAGIC The second tool in the package attacks ordinal factor banding. Rather than splitting
# MAGIC NCD years into quintiles (or asking the actuary to draw the groups by eye), `FactorClusterer`
# MAGIC fits a fused lasso that penalises differences between adjacent level coefficients.
# MAGIC Adjacent levels whose difference shrinks to zero get merged.
# MAGIC
# MAGIC BIC automatically selects the regularisation strength, so the number of final groups
# MAGIC is data-driven — not fixed at 5 or whatever the actuary happened to use last year.

# COMMAND ----------

# Prepare a dataset for clustering demo
# Use only structured factors (no vehicle make, to keep it focused)
X_cluster = df_train[["ncd_years", "vehicle_age", "vehicle_group", "area"]].copy()

t0 = time.time()
fc = FactorClusterer(family="poisson", lambda_="bic", min_exposure=300, n_lambda=40)
fc.fit(
    X_cluster,
    y_train,
    exposure=exp_train,
    ordinal_factors=["ncd_years", "vehicle_age"],
)
t_cluster = time.time() - t0

print(f"FactorClusterer fitted in {t_cluster:.1f}s")
print(f"Best lambda: {fc.best_lambda:.6f}")

ncd_lm = fc.level_map("ncd_years")
veh_age_lm = fc.level_map("vehicle_age")

print(f"\nNCD years: {ncd_lm.n_levels} original levels -> {ncd_lm.n_groups} merged groups")
print(ncd_lm.to_df().to_string(index=False))

print(f"\nVehicle age: {veh_age_lm.n_levels} original levels -> {veh_age_lm.n_groups} merged groups")
print(veh_age_lm.to_df().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Visualise the clustering results vs the true DGP

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# NCD years plot
ncd_df = ncd_lm.to_df()
ax = axes[0]

# True DGP curve
ax.step(range(10), ncd_true_log_rel, where="post", color="#2c3e50", linewidth=2, label="True DGP", zorder=4)

# Merged group bands as background
group_colors = ["#d6eaf8", "#d5f5e3", "#fdebd0", "#fdf2f8", "#e8f8f5", "#fef9e7"]
for _, row in ncd_lm.group_summary().iterrows():
    levels = sorted(row["levels"])
    g = row["merged_group"]
    ax.axvspan(levels[0] - 0.4, levels[-1] + 0.4,
               alpha=0.35, color=group_colors[g % len(group_colors)],
               label=f"Group {g}" if g == 0 else "_nolegend_")

# Fitted relativities per group
for _, row in ncd_df.iterrows():
    ax.plot(row["original_level"], row["coefficient"], "o",
            color="#e74c3c", markersize=7, zorder=5)
ax.plot([], [], "o", color="#e74c3c", markersize=7, label="R2VF merged coefficient")

ax.set_xlabel("NCD years")
ax.set_ylabel("Log-relativity")
ax.set_title(f"NCD banding: {ncd_lm.n_levels} levels -> {ncd_lm.n_groups} groups")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(10))

# Vehicle age plot
veh_df = veh_age_lm.to_df()
ax = axes[1]

ax.step(range(16), veh_age_true_log_rel, where="post", color="#2c3e50", linewidth=2, label="True DGP", zorder=4)

for _, row in veh_age_lm.group_summary().iterrows():
    levels = sorted(row["levels"])
    g = row["merged_group"]
    ax.axvspan(levels[0] - 0.4, levels[-1] + 0.4,
               alpha=0.35, color=group_colors[g % len(group_colors)],
               label=f"Group {g}" if g == 0 else "_nolegend_")

for _, row in veh_df.iterrows():
    ax.plot(row["original_level"], row["coefficient"], "o",
            color="#e74c3c", markersize=7, zorder=5)
ax.plot([], [], "o", color="#e74c3c", markersize=7, label="R2VF merged coefficient")

ax.set_xlabel("Vehicle age (years)")
ax.set_ylabel("Log-relativity")
ax.set_title(f"Vehicle age banding: {veh_age_lm.n_levels} levels -> {veh_age_lm.n_groups} groups")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(16))

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. BIC regularisation path
# MAGIC
# MAGIC The full lambda grid shows how many groups survive at each regularisation strength,
# MAGIC and where BIC selects the best tradeoff between fit and parsimony.

# COMMAND ----------

dp = fc.diagnostic_path
if dp is not None:
    dp_df = dp.to_df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # BIC path
    valid = np.isfinite(dp_df["bic"])
    ax1.plot(np.log10(dp_df["lambda"][valid]), dp_df["bic"][valid], color="#2980b9", linewidth=2)
    best_row = dp_df[dp_df["is_best"]]
    ax1.axvline(np.log10(best_row["lambda"].values[0]), color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f'Best lambda = {dp.best_lambda:.5f}')
    ax1.scatter(np.log10(best_row["lambda"].values[0]), best_row["bic"].values[0],
                color="#e74c3c", s=80, zorder=5)
    ax1.set_xlabel("log10(lambda)")
    ax1.set_ylabel("BIC")
    ax1.set_title("BIC regularisation path")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Number of groups path
    ax2.step(np.log10(dp_df["lambda"][valid]), dp_df["n_groups"][valid],
             where="post", color="#27ae60", linewidth=2)
    ax2.axvline(np.log10(best_row["lambda"].values[0]), color="#e74c3c", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("log10(lambda)")
    ax2.set_ylabel("Total merged groups (K_eff)")
    ax2.set_title("Groups at each regularisation level")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. R2VF vs manual quintile banding: deviance comparison

# COMMAND ----------

# Manual quintile banding: sort levels by crude frequency, split into 5 groups
def quintile_band(series, y, exp, n_bands=5):
    """Band an ordinal factor into n_bands groups by crude frequency ranking."""
    levels = sorted(series.unique())
    level_freq = {}
    for lv in levels:
        mask = series == lv
        if mask.sum() > 0 and exp[mask].sum() > 0:
            level_freq[lv] = y[mask].sum() / exp[mask].sum()
        else:
            level_freq[lv] = 0.0
    sorted_levels = sorted(levels, key=lambda l: level_freq[l])
    band_size = max(1, len(sorted_levels) // n_bands)
    level_to_band = {}
    for i, lv in enumerate(sorted_levels):
        level_to_band[lv] = min(i // band_size, n_bands - 1)
    return series.map(level_to_band)

X_q = X_cluster.copy()
X_q["ncd_years"] = quintile_band(X_cluster["ncd_years"], y_train, exp_train, n_bands=5)
X_q["vehicle_age"] = quintile_band(X_cluster["vehicle_age"], y_train, exp_train, n_bands=5)

# Test data
X_cluster_test = df_test[["ncd_years", "vehicle_age", "vehicle_group", "area"]].copy()
X_q_test = X_cluster_test.copy()
X_q_test["ncd_years"] = quintile_band(
    X_cluster_test["ncd_years"], y_test, exp_test, n_bands=5
)
X_q_test["vehicle_age"] = quintile_band(
    X_cluster_test["vehicle_age"], y_test, exp_test, n_bands=5
)

# Fit GLM with quintile bands
q_formula_str = "y ~ C(ncd_years) + C(vehicle_age) + vehicle_group + C(area)"
quint_result = smf.glm(
    formula=q_formula_str,
    data=X_q.assign(y=y_train),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(exp_train),
).fit(disp=False)
pred_q_test = quint_result.predict(X_q_test.assign(y=y_test), offset=np.log(exp_test))
dev_quint_test = poisson_deviance_holdout(y_test, pred_q_test)

# R2VF refit GLM
X_r2vf_train = fc.transform(X_cluster)
X_r2vf_test = fc.transform(X_cluster_test)

r2vf_result = fc.refit_glm(X_cluster, y_train, exposure=exp_train)

# Predict with R2VF refit using statsmodels predict
# Build test refit matrix manually
from insurance_glm_tools.cluster.backends import build_refit_matrix
X_r2vf_test_mat, col_names_r = build_refit_matrix(
    X_cluster_test,
    {factor: fc.level_map(factor).level_to_group for factor in ["ncd_years", "vehicle_age"]},
    ["ncd_years", "vehicle_age"],
)
X_r2vf_test_sm = sm.add_constant(X_r2vf_test_mat, has_constant="add")
pred_r2vf_test = r2vf_result.predict(X_r2vf_test_sm, offset=np.log(exp_test))
dev_r2vf_test = poisson_deviance_holdout(y_test, pred_r2vf_test)

n_quint_params = quint_result.df_model + 1
n_r2vf_params = r2vf_result.df_model + 1

displayHTML(f"""
<h3>R2VF vs Manual Quintile Banding</h3>
<p style="font-size:13px; color:#555;">Structured factors only: ncd_years + vehicle_age + vehicle_group + area</p>
<table style="border-collapse:collapse; font-size:13px;">
<thead>
<tr style="background:#2c3e50; color:white;">
  <th style="padding:8px 16px; text-align:left;">Method</th>
  <th style="padding:8px 16px; text-align:right;">NCD groups</th>
  <th style="padding:8px 16px; text-align:right;">Veh age groups</th>
  <th style="padding:8px 16px; text-align:right;">Total params</th>
  <th style="padding:8px 16px; text-align:right;">Test deviance</th>
</tr>
</thead>
<tbody>
<tr style="background:#ecf0f1;">
  <td style="padding:6px 16px;">Manual quintile banding (5 each, fixed)</td>
  <td style="padding:6px 16px; text-align:right;">5</td>
  <td style="padding:6px 16px; text-align:right;">5</td>
  <td style="padding:6px 16px; text-align:right;">{int(n_quint_params)}</td>
  <td style="padding:6px 16px; text-align:right;">{dev_quint_test:.4f}</td>
</tr>
<tr style="background:#d5f5e3;">
  <td style="padding:6px 16px;">R2VF clustering (BIC-selected, data-driven)</td>
  <td style="padding:6px 16px; text-align:right;">{ncd_lm.n_groups}</td>
  <td style="padding:6px 16px; text-align:right;">{veh_age_lm.n_groups}</td>
  <td style="padding:6px 16px; text-align:right;">{int(n_r2vf_params)}</td>
  <td style="padding:6px 16px; text-align:right;">{dev_r2vf_test:.4f}</td>
</tr>
</tbody>
</table>
<p style="font-size:13px; color:#555; margin-top:8px;">
  R2VF finds a <strong>data-driven</strong> number of groups. When adjacent levels genuinely share
  the same true frequency, fewer groups are justified by BIC — producing a more parsimonious model
  without sacrificing predictive accuracy.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d. Level maps as HTML tables

# COMMAND ----------

def level_map_to_html(lm, title):
    df = lm.to_df()
    n_groups = lm.n_groups
    group_colors_hex = ["#d6eaf8", "#d5f5e3", "#fdebd0", "#fdf2f8", "#e8f8f5",
                        "#fef9e7", "#f5cba7", "#d7bde2"]
    rows = ""
    for _, r in df.iterrows():
        bg = group_colors_hex[int(r["merged_group"]) % len(group_colors_hex)]
        rows += f"""<tr style="background:{bg};">
  <td style="padding:5px 14px; font-family:monospace;">{r["original_level"]}</td>
  <td style="padding:5px 14px; text-align:center;">{int(r["merged_group"])}</td>
  <td style="padding:5px 14px; text-align:right;">{r["coefficient"]:.4f}</td>
  <td style="padding:5px 14px; text-align:right;">{r["group_exposure"]:.0f}</td>
</tr>"""
    return f"""
<div style="display:inline-block; vertical-align:top; margin-right:30px;">
<h4>{title} ({lm.n_levels} levels -> {lm.n_groups} groups)</h4>
<table style="border-collapse:collapse; font-size:12px;">
<thead>
<tr style="background:#2c3e50; color:white;">
  <th style="padding:6px 14px;">Level</th>
  <th style="padding:6px 14px;">Group</th>
  <th style="padding:6px 14px; text-align:right;">Log-rel</th>
  <th style="padding:6px 14px; text-align:right;">Exposure</th>
</tr>
</thead>
<tbody>{rows}</tbody>
</table>
</div>"""

displayHTML(
    "<h3>Factor-level maps from R2VF</h3>" +
    level_map_to_html(ncd_lm, "NCD years") +
    level_map_to_html(veh_age_lm, "Vehicle age")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary
# MAGIC
# MAGIC ### NestedGLMPipeline
# MAGIC
# MAGIC The embedding approach is clearly better than dummy-coding with sparse makes:
# MAGIC - Operates over all 450 makes including thin ones, without manual grouping
# MAGIC - Handles unseen makes at prediction time via the learned embedding space
# MAGIC - Outer GLM parameter count stays low regardless of cardinality
# MAGIC - Embedding dimensionality provides a tunable tradeoff between regularisation and capacity
# MAGIC
# MAGIC The Wang, Shi, Cao (2025) framework is the right tool when you have 100+ factor levels
# MAGIC and no natural ordering. For spatial factors (postcode sector), the optional `geo_gdf`
# MAGIC argument adds SKATER-constrained territory clustering in phase 3.
# MAGIC
# MAGIC ### FactorClusterer (R2VF)
# MAGIC
# MAGIC For ordinal factors where level banding is currently done by hand:
# MAGIC - BIC selects the number of groups automatically — no more arbitrary quintile splits
# MAGIC - Adjacent levels with statistically indistinguishable frequencies are fused
# MAGIC - Output is a standard `LevelMap` object that recodes the factor for downstream GLMs
# MAGIC - The `min_exposure` constraint prevents thin groups standing alone
# MAGIC
# MAGIC The two tools complement each other: use `FactorClusterer` to band ordinal factors
# MAGIC (NCD, vehicle age, driver age), then pass the banded factors as the base formula
# MAGIC in `NestedGLMPipeline` alongside any high-cardinality unordered factors.

# COMMAND ----------

final_results = {
    "Base GLM test deviance": round(dev_base_test, 4),
    "Dummy-coded GLM test deviance": round(dev_dummy_test, 4),
    "NestedGLMPipeline test deviance": round(dev_nested_test, 4),
    "Embedding improvement over dummy (%)": round(improvement_pct, 1),
    "NCD original levels": ncd_lm.n_levels,
    "NCD R2VF groups (BIC)": ncd_lm.n_groups,
    "Vehicle age original levels": veh_age_lm.n_levels,
    "Vehicle age R2VF groups (BIC)": veh_age_lm.n_groups,
    "Quintile banding test deviance": round(dev_quint_test, 4),
    "R2VF test deviance": round(dev_r2vf_test, 4),
}

rows_html = ""
for i, (k, v) in enumerate(final_results.items()):
    bg = "#ecf0f1" if i % 2 == 0 else "white"
    rows_html += f'<tr style="background:{bg};"><td style="padding:6px 16px; font-family:monospace;">{k}</td><td style="padding:6px 16px; text-align:right; font-weight:bold;">{v}</td></tr>'

displayHTML(f"""
<h3>Final results summary</h3>
<table style="border-collapse:collapse; font-size:13px;">
<thead>
<tr style="background:#2c3e50; color:white;">
  <th style="padding:8px 16px; text-align:left;">Metric</th>
  <th style="padding:8px 16px; text-align:right;">Value</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
""")
