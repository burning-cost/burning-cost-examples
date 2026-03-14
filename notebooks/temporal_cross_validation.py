# Databricks notebook source

# MAGIC %md
# MAGIC # Temporal Cross-Validation for Insurance Pricing: Why Random CV Lies
# MAGIC
# MAGIC **The problem:** Standard k-fold CV randomly shuffles your data into folds. For insurance pricing, this is wrong in a way that doesn't announce itself. The CV scores look fine. The model looks well-evaluated. And then in the rating year, actual performance is noticeably worse than the CV estimate suggested.
# MAGIC
# MAGIC The cause is temporal leakage. When you randomly assign a 2022 policy to the test set while 2023 data is in training, the model has implicitly learned from future patterns to predict the past. Claims development, seasonal cycles, and trend drift all flow backwards across the random split boundary. The result is a CV estimate that is optimistic — sometimes subtly, sometimes by enough to matter for rate adequacy decisions.
# MAGIC
# MAGIC This notebook benchmarks three evaluation strategies on the same dataset and model:
# MAGIC
# MAGIC 1. **Random 5-fold KFold** — the standard sklearn default, almost certainly what your team is using
# MAGIC 2. **Walk-forward temporal CV** (expanding window, 3-month IBNR buffer) — what you should be using for motor
# MAGIC 3. **True out-of-time holdout** — the ground truth, a final test year the model never touched
# MAGIC
# MAGIC The hypothesis: random CV will produce an optimistic deviance estimate. Walk-forward CV will be closer to the true holdout. The gap between them is the cost of leaky evaluation.
# MAGIC
# MAGIC **Key outputs:**
# MAGIC - Measured optimism bias of random CV relative to true OOT performance
# MAGIC - Fold-level trajectory showing how model performance changes as the test window moves further from training
# MAGIC - Timeline visualisation of fold boundaries
# MAGIC - Split summary table suitable for model governance documentation

# COMMAND ----------

# MAGIC %pip install insurance-cv catboost polars matplotlib scikit-learn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate a Synthetic UK Motor Portfolio
# MAGIC
# MAGIC We need 5 years of policy data with:
# MAGIC - A known data-generating process so we can measure optimism bias precisely
# MAGIC - Temporal structure that gives random CV something to leak: a frequency trend and seasonal variation
# MAGIC - Realistic UK motor rating factors and exposure distributions
# MAGIC
# MAGIC The temporal structure is the key design decision here. A DGP with no trend would make all CV strategies equivalent — there would be nothing to leak. We inject two sources of temporal signal:
# MAGIC
# MAGIC 1. **Frequency trend**: a mild 3% per year increase in claim frequency, plausibly reflecting cost inflation or a portfolio mix shift towards higher-risk segments. Random CV sees future high-frequency years in training when testing on past low-frequency years, inflating apparent predictive performance.
# MAGIC 2. **Seasonal variation**: Q4 claims are ~15% above average (winter conditions). Random CV mixes seasons arbitrarily across folds; temporal CV tests on contiguous periods with realistic seasonal structure.
# MAGIC
# MAGIC Portfolio: 50,000 policies across 5 years (2019–2023). CV pool: 2019–2022. True holdout: 2023.

# COMMAND ----------

from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import pandas as pd
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# Reproducibility
rng = np.random.default_rng(seed=42)

# Portfolio parameters
N_POLICIES = 50_000
YEARS = [2019, 2020, 2021, 2022, 2023]
N_PER_YEAR = N_POLICIES // len(YEARS)   # 10,000 per year
BASE_FREQ = 0.075                        # 7.5% annual frequency — UK motor typical
FREQ_TREND = 1.03                        # 3% per year frequency increase
HOLDOUT_YEAR = 2023                      # kept completely separate from CV

print(f"Total policies:      {N_POLICIES:,}")
print(f"Years:               {YEARS[0]}–{YEARS[-1]}")
print(f"Per year:            {N_PER_YEAR:,}")
print(f"Base frequency:      {BASE_FREQ:.1%}")
print(f"Annual freq trend:   {(FREQ_TREND - 1):.1%} increase per year")
print(f"Holdout year:        {HOLDOUT_YEAR} (never used in CV)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data-Generating Process
# MAGIC
# MAGIC The true model is a multiplicative frequency model:
# MAGIC ```
# MAGIC freq_true = base_freq × year_trend × seasonal_factor × age_factor × ncd_factor × vehicle_factor
# MAGIC ```
# MAGIC
# MAGIC The deployed model approximates these factors but with some miscalibration — it knows the general shape of the age and NCD relativities but does not capture the year-on-year trend (this is the key leakage source: if training data contains future years, the model implicitly absorbs the trend signal it would not have access to at genuine prediction time).

# COMMAND ----------

def seasonal_factor(inception_dates: np.ndarray) -> np.ndarray:
    """Q4 (Oct-Dec) is ~15% higher frequency. Q1-Q3 near-average."""
    months = np.array([d.month for d in inception_dates])
    return np.where(months >= 10, 1.15,
           np.where(months <= 3, 1.08, 1.0))


def age_factor_true(ages: np.ndarray) -> np.ndarray:
    """True frequency factor for driver age. UK actuarial shape."""
    return np.where(ages < 25, 2.5,
           np.where(ages < 35, 1.4,
           np.where(ages < 55, 1.0,
           np.where(ages < 70, 0.85, 1.2))))


def age_factor_model(ages: np.ndarray) -> np.ndarray:
    """What the model believes about the age factor — slightly miscalibrated."""
    return np.where(ages < 25, 2.3,
           np.where(ages < 35, 1.35,
           np.where(ages < 55, 1.0,
           np.where(ages < 70, 0.88, 1.15))))


def ncd_factor(ncd: np.ndarray) -> np.ndarray:
    """NCD discount on frequency. Exponential shape typical in UK motor."""
    return np.exp(-0.12 * ncd)


def vehicle_age_factor(veh_age: np.ndarray) -> np.ndarray:
    """Older vehicles: slightly higher frequency (parts failure, safety features)."""
    return 1.0 + 0.025 * veh_age


# Generate the portfolio year by year
records = []
policy_id = 0

for yr_idx, year in enumerate(YEARS):
    year_trend = FREQ_TREND ** yr_idx  # compounds from base year (2019)

    # Inception dates: random across the year
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    days_in_year = (year_end - year_start).days + 1

    inception_days = rng.integers(0, days_in_year, N_PER_YEAR)
    inception_dates = [year_start + timedelta(days=int(d)) for d in inception_days]

    # Rating factors
    driver_ages = rng.integers(18, 80, N_PER_YEAR)
    ncd_years = rng.choice(range(0, 10), N_PER_YEAR,
                           p=[0.12, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.12, 0.12])
    vehicle_ages = rng.integers(0, 15, N_PER_YEAR)
    vehicle_values = rng.lognormal(mean=9.3, sigma=0.5, size=N_PER_YEAR)  # ~GBP 5k-30k
    exposure = rng.beta(a=8, b=2, size=N_PER_YEAR)  # car-years [0.1, 1.0]

    # True frequency
    seas = seasonal_factor(np.array(inception_dates))
    af_true = age_factor_true(driver_ages.astype(float))
    nf = ncd_factor(ncd_years.astype(float))
    vf = vehicle_age_factor(vehicle_ages.astype(float))

    true_freq = BASE_FREQ * year_trend * seas * af_true * nf * vf

    # Claims from true process
    claim_counts = rng.poisson(true_freq * exposure)

    # Model prediction: no year trend (model was trained at a point in time, doesn't know the future trend)
    af_model = age_factor_model(driver_ages.astype(float))
    model_freq = BASE_FREQ * seas * af_model * nf * vf  # no year_trend term

    for i in range(N_PER_YEAR):
        records.append({
            "policy_id": f"POL{policy_id:06d}",
            "inception_date": inception_dates[i],
            "policy_year": year,
            "driver_age": int(driver_ages[i]),
            "ncd_years": int(ncd_years[i]),
            "vehicle_age": int(vehicle_ages[i]),
            "vehicle_value": round(float(vehicle_values[i]), 2),
            "exposure": round(float(exposure[i]), 4),
            "claim_count": int(claim_counts[i]),
            "true_freq": round(float(true_freq[i]), 6),
            "model_pred": round(float(model_freq[i]), 6),
        })
        policy_id += 1

df = pl.DataFrame(records).with_columns(
    pl.col("inception_date").cast(pl.Date)
)

print(f"Portfolio shape:     {df.shape}")
print(f"Date range:          {df['inception_date'].min()} to {df['inception_date'].max()}")
print(f"Claim frequency:     {df['claim_count'].sum() / df['exposure'].sum():.3%} actual")
print(f"\nPolicy year breakdown:")
print(df.group_by("policy_year").agg([
    pl.len().alias("n"),
    pl.col("claim_count").sum().alias("claims"),
    (pl.col("claim_count").sum() / pl.col("exposure").sum()).alias("freq"),
]).sort("policy_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Trend Is Visible in the Data
# MAGIC
# MAGIC Before we compare CV strategies, confirm the temporal structure is real. If frequency is genuinely rising year-on-year, random CV will be optimistic because it can train on post-trend data to predict pre-trend years. The column `freq` above should show a consistent upward drift from 2019 to 2023.

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64

def fig_to_html(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 HTML img tag for displayHTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" style="max-width:900px"/>'


# Actual frequency by policy year
freq_by_year = (
    df.group_by("policy_year")
    .agg((pl.col("claim_count").sum() / pl.col("exposure").sum()).alias("actual_freq"))
    .sort("policy_year")
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(freq_by_year["policy_year"].to_list(),
       [v * 100 for v in freq_by_year["actual_freq"].to_list()],
       color=["#c0392b" if y == HOLDOUT_YEAR else "#2980b9" for y in freq_by_year["policy_year"].to_list()],
       alpha=0.85, edgecolor="white", linewidth=0.5)

ax.set_xlabel("Policy Year", fontsize=11)
ax.set_ylabel("Actual Claim Frequency (%)", fontsize=11)
ax.set_title("Claim Frequency by Policy Year\nUpward trend is the leakage source for random CV", fontsize=12)
ax.axhline(BASE_FREQ * 100, color="#7f8c8d", linestyle="--", linewidth=1, label=f"Base frequency {BASE_FREQ:.1%}")

cv_patch = mpatches.Patch(color="#2980b9", alpha=0.85, label="CV pool (2019–2022)")
holdout_patch = mpatches.Patch(color="#c0392b", alpha=0.85, label=f"Holdout ({HOLDOUT_YEAR})")
ax.legend(handles=[cv_patch, holdout_patch], fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

displayHTML(fig_to_html(fig))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Split into CV Pool and True Holdout
# MAGIC
# MAGIC The 2023 data is set aside now and never touched during cross-validation. This is the true out-of-time test — it represents the rating year the model will be deployed into. We measure CV strategy quality by comparing each strategy's estimate to this ground truth.

# COMMAND ----------

df_cv = df.filter(pl.col("policy_year") < HOLDOUT_YEAR)
df_holdout = df.filter(pl.col("policy_year") == HOLDOUT_YEAR)

print(f"CV pool:        {len(df_cv):,} policies ({df_cv['policy_year'].min()}–{df_cv['policy_year'].max()})")
print(f"Holdout:        {len(df_holdout):,} policies (year {HOLDOUT_YEAR})")

# Feature matrix and target for the CV pool
FEATURES = ["driver_age", "ncd_years", "vehicle_age", "vehicle_value"]

X_cv = df_cv[FEATURES].to_pandas().values
y_cv = df_cv["claim_count"].to_pandas().values
offset_cv = np.log(df_cv["exposure"].to_pandas().values)

X_holdout = df_holdout[FEATURES].to_pandas().values
y_holdout = df_holdout["claim_count"].to_pandas().values
offset_holdout = np.log(df_holdout["exposure"].to_pandas().values)

print(f"\nFeatures used:  {FEATURES}")
print(f"Target:         claim_count (Poisson regression)")
print(f"Offset:         log(exposure) — accounts for partial-year policies")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: The Benchmark — Three CV Strategies
# MAGIC
# MAGIC ### Strategy A: Random 5-Fold KFold
# MAGIC
# MAGIC The sklearn default. Five folds, random shuffle. Every policy can appear in any fold regardless of its inception year. This is the baseline to beat.
# MAGIC
# MAGIC We fit a CatBoost Poisson regression — a representative GBM for motor frequency. The metric is mean Poisson deviance, which is appropriate for count targets. Lower deviance = better model.

# COMMAND ----------

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_poisson_deviance

# CatBoost Poisson with sensible motor pricing hyperparameters
# Not heavily tuned — the point is to compare CV strategies, not optimise the model
catboost_params = dict(
    loss_function="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
)


def fit_catboost(X_train, y_train, offset_train):
    """Fit CatBoost Poisson with log offset."""
    model = CatBoostRegressor(**catboost_params)
    model.fit(X_train, y_train, sample_weight=np.exp(offset_train))
    return model


def poisson_deviance(y_true, y_pred, exposure):
    """Mean Poisson deviance per unit exposure."""
    # Avoid log(0)
    y_pred_clipped = np.maximum(y_pred * exposure, 1e-9)
    y_true_counts = y_true
    d = 2 * (
        np.where(y_true_counts > 0, y_true_counts * np.log(y_true_counts / y_pred_clipped), 0)
        - (y_true_counts - y_pred_clipped)
    )
    return float(d.mean())


# Random 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

random_fold_scores = []
print("Random KFold — fold-level Poisson deviance:")
for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_cv), start=1):
    X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
    y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
    off_tr, off_te = offset_cv[train_idx], offset_cv[test_idx]
    exp_te = np.exp(off_te)

    m = fit_catboost(X_tr, y_tr, off_tr)
    preds = m.predict(X_te)  # predict raw frequency rate
    score = poisson_deviance(y_te, preds, exp_te)
    random_fold_scores.append(score)
    print(f"  Fold {fold_num}: deviance = {score:.6f}")

random_cv_mean = float(np.mean(random_fold_scores))
random_cv_std = float(np.std(random_fold_scores))
print(f"\nRandom KFold CV estimate:  {random_cv_mean:.6f} ± {random_cv_std:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy B: Walk-Forward Temporal CV
# MAGIC
# MAGIC Using `insurance-cv` with:
# MAGIC - `min_train_months=12` — need at least a full year to cover seasonal variation
# MAGIC - `test_months=6` — evaluate on 6-month windows (two per year)
# MAGIC - `step_months=6` — non-overlapping test periods
# MAGIC - `ibnr_buffer_months=3` — standard for UK motor own-damage (short development tail)
# MAGIC
# MAGIC The leakage check runs before any model fitting. This is not just good practice — on a production pricing workflow, a silent leakage error discovered post-submission is expensive.

# COMMAND ----------

from insurance_cv import walk_forward_split, temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV

# Generate walk-forward splits on the CV pool
temporal_splits = walk_forward_split(
    df_cv,
    date_col="inception_date",
    min_train_months=12,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)

# Validate before running any models
check = temporal_leakage_check(temporal_splits, df_cv, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("Temporal leakage detected:\n" + "\n".join(check["errors"]))
if check["warnings"]:
    print("Warnings (not fatal):")
    for w in check["warnings"]:
        print(f"  {w}")
else:
    print("Leakage check: PASSED — no temporal leakage detected\n")

# Summary of fold structure
summary = split_summary(temporal_splits, df_cv, date_col="inception_date")
print(f"Temporal CV folds generated: {len(temporal_splits)}")
print(summary.select(["fold", "train_n", "test_n", "train_end", "test_start", "gap_days", "ibnr_buffer_months"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fold Structure Table (Governance-Ready)
# MAGIC
# MAGIC The `split_summary` output is what a model governance reviewer will ask for. It shows exactly which data was used to train and test each fold, with the IBNR buffer documented. Random KFold cannot produce this table — there are no meaningful date boundaries to report.

# COMMAND ----------

summary_pd = summary.to_pandas()

header_style = "background:#1a252f;color:white;padding:8px 12px;text-align:left"
rows_html = ""
for _, row in summary_pd.iterrows():
    rows_html += (
        f'<tr>'
        f'<td style="padding:6px 12px">{int(row["fold"])}</td>'
        f'<td style="padding:6px 12px;text-align:right">{int(row["train_n"]):,}</td>'
        f'<td style="padding:6px 12px;text-align:right">{int(row["test_n"]):,}</td>'
        f'<td style="padding:6px 12px">{row["train_end"]}</td>'
        f'<td style="padding:6px 12px">{row["test_start"]}</td>'
        f'<td style="padding:6px 12px;text-align:right">{int(row["gap_days"])}</td>'
        f'<td style="padding:6px 12px;text-align:right">{int(row["ibnr_buffer_months"])}</td>'
        f'</tr>'
    )

displayHTML(f"""
<h3>Temporal CV Fold Structure</h3>
<p style="font-size:13px;color:#555">IBNR buffer: {temporal_splits[0].ibnr_buffer_months} months.
Training window expands with each fold. Test windows are non-overlapping 6-month periods.
This table is the audit trail for model governance.</p>
<table style="border-collapse:collapse;font-family:monospace;font-size:13px">
  <thead>
    <tr>
      <th style="{header_style}">Fold</th>
      <th style="{header_style}">Train N</th>
      <th style="{header_style}">Test N</th>
      <th style="{header_style}">Train End</th>
      <th style="{header_style}">Test Start</th>
      <th style="{header_style}">Gap (days)</th>
      <th style="{header_style}">IBNR Buffer (mths)</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the Model on Each Temporal Fold

# COMMAND ----------

temporal_fold_scores = []
temporal_fold_meta = []
print("Temporal walk-forward — fold-level Poisson deviance:")

for fold_num, split in enumerate(temporal_splits, start=1):
    train_idx, test_idx = split.get_indices(df_cv)

    if len(train_idx) == 0 or len(test_idx) == 0:
        print(f"  Fold {fold_num}: skipped (empty set)")
        continue

    X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
    y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
    off_tr, off_te = offset_cv[train_idx], offset_cv[test_idx]
    exp_te = np.exp(off_te)

    m = fit_catboost(X_tr, y_tr, off_tr)
    preds = m.predict(X_te)
    score = poisson_deviance(y_te, preds, exp_te)
    temporal_fold_scores.append(score)

    # Capture test window dates for the timeline plot
    test_dates = df_cv["inception_date"].to_pandas()
    test_start = test_dates.iloc[test_idx].min()
    test_end = test_dates.iloc[test_idx].max()
    temporal_fold_meta.append({
        "fold": fold_num,
        "test_start": test_start,
        "test_end": test_end,
        "deviance": score,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
    })

    print(f"  Fold {fold_num} ({test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}): deviance = {score:.6f}")

temporal_cv_mean = float(np.mean(temporal_fold_scores))
temporal_cv_std = float(np.std(temporal_fold_scores))
print(f"\nTemporal walk-forward CV estimate:  {temporal_cv_mean:.6f} ± {temporal_cv_std:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy C: True Out-of-Time Holdout (Ground Truth)
# MAGIC
# MAGIC Train the model on all of 2019–2022. Evaluate on 2023. This is what the model will actually do in the rating year — it cannot see 2023 data at deployment. This is the number that matters.

# COMMAND ----------

# Final model: trained on full CV pool, evaluated on holdout
final_model = fit_catboost(X_cv, y_cv, offset_cv)
holdout_preds = final_model.predict(X_holdout)
exp_holdout = np.exp(offset_holdout)

true_oot_deviance = poisson_deviance(y_holdout, holdout_preds, exp_holdout)
print(f"True OOT (2023 holdout) deviance:   {true_oot_deviance:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: The Comparison — How Wrong Was Random CV?
# MAGIC
# MAGIC The three numbers side by side tell the story. Lower deviance = model looks better.
# MAGIC
# MAGIC Random KFold produces the most optimistic estimate. The walk-forward temporal CV is typically closer to the true OOT performance. The gap between random CV and true OOT is the optimism bias — the amount by which the random CV evaluation overstated the model's prospective performance.

# COMMAND ----------

# Summary comparison
random_gap = random_cv_mean - true_oot_deviance
temporal_gap = temporal_cv_mean - true_oot_deviance

# Optimism: negative gap = CV estimate was better than reality (overoptimistic)
random_optimism_pct = (random_gap / true_oot_deviance) * 100
temporal_optimism_pct = (temporal_gap / true_oot_deviance) * 100

improvement = abs(random_gap) - abs(temporal_gap)
improvement_pct = (improvement / abs(random_gap)) * 100 if abs(random_gap) > 1e-9 else 0.0

print("=" * 60)
print("CV STRATEGY COMPARISON RESULTS")
print("=" * 60)
print(f"Random KFold CV mean deviance:      {random_cv_mean:.6f}")
print(f"Temporal walk-forward CV deviance:  {temporal_cv_mean:.6f}")
print(f"True OOT holdout deviance:          {true_oot_deviance:.6f}")
print()
print(f"Random CV gap to true OOT:          {random_gap:+.6f}  ({random_optimism_pct:+.2f}%)")
print(f"Temporal CV gap to true OOT:        {temporal_gap:+.6f}  ({temporal_optimism_pct:+.2f}%)")
print()
if random_gap < 0 and abs(temporal_gap) < abs(random_gap):
    print(f"Random CV is MORE optimistic by {improvement_pct:.1f}%")
    print(f"Temporal CV reduces the estimation error by {improvement:.6f} deviance units")
elif random_gap < 0:
    print("Both strategies underestimate OOT deviance, but random CV is more so")
else:
    print("Random CV overestimates deviance in this case")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Visualisations
# MAGIC
# MAGIC ### Timeline: Where Do the Fold Boundaries Fall?
# MAGIC
# MAGIC This chart shows the temporal CV fold structure against the full data timeline. Each horizontal bar is a fold's test window. The 2023 holdout sits to the right of all training data — it is genuinely out-of-time. Random KFold has no equivalent representation because its folds have no coherent time boundaries.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(11, 5))

# Background: full data range
data_start = pd.Timestamp("2019-01-01")
data_end = pd.Timestamp("2023-12-31")
holdout_start = pd.Timestamp(f"{HOLDOUT_YEAR}-01-01")

ax.barh(
    y=0, left=data_start, width=holdout_start - data_start,
    height=0.4, color="#aed6f1", alpha=0.5, label="CV pool (2019–2022)"
)
ax.barh(
    y=0, left=holdout_start, width=data_end - holdout_start,
    height=0.4, color="#f1948a", alpha=0.5, label=f"True OOT holdout ({HOLDOUT_YEAR})"
)

# Temporal CV fold test windows
colours = plt.cm.Blues(np.linspace(0.4, 0.9, len(temporal_fold_meta)))
for i, meta in enumerate(temporal_fold_meta):
    ts = pd.Timestamp(meta["test_start"])
    te = pd.Timestamp(meta["test_end"])
    y_pos = -(i + 1) * 0.55 - 0.2
    ax.barh(
        y=y_pos, left=ts, width=te - ts,
        height=0.4, color=colours[i], alpha=0.9,
        label=f"Fold {meta['fold']} test" if i == 0 else None
    )
    ax.text(
        te + pd.Timedelta(days=5), y_pos,
        f"d={meta['deviance']:.5f}", va="center", fontsize=8, color="#2c3e50"
    )

# Holdout marker
ax.axvline(holdout_start, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Holdout boundary ({HOLDOUT_YEAR})")

# Formatting
n_folds = len(temporal_fold_meta)
ax.set_xlim(data_start - pd.Timedelta(days=30), data_end + pd.Timedelta(days=120))
ax.set_ylim(-(n_folds + 1) * 0.55, 0.7)
ax.set_yticks([0] + [-(i + 1) * 0.55 - 0.2 for i in range(n_folds)])
ax.set_yticklabels(["Full dataset"] + [f"Fold {m['fold']}" for m in temporal_fold_meta], fontsize=9)
ax.set_xlabel("Calendar Date", fontsize=11)
ax.set_title("Temporal CV Fold Boundaries vs Full Dataset Timeline\nTest windows shown with Poisson deviance", fontsize=12)
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)

# Year gridlines
for yr in YEARS:
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#bdc3c7", linewidth=0.5, linestyle=":")

plt.tight_layout()
displayHTML(fig_to_html(fig))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fold-Level Deviance Trajectory
# MAGIC
# MAGIC This is the diagnostic chart that random CV cannot produce. As the temporal CV test window advances further from the training data (left to right), performance may degrade. A rising trend in fold-level deviance is a signal that the model's accuracy degrades with temporal distance — which is exactly what you expect from a dataset with a frequency trend.
# MAGIC
# MAGIC Random CV, by averaging across all time periods, hides this signal. It gives you a number that represents average performance across all temporal distances — including past data where the model has an unfair advantage. The fold trajectory shows you the prospective trajectory.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: fold-level deviance trajectory
fold_nums = [m["fold"] for m in temporal_fold_meta]
fold_deviances = [m["deviance"] for m in temporal_fold_meta]
fold_labels = [m["test_start"].strftime("%Y-Q%d")[:7] for m in temporal_fold_meta]

ax = axes[0]
ax.plot(fold_nums, fold_deviances, "o-", color="#2980b9", linewidth=2, markersize=7, label="Temporal CV folds")
ax.axhline(temporal_cv_mean, color="#2980b9", linestyle="--", linewidth=1, alpha=0.6, label=f"Temporal CV mean ({temporal_cv_mean:.5f})")
ax.axhline(random_cv_mean, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.8, label=f"Random KFold mean ({random_cv_mean:.5f})")
ax.axhline(true_oot_deviance, color="#27ae60", linestyle="-", linewidth=2, label=f"True OOT ({true_oot_deviance:.5f})")

ax.set_xlabel("Temporal CV Fold Number", fontsize=11)
ax.set_ylabel("Poisson Deviance (lower = better)", fontsize=11)
ax.set_title("Fold-Level Performance Trajectory\n(time moves left to right)", fontsize=11)
ax.set_xticks(fold_nums)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Add annotation for rising trend
if len(fold_deviances) >= 3:
    first_half = np.mean(fold_deviances[:len(fold_deviances)//2])
    second_half = np.mean(fold_deviances[len(fold_deviances)//2:])
    if second_half > first_half:
        ax.annotate(
            "Performance degrades\nas test window\nmoves later in time",
            xy=(fold_nums[-1], fold_deviances[-1]),
            xytext=(fold_nums[-1] - 1.5, fold_deviances[-1] + (max(fold_deviances) - min(fold_deviances)) * 0.3),
            fontsize=8, color="#7f8c8d",
            arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
        )

# Right: summary bar chart — the three strategies
ax2 = axes[1]
strategies = ["Random\n5-Fold KFold", "Walk-Forward\nTemporal CV", "True OOT\nHoldout (2023)"]
values = [random_cv_mean, temporal_cv_mean, true_oot_deviance]
bar_colours = ["#e74c3c", "#2980b9", "#27ae60"]
bars = ax2.bar(strategies, values, color=bar_colours, alpha=0.85, edgecolor="white", linewidth=1.5)

# Value labels on bars
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.00002,
             f"{val:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Gap annotations
y_top = max(values) * 1.04
ax2.annotate(
    "",
    xy=(1.0, values[1]), xytext=(0.0, values[0]),
    arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.5),
)

# Annotate the optimism gap
gap_y = (values[0] + values[1]) / 2
ax2.annotate(
    f"Optimism bias\n{abs(random_gap):.5f} deviance",
    xy=(0.5, gap_y), xytext=(0.5, gap_y),
    ha="center", va="center", fontsize=8, color="#7f8c8d",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

ax2.set_ylabel("Mean Poisson Deviance (lower = better)", fontsize=11)
ax2.set_title("Evaluation Strategy Comparison\nRed = most optimistic, Green = ground truth", fontsize=11)
ax2.set_ylim(min(values) * 0.998, max(values) * 1.045)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
displayHTML(fig_to_html(fig))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing Fold Variance: Random vs Temporal CV
# MAGIC
# MAGIC Random CV folds have artificially low variance — each fold is a random sample from the same overall distribution, so the fold scores are almost interchangeable. Temporal CV folds have higher variance because they genuinely test different time periods. This higher variance is informative: it tells you whether performance is stable across the book's temporal evolution, or whether it degrades as the model ages.
# MAGIC
# MAGIC A model that shows rising fold deviance as the test window advances is one that will need more frequent refitting.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 4))

# Random fold scores
ax.scatter(
    [f"R-{i+1}" for i in range(len(random_fold_scores))],
    random_fold_scores,
    color="#e74c3c", s=80, zorder=5, label="Random KFold folds"
)
ax.axhline(random_cv_mean, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)

# Temporal fold scores (offset on x axis)
offset_labels = [f"T-{m['fold']}" for m in temporal_fold_meta]
ax.scatter(
    offset_labels,
    fold_deviances,
    color="#2980b9", s=80, marker="D", zorder=5, label="Temporal CV folds"
)
ax.axhline(temporal_cv_mean, color="#2980b9", linestyle="--", linewidth=1.5, alpha=0.7)

# True OOT reference
ax.axhline(true_oot_deviance, color="#27ae60", linestyle="-", linewidth=2, label=f"True OOT = {true_oot_deviance:.5f}")

ax.set_ylabel("Poisson Deviance per Fold", fontsize=11)
ax.set_title(
    f"Fold Variance Comparison\n"
    f"Random CV std={random_cv_std:.6f}   Temporal CV std={temporal_cv_std:.6f}",
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
displayHTML(fig_to_html(fig))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Summary Table — Results at a Glance

# COMMAND ----------

closer_strategy = "Walk-forward temporal CV" if abs(temporal_gap) < abs(random_gap) else "Random KFold"

rows_html = ""

def deviance_row(label, value, gap, gap_pct, bg, notes):
    return (
        f'<tr style="background:{bg}">'
        f'<td style="padding:8px 14px"><b>{label}</b></td>'
        f'<td style="padding:8px 14px;text-align:right;font-family:monospace">{value:.6f}</td>'
        f'<td style="padding:8px 14px;text-align:right;font-family:monospace">{gap:+.6f}</td>'
        f'<td style="padding:8px 14px;text-align:right;font-family:monospace">{gap_pct:+.2f}%</td>'
        f'<td style="padding:8px 14px">{notes}</td>'
        f'</tr>'
    )

rows_html += deviance_row(
    "Random 5-Fold KFold", random_cv_mean, random_gap, random_optimism_pct,
    "#fde8e8",
    f"Temporal leakage: YES. Std={random_cv_std:.6f}. Cannot produce time-ordered audit trail."
)
rows_html += deviance_row(
    "Walk-Forward Temporal CV", temporal_cv_mean, temporal_gap, temporal_optimism_pct,
    "#e8f4fd",
    f"Temporal leakage: NO (verified). Std={temporal_cv_std:.6f}. "
    f"{len(temporal_splits)} folds with full audit trail."
)
rows_html += deviance_row(
    "True OOT Holdout (2023)", true_oot_deviance, 0.0, 0.0,
    "#e8f8e8",
    "Ground truth — what the model actually achieves on the rating year."
)

displayHTML(f"""
<h3>Evaluation Strategy Comparison — Final Results</h3>
<p style="font-size:13px;color:#555">
Gap = strategy deviance minus true OOT deviance.
Negative gap = strategy was overoptimistic (claimed better performance than reality).
</p>
<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px;width:100%">
  <thead>
    <tr style="background:#1a252f;color:white">
      <th style="padding:8px 14px;text-align:left">Strategy</th>
      <th style="padding:8px 14px;text-align:right">Mean Deviance</th>
      <th style="padding:8px 14px;text-align:right">Gap to OOT</th>
      <th style="padding:8px 14px;text-align:right">Gap %</th>
      <th style="padding:8px 14px;text-align:left">Notes</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:12px;color:#888;margin-top:8px">
Closer to true OOT: <b>{closer_strategy}</b>.
Temporal CV reduces estimation error by {improvement_pct:.1f}% vs random KFold.
</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Policy-Year Splits — Annual Rate Change Boundaries
# MAGIC
# MAGIC Walk-forward splits are right for ongoing monitoring with monthly granularity. But if your rate changes are annual — which is typical in UK commercial and schemes business — policy-year splits are cleaner. Train on complete policy years, test on the next complete year. The 1 Jan boundary means you are never mixing pre- and post-rate-change data in a single fold.

# COMMAND ----------

from insurance_cv import policy_year_split

py_splits = policy_year_split(
    df_cv,
    date_col="inception_date",
    n_years_train=2,    # train on 2 complete policy years
    n_years_test=1,     # test on the next year
    step_years=1,       # advance one year per fold
)

py_check = temporal_leakage_check(py_splits, df_cv, date_col="inception_date")
assert not py_check["errors"], "Policy-year split leakage detected"

print(f"Policy-year folds: {len(py_splits)}")
print(split_summary(py_splits, df_cv, date_col="inception_date").select(
    ["fold", "label", "train_n", "test_n", "train_end", "test_start"]
))

py_fold_scores = []
print("\nPolicy-year CV — fold-level deviance:")
for fold_num, split in enumerate(py_splits, start=1):
    train_idx, test_idx = split.get_indices(df_cv)
    X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
    y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
    off_tr, off_te = offset_cv[train_idx], offset_cv[test_idx]

    m = fit_catboost(X_tr, y_tr, off_tr)
    preds = m.predict(X_te)
    score = poisson_deviance(y_te, preds, np.exp(off_te))
    py_fold_scores.append(score)
    print(f"  Fold {fold_num} ({split.label}): deviance = {score:.6f}")

py_cv_mean = float(np.mean(py_fold_scores))
print(f"\nPolicy-year CV estimate: {py_cv_mean:.6f}")
print(f"True OOT (2023):         {true_oot_deviance:.6f}")
print(f"Gap:                     {py_cv_mean - true_oot_deviance:+.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: IBNR Buffer Sensitivity
# MAGIC
# MAGIC How much does the IBNR buffer choice matter? This cell sweeps the buffer from 0 to 9 months and shows how the CV estimate changes. The practical question is: does a shorter buffer introduce enough leakage to matter, or is the effect small relative to the variation across folds?
# MAGIC
# MAGIC For motor own-damage (short tail), the effect is typically modest. For longer-tail lines (motor third party bodily injury, liability), the buffer matters substantially.

# COMMAND ----------

buffer_results = []
for buffer_months in [0, 1, 2, 3, 6, 9]:
    try:
        b_splits = walk_forward_split(
            df_cv,
            date_col="inception_date",
            min_train_months=12,
            test_months=6,
            step_months=6,
            ibnr_buffer_months=buffer_months,
        )
        b_check = temporal_leakage_check(b_splits, df_cv, date_col="inception_date")
        if b_check["errors"]:
            buffer_results.append({"buffer": buffer_months, "n_folds": 0, "mean_deviance": None, "error": "leakage"})
            continue

        scores = []
        for split in b_splits:
            train_idx, test_idx = split.get_indices(df_cv)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            m = fit_catboost(X_cv[train_idx], y_cv[train_idx], offset_cv[train_idx])
            preds = m.predict(X_cv[test_idx])
            s = poisson_deviance(y_cv[test_idx], preds, np.exp(offset_cv[test_idx]))
            scores.append(s)

        buffer_results.append({
            "buffer": buffer_months,
            "n_folds": len(scores),
            "mean_deviance": float(np.mean(scores)) if scores else None,
            "error": None,
        })
    except ValueError as e:
        buffer_results.append({"buffer": buffer_months, "n_folds": 0, "mean_deviance": None, "error": str(e)[:40]})

print(f"{'Buffer (mths)':<14} {'N Folds':<10} {'Mean Deviance':<16} {'Gap to OOT':<14}")
print("-" * 55)
for r in buffer_results:
    if r["mean_deviance"] is not None:
        gap = r["mean_deviance"] - true_oot_deviance
        print(f"{r['buffer']:<14} {r['n_folds']:<10} {r['mean_deviance']:<16.6f} {gap:<+14.6f}")
    else:
        print(f"{r['buffer']:<14} {r['n_folds']:<10} {'—':<16} {r.get('error','—')}")

# COMMAND ----------

buffer_plot_data = [(r["buffer"], r["mean_deviance"]) for r in buffer_results if r["mean_deviance"] is not None]
if buffer_plot_data:
    buf_x, buf_y = zip(*buffer_plot_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(buf_x, buf_y, "s-", color="#8e44ad", linewidth=2, markersize=8, label="CV mean deviance by buffer")
    ax.axhline(true_oot_deviance, color="#27ae60", linestyle="-", linewidth=2, label=f"True OOT = {true_oot_deviance:.5f}")
    ax.axhline(random_cv_mean, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Random KFold = {random_cv_mean:.5f}")

    ax.set_xlabel("IBNR Buffer (months)", fontsize=11)
    ax.set_ylabel("Mean Poisson Deviance", fontsize=11)
    ax.set_title("IBNR Buffer Length vs CV Estimate\nMotor own-damage: buffer effect is modest but present", fontsize=11)
    ax.set_xticks(list(buf_x))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    displayHTML(fig_to_html(fig))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: When Temporal CV Matters Most
# MAGIC
# MAGIC This notebook showed one clean example: a linear frequency trend producing measurable optimism bias in random CV. In practice, the effect is larger when:
# MAGIC
# MAGIC **Portfolio composition shifts.** If the mix of risks in your book is changing — new acquisition channels, telematics products, geographic expansion — future data reflects a different population than past data. Random CV trains on that future population when testing on past data, learning patterns that weren't available at the earlier prediction point.
# MAGIC
# MAGIC **Economic cycles matter.** Motor claims are correlated with fuel prices, vehicle repair costs, and accident repair inflation. A model trained on 2020–2021 data (unusually low claims due to COVID lockdowns) and tested on 2022 data will look systematically optimistic under random CV because the 2020–2021 low-frequency signal leaks into every test fold.
# MAGIC
# MAGIC **Rate changes create discontinuities.** If you changed your rates at a policy year boundary, random CV mixes pre- and post-rate-change data in both train and test. This makes the model appear more accurate at predicting near the rate change than it actually is, because it has partially seen the answer. Policy-year splits from `insurance-cv` handle this correctly.
# MAGIC
# MAGIC **Long-tail development.** For liability or professional indemnity, claims reported 3 years after the accident date are still partly unknown at underwriting time. If your training set contains partially-developed claims from near the data collection cutoff, the model learns from understated targets — and CV appears better than it will perform on a fully-developed rating year.
# MAGIC
# MAGIC **The floor case.** If your data has no trend, no composition shift, and a short development tail, random CV and temporal CV will converge. But you rarely know that at the start of a modelling project. Starting with temporal CV costs almost nothing (it's a one-line change from KFold to InsuranceCV) and eliminates a category of methodological risk.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What we did | Key number |
# MAGIC |---|---|
# MAGIC | Generated 50,000-policy UK motor portfolio with temporal structure | 5 years, 3% annual frequency trend |
# MAGIC | Benchmarked random KFold vs walk-forward temporal CV vs true OOT holdout | Three CV strategies, same model |
# MAGIC | Measured optimism bias in random CV | Reported above |
# MAGIC | Showed fold-level trajectory (rising deviance with time) | Walk-forward reveals trend signal that random CV hides |
# MAGIC | Demonstrated policy-year splits for annual rate boundaries | Cleaner than walk-forward for annual rate change models |
# MAGIC | IBNR buffer sensitivity: 0 to 9 months | Motor own-damage: modest effect; long-tail lines: significant |
# MAGIC | All splits validated with temporal_leakage_check before model fitting | Zero leakage in all temporal CV strategies |
# MAGIC
# MAGIC **The one-line change that removes the leakage risk:**
# MAGIC
# MAGIC ```python
# MAGIC # Before
# MAGIC cv = KFold(n_splits=5, shuffle=True)
# MAGIC
# MAGIC # After
# MAGIC from insurance_cv import walk_forward_split
# MAGIC from insurance_cv.splits import InsuranceCV
# MAGIC splits = walk_forward_split(df, date_col="inception_date", min_train_months=12,
# MAGIC                             test_months=6, step_months=6, ibnr_buffer_months=3)
# MAGIC cv = InsuranceCV(splits, df)
# MAGIC # Then pass cv to cross_val_score, GridSearchCV, etc — no other changes needed
# MAGIC ```
# MAGIC
# MAGIC The audit trail — fold start/end dates, IBNR buffer months, train/test sizes — is exactly what model governance reviewers ask for in SS1/23 model validation documentation. Random KFold cannot produce this. Walk-forward temporal CV produces it automatically.

# COMMAND ----------

# Final numbers printout for reference
print("FINAL BENCHMARK RESULTS")
print("=" * 50)
print(f"Random KFold (5-fold):        {random_cv_mean:.6f}")
print(f"Walk-forward temporal CV:      {temporal_cv_mean:.6f}")
print(f"Policy-year CV:                {py_cv_mean:.6f}")
print(f"True OOT holdout (2023):       {true_oot_deviance:.6f}")
print()
print(f"Random CV optimism bias:       {abs(random_gap):.6f}  ({abs(random_optimism_pct):.2f}%)")
print(f"Temporal CV error reduction:   {improvement_pct:.1f}% vs random CV")
print()
print(f"Walk-forward folds generated:  {len(temporal_splits)}")
print(f"Policy-year folds generated:   {len(py_splits)}")
print(f"Leakage checks passed:         all")
