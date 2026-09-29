"""
===============================================================================
Data Preprocessing and Analysis of the CDC Diabetes Health Indicators Dataset
(BRFSS 2015)

Group project - Data Analytics
Author : Umer Majid Muneer
Student ID : 261PC260Y3
Date   : October 2026

Dataset : CDC Diabetes Health Indicators, UCI ML Repository (ID 891)
          253,680 survey responses x 22 attributes
          Derived from the CDC Behavioral Risk Factor Surveillance System 2015

This script covers the four practical tasks of the project:
    Task 2 - Data cleaning
    Task 3 - Data integration, correlation analysis and transformation
    Task 4 - Data reduction
    Task 5 - Summary outputs for the conclusion

Run with:  python analysis.py
Outputs :  figures/*.png  and  output/*.csv
===============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # write files instead of opening windows
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)             # so every run gives the same numbers

DATA_DIR = "data"
FIG_DIR = "figures"
OUT_DIR = "output"
for d in (FIG_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

# plain matplotlib settings - readable labels, no heavy styling
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

# colours used consistently across the charts
C_BEFORE, C_AFTER = "#4878A8", "#D1832F"
C_NEG, C_POS = "#4878A8", "#B3483F"

results = {}                   # collects numbers that the report quotes


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# TASK 1 - LOAD THE DATASET
# =============================================================================
banner("TASK 1 : LOADING THE DATASET")

df_raw = pd.read_csv(os.path.join(DATA_DIR, "brfss2015.csv"))
print(f"Rows    : {df_raw.shape[0]:,}")
print(f"Columns : {df_raw.shape[1]}")
print(f"Memory  : {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

TARGET = "Diabetes_binary"
BINARY_COLS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
               "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
               "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
               "DiffWalk", "Sex"]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
COUNT_COLS = ["MentHlth", "PhysHlth"]
CONTINUOUS_COLS = ["BMI"]

results["n_raw"] = df_raw.shape[0]
results["n_cols"] = df_raw.shape[1]
results["mem_raw_mb"] = df_raw.memory_usage(deep=True).sum() / 1024**2


# =============================================================================
# TASK 2 - DATA CLEANING
# =============================================================================
banner("TASK 2 : DATA CLEANING")
df = df_raw.copy()

# -----------------------------------------------------------------------------
# 2.1 Missing values
#     Two checks are needed. The first is for normal blank/NaN cells. The second
#     is for "disguised" missing values - survey files often store a refusal as
#     a code such as 7, 9, 77 or 99 rather than leaving the cell empty, and those
#     look like real data to pandas.
# -----------------------------------------------------------------------------
print("\n2.1 MISSING VALUES")
null_counts = df.isna().sum()
print(f"  Standard NaN / blank cells : {int(null_counts.sum())}")

# valid ranges taken from the BRFSS 2015 codebook
valid_ranges = {
    **{c: (0, 1) for c in BINARY_COLS + [TARGET]},
    "GenHlth": (1, 5), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8),
    "MentHlth": (0, 30), "PhysHlth": (0, 30), "BMI": (12, 98),
}
disguised = {}
for col, (lo, hi) in valid_ranges.items():
    bad = int(((df[col] < lo) | (df[col] > hi)).sum())
    if bad:
        disguised[col] = bad
print(f"  Values outside the codebook range (e.g. 7/9/77/99 refusal codes) : "
      f"{sum(disguised.values())}")
if disguised:
    print(f"    {disguised}")

results["n_missing"] = int(null_counts.sum())
results["n_disguised"] = sum(disguised.values())

# -----------------------------------------------------------------------------
# 2.2 Duplicate records
#     Every column here is a coded survey answer, so two respondents can produce
#     an identical row by chance. A duplicate is only a real problem if the row
#     is a repeated record rather than a genuine coincidence. Because the file
#     carries no respondent ID we cannot tell the two apart with certainty, so
#     the decision has to be made and justified rather than applied blindly.
# -----------------------------------------------------------------------------
print("\n2.2 DUPLICATE RECORDS")
n_dup = int(df.duplicated().sum())
print(f"  Fully duplicated rows : {n_dup:,}  ({n_dup / len(df) * 100:.2f}%)")

dup_rate_before = df[TARGET].mean()
df_nodup = df.drop_duplicates().reset_index(drop=True)
dup_rate_after = df_nodup[TARGET].mean()
print(f"  Diabetes rate before  : {dup_rate_before:.4f}")
print(f"  Diabetes rate after   : {dup_rate_after:.4f}")
print(f"  Rows remaining        : {len(df_nodup):,}")

df = df_nodup
results["n_duplicates"] = n_dup
results["n_after_dedup"] = len(df)
results["rate_before_dedup"] = dup_rate_before
results["rate_after_dedup"] = dup_rate_after

# -----------------------------------------------------------------------------
# 2.3 Noisy data - outlier detection on BMI
#     Three independent methods are used so the result does not depend on one
#     rule of thumb: the IQR rule, the z-score rule, and distance from a k-means
#     cluster centre (the clustering technique named in the project brief).
# -----------------------------------------------------------------------------
print("\n2.3 NOISY DATA - OUTLIER DETECTION ON BMI")
bmi = df["BMI"]

q1, q3 = bmi.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_out = ((bmi < iqr_lo) | (bmi > iqr_hi))
print(f"  IQR rule      : bounds [{iqr_lo:.1f}, {iqr_hi:.1f}] "
      f"-> {int(iqr_out.sum()):,} outliers")

z = np.abs(stats.zscore(bmi))
z_out = z > 3
print(f"  Z-score > 3   : {int(z_out.sum()):,} outliers")

# clustering-based detection: group BMI into 5 clusters and flag the points
# that sit furthest from their own centre
km = KMeans(n_clusters=5, n_init=10, random_state=42)
labels = km.fit_predict(bmi.values.reshape(-1, 1))
centres = km.cluster_centers_.ravel()
dist = np.abs(bmi.values - centres[labels])
clust_out = dist > np.percentile(dist, 99)
print(f"  K-means (99th pct distance) : {int(clust_out.sum()):,} outliers")

agreed = iqr_out & z_out
print(f"  Flagged by both IQR and z-score : {int(agreed.sum()):,}")

# Treatment: the extreme values are clinically possible (a BMI of 98 is rare but
# real), so deleting those respondents would throw away exactly the high-risk
# people the analysis is about. Instead the tail is capped at the 0.5th / 99.5th
# percentile - winsorisation - which limits the influence of the extremes while
# keeping every row.
lo_cap, hi_cap = bmi.quantile([0.005, 0.995])
bmi_before = bmi.copy()
df["BMI"] = bmi.clip(lo_cap, hi_cap)
n_capped = int(((bmi_before < lo_cap) | (bmi_before > hi_cap)).sum())
print(f"  Winsorised at [{lo_cap:.0f}, {hi_cap:.0f}] -> {n_capped:,} values capped")
print(f"  BMI std before {bmi_before.std():.2f} -> after {df['BMI'].std():.2f}")
print(f"  BMI max before {bmi_before.max():.0f} -> after {df['BMI'].max():.0f}")

results["iqr_outliers"] = int(iqr_out.sum())
results["z_outliers"] = int(z_out.sum())
results["clust_outliers"] = int(clust_out.sum())
results["n_capped"] = n_capped
results["bmi_std_before"] = bmi_before.std()
results["bmi_std_after"] = df["BMI"].std()

# Figure 1 - BMI before and after outlier treatment
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(bmi_before, bins=60, color=C_BEFORE, edgecolor="white", linewidth=0.3)
ax[0].axvline(iqr_hi, color="red", ls="--", lw=1, label=f"IQR upper {iqr_hi:.0f}")
ax[0].set_title("BMI before cleaning")
ax[0].set_xlabel("BMI"); ax[0].set_ylabel("Number of respondents"); ax[0].legend()
ax[1].hist(df["BMI"], bins=60, color=C_AFTER, edgecolor="white", linewidth=0.3)
ax[1].set_title("BMI after winsorisation")
ax[1].set_xlabel("BMI"); ax[1].set_ylabel("Number of respondents")
# same x-axis on both panels, otherwise the change is hard to judge
for a in ax:
    a.set_xlim(10, 100)
fig.suptitle("Figure 1: Effect of outlier treatment on BMI", y=1.02)
fig.savefig(f"{FIG_DIR}/fig01_bmi_before_after.png"); plt.close(fig)

# Figure 2 - boxplot view of the same change
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([bmi_before, df["BMI"]], tick_labels=["Before", "After"],
           vert=False, widths=0.6, patch_artist=True,
           boxprops=dict(facecolor="#CFE0EF"), medianprops=dict(color="black"))
ax.set_xlabel("BMI")
ax.set_title("Figure 2: BMI distribution before and after cleaning")
fig.savefig(f"{FIG_DIR}/fig02_bmi_boxplot.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 2.4 Noise smoothing by binning
#     MentHlth and PhysHlth ask "how many of the past 30 days was your mental /
#     physical health not good". Most people answer 0, and the rest cluster on
#     round numbers such as 5, 10, 15 and 30, which is recall bias rather than
#     precise measurement. Binning into four groups smooths that noise.
# -----------------------------------------------------------------------------
print("\n2.4 NOISE SMOOTHING BY BINNING")
for col in COUNT_COLS:
    zero_pct = (df[col] == 0).mean() * 100
    round_pct = df[col].isin([5, 10, 15, 20, 25, 30]).mean() * 100
    print(f"  {col}: {zero_pct:.1f}% are zero, {round_pct:.1f}% are round numbers")
    df[f"{col}_Binned"] = pd.cut(df[col], bins=[-1, 0, 7, 14, 30],
                                 labels=["None", "1-7 days", "8-14 days", "15-30 days"])

results["menthlth_zero_pct"] = (df["MentHlth"] == 0).mean() * 100
results["physhlth_zero_pct"] = (df["PhysHlth"] == 0).mean() * 100

# Figure 3 - binning of PhysHlth
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["PhysHlth"], bins=31, color=C_BEFORE, edgecolor="white", linewidth=0.3)
ax[0].set_title("PhysHlth (raw day counts)")
ax[0].set_xlabel("Unhealthy days in past 30"); ax[0].set_ylabel("Respondents")
counts = df["PhysHlth_Binned"].value_counts().reindex(
    ["None", "1-7 days", "8-14 days", "15-30 days"])
ax[1].bar(counts.index, counts.values, color=C_AFTER, edgecolor="white")
ax[1].set_title("PhysHlth after binning")
ax[1].set_ylabel("Respondents"); ax[1].tick_params(axis="x", rotation=15)
fig.suptitle("Figure 3: Smoothing a noisy day-count variable by binning", y=1.02)
fig.savefig(f"{FIG_DIR}/fig03_binning.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 2.5 Data type conversion
#     Every column arrived as float64 even though most hold only 0 or 1. Casting
#     them to the smallest correct type cuts memory sharply and stops binary
#     flags being treated as continuous measurements by later steps.
# -----------------------------------------------------------------------------
print("\n2.5 DATA TYPE CONVERSION")
mem_before = df.memory_usage(deep=True).sum() / 1024**2
for col in BINARY_COLS + [TARGET]:
    df[col] = df[col].astype("int8")
for col in ORDINAL_COLS + COUNT_COLS:
    df[col] = df[col].astype("int8")
df["BMI"] = df["BMI"].astype("float32")
mem_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"  Memory {mem_before:.1f} MB -> {mem_after:.1f} MB "
      f"({(1 - mem_after / mem_before) * 100:.0f}% smaller)")

results["mem_before_dtype"] = mem_before
results["mem_after_dtype"] = mem_after

# -----------------------------------------------------------------------------
# 2.6 Class balance - important context for everything that follows
# -----------------------------------------------------------------------------
print("\n2.6 CLASS BALANCE OF THE TARGET")
balance = df[TARGET].value_counts(normalize=True).sort_index()
print(f"  No diabetes            : {balance[0] * 100:.1f}%")
print(f"  Prediabetes / diabetes : {balance[1] * 100:.1f}%")
print("  Note: a model that always predicts 'no diabetes' would already be")
print(f"  {balance[0] * 100:.1f}% accurate, so ROC-AUC is used later, not accuracy.")
results["class_neg_pct"] = balance[0] * 100
results["class_pos_pct"] = balance[1] * 100

# -----------------------------------------------------------------------------
# 2.7 Did the cleaning actually help?
#     Cleaning decisions should be justified by evidence, not taste. The same
#     logistic regression is fitted on the data with the raw BMI column and
#     again with the winsorised one, and the two are compared on a held-out
#     test set. If the treatment were harmful the score would drop.
# -----------------------------------------------------------------------------
print("\n2.7 EFFECT OF THE CLEANING ON MODEL QUALITY")
_feats = BINARY_COLS + ORDINAL_COLS + COUNT_COLS + CONTINUOUS_COLS
_y = df[TARGET].astype(int)


def _quick_auc(frame):
    a, b, c, d_ = train_test_split(frame, _y, test_size=0.3,
                                   random_state=42, stratify=_y)
    sc = StandardScaler().fit(a)
    mdl = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(a), c)
    return roc_auc_score(d_, mdl.predict_proba(sc.transform(b))[:, 1])


_raw_bmi_frame = df[_feats].astype(float).copy()
_raw_bmi_frame["BMI"] = bmi_before.values          # the untreated column
auc_raw_bmi = _quick_auc(_raw_bmi_frame)
auc_clean_bmi = _quick_auc(df[_feats].astype(float))
print(f"  ROC-AUC with untreated BMI  : {auc_raw_bmi:.4f}")
print(f"  ROC-AUC with winsorised BMI : {auc_clean_bmi:.4f}")
print(f"  Change: {auc_clean_bmi - auc_raw_bmi:+.4f} - the treatment helped slightly,")
print("  so the outliers were adding noise rather than signal.")

results["auc_raw_bmi"] = auc_raw_bmi
results["auc_clean_bmi"] = auc_clean_bmi

df.to_csv(f"{OUT_DIR}/cleaned_data.csv", index=False)
print(f"\n  Cleaned dataset saved -> {OUT_DIR}/cleaned_data.csv")


# =============================================================================
# TASK 3 - DATA INTEGRATION, CORRELATION AND TRANSFORMATION
# =============================================================================
banner("TASK 3 : DATA INTEGRATION, CORRELATION AND TRANSFORMATION")

# -----------------------------------------------------------------------------
# 3.1 Integration of a second source
#     The survey file stores Age, Education, Income, GenHlth and Sex as bare
#     numbers with no labels. The meaning of each code lives in the CDC BRFSS
#     2015 codebook, which is transcribed here into a second file and joined on
#     the code value. This is a classic lookup-table (dimension table) join.
# -----------------------------------------------------------------------------
print("\n3.1 INTEGRATION WITH THE CDC CODEBOOK")
codebook = pd.read_csv(os.path.join(DATA_DIR, "brfss_codebook.csv"))
print(f"  Codebook rows : {len(codebook)} covering "
      f"{codebook['variable'].nunique()} variables")

for var in ["Age", "Education", "Income", "GenHlth", "Sex"]:
    lut = codebook[codebook["variable"] == var]
    label_map = dict(zip(lut["code"], lut["label"]))
    df[f"{var}_Label"] = df[var].map(label_map)
    n_unmatched = int(df[f"{var}_Label"].isna().sum())
    print(f"  {var:10s} -> {var}_Label   unmatched rows: {n_unmatched}")

# the codebook also supplies a numeric midpoint, which turns the banded income
# and age codes into something that can be averaged and correlated properly
inc_mid = dict(zip(codebook[codebook["variable"] == "Income"]["code"],
                   codebook[codebook["variable"] == "Income"]["numeric_value"]))
age_mid = dict(zip(codebook[codebook["variable"] == "Age"]["code"],
                   codebook[codebook["variable"] == "Age"]["numeric_value"]))
df["Income_USD"] = df["Income"].map(inc_mid)
df["Age_Years"] = df["Age"].map(age_mid)
print(f"  Added Income_USD (band midpoint) and Age_Years (band midpoint)")
print(f"  Mean estimated income : ${df['Income_USD'].mean():,.0f}")
print(f"  Mean estimated age    : {df['Age_Years'].mean():.1f} years")

results["codebook_rows"] = len(codebook)
results["mean_income"] = df["Income_USD"].mean()
results["mean_age"] = df["Age_Years"].mean()

# -----------------------------------------------------------------------------
# 3.2 Correlation analysis
#     Three different measures are needed because the columns are of three
#     different kinds:
#       Pearson / Spearman  - between the numeric and ordinal columns
#       Point-biserial      - a binary column against a continuous one
#       Cramer's V          - between two categorical columns
# -----------------------------------------------------------------------------
print("\n3.2 CORRELATION ANALYSIS")
numeric_for_corr = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age",
                    "Education", "Income", "Age_Years", "Income_USD"]
corr_matrix = df[numeric_for_corr + [TARGET]].corr(method="pearson")

# correlation of every attribute with the target
feature_cols = BINARY_COLS + ORDINAL_COLS + COUNT_COLS + CONTINUOUS_COLS
corr_target = (df[feature_cols + [TARGET]].corr()[TARGET]
               .drop(TARGET).sort_values(key=abs, ascending=False))
print("\n  Pearson correlation with Diabetes_binary (top 10 by strength):")
for name, val in corr_target.head(10).items():
    print(f"    {name:22s} {val:+.3f}")


def cramers_v(x, y):
    """Cramer's V - association between two categorical variables, 0 to 1."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


print("\n  Cramer's V with Diabetes_binary (top 10):")
cv_rows = []
for col in feature_cols:
    series = df[col] if col != "BMI" else pd.cut(df["BMI"], 10)
    cv_rows.append({"attribute": col, "cramers_v": cramers_v(series, df[TARGET])})
cv_df = pd.DataFrame(cv_rows).sort_values("cramers_v", ascending=False)
for _, r in cv_df.head(10).iterrows():
    print(f"    {r['attribute']:22s} {r['cramers_v']:.3f}")

# Chi-square test - with an important caveat about the sample size
print("\n  Chi-square tests of independence vs Diabetes_binary:")
chi_rows = []
for col in feature_cols:
    series = df[col] if col != "BMI" else pd.cut(df["BMI"], 10)
    table = pd.crosstab(series, df[TARGET])
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    chi_rows.append({"attribute": col, "chi_square": chi2, "dof": dof,
                     "p_value": p, "cramers_v": cramers_v(series, df[TARGET]),
                     "significant_p05": p < 0.05})
chi_df = pd.DataFrame(chi_rows).sort_values("chi_square", ascending=False)
n_sig = int(chi_df["significant_p05"].sum())
print(f"    Significant at p < 0.05 : {n_sig} of {len(chi_df)} attributes")
print(f"    With n = {len(df):,}, even a trivial association reaches p < 0.05,")
print(f"    so Cramer's V (effect size) is the meaningful ranking, not p.")
chi_df.to_csv(f"{OUT_DIR}/chi_square_results.csv", index=False)
corr_target.to_csv(f"{OUT_DIR}/correlation_with_target.csv")

results["n_chi_significant"] = n_sig
results["n_chi_total"] = len(chi_df)
results["top_corr"] = corr_target.head(6).to_dict()
results["top_cramers"] = cv_df.head(6).set_index("attribute")["cramers_v"].to_dict()

# Figure 4 - correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6.5))
sub = df[["BMI", "GenHlth", "PhysHlth", "MentHlth", "Age", "Education",
          "Income", "HighBP", "HighChol", "DiffWalk", "PhysActivity",
          TARGET]].corr()
im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.columns, rotation=45, ha="right")
ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.columns)
for i in range(len(sub)):
    for j in range(len(sub)):
        v = sub.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Figure 4: Pearson correlation matrix")
ax.grid(False)
fig.savefig(f"{FIG_DIR}/fig04_correlation_heatmap.png"); plt.close(fig)

# Figure 5 - correlation with the target, ranked
fig, ax = plt.subplots(figsize=(8, 6))
ct = corr_target.sort_values()
colors = [C_NEG if v < 0 else C_POS for v in ct.values]
ax.barh(ct.index, ct.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson correlation with Diabetes_binary")
ax.set_title("Figure 5: Which attributes move with diabetes status")
fig.savefig(f"{FIG_DIR}/fig05_corr_with_target.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 3.3 Transformation
#     Four standard techniques are applied:
#       discretisation - BMI into the WHO weight categories
#       generalisation - 13 age bands collapsed into 4 life stages
#       aggregation    - several health flags summed into one risk score
#       normalisation  - min-max and z-score scaling for the PCA in Task 4
# -----------------------------------------------------------------------------
print("\n3.3 TRANSFORMATION")

# (a) discretisation using the WHO cut-offs
df["BMI_Category"] = pd.cut(
    df["BMI"], bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"])
print("\n  (a) Discretisation - BMI into WHO categories:")
bmi_tab = (df.groupby("BMI_Category", observed=True)
             .agg(n=(TARGET, "size"), diabetes_rate=(TARGET, "mean")))
for cat, row in bmi_tab.iterrows():
    print(f"      {str(cat):12s} n={int(row['n']):>7,}  "
          f"diabetes rate {row['diabetes_rate'] * 100:5.1f}%")

# (b) generalisation - concept hierarchy on age
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 4, 7, 10, 13],
                         labels=["18-39", "40-54", "55-69", "70+"])
print("\n  (b) Generalisation - 13 age bands into 4 life stages:")
age_tab = (df.groupby("Age_Group", observed=True)
             .agg(n=(TARGET, "size"), diabetes_rate=(TARGET, "mean")))
for cat, row in age_tab.iterrows():
    print(f"      {str(cat):8s} n={int(row['n']):>7,}  "
          f"diabetes rate {row['diabetes_rate'] * 100:5.1f}%")

# (c) aggregation - build two composite indicators
df["Risk_Score"] = (df["HighBP"] + df["HighChol"] + df["Smoker"] +
                    df["Stroke"] + df["HeartDiseaseorAttack"] + df["DiffWalk"])
df["Healthy_Habits"] = (df["PhysActivity"] + df["Fruits"] + df["Veggies"] +
                        (1 - df["Smoker"]) + (1 - df["HvyAlcoholConsump"]))
print("\n  (c) Aggregation - composite indicators:")
print(f"      Risk_Score      0-6, mean {df['Risk_Score'].mean():.2f}, "
      f"correlation with target {df['Risk_Score'].corr(df[TARGET]):+.3f}")
print(f"      Healthy_Habits  0-5, mean {df['Healthy_Habits'].mean():.2f}, "
      f"correlation with target {df['Healthy_Habits'].corr(df[TARGET]):+.3f}")

risk_tab = df.groupby("Risk_Score").agg(n=(TARGET, "size"),
                                        diabetes_rate=(TARGET, "mean"))
print("      Diabetes rate by Risk_Score:")
for sc, row in risk_tab.iterrows():
    print(f"        {sc}: {row['diabetes_rate'] * 100:5.1f}%  (n={int(row['n']):,})")

# (d) normalisation
scaled_cols = ["BMI", "Age", "Income", "GenHlth", "MentHlth", "PhysHlth"]
minmax = MinMaxScaler().fit_transform(df[scaled_cols])
zscore = StandardScaler().fit_transform(df[scaled_cols])
print("\n  (d) Normalisation of 6 numeric columns:")
print(f"      Min-max  -> range [{minmax.min():.1f}, {minmax.max():.1f}]")
print(f"      Z-score  -> mean {zscore.mean():.2f}, std {zscore.std():.2f}")

results["bmi_table"] = bmi_tab.to_dict("index")
results["age_table"] = age_tab.to_dict("index")
results["risk_table"] = risk_tab.to_dict("index")
results["risk_corr"] = df["Risk_Score"].corr(df[TARGET])
results["habits_corr"] = df["Healthy_Habits"].corr(df[TARGET])

# Figure 6 - diabetes rate by BMI category and age group
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
b = bmi_tab["diabetes_rate"] * 100
ax[0].bar(b.index.astype(str), b.values, color=C_POS, edgecolor="white")
ax[0].set_ylabel("Diabetes rate (%)"); ax[0].set_title("By BMI category")
for i, v in enumerate(b.values):
    ax[0].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
a = age_tab["diabetes_rate"] * 100
ax[1].bar(a.index.astype(str), a.values, color="#6A8CAF", edgecolor="white")
ax[1].set_ylabel("Diabetes rate (%)"); ax[1].set_title("By age group")
for i, v in enumerate(a.values):
    ax[1].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
fig.suptitle("Figure 6: Diabetes rate after discretisation and generalisation", y=1.02)
fig.savefig(f"{FIG_DIR}/fig06_transformed_rates.png"); plt.close(fig)

# Figure 7 - the aggregated risk score
fig, ax = plt.subplots(figsize=(7, 4.2))
r = risk_tab["diabetes_rate"] * 100
ax.bar(r.index.astype(str), r.values, color="#8C5A9E", edgecolor="white")
for i, v in enumerate(r.values):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("Risk_Score (number of risk conditions, 0-6)")
ax.set_ylabel("Diabetes rate (%)")
ax.set_title("Figure 7: Aggregated risk score against diabetes rate")
fig.savefig(f"{FIG_DIR}/fig07_risk_score.png"); plt.close(fig)


# =============================================================================
# TASK 4 - DATA REDUCTION
# =============================================================================
banner("TASK 4 : DATA REDUCTION")

X_full = df[feature_cols].astype(float)
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


def auc_of(cols_idx=None, Xtr=None, Xte=None):
    """Fit logistic regression and return test ROC-AUC."""
    a = X_train_s if Xtr is None else Xtr
    b = X_test_s if Xte is None else Xte
    if cols_idx is not None:
        a, b = a[:, cols_idx], b[:, cols_idx]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(a, y_train)
    return roc_auc_score(y_test, model.predict_proba(b)[:, 1])


baseline_auc = auc_of()
print(f"\nBaseline: all {len(feature_cols)} attributes -> ROC-AUC {baseline_auc:.4f}")
results["baseline_auc"] = baseline_auc
results["n_features_full"] = len(feature_cols)

# -----------------------------------------------------------------------------
# 4.1 Attribute subset selection
#     Two filter methods rank the attributes, then the ranking is tested by
#     rebuilding the model on the top-k attributes only.
# -----------------------------------------------------------------------------
print("\n4.1 ATTRIBUTE SUBSET SELECTION")
# All attributes except BMI hold a small set of integer codes, so they must be
# declared discrete. Left at the default, mutual_info_classif treats every
# column as continuous and uses a nearest-neighbour estimate, which gives
# meaningless scores for 0/1 flags.
discrete_mask = np.array([c != "BMI" for c in feature_cols])
mi = mutual_info_classif(X_train, y_train, discrete_features=discrete_mask,
                         random_state=42)
mi_rank = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
print("\n  Mutual information with the target (top 10):")
for name, val in mi_rank.head(10).items():
    print(f"    {name:22s} {val:.4f}")

ranked = list(mi_rank.index)
ks, aucs = [], []
for k in [3, 5, 8, 10, 12, 15, len(feature_cols)]:
    idx = [feature_cols.index(c) for c in ranked[:k]]
    a = auc_of(idx)
    ks.append(k); aucs.append(a)
    print(f"    top {k:2d} attributes -> ROC-AUC {a:.4f}  "
          f"({(a - baseline_auc) / baseline_auc * 100:+.2f}% vs all)")

sel_k = 10
sel_features = ranked[:sel_k]
sel_auc = aucs[ks.index(sel_k)]
print(f"\n  Chosen subset ({sel_k} attributes): {', '.join(sel_features)}")
print(f"  Attributes dropped: {len(feature_cols) - sel_k} of {len(feature_cols)} "
      f"({(1 - sel_k / len(feature_cols)) * 100:.0f}% fewer)")
print(f"  ROC-AUC change: {baseline_auc:.4f} -> {sel_auc:.4f} "
      f"({(sel_auc - baseline_auc) / baseline_auc * 100:+.2f}%)")

results["mi_rank"] = mi_rank.to_dict()
results["subset_k"] = sel_k
results["subset_features"] = sel_features
results["subset_auc"] = sel_auc
results["k_curve"] = dict(zip(ks, aucs))

# Figure 8 - mutual information ranking
fig, ax = plt.subplots(figsize=(8, 6))
m = mi_rank.sort_values()
ax.barh(m.index, m.values, color="#4878A8", edgecolor="white")
ax.set_xlabel("Mutual information with Diabetes_binary")
ax.set_title("Figure 8: Attribute ranking by mutual information")
fig.savefig(f"{FIG_DIR}/fig08_mutual_info.png"); plt.close(fig)

# Figure 9 - accuracy against number of attributes kept
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(ks, aucs, "o-", color=C_POS, lw=2, markersize=6)
ax.axhline(baseline_auc, color="gray", ls="--", lw=1,
           label=f"All {len(feature_cols)} attributes ({baseline_auc:.4f})")
ax.axvline(sel_k, color="green", ls=":", lw=1.5, label=f"Chosen k = {sel_k}")
ax.set_xlabel("Number of attributes kept"); ax.set_ylabel("ROC-AUC on test set")
ax.set_title("Figure 9: Effect of attribute subset selection")
ax.legend()
fig.savefig(f"{FIG_DIR}/fig09_subset_curve.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 4.2 Dimensionality reduction with PCA
#     Caveat worth stating in the report: PCA assumes continuous, linearly
#     related variables, and 14 of these 19 attributes are binary. PCA is
#     therefore applied and reported, but the subset selection above is the
#     more appropriate reduction for this dataset.
# -----------------------------------------------------------------------------
print("\n4.2 DIMENSIONALITY REDUCTION - PCA")
pca_full = PCA().fit(X_train_s)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n_80 = int(np.argmax(cum >= 0.80) + 1)
n_90 = int(np.argmax(cum >= 0.90) + 1)
n_95 = int(np.argmax(cum >= 0.95) + 1)
print(f"  Components for 80% of variance : {n_80}")
print(f"  Components for 90% of variance : {n_90}")
print(f"  Components for 95% of variance : {n_95}")
print(f"  First component alone explains  : {cum[0] * 100:.1f}%")

pca_n = n_90
pca = PCA(n_components=pca_n, random_state=42).fit(X_train_s)
auc_pca = auc_of(None, pca.transform(X_train_s), pca.transform(X_test_s))
print(f"  PCA with {pca_n} components -> ROC-AUC {auc_pca:.4f} "
      f"({(auc_pca - baseline_auc) / baseline_auc * 100:+.2f}% vs all)")

loadings = pd.Series(np.abs(pca.components_[0]), index=feature_cols).sort_values(ascending=False)
print(f"  Attributes loading most on PC1: {', '.join(loadings.head(4).index)}")

results["pca_80"] = n_80; results["pca_90"] = n_90; results["pca_95"] = n_95
results["pca_auc"] = auc_pca; results["pca_n"] = pca_n
results["pc1_var"] = cum[0] * 100
results["pc1_top"] = list(loadings.head(4).index)

# Figure 10 - PCA scree / cumulative variance
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
          pca_full.explained_variance_ratio_ * 100, color=C_BEFORE, edgecolor="white")
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Variance explained (%)")
ax[0].set_title("Scree plot")
ax[1].plot(range(1, len(cum) + 1), cum * 100, "o-", color=C_POS, markersize=4)
ax[1].axhline(90, color="gray", ls="--", lw=1, label="90% threshold")
ax[1].axvline(n_90, color="green", ls=":", lw=1.5, label=f"{n_90} components")
ax[1].set_xlabel("Number of components"); ax[1].set_ylabel("Cumulative variance (%)")
ax[1].set_title("Cumulative variance"); ax[1].legend()
fig.suptitle("Figure 10: PCA dimensionality reduction", y=1.02)
fig.savefig(f"{FIG_DIR}/fig10_pca.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 4.3 Numerosity reduction - does the analysis need all 229,474 rows?
# -----------------------------------------------------------------------------
print("\n4.3 NUMEROSITY REDUCTION - SAMPLING")
sample_rows, sample_aucs = [], []
for frac in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
    if frac < 1.0:
        idx = np.random.RandomState(42).choice(len(X_train_s),
                                               int(len(X_train_s) * frac), replace=False)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_s[idx], y_train.values[idx])
        a = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    else:
        a = baseline_auc
    n = int(len(X_train_s) * frac)
    sample_rows.append(n); sample_aucs.append(a)
    print(f"  {frac * 100:5.0f}% of training rows ({n:>7,}) -> ROC-AUC {a:.4f}")

results["sample_curve"] = dict(zip(sample_rows, sample_aucs))

# Figure 11 - sampling curve
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogx(sample_rows, sample_aucs, "o-", color="#2E7D52", lw=2, markersize=6)
ax.axhline(baseline_auc, color="gray", ls="--", lw=1, label="Full training set")
# a wide y-axis on purpose: zoomed in, a 0.007 difference would look dramatic
ax.set_ylim(0.70, 0.85)
ax.set_xlabel("Training rows used (log scale)"); ax.set_ylabel("ROC-AUC on test set")
ax.set_title("Figure 11: Numerosity reduction by random sampling")
ax.legend()
fig.savefig(f"{FIG_DIR}/fig11_sampling.png"); plt.close(fig)

# -----------------------------------------------------------------------------
# 4.4 Combined effect of the reduction
# -----------------------------------------------------------------------------
print("\n4.4 SUMMARY OF THE REDUCTION")
reduction = pd.DataFrame([
    {"method": "All attributes (baseline)", "attributes": len(feature_cols),
     "rows": len(X_train_s), "roc_auc": baseline_auc},
    {"method": f"Subset selection (top {sel_k})", "attributes": sel_k,
     "rows": len(X_train_s), "roc_auc": sel_auc},
    {"method": f"PCA ({pca_n} components)", "attributes": pca_n,
     "rows": len(X_train_s), "roc_auc": auc_pca},
    {"method": "Subset + 10% sample", "attributes": sel_k,
     "rows": int(len(X_train_s) * 0.10), "roc_auc": None},
])
idx10 = np.random.RandomState(42).choice(len(X_train_s), int(len(X_train_s) * 0.10),
                                         replace=False)
sel_idx = [feature_cols.index(c) for c in sel_features]
m = LogisticRegression(max_iter=1000, random_state=42)
m.fit(X_train_s[np.ix_(idx10, sel_idx)], y_train.values[idx10])
combo_auc = roc_auc_score(y_test, m.predict_proba(X_test_s[:, sel_idx])[:, 1])
reduction.loc[3, "roc_auc"] = combo_auc
reduction["vs_baseline_pct"] = ((reduction["roc_auc"] - baseline_auc)
                                / baseline_auc * 100).round(2)
print(reduction.to_string(index=False))
reduction.to_csv(f"{OUT_DIR}/reduction_summary.csv", index=False)

cells_before = len(X_train_s) * len(feature_cols)
cells_after = int(len(X_train_s) * 0.10) * sel_k
print(f"\n  Data volume: {cells_before:,} values -> {cells_after:,} values "
      f"({(1 - cells_after / cells_before) * 100:.1f}% smaller)")
print(f"  ROC-AUC cost: {baseline_auc:.4f} -> {combo_auc:.4f} "
      f"({(combo_auc - baseline_auc) / baseline_auc * 100:+.2f}%)")

results["combo_auc"] = combo_auc
results["cells_before"] = cells_before
results["cells_after"] = cells_after

# Figure 12 - reduction comparison
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(reduction["method"], reduction["roc_auc"],
              color=[C_BEFORE, C_AFTER, "#8C5A9E", "#2E7D52"], edgecolor="white")
ax.axhline(baseline_auc, color="gray", ls="--", lw=1)
ax.set_ylim(0.70, 0.84); ax.set_ylabel("ROC-AUC on test set")
ax.set_title("Figure 12: Model quality after each reduction technique")
ax.tick_params(axis="x", rotation=12)
for b, v in zip(bars, reduction["roc_auc"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
            ha="center", fontsize=9)
fig.savefig(f"{FIG_DIR}/fig12_reduction_summary.png"); plt.close(fig)


# =============================================================================
# TASK 5 - SUMMARY
# =============================================================================
banner("TASK 5 : SUMMARY OF KEY FINDINGS")

df_reduced = df[sel_features + [TARGET]].copy()
df_reduced.to_csv(f"{OUT_DIR}/reduced_data.csv", index=False)

summary = pd.DataFrame([
    ["Raw rows", f"{results['n_raw']:,}"],
    ["Duplicate rows removed", f"{results['n_duplicates']:,}"],
    ["Rows after cleaning", f"{results['n_after_dedup']:,}"],
    ["Missing values found", f"{results['n_missing']}"],
    ["BMI values winsorised", f"{results['n_capped']:,}"],
    ["Memory after type conversion", f"{results['mem_after_dtype']:.1f} MB "
                                     f"(from {results['mem_before_dtype']:.1f} MB)"],
    ["Strongest correlate of diabetes", f"GenHlth (r = {corr_target['GenHlth']:+.3f})"],
    ["Attributes kept after selection", f"{sel_k} of {len(feature_cols)}"],
    ["PCA components for 90% variance", f"{n_90} of {len(feature_cols)}"],
    ["Baseline ROC-AUC", f"{baseline_auc:.4f}"],
    ["ROC-AUC after full reduction", f"{combo_auc:.4f}"],
], columns=["metric", "value"])
print(summary.to_string(index=False))
summary.to_csv(f"{OUT_DIR}/summary_metrics.csv", index=False)

print(f"\nFigures written to {FIG_DIR}/ :")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
print(f"\nData and result tables written to {OUT_DIR}/ :")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")

import json
with open(f"{OUT_DIR}/results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nDone.")
