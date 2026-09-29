"""
Diabetes Health Indicators Dashboard (CDC BRFSS 2015)
Run:  streamlit run dashboard.py
Reads the cleaned data and result files produced by analysis.py.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
OUT = BASE / "output"
DATA = BASE / "data"
TARGET = "Diabetes_binary"

# Palette: categorical slots 1-2 plus a one-hue sequential ramp.
C_NO = "#2a78d6"      # No diabetes (blue)
C_YES = "#eb6834"     # Diabetes (orange)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS_COLORS = {"No diabetes": C_NO, "Prediabetes or diabetes": C_YES}

st.set_page_config(page_title="Diabetes Indicators Dashboard",
                   page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(OUT / "cleaned_data.csv")
    # pandas reads the "None" bin label as missing; restore it.
    for c in ["MentHlth_Binned", "PhysHlth_Binned"]:
        df[c] = df[c].fillna("None")
    cb = pd.read_csv(DATA / "brfss_codebook.csv")

    # Integration: join readable labels from the codebook.
    for var in ["Age", "Education", "Income", "GenHlth", "Sex", TARGET]:
        lab = cb[cb["variable"] == var][["code", "label"]]
        mapping = dict(zip(lab["code"], lab["label"]))
        df[f"{var}_Label"] = df[var].map(mapping)

    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                                labels=["Underweight", "Normal", "Overweight", "Obese"])
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 4, 7, 10, 13],
                             labels=["18-39", "40-54", "55-69", "70+"])
    df["Risk_Score"] = (df["HighBP"] + df["HighChol"] + df["Smoker"] + df["Stroke"]
                        + df["HeartDiseaseorAttack"] + df["DiffWalk"])
    df["Healthy_Habits"] = (df["PhysActivity"] + df["Fruits"] + df["Veggies"]
                            + (1 - df["Smoker"]) + (1 - df["HvyAlcoholConsump"]))
    return df, cb


@st.cache_data
def load_results():
    res = json.loads((OUT / "results.json").read_text())
    chi = pd.read_csv(OUT / "chi_square_results.csv")
    corr = pd.read_csv(OUT / "correlation_with_target.csv", index_col=0)
    red = pd.read_csv(OUT / "reduction_summary.csv")
    return res, chi, corr, red


df, codebook = load_data()
res, chi, corr, red = load_results()

ORDER = {
    "Age_Label": codebook[codebook.variable == "Age"].label.tolist(),
    "Education_Label": codebook[codebook.variable == "Education"].label.tolist(),
    "Income_Label": codebook[codebook.variable == "Income"].label.tolist(),
    "GenHlth_Label": codebook[codebook.variable == "GenHlth"].label.tolist(),
    "BMI_Category": ["Underweight", "Normal", "Overweight", "Obese"],
    "Age_Group": ["18-39", "40-54", "55-69", "70+"],
    "MentHlth_Binned": ["None", "1-7 days", "8-14 days", "15-30 days"],
    "PhysHlth_Binned": ["None", "1-7 days", "8-14 days", "15-30 days"],
}

BINARY_FACTORS = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check (5 yrs)", "Smoker": "Smoker (100+ cigs)",
    "Stroke": "Had a stroke", "HeartDiseaseorAttack": "Heart disease / attack",
    "PhysActivity": "Physically active", "Fruits": "Eats fruit daily",
    "Veggies": "Eats vegetables daily", "HvyAlcoholConsump": "Heavy drinker",
    "AnyHealthcare": "Has healthcare cover", "NoDocbcCost": "Skipped doctor (cost)",
    "DiffWalk": "Difficulty walking",
}

GROUP_DIMS = {
    "BMI category": "BMI_Category", "Age group": "Age_Group",
    "Age (5-yr band)": "Age_Label", "General health": "GenHlth_Label",
    "Income": "Income_Label", "Education": "Education_Label",
    "Sex": "Sex_Label", "Risk score (0-6)": "Risk_Score",
    "Healthy habits (0-5)": "Healthy_Habits",
    "Poor mental health days": "MentHlth_Binned",
    "Poor physical health days": "PhysHlth_Binned",
}


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend_title_text="",
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


def rate_by(data, col):
    g = (data.groupby(col, observed=True)[TARGET]
         .agg(n="size", rate="mean").reset_index())
    g["rate_pct"] = g["rate"] * 100
    if col in ORDER:
        g[col] = pd.Categorical(g[col].astype(str), ORDER[col], ordered=True)
        g = g.sort_values(col)
    return g


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🩺 Filters")
st.sidebar.caption("Filters apply to the Overview, Risk Factors and Explorer tabs.")

sex_sel = st.sidebar.multiselect("Sex", ["Female", "Male"], ["Female", "Male"])
age_sel = st.sidebar.multiselect("Age group", ORDER["Age_Group"], ORDER["Age_Group"])
bmi_sel = st.sidebar.multiselect("BMI category", ORDER["BMI_Category"], ORDER["BMI_Category"])
gh_sel = st.sidebar.multiselect("General health", ORDER["GenHlth_Label"], ORDER["GenHlth_Label"])
inc_range = st.sidebar.select_slider("Income band", options=list(range(1, 9)), value=(1, 8),
                                     format_func=lambda c: ORDER["Income_Label"][c - 1])
bmi_range = st.sidebar.slider("BMI range", float(df.BMI.min()), float(df.BMI.max()),
                              (float(df.BMI.min()), float(df.BMI.max())), step=1.0)

f = df[df.Sex_Label.isin(sex_sel)
       & df.Age_Group.astype(str).isin(age_sel)
       & df.BMI_Category.astype(str).isin(bmi_sel)
       & df.GenHlth_Label.isin(gh_sel)
       & df.Income.between(*inc_range)
       & df.BMI.between(*bmi_range)]

st.sidebar.markdown("---")
st.sidebar.metric("Respondents selected", f"{len(f):,}", f"{len(f) / len(df):.1%} of cleaned data",
                  delta_color="off")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Diabetes Health Indicators Dashboard")
st.caption("CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 · UCI ML Repository ID 891 · "
           f"{len(df):,} respondents after cleaning")

if f.empty:
    st.warning("No respondents match the current filters. Widen the selection in the sidebar.")
    st.stop()

tabs = st.tabs(["📊 Overview", "⚠️ Risk Factors", "🔎 Explorer",
                "🧹 Data Cleaning", "🧬 Feature Selection & Reduction", "📋 Data"])

# ==========================================================================
# 1. Overview
# ==========================================================================
with tabs[0]:
    overall = df[TARGET].mean()
    rate = f[TARGET].mean()
    k = st.columns(5)
    k[0].metric("Respondents", f"{len(f):,}")
    k[1].metric("Diabetes rate", f"{rate:.1%}",
                f"{(rate - overall) * 100:+.1f} pts vs all", delta_color="inverse")
    k[2].metric("Average BMI", f"{f.BMI.mean():.1f}")
    k[3].metric("High blood pressure", f"{f.HighBP.mean():.1%}")
    k[4].metric("Physically active", f"{f.PhysActivity.mean():.1%}")

    c1, c2 = st.columns([1, 2])
    with c1:
        cls = f[f"{TARGET}_Label"].value_counts().reset_index()
        cls.columns = ["Status", "Count"]
        fig = px.pie(cls, names="Status", values="Count", hole=0.6,
                     color="Status", color_discrete_map=STATUS_COLORS,
                     title="Class balance")
        fig.update_traces(textinfo="percent", marker_line_width=2,
                          marker_line_color="white", sort=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Only about 15% have diabetes, so accuracy is misleading. "
                   "Always guessing 'no diabetes' would score about 85%. Models here are judged by ROC-AUC.")
    with c2:
        g = rate_by(f, "Age_Label")
        fig = px.bar(g, x="Age_Label", y="rate_pct", title="Diabetes rate by age band",
                     labels={"Age_Label": "Age", "rate_pct": "Diabetes rate (%)"},
                     custom_data=["n"], color_discrete_sequence=[C_NO])
        fig.update_traces(hovertemplate="%{x}<br>Rate: %{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        fig.add_hline(y=rate * 100, line_dash="dot", line_color="gray",
                      annotation_text=f"Selection average {rate:.1%}", annotation_position="top left")
        st.plotly_chart(style(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        g = rate_by(f, "BMI_Category")
        fig = px.bar(g, x="BMI_Category", y="rate_pct", text="rate_pct",
                     title="Diabetes rate by BMI category (WHO)",
                     labels={"BMI_Category": "", "rate_pct": "Diabetes rate (%)"},
                     custom_data=["n"], color_discrete_sequence=[C_NO])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          hovertemplate="%{x}<br>Rate: %{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
    with c4:
        g = rate_by(f, "GenHlth_Label")
        fig = px.bar(g, x="GenHlth_Label", y="rate_pct", text="rate_pct",
                     title="Diabetes rate by self-rated general health",
                     labels={"GenHlth_Label": "", "rate_pct": "Diabetes rate (%)"},
                     custom_data=["n"], color_discrete_sequence=[C_NO])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          hovertemplate="%{x}<br>Rate: %{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")

# ==========================================================================
# 2. Risk factors
# ==========================================================================
with tabs[1]:
    st.subheader("How much does each yes/no factor change the diabetes rate?")
    rows = []
    for col, name in BINARY_FACTORS.items():
        grp = f.groupby(col)[TARGET].mean()
        if len(grp) == 2:
            rows.append({"Factor": name, "Without": grp.get(0, float("nan")) * 100,
                         "With": grp.get(1, float("nan")) * 100,
                         "Prevalence": f[col].mean() * 100})
    bf = pd.DataFrame(rows)
    bf["Difference"] = bf["With"] - bf["Without"]
    bf = bf.sort_values("Difference")

    fig = go.Figure()
    for _, r in bf.iterrows():
        fig.add_shape(type="line", x0=r.Without, x1=r.With, y0=r.Factor, y1=r.Factor,
                      line=dict(color="rgba(128,128,128,0.5)", width=2))
    fig.add_trace(go.Scatter(x=bf.Without, y=bf.Factor, mode="markers", name="Without factor",
                             marker=dict(size=11, color=C_NO, line=dict(width=2, color="white")),
                             customdata=bf[["Prevalence"]],
                             hovertemplate="%{y}<br>Without: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=bf.With, y=bf.Factor, mode="markers", name="With factor",
                             marker=dict(size=11, color=C_YES, line=dict(width=2, color="white")),
                             customdata=bf[["Prevalence", "Difference"]],
                             hovertemplate="%{y}<br>With: %{x:.1f}%<br>Difference: %{customdata[1]:+.1f} pts"
                                           "<br>Prevalence: %{customdata[0]:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Diabetes rate (%)", yaxis_title="",
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(style(fig, 520), width="stretch")
    st.caption("Sorted by the gap between the two dots. Heart disease, stroke, high blood pressure and "
               "difficulty walking show the widest gaps. Protective habits sit near the bottom with negative gaps.")

    c1, c2 = st.columns(2)
    with c1:
        g = rate_by(f, "Risk_Score")
        fig = px.bar(g, x="Risk_Score", y="rate_pct", text="rate_pct", custom_data=["n"],
                     title="Aggregated risk score vs diabetes rate",
                     labels={"Risk_Score": "Number of risk conditions (0-6)", "rate_pct": "Diabetes rate (%)"},
                     color_discrete_sequence=[C_YES])
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                          hovertemplate="Score %{x}<br>Rate: %{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Risk score counts high blood pressure, high cholesterol, smoking, stroke, "
                   "heart disease and difficulty walking.")
    with c2:
        g = rate_by(f, "Healthy_Habits")
        fig = px.bar(g, x="Healthy_Habits", y="rate_pct", text="rate_pct", custom_data=["n"],
                     title="Healthy habits score vs diabetes rate",
                     labels={"Healthy_Habits": "Number of healthy habits (0-5)", "rate_pct": "Diabetes rate (%)"},
                     color_discrete_sequence=[C_NO])
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                          hovertemplate="Score %{x}<br>Rate: %{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Habits score counts physical activity, fruit, vegetables, not smoking and not drinking heavily.")

    c3, c4 = st.columns(2)
    with c3:
        g = rate_by(f, "Income_Label")
        fig = px.bar(g, y="Income_Label", x="rate_pct", orientation="h", custom_data=["n"],
                     title="Diabetes rate by household income",
                     labels={"Income_Label": "", "rate_pct": "Diabetes rate (%)"},
                     color_discrete_sequence=[C_NO])
        fig.update_traces(hovertemplate="%{y}<br>Rate: %{x:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
    with c4:
        g = rate_by(f, "Education_Label")
        g["Short"] = g["Education_Label"].astype(str).str.split("(").str[-1].str.rstrip(")")
        fig = px.bar(g, y="Short", x="rate_pct", orientation="h", custom_data=["n", "Education_Label"],
                     title="Diabetes rate by education",
                     labels={"Short": "", "rate_pct": "Diabetes rate (%)"},
                     color_discrete_sequence=[C_NO])
        fig.update_traces(hovertemplate="%{customdata[1]}<br>Rate: %{x:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")

# ==========================================================================
# 3. Explorer
# ==========================================================================
with tabs[2]:
    st.subheader("Cross two dimensions")
    e1, e2 = st.columns(2)
    dim_x = e1.selectbox("Columns", list(GROUP_DIMS), index=1)
    dim_y = e2.selectbox("Rows", list(GROUP_DIMS), index=0)
    cx, cy = GROUP_DIMS[dim_x], GROUP_DIMS[dim_y]
    if cx == cy:
        st.info("Pick two different dimensions.")
    else:
        pv = f.groupby([cy, cx], observed=True)[TARGET].agg(["mean", "size"]).reset_index()
        rate_mat = pv.pivot(index=cy, columns=cx, values="mean") * 100
        n_mat = pv.pivot(index=cy, columns=cx, values="size")
        if cy in ORDER:
            rate_mat = rate_mat.reindex([v for v in ORDER[cy] if v in rate_mat.index.astype(str)])
            n_mat = n_mat.reindex(rate_mat.index)
        if cx in ORDER:
            cols = [v for v in ORDER[cx] if v in rate_mat.columns.astype(str)]
            rate_mat, n_mat = rate_mat[cols], n_mat[cols]
        fig = go.Figure(go.Heatmap(
            z=rate_mat.values, x=[str(c) for c in rate_mat.columns], y=[str(i) for i in rate_mat.index],
            customdata=n_mat.values, colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
            text=rate_mat.round(1).values, texttemplate="%{text}%", xgap=2, ygap=2,
            colorbar=dict(title="Rate %"),
            hovertemplate=f"{dim_y}: %{{y}}<br>{dim_x}: %{{x}}<br>Diabetes rate: %{{z:.1f}}%"
                          "<br>n = %{customdata:,}<extra></extra>"))
        fig.update_layout(title=f"Diabetes rate (%) by {dim_y.lower()} and {dim_x.lower()}",
                          xaxis_title=dim_x, yaxis_title=dim_y)
        st.plotly_chart(style(fig, 480), width="stretch")
        st.caption("Cells with few respondents can be noisy. Hover to see the count behind each cell.")

    st.subheader("Distribution of a numeric attribute by diabetes status")
    num = st.selectbox("Attribute", ["BMI", "Age", "GenHlth", "MentHlth", "PhysHlth", "Income", "Education"])
    norm = st.toggle("Show as percentage within each group", value=True)
    fig = px.histogram(f, x=num, color=f"{TARGET}_Label", barmode="overlay", opacity=0.65,
                       histnorm="percent" if norm else None, nbins=45 if num == "BMI" else None,
                       color_discrete_map=STATUS_COLORS,
                       labels={f"{TARGET}_Label": "Status"})
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0),
                      yaxis_title="% of group" if norm else "Respondents", bargap=0.05)
    st.plotly_chart(style(fig, 400), width="stretch")
    means = f.groupby(f"{TARGET}_Label")[num].mean()
    st.caption(" · ".join(f"Mean {num} for {k.lower()}: {v:.2f}" for k, v in means.items()))

# ==========================================================================
# 4. Data cleaning (from analysis.py outputs, not filtered)
# ==========================================================================
with tabs[3]:
    st.info("This tab shows results from the full cleaning pipeline in analysis.py. Sidebar filters do not apply.")
    k = st.columns(4)
    k[0].metric("Raw rows", f"{res['n_raw']:,}")
    k[1].metric("Duplicates removed", f"{res['n_duplicates']:,}",
                f"{res['n_duplicates'] / res['n_raw']:.2%} of rows", delta_color="off")
    k[2].metric("Missing / out-of-range", f"{res['n_missing'] + res['n_disguised']}")
    k[3].metric("BMI values winsorised", f"{res['n_capped']:,}")

    c1, c2 = st.columns(2)
    with c1:
        d = pd.DataFrame({"Stage": ["Before de-duplication", "After de-duplication"],
                          "Rate": [res["rate_before_dedup"] * 100, res["rate_after_dedup"] * 100]})
        fig = px.bar(d, x="Stage", y="Rate", text="Rate", title="Duplicate removal shifted the diabetes rate",
                     labels={"Stage": "", "Rate": "Diabetes rate (%)"}, color_discrete_sequence=[C_YES])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", width=0.45)
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Duplicated rows were mostly healthy respondents, so removing them raised the rate by 1.4 points.")
    with c2:
        d = pd.DataFrame({"Method": ["IQR rule", "Z-score > 3", "K-means distance"],
                          "Outliers": [res["iqr_outliers"], res["z_outliers"], res["clust_outliers"]]})
        fig = px.bar(d, x="Method", y="Outliers", text="Outliers",
                     title="BMI outliers flagged by three methods",
                     labels={"Method": ""}, color_discrete_sequence=[C_NO])
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", width=0.5)
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Each rule gives a different count. Values were capped at 17 and 56 rather than deleted.")

    c3, c4 = st.columns(2)
    with c3:
        raw_bmi = pd.read_csv(DATA / "brfss2015.csv", usecols=["BMI"])["BMI"] if st.checkbox(
            "Load raw BMI to compare before and after (reads the raw file)") else None
        if raw_bmi is not None:
            comp = pd.concat([pd.DataFrame({"BMI": raw_bmi, "Version": "Raw"}),
                              pd.DataFrame({"BMI": df.BMI, "Version": "Winsorised"})])
            fig = px.box(comp, x="Version", y="BMI", color="Version", points=False,
                         color_discrete_sequence=[C_YES, C_NO], title="BMI before and after winsorisation")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig), width="stretch")
        else:
            st.metric("BMI standard deviation", f"{res['bmi_std_after']:.2f}",
                      f"{res['bmi_std_after'] - res['bmi_std_before']:.2f} after winsorising", delta_color="off")
            st.metric("Memory after type conversion", f"{res['mem_after_dtype']:.1f} MB",
                      f"from {res['mem_before_dtype']:.1f} MB", delta_color="off")
    with c4:
        d = pd.DataFrame({"BMI column": ["Untreated", "Winsorised"],
                          "AUC": [res["auc_raw_bmi"], res["auc_clean_bmi"]]})
        fig = px.bar(d, x="BMI column", y="AUC", text="AUC", title="Model ROC-AUC with each BMI version",
                     labels={"BMI column": ""}, color_discrete_sequence=[C_NO])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", width=0.45)
        fig.update_yaxes(range=[0.80, 0.815])
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Same model and split. The y-axis is zoomed in, so the gain is small: +0.0015.")

    st.subheader("Recall bias in health-day answers, smoothed by binning")
    b1, b2 = st.columns(2)
    for col, box in [("MentHlth_Binned", b1), ("PhysHlth_Binned", b2)]:
        g = rate_by(df, col)
        fig = px.bar(g, x=col, y="n", custom_data=["rate_pct"],
                     title=("Mental" if col.startswith("Ment") else "Physical") + " health: poor days in last 30",
                     labels={col: "", "n": "Respondents"}, color_discrete_sequence=[C_NO])
        fig.update_traces(hovertemplate="%{x}<br>n = %{y:,}<br>Diabetes rate: %{customdata[0]:.1f}%<extra></extra>")
        box.plotly_chart(style(fig, 320), width="stretch")

# ==========================================================================
# 5. Feature selection & reduction
# ==========================================================================
with tabs[4]:
    st.info("Results from analysis.py on the full cleaned data. Sidebar filters do not apply.")
    c1, c2 = st.columns(2)
    with c1:
        cr = corr.drop(index=TARGET, errors="ignore").reset_index()
        cr.columns = ["Attribute", "r"]
        cr = cr.sort_values("r")
        cr["Direction"] = cr["r"].apply(lambda v: "Positive" if v >= 0 else "Negative")
        fig = px.bar(cr, x="r", y="Attribute", orientation="h", color="Direction",
                     color_discrete_map={"Positive": C_YES, "Negative": C_NO},
                     title="Pearson correlation with diabetes")
        fig.update_traces(hovertemplate="%{y}<br>r = %{x:+.3f}<extra></extra>")
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0), yaxis_title="")
        st.plotly_chart(style(fig, 560), width="stretch")
    with c2:
        mi = pd.Series(res["mi_rank"]).sort_values().reset_index()
        mi.columns = ["Attribute", "MI"]
        mi["Selected"] = mi["Attribute"].isin(res["subset_features"]).map(
            {True: "Selected (top 10)", False: "Dropped"})
        fig = px.bar(mi, x="MI", y="Attribute", orientation="h", color="Selected",
                     color_discrete_map={"Selected (top 10)": C_NO, "Dropped": "#b5b4ae"},
                     title="Mutual information with diabetes")
        fig.update_traces(hovertemplate="%{y}<br>MI = %{x:.4f}<extra></extra>")
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0), yaxis_title="")
        st.plotly_chart(style(fig, 560), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        kc = pd.Series(res["k_curve"]).reset_index()
        kc.columns = ["k", "AUC"]
        kc["k"] = kc["k"].astype(int)
        fig = px.line(kc, x="k", y="AUC", markers=True, title="ROC-AUC vs number of attributes kept",
                      labels={"k": "Attributes kept"}, color_discrete_sequence=[C_NO])
        fig.update_traces(marker_size=9, line_width=2, hovertemplate="k = %{x}<br>AUC = %{y:.4f}<extra></extra>")
        fig.add_vline(x=res["subset_k"], line_dash="dot", line_color="gray",
                      annotation_text=f"Chosen k = {res['subset_k']}")
        st.plotly_chart(style(fig), width="stretch")
    with c4:
        sc = pd.Series(res["sample_curve"]).reset_index()
        sc.columns = ["Rows", "AUC"]
        sc["Rows"] = sc["Rows"].astype(int)
        fig = px.line(sc, x="Rows", y="AUC", markers=True, log_x=True,
                      title="ROC-AUC vs training sample size", labels={"Rows": "Training rows (log scale)"},
                      color_discrete_sequence=[C_NO])
        fig.update_traces(marker_size=9, line_width=2, hovertemplate="%{x:,} rows<br>AUC = %{y:.4f}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Reduction strategies compared")
    r = red.copy()
    r["label"] = r["method"] + "<br>" + r["attributes"].astype(str) + " attrs · " + r["rows"].map("{:,}".format) + " rows"
    fig = px.bar(r, x="roc_auc", y="label", orientation="h", text="roc_auc",
                 labels={"roc_auc": "ROC-AUC", "label": ""}, color_discrete_sequence=[C_NO])
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_xaxes(range=[0.79, 0.815])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 330), width="stretch")
    st.caption(f"Keeping 10 attributes and 10% of rows cuts the data from {res['cells_before']:,} to "
               f"{res['cells_after']:,} cells while losing only {(1 - res['combo_auc'] / res['baseline_auc']):.2%} "
               "of ROC-AUC. The x-axis is zoomed in.")

    with st.expander("Chi-square test results (all 21 attributes significant at p < 0.05)"):
        st.dataframe(chi.rename(columns={"cramers_v": "Cramér's V"}).style.format(
            {"chi_square": "{:,.1f}", "p_value": "{:.2e}", "Cramér's V": "{:.3f}"}),
            width="stretch", hide_index=True)

# ==========================================================================
# 6. Data table
# ==========================================================================
with tabs[5]:
    st.subheader("Filtered respondents")
    show_cols = [TARGET + "_Label", "Sex_Label", "Age_Label", "BMI", "BMI_Category", "GenHlth_Label",
                 "Income_Label", "Education_Label", "Risk_Score", "Healthy_Habits"] + list(BINARY_FACTORS)
    st.dataframe(f[show_cols].head(1000), width="stretch", hide_index=True)
    st.caption(f"Showing the first 1,000 of {len(f):,} rows.")
    st.download_button("Download filtered data as CSV", f.to_csv(index=False).encode("utf-8"),
                       "diabetes_filtered.csv", "text/csv")
    with st.expander("Codebook used for label integration"):
        st.dataframe(codebook, width="stretch", hide_index=True)
