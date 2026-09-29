# diabetes_analysis

Data preprocessing and analysis of the **CDC Diabetes Health Indicators** dataset (BRFSS 2015, UCI ML Repository ID 891), with an interactive Streamlit dashboard.

- 253,680 survey responses × 22 attributes
- Target: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes)

## Project structure

| Path | Contents |
|---|---|
| `analysis.py` | Full pipeline: cleaning, integration, correlation, transformation, reduction |
| `dashboard.py` | Streamlit dashboard built on the pipeline outputs |
| `data/brfss2015.csv` | Raw dataset |
| `data/brfss_codebook.csv` | Code-to-label lookup from the CDC BRFSS 2015 codebook |
| `output/` | Cleaned data, reduced data, statistical results |
| `figures/` | Charts produced by the pipeline |

## Setup

```bash
pip install -r requirements.txt
```

## Run the analysis

```bash
python analysis.py
```

Regenerates everything in `output/` and `figures/`. Takes about 3 minutes.

## Run the dashboard

```bash
streamlit run dashboard.py
```

Tabs:

1. **Overview** – key metrics, class balance, diabetes rate by age, BMI and general health
2. **Risk Factors** – rate with and without each factor, composite risk and healthy-habits scores, income and education
3. **Explorer** – heatmap of diabetes rate across any two dimensions, distributions by diabetes status
4. **Data Cleaning** – duplicates, BMI outlier detection, winsorisation, binning
5. **Feature Selection & Reduction** – correlation, mutual information, subset size and sample size curves, PCA comparison
6. **Data** – filtered table, CSV download and codebook

Sidebar filters (sex, age group, BMI, general health, income) apply to the first three tabs.

## Key results

| Metric | Value |
|---|---|
| Duplicate rows removed | 24,206 |
| Rows after cleaning | 229,474 |
| Diabetes rate after cleaning | 15.3% |
| Strongest correlate | General health (r = +0.277) |
| Baseline ROC-AUC (21 attributes) | 0.8110 |
| ROC-AUC with top 10 attributes | 0.8059 |
