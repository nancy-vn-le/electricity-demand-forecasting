# Australian Electricity Demand Forecasting

End-to-end time series forecasting project using publicly available AEMO (Australian Energy Market Operator) data for New South Wales. Demonstrates the full data science workflow from data acquisition through to operational interpretation — built as a portfolio project targeting data analyst/data science roles in the Australian energy sector.

---

## Results

| Model | RMSE (MW) | MAPE (%) | Granularity |
|-------|-----------|----------|-------------|
| **XGBoost** | **674.7** | **7.43%** | 30-min |
| Holt-Winters | 1,592.3 | 16.20% | 30-min |
| Naive Baseline | 1,672.2 | 17.63% | 30-min |
| SARIMA(1,1,1)(1,1,1,7) | 1,624.1 | 19.73% | 30-min |
| ARIMA(2,1,1) | 1,599.5 | 19.88% | 30-min |

XGBoost achieves a **57.9% improvement in MAPE over the naive baseline**, driven by lag features that capture the weekly seasonal pattern alongside calendar features that encode the hour-of-day and weekday/weekend split.

ARIMA and SARIMA were fit on daily-aggregated data for computational tractability and then upsampled to 30-min resolution using forward-fill. Their 30-min MAPE is penalised by the ~17.6% intra-day demand swing — at daily granularity, ARIMA (13.5% daily MAPE) and SARIMA (13.5% daily MAPE) are competitive with Holt-Winters.

## Key Findings

- **Seasonal patterns dominate:** NSW electricity demand has a strong weekly cycle (weekday demand ~15–20% higher than weekend) and an annual dual-peak pattern — summer cooling peaks (Dec–Feb) and a secondary winter heating peak (Jun–Aug).
- **XGBoost wins at 30-minute resolution:** Lag and calendar features allow it to capture both level and intra-day shape simultaneously. MAPE 7.43%.
- **ARIMA/SARIMA are penalised at 30-min resolution:** Their flat daily forecasts are blind to intra-day shape — but they perform comparably at daily granularity.
- **All models struggle at demand peaks:** Morning ramp-up (6–9am) and evening peak (5–8pm) have the highest forecast errors — these periods are most sensitive to temperature, and none of the models here include weather data. Adding temperature as an input would likely halve MAPE.
- **COVID-19 structural break (2020):** Visible as a step-down in demand during lockdowns — any model trained across this break must account for it or exclude the period.

## Sample Forecast

![Forecast vs Actual](outputs/figures/forecast_comparison.png)

*All five models forecast against actual demand for the 4-week held-out test set. Zoomed panel shows detail for the first two weeks.*

---

## Models Compared

| Model | Description | Granularity | Relative Complexity |
|-------|-------------|-------------|---------------------|
| **Seasonal Naive** | Forecast = demand 1 week ago | 30-min | Baseline |
| **ARIMA(2,1,1)** | AutoRegressive Integrated Moving Average | Daily (upsampled) | Low |
| **SARIMA(1,1,1)(1,1,1,7)** | ARIMA + weekly seasonal component | Daily (upsampled) | Medium |
| **Holt-Winters** | Triple exponential smoothing (level + trend + seasonal) | 30-min | Medium |
| **XGBoost** | Gradient boosting with lag and calendar features | 30-min | High |

---

## Project Structure

```
project-energy/
├── data/
│   ├── raw/              # AEMO CSV files (gitignored — run download script)
│   └── processed/        # Cleaned data (gitignored — reproducible from raw)
├── notebooks/
│   └── electricity_demand_forecasting.ipynb   # Main analysis (all sections)
├── src/
│   ├── download_aemo.py  # Download AEMO data for a configurable date range
│   └── utils.py          # Shared metrics (RMSE, MAPE), feature engineering, plot helpers
├── outputs/
│   └── figures/          # Saved charts (committed — used in this README)
├── requirements.txt
└── README.md
```

---

## How to Reproduce

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the data**
```bash
python src/download_aemo.py
```
This downloads AEMO NSW1 dispatch data for 2019–2024 (~175,000 rows). Takes ~5–10 minutes depending on connection speed. Edit `REGION`, `START_YEAR`, `END_YEAR` at the top of the script to change the scope.

**3. Run the notebook**
```bash
jupyter notebook notebooks/electricity_demand_forecasting.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All). Total runtime approximately 10–20 minutes depending on hardware (SARIMA and Holt-Winters are the slow steps).

---

## Methodology

The notebook follows this structure:

1. **Data Loading & Cleaning** — parse AEMO settlement dates with correct Australian timezone handling (AEST/AEDT); check for gaps; verify 30-minute interval regularity
2. **EDA** — trend decomposition, weekday vs weekend analysis, hour-of-day profile, monthly seasonality, ACF/PACF plots
3. **Stationarity Testing** — ADF test; first-differencing if needed
4. **Train/Test Split** — last 4 weeks as test, strictly time-ordered (no random splitting)
5. **Models** — each model explained, fitted, and evaluated with RMSE and MAPE
6. **Evaluation** — side-by-side model comparison, residual heatmap by hour and day of week
7. **Practical Implications** — when models fail, what an energy company would do next

Every modelling decision includes a plain-language explanation and an "interview answer" note — the notebook is designed to build intuition, not just produce results.

---

## Dataset

**Source:** AEMO (Australian Energy Market Operator)  
**URL pattern:** `https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_NSW1.csv`  
**Coverage:** NSW1 region, 30-minute dispatch intervals, 2019–2024 (72 monthly files)  
**Columns:** `TOTALDEMAND` (MW demand served by the grid) and `RRP` (regional reference price, $/MWh)  
**Licence:** AEMO data is publicly available under the [AEMO Copyright Notice](https://www.aemo.com.au/about/privacy-and-legal-notices/copyright-permissions)

---

## Skills Demonstrated

- Time series analysis: seasonal decomposition, ACF/PACF, ADF stationarity test
- Statistical forecasting: ARIMA, SARIMA, Holt-Winters exponential smoothing
- Machine learning for time series: XGBoost with lag/calendar feature engineering, no-leakage train/test splits
- Model evaluation: RMSE, MAPE, residual analysis, error heatmaps
- Python data stack: pandas, NumPy, statsmodels, scikit-learn, XGBoost, matplotlib, seaborn
- Data engineering: downloading and parsing AEMO public CSV archives programmatically; timezone-aware datetime handling (AEST/AEDT, DST transitions)
- Domain interpretation: electricity market context, operational forecasting use cases

---

## What's Next (Potential Extensions)

- **Add weather data:** Integrate BOM temperature forecasts as an exogenous variable (SARIMAX, XGBoost feature) — expected to halve MAPE
- **Probabilistic forecasting:** XGBoost Quantile Regression or SARIMA prediction intervals for risk management
- **Prophet:** Facebook's Prophet model handles holidays and trend changepoints well — good comparison point
- **Streamlit dashboard:** Interactive visualisation of forecasts for non-technical stakeholders
- **Multi-state:** Extend to VIC1 or full NEM-wide demand
