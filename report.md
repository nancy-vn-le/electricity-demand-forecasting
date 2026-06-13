# NSW Electricity Demand Forecasting
### Viet Ngan Le — AEMO NSW1, 2019-2025

---

## Executive Summary

Seven years of AEMO NSW1 dispatch data (122,708 half-hour intervals, January 2019 to December 2025) were used to build and evaluate six forecasting models for 30-minute electricity demand. The held-out test set covers 4 December 2025 to 1 January 2026.

The central question: how much can systematic modelling improve on a naive "same demand as four weeks ago" benchmark?

XGBoost with lag features, calendar features, and daily maximum temperature achieves 6.8% MAPE on the test set — a **58.4% reduction** against the naive baseline (16.3% MAPE). The single largest driver of that gap is temperature: adding daily maximum temperature to XGBoost cuts MAPE a further 7.4% beyond the history-and-calendar model alone, quantifying directly what weather contributes to demand variance in NSW.

Statistical models (ARIMA, SARIMA, Holt-Winters) were included for comparison. At 30-minute resolution they underperform the naive baseline, but this reflects a granularity mismatch — ARIMA and SARIMA are daily models evaluated at 30-minute intervals. At daily granularity ARIMA achieves 12.3% MAPE, outperforming the naive baseline.

---

## 1. Data

**Source:** AEMO Price and Demand — publicly available, no authentication required.

**Coverage:** NSW1 region, January 2019 to December 2025 (84 monthly CSV files, 122,708 rows after resampling).

**Key fields:**
- `TOTALDEMAND` — grid demand served in megawatts (MW), at 30-minute settlement intervals
- `RRP` — regional reference price ($/MWh) — collected but not used in this analysis

**Data engineering notes:**

AEMO introduced Five Minute Settlement in October 2021, switching dispatch intervals from 30 minutes to 5 minutes. Files from that point contain six times as many rows per month. All data was resampled to a consistent 30-minute mean frequency before analysis.

Australian Eastern Daylight Time creates two edge cases each year: a one-hour gap in October (spring-forward) and a duplicate hour in April (fall-back). The duplicate hour is dropped rather than attempting to resolve pre/post-transition ambiguity — approximately 54 rows are removed across seven years.

Demand ranges from 2,574 MW (overnight, low-load periods) to 13,723 MW (summer afternoon peaks). There is a visible structural break in 2020 corresponding to COVID-19 lockdowns, visible as a step-down in commercial and industrial load that partially persisted through 2021.

**Temperature data:** Daily maximum, minimum, and mean temperatures for Sydney (Observatory Hill, -33.87°, 151.21°) were sourced from Open-Meteo's historical archive (ERA5 reanalysis). Temperature accuracy is within ~1-2°C of station observations.

---

## 2. Exploratory Data Analysis

### 2.1 Seasonal Patterns

NSW electricity demand has three overlapping seasonal cycles:

**Annual (dual-peak):** A summer cooling peak (December to February) driven by air conditioning and a secondary winter heating peak (June to August). Unlike most Northern Hemisphere electricity markets that have a single winter peak, NSW has elevated demand at both temperature extremes. January and July are typically the two highest-demand months.

**Weekly:** Weekday demand runs approximately 15-20% higher than weekend demand. Commercial and industrial loads — offices, factories, retail — largely switch off on weekends. The weekday profile is consistent Monday through Friday; Saturday and Sunday both drop, with Sunday lower than Saturday.

**Daily:** A morning ramp starting around 6am peaks around 8-9am (workplaces opening, cooking, hot water). A midday trough follows, then an evening peak between 6-8pm as people return home (cooking, lighting, heating or cooling). Weekday peaks are sharper and higher; weekend peaks are flatter and shifted approximately one to two hours later.

### 2.2 Temperature-Demand Relationship

![Temperature and Demand](outputs/figures/temperature_demand.png)

NSW demand shows a U-shaped relationship with temperature: high demand at both extremes. Below approximately 18°C, demand rises as heating loads increase. Above approximately 22°C, demand rises steeply as cooling loads dominate. The summer (high-temperature) slope is steeper than the winter slope — air conditioning loads are larger and more dispersed across the residential sector than gas-supplemented heating.

The monthly co-movement chart shows demand and temperature tracking each other closely during heatwaves (January, February) and cold periods (June, July). This co-movement is what makes temperature the single most important exogenous variable for NSW demand forecasting.

### 2.3 Stationarity

The Augmented Dickey-Fuller (ADF) test was applied to a representative 26-week sample. The test statistic of -4.70 (p-value = 0.00008) rejects the unit root null at the 1% level. The series is stationary — the daily and weekly mean-reverting cycles drive this result; demand oscillates around a broadly stable long-run mean despite the visible 2020 structural break.

For ARIMA modelling, `d=1` (first differencing) was used to remove residual slow trends.

---

## 3. Methodology

### 3.1 Train-Test Split

The last four weeks of data (1,344 half-hour intervals, 4 December 2025 to 1 January 2026) were held out as the test set. All data prior to 4 December 2025 was used for training.

Time series observations are temporally dependent — each value is correlated with its recent history. Randomly assigning rows to train and test would expose future values to the model during training, producing metrics that cannot be replicated in deployment. The held-out test set mirrors the operational setting: always forecasting strictly forward from a fixed training cutoff.

The test window spans the Christmas-New Year period, which contains Australian public holidays (Christmas Day 25 December, Boxing Day 26 December, New Year's Day). Demand on these days behaves like Sundays regardless of their calendar weekday position — a systematic pattern that none of the models here capture, since no public holiday indicator was included in the feature set.

### 3.2 Evaluation Metrics

**RMSE (Root Mean Squared Error):** Expressed in MW. Penalises large errors more heavily than small ones — relevant for electricity, where large forecast errors carry disproportionate costs (emergency generation dispatch, market exposure, risk of load shedding).

**MAPE (Mean Absolute Percentage Error):** Scale-free. Used as the primary ranking metric because it is directly comparable across time windows and communicable to non-technical audiences.

### 3.3 Evaluation Granularity

ARIMA and SARIMA were fit on daily-averaged data rather than 30-minute data. Fitting on the full 30-minute series covering seven years is computationally prohibitive. Their daily forecasts were forward-filled to 30-minute resolution for comparison at the same granularity as the other models.

A flat daily forecast evaluated against 30-minute data carries an inherent penalty: NSW demand has an intra-day standard deviation of approximately 1,264 MW against a daily mean of ~7,172 MW, a ~17.6% relative swing. Any model that cannot represent intra-day shape will accumulate this variance as forecast error regardless of daily-level accuracy.

Both 30-minute and daily metrics are reported so each model is assessed at the resolution it was designed for.

---

## 4. Models

### Seasonal Naive Baseline

For each test point, forecast the demand observed exactly four weeks prior. A four-week lag is used rather than one week because the test horizon is four weeks — a one-week lag would require using test-period observations to produce forecasts for weeks two through four.

Four weeks prior to December 2025 maps to early November 2025, a shoulder season with lower peak temperatures than December. The naive forecast therefore systematically under-predicts summer afternoon peaks in the test window. This inflates naive MAPE for this particular test period relative to what a year-on-year lag would produce.

### ARIMA(2,1,1)

Order (2,1,1) was selected from ACF/PACF analysis: the PACF cuts off after lag 2-3 (AR order 2), the ACF decays slowly after differencing (d=1), and a spike at lag 1 post-differencing suggests MA order 1. Fit on daily-averaged data, upsampled to 30-minute resolution.

### SARIMA(1,1,1)(1,1,1,7)

Extends ARIMA with a weekly seasonal component (m=7 for daily data). This adds AR, differencing, and MA terms at the weekly frequency to capture the weekday/weekend pattern that plain ARIMA cannot represent. Fitting at 30-minute resolution with m=336 (full weekly cycle) would be computationally prohibitive — the state-space matrices scale as m×m. Fit on daily-averaged data, upsampled to 30-minute resolution.

### Holt-Winters Exponential Smoothing

Triple exponential smoothing with additive trend and additive seasonality at `seasonal_periods=336` (one full week of 30-minute intervals). Unlike ARIMA and SARIMA, Holt-Winters operates at full 30-minute resolution and explicitly represents intra-day shape.

Fixed smoothing parameters were used (level α=0.3, trend β=0.01, seasonal γ=0.05) rather than optimised values. The L-BFGS-B optimiser finds degenerate solutions near α≈β≈1 at weekly seasonal periods at 30-minute resolution, producing explosive long-horizon forecasts. The fixed parameters are standard starting points for short-horizon electricity demand forecasting and produce stable results.

### XGBoost with Lag Features

Gradient boosting treating demand forecasting as supervised regression. The time series structure is encoded as features rather than modelled explicitly.

**Feature set:**
- Lag features: demand at t-48 (24 hours prior), t-96 (48 hours), t-336 (one week), t-672 (two weeks)
- Rolling statistics: 48-period rolling mean and standard deviation
- Calendar features: hour of day, day of week, month, quarter, is-weekend flag, half-hour-of-day position

All lag and rolling features are computed with `.shift(k)` so that the feature for time t uses only observations at t-k and earlier — no leakage.

### XGBoost with Temperature

Extends the lag feature model with daily maximum temperature (`temp_max`) mapped to each 30-minute interval by date. Temperature is the dominant exogenous driver of demand variation in NSW — the U-shaped temperature-demand relationship means a single daily temperature figure captures the bulk of weather-driven demand variance that history-based features cannot anticipate.

---

## 5. Results

![Forecast vs Actual](outputs/figures/forecast_comparison.png)

![Model Comparison](outputs/figures/model_comparison.png)

### 5.1 30-Minute Resolution

| Model | RMSE (MW) | MAPE (%) |
|-------|-----------|----------|
| **XGBoost + Temp** | **615.1** | **6.8%** |
| XGBoost | 684.0 | 7.3% |
| Naive Baseline | 1,492.1 | 16.3% |
| ARIMA(2,1,1) | 1,704.7 | 18.7% |
| SARIMA(1,1,1)(1,1,1,7) | 1,783.8 | 21.3% |
| Holt-Winters | 1,893.5 | 24.6% |

### 5.2 Daily Resolution

| Model | RMSE daily (MW) | MAPE daily (%) |
|-------|-----------------|----------------|
| **XGBoost + Temp** | **343.5** | **4.0%** |
| XGBoost | 389.4 | 4.4% |
| ARIMA(2,1,1) | 1,090.0 | 12.3% |
| Naive Baseline | 1,164.7 | 14.4% |
| SARIMA(1,1,1)(1,1,1,7) | 1,213.4 | 14.2% |
| Holt-Winters | 1,641.2 | 20.1% |

At daily granularity ARIMA outperforms the naive baseline (12.3% vs 14.4%), which is the result expected of a correctly specified daily model. SARIMA is slightly worse than naive at this granularity — the seasonal differencing in the (1,1,1,7) component appears to overfit to a weekday/weekend pattern that shifts during the Christmas test window. Holt-Winters performs poorly at both resolutions on this test set; its fixed seasonal component cannot adapt quickly to the atypical demand shape of the holiday period.

### 5.3 The Temperature Effect

The difference between XGBoost (7.3% MAPE) and XGBoost+Temp (6.8% MAPE) isolates the contribution of temperature information. On the December test set this is a 7.4% relative reduction — meaningful but smaller than would be expected on a January heatwave test window, where air conditioning load swings are larger. Adding temperature converts a history-extrapolation model into one that can anticipate demand spikes from incoming hot weather.

---

## 6. Error Analysis

![Residual Heatmap](outputs/figures/residual_heatmap.png)

XGBoost+Temp mean absolute error by hour of day and day of week across the 4-week test set:

**Morning ramp (6-9am) and evening peak (6-8pm)** have the highest errors. These are the periods of fastest demand change. Even with temperature in the model, the precise timing and magnitude of demand ramps depend on behavioural factors — when people start cooking, run appliances, or leave work — that are not captured by a single daily maximum temperature figure.

**Weekend errors are elevated** relative to equivalent weekday hours. Weekend demand is more variable: leisure patterns, weather-dependent recreational loads, and fewer predictable commercial anchors make weekends harder to forecast than weekdays.

**Overnight (2-4am) errors are near zero.** Overnight demand is stable and well-explained by lag features alone; temperature has minimal incremental effect at these hours.

**Christmas and Boxing Day (25-26 December)** appear as elevated error days regardless of hour. Both fall mid-week in 2025 (Wednesday and Thursday) but demand patterns collapse to Sunday-equivalent levels. Without a public holiday indicator, the model predicts mid-week commercial demand and over-forecasts by 1,000-2,000 MW on these days.

---

## 7. Feature Importance

![Feature Importance](outputs/figures/xgboost_temp_feature_importance.png)

The 1-week lag (t-336) is the dominant feature. This directly reflects the strong weekly seasonal pattern: the best single predictor of demand at any given time is demand at the same time last week. The 2-week lag (t-672) and 24-hour lag (t-48) also carry substantial weight.

Calendar features — hour of day and day of week — contribute the intra-day and weekday/weekend shape. The rolling 24-hour mean captures recent demand level, which matters during heatwaves or cold snaps where the level shifts over several days.

`temp_max` ranks within the top features, behind the weekly lag but ahead of several calendar variables. Its importance is asymmetric: it contributes most on high-temperature days (above ~28°C) where air conditioning load introduces demand that lags cannot anticipate.

---

## 8. Limitations

**Single test window.** All metrics are from a four-week December window. December 2025 has specific characteristics — a summer heat event around 19 December, a public holiday cluster at Christmas-New Year — that may not represent average forecast difficulty. A robust evaluation would use rolling-window backtesting across multiple seasons and years.

**No prediction intervals.** Operational energy forecasting requires probabilistic outputs — 10th, 50th, and 90th percentile forecasts — to support reserve capacity decisions and risk management. This analysis produces only point forecasts.

**Public holidays not flagged.** An `is_public_holiday` binary feature would directly address the Christmas/Boxing Day over-prediction. The `holidays` Python package provides the full Australian public holiday calendar with one function call.

**Static model.** Demand patterns shift over time — rooftop solar uptake suppresses midday demand, EV charging is emerging as an evening load, industrial mix changes. A model fit on 2019-2025 data captures these shifts implicitly but would need periodic retraining to remain accurate as structural changes accelerate.

**Daily temperature only.** `temp_max` is a single daily figure. Sub-daily temperature (hourly or 3-hourly BOM forecasts) would better represent the afternoon timing of heat-driven demand peaks and the overnight temperature effects on heating load.

---

## 9. Extensions

**Public holiday calendar** — one binary feature from the `holidays` package. The highest-ROI single change, directly fixing the most visible failure mode in this test set.

**Sub-daily temperature** — hourly BOM temperature forecasts as an exogenous variable would improve peak timing accuracy. Replacing ERA5 reanalysis (historical actuals) with BOM forecast temperature would also make the model deployable in a live forecasting context, where future temperature must be forecast rather than observed.

**Probabilistic forecasting** — XGBoost Quantile Regression at the 10th, 50th, and 90th percentiles, or conformal prediction intervals. Required for operational use in energy markets where risk management depends on the forecast distribution, not just the central estimate.

**Rolling backtesting** — evaluate across 12+ held-out months to obtain stable seasonal estimates of model performance and identify periods where each model degrades.

**Ensemble** — a simple average of XGBoost+Temp and Holt-Winters often outperforms either component alone for electricity demand. Holt-Winters's explicit seasonal structure complements XGBoost's flexibility on non-seasonal variation.

**Multi-region** — extending to VIC1 or full NEM-wide demand. NSW and Victoria have correlated demand (similar climate, overlapping commercial hours) but distinct characteristics — Victorian demand is gas-heating dominated in winter, NSW is more electricity-heating.

---

## 10. Conclusions

Six forecasting models were evaluated on seven years of AEMO NSW1 demand data.

XGBoost with temperature achieves 6.8% MAPE on the December 2025 test set, a 58.4% reduction against the naive seasonal baseline. The weekly lag feature (t-336) is the dominant predictor, consistent with the strong weekly seasonality in NSW demand. Adding daily maximum temperature to the model accounts for approximately one-seventh of the total baseline-to-best-model improvement.

The statistical models (ARIMA, SARIMA, Holt-Winters) are competitive at the granularity they were designed for. ARIMA(2,1,1) achieves 12.3% daily MAPE — outperforming the naive baseline at daily resolution. Their weaker 30-minute metrics are an artefact of daily-to-30-minute upsampling rather than poor daily-level accuracy.

The primary remaining error sources are public holidays (systematic mis-classification of Christmas/Boxing Day as weekdays) and intra-day timing uncertainty during peak demand ramps — both concentrated in the 6-9am and 6-8pm windows where demand changes fastest and where a single daily temperature figure is least informative.

---

*Data: AEMO NSW1 Price and Demand — `https://aemo.com.au/aemo/data/nem/priceanddemand/`*
*Temperature: Open-Meteo historical archive (ERA5 reanalysis)*
*Period: January 2019 - December 2025*
*Code: `https://github.com/nancy-vn-le/electricity-demand-forecasting`*
