"""Streamlit dashboard for NSW Electricity Demand Forecasting.

Run with:
    streamlit run app.py

Requires outputs/forecasts.csv and outputs/metrics.csv to exist.
Generate them by running the notebook end-to-end first.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSW Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    forecasts = pd.read_csv("outputs/forecasts.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv("outputs/metrics.csv", index_col=0)
    return forecasts, metrics


try:
    forecasts, metrics = load_data()
except FileNotFoundError:
    st.error(
        "outputs/forecasts.csv not found. "
        "Run the notebook end-to-end first (Kernel → Restart & Run All)."
    )
    st.stop()

MODEL_COLOURS = {
    "Naive Baseline":           "#9E9E9E",
    "ARIMA":                    "#E91E63",
    "SARIMA":                   "#9C27B0",
    "Holt-Winters":             "#4CAF50",
    "XGBoost":                  "#F44336",
    "XGBoost + Temp":           "#FF6F00",
}

model_cols = [c for c in forecasts.columns if c not in ("actual", "temp_max")]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("NSW Electricity Demand Forecasting")
st.markdown(
    "End-to-end time series forecasting using six years of AEMO data (2019-2024). "
    "Select models below to compare forecasts against actual demand."
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Display Options")

selected_models = st.sidebar.multiselect(
    "Models to display",
    options=model_cols,
    default=["XGBoost + Temp", "Holt-Winters", "Naive Baseline"],
)

show_temp = st.sidebar.checkbox("Overlay temperature", value=True)

zoom_options = {"Full 4 weeks": None, "Last 2 weeks": 2, "Last 1 week": 1}
zoom_label = st.sidebar.radio("Zoom", options=list(zoom_options.keys()), index=1)
zoom_weeks = zoom_options[zoom_label]

# ---------------------------------------------------------------------------
# Filter data by zoom
# ---------------------------------------------------------------------------
plot_df = forecasts.copy()
if zoom_weeks:
    cutoff = plot_df.index[-1] - pd.Timedelta(weeks=zoom_weeks)
    plot_df = plot_df[plot_df.index >= cutoff]

# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df["actual"],
    name="Actual",
    line=dict(color="black", width=1.5),
))

for model in selected_models:
    if model in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[model],
            name=model,
            line=dict(color=MODEL_COLOURS.get(model, "#888"), width=1),
            opacity=0.85,
        ))

if show_temp and "temp_max" in plot_df.columns:
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["temp_max"],
        name="Max Temp (°C)",
        yaxis="y2",
        line=dict(color="#FF7043", width=1, dash="dot"),
        opacity=0.6,
    ))

fig.update_layout(
    title="Forecast vs Actual — 4-Week Test Period (Dec 2024)",
    xaxis_title="Date",
    yaxis_title="Demand (MW)",
    yaxis2=dict(
        title="Max Temperature (°C)",
        overlaying="y",
        side="right",
        showgrid=False,
        range=[0, 60],
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=480,
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
st.subheader("Model Performance — 4-Week Test Set")

display_metrics = metrics.copy()
display_metrics.columns = ["RMSE (MW)", "MAPE (%)"]
display_metrics["RMSE (MW)"] = display_metrics["RMSE (MW)"].map("{:.1f}".format)
display_metrics["MAPE (%)"] = display_metrics["MAPE (%)"].map("{:.2f}%".format)

naive_mape = float(metrics.loc["Naive Baseline", "mape"])

def highlight_best(row):
    mape_val = float(row["MAPE (%)"].replace("%", ""))
    if mape_val == metrics["mape"].min():
        return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
    elif mape_val > naive_mape:
        return ["background-color: #fff3e0"] * len(row)
    return [""] * len(row)

st.dataframe(
    display_metrics.style.apply(highlight_best, axis=1),
    use_container_width=True,
)
st.caption("Green = best model. Orange = worse than naive baseline at 30-min resolution (ARIMA/SARIMA are daily models — see notebook for daily-granularity evaluation).")

# ---------------------------------------------------------------------------
# Key insights
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Key Findings")

best_model = metrics["mape"].idxmin()
best_mape = metrics["mape"].min()
improvement = (naive_mape - best_mape) / naive_mape * 100

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_model)
col2.metric("Best MAPE", f"{best_mape:.2f}%")
col3.metric("Improvement vs Naive", f"{improvement:.1f}%")

st.markdown(
    """
**Why XGBoost wins:** Lag features (especially 1-week lag) capture the strong weekly seasonal pattern.
Calendar features (hour-of-day, weekday/weekend) capture the systematic intra-day and weekly shape.
Temperature captures the demand surge on hot summer days that pure history-based models miss.

**Why ARIMA/SARIMA appear weaker here:** They were fit on daily-averaged data and upsampled to 30-min
by forward-fill. The flat daily forecast is penalised by the ~17.6% intra-day demand swing.
At daily granularity they are competitive with Holt-Winters (~13-14% MAPE).

**Biggest remaining gap:** Public holidays (Christmas Day falls in the test window).
These behave like Sundays but all models treat them as weekdays, causing systematic over-prediction.
    """
)

st.divider()
st.caption(
    "Data: AEMO NSW1 Price and Demand | Temperature: Open-Meteo (ERA5) | "
    "Code: github.com/nancy-vn-le/electricity-demand-forecasting"
)
