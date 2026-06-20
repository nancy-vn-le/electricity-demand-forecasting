"""Streamlit dashboard for NSW Electricity Demand Forecasting.

Run with:
    streamlit run app.py

Requires outputs/forecasts.csv and outputs/metrics.csv to exist.
Generate them by running the notebook end-to-end first.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NSW Electricity Demand Forecasting",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Colors - single source of truth for pills, chart lines, and CSS
# ---------------------------------------------------------------------------
MODEL_COLOURS = {
    "Actual":         "#FFFFFF",
    "Naive Baseline": "#888888",
    "ARIMA":          "#EF553B",
    "SARIMA":         "#AB63FA",
    "Holt-Winters":   "#00CC96",
    "XGBoost":        "#FFA15A",
    "XGBoost + Temp": "#FF6692",
}

# Light-background models need dark text when pill is selected
_DARK_TEXT = {"#00CC96", "#FFA15A"}

MODEL_ORDER = [
    "Naive Baseline", "ARIMA", "SARIMA",
    "Holt-Winters", "XGBoost", "XGBoost + Temp",
]

METRIC_NAME = {
    "ARIMA":  "ARIMA(2,1,1)",
    "SARIMA": "SARIMA(1,1,1)(1,1,1,7)",
}

# ---------------------------------------------------------------------------
# CSS - metric cards + colored pills
# ---------------------------------------------------------------------------
def _pill_css(model: str, color: str) -> str:
    text = "#1e2130" if color.upper() in _DARK_TEXT else "#ffffff"
    # Streamlit 1.40+ renders st.pills buttons inside [data-testid="stPills"]
    # Each button carries aria-label=<option text> and aria-selected="true|false"
    base = f'[data-testid="stPills"] button[aria-label="{model}"]'
    return f"""
    {base} {{
        border: 2px solid {color} !important;
        color: {color} !important;
        background-color: transparent !important;
        border-radius: 999px !important;
        transition: background-color 0.15s, color 0.15s !important;
    }}
    {base}[aria-selected="true"] {{
        background-color: {color} !important;
        color: {text} !important;
    }}
    {base}:hover {{
        opacity: 0.8 !important;
    }}
    """

pill_styles = "\n".join(
    _pill_css(m, c)
    for m, c in MODEL_COLOURS.items()
    if m != "Actual"
)

st.markdown(
    f"""
    <style>
    /* Metric cards */
    [data-testid="metric-container"] {{
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
    }}
    [data-testid="stMetricLabel"] > div {{
        font-size: 0.7rem !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888 !important;
    }}
    [data-testid="stMetricValue"] > div {{ font-size: 2rem; }}
    [data-testid="stMetricDelta"] > div {{ font-size: 0.75rem !important; }}

    /* Pills container */
    [data-testid="stPills"] {{
        gap: 8px !important;
    }}
    [data-testid="stPills"] button {{
        font-size: 0.78rem !important;
        padding: 4px 14px !important;
        font-weight: 500 !important;
    }}

    {pill_styles}

    /* Hide sidebar entirely */
    section[data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"]  {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
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
        "Run the notebook end-to-end first (Kernel -> Restart & Run All)."
    )
    st.stop()

model_cols = [m for m in MODEL_ORDER if m in forecasts.columns]

# ---------------------------------------------------------------------------
# Derived KPI values
# ---------------------------------------------------------------------------
best_model = metrics["mape"].idxmin()
best_rmse  = metrics.loc[best_model, "rmse"]
best_mape  = metrics.loc[best_model, "mape"]
naive_rmse = metrics.loc["Naive Baseline", "rmse"]
naive_mape = metrics.loc["Naive Baseline", "mape"]

peak_idx    = forecasts["actual"].idxmax()
peak_demand = forecasts["actual"].max()
h = peak_idx.hour
am_pm = "am" if h < 12 else "pm"
h12 = h % 12 or 12
peak_label = f"{peak_idx.strftime('%b')} {peak_idx.day} at {h12} {am_pm}"

test_month = forecasts.index.min().strftime("%b %Y").upper()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_btns = st.columns([7, 3])
with col_title:
    st.markdown(
        "<p style='color:#888; font-size:0.72rem; letter-spacing:0.15em; margin-bottom:0;'>"
        f"NSW AEMO &middot; {test_month} &middot; 4-WEEK TEST</p>",
        unsafe_allow_html=True,
    )
    st.title("Electricity Demand Forecast")
with col_btns:
    st.markdown("<div style='padding-top:2.2rem; display:flex; justify-content:flex-end; gap:0.5rem;'>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.link_button(
            "Full Report",
            "https://github.com/nancy-vn-le/electricity-demand-forecasting/blob/main/report.md",
            use_container_width=True,
        )
    with b2:
        st.link_button(
            "GitHub",
            "https://github.com/nancy-vn-le/electricity-demand-forecasting",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Best RMSE",
    f"{best_rmse:,.0f} MW",
    delta=best_model,
    delta_color="off",
    help="Lowest RMSE across all models on the 4-week test set",
)
kpi2.metric(
    "Best MAPE",
    f"{best_mape:.1f} %",
    delta=best_model,
    delta_color="off",
    help="Lowest MAPE across all models - primary ranking metric",
)
kpi3.metric(
    "Naive RMSE",
    f"{naive_rmse:,.0f} MW",
    delta="baseline",
    delta_color="off",
    help="Seasonal naive (same demand 4 weeks prior) - the bar to beat",
)
kpi4.metric(
    "Peak Demand",
    f"{peak_demand / 1000:.1f} GW",
    delta=peak_label,
    delta_color="off",
    help="Highest actual demand recorded in the test period",
)
st.divider()

# ---------------------------------------------------------------------------
# Chart section header + all inline controls (pills, temp overlay, zoom)
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='color:#aaa; font-size:0.72rem; letter-spacing:0.12em; margin-bottom:0.4rem;'>"
    "FORECAST VS ACTUAL - TOGGLE MODELS</p>",
    unsafe_allow_html=True,
)

zoom_options = {"Full 4 weeks": None, "Last 2 weeks": 2, "Last 1 week": 1}

c_badge, c_pills, c_temp, c_zoom = st.columns([1, 7, 1.8, 1.5])
with c_badge:
    st.markdown(
        "<div style='padding-top:6px; font-size:0.82rem; color:#e0e0e0;'>"
        "<span style='color:#fff; font-size:1.1rem;'>&#9679;</span>&nbsp;Actual</div>",
        unsafe_allow_html=True,
    )
with c_pills:
    selected_models = st.pills(
        label="models",
        options=model_cols,
        selection_mode="multi",
        default=model_cols,
        label_visibility="collapsed",
    )
with c_temp:
    show_temp = st.checkbox("Overlay temp", value=True, key="show_temp")
with c_zoom:
    zoom_label = st.selectbox(
        "Zoom",
        list(zoom_options.keys()),
        index=1,
        key="zoom_select",
        label_visibility="collapsed",
    )

zoom_weeks = zoom_options[zoom_label]

# Normalize to empty list if nothing is selected
if selected_models is None:
    selected_models = []

# ---------------------------------------------------------------------------
# Filter data for zoom window
# ---------------------------------------------------------------------------
plot_df = forecasts.copy()
if zoom_weeks:
    cutoff = plot_df.index[-1] - pd.Timedelta(weeks=zoom_weeks)
    plot_df = plot_df[plot_df.index >= cutoff]

# ---------------------------------------------------------------------------
# Plotly forecast chart
# ---------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df["actual"],
    name="Actual",
    line=dict(color=MODEL_COLOURS["Actual"], width=2),
    hovertemplate="%{y:,.0f} MW<extra>Actual</extra>",
))

for model in model_cols:
    if model in selected_models and model in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[model],
            name=model,
            line=dict(color=MODEL_COLOURS.get(model, "#888"), width=1.5),
            hovertemplate=f"%{{y:,.0f}} MW<extra>{model}</extra>",
        ))

if show_temp and "temp_max" in plot_df.columns:
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["temp_max"],
        name="Max Temp (C)",
        yaxis="y2",
        line=dict(color="#FF7043", width=1, dash="dot"),
        opacity=0.7,
        hovertemplate="%{y:.1f} C<extra>Max Temp</extra>",
    ))

fig.update_layout(
    paper_bgcolor="#1e2130",
    plot_bgcolor="#1e2130",
    font=dict(color="#e0e0e0"),
    xaxis=dict(showgrid=True, gridcolor="#2d3250", zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(title="Demand (MW)", showgrid=True, gridcolor="#2d3250", zeroline=False),
    yaxis2=dict(
        title="Max Temp (C)", overlaying="y", side="right",
        showgrid=False, range=[0, 60],
        tickfont=dict(color="#FF7043"), title_font=dict(color="#FF7043"),
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        bgcolor="rgba(30,33,48,0.8)", bordercolor="#2d3250", borderwidth=1,
    ),
    height=480,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=60, b=10),
)

st.plotly_chart(fig, width="stretch")
st.caption("Hover chart for values · Scroll/drag to zoom and pan")
st.divider()

# ---------------------------------------------------------------------------
# Performance table with progress bars
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='color:#aaa; font-size:0.72rem; letter-spacing:0.12em; margin-bottom:0.4rem;'>"
    "MODEL PERFORMANCE - 4-WEEK TEST SET</p>",
    unsafe_allow_html=True,
)

perf = metrics.copy().reset_index().sort_values("rmse").reset_index(drop=True)
perf.columns = ["Model", "rmse", "mape"]

rmse_max = float(perf["rmse"].max()) * 1.08
mape_max = float(perf["mape"].max()) * 1.08

st.dataframe(
    perf,
    column_config={
        "Model": st.column_config.TextColumn("MODEL", width="large"),
        "rmse": st.column_config.ProgressColumn(
            "RMSE (MW)",
            help="Root Mean Squared Error - lower is better",
            min_value=0,
            max_value=rmse_max,
            format="%.1f",
        ),
        "mape": st.column_config.ProgressColumn(
            "MAPE (%)",
            help="Mean Absolute Percentage Error - lower is better",
            min_value=0,
            max_value=mape_max,
            format="%.2f%%",
        ),
    },
    hide_index=True,
    width="stretch",
    height=260,
)
st.caption(
    "ARIMA/SARIMA were fit on daily data and upsampled - their 30-min MAPE reflects the ~17.6% "
    "intra-day swing a flat daily forecast cannot represent. At daily granularity ARIMA achieves "
    "12.3% MAPE, outperforming the naive baseline."
)
st.divider()

# ---------------------------------------------------------------------------
# Key insights
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='color:#aaa; font-size:0.72rem; letter-spacing:0.12em; margin-bottom:0.8rem;'>"
    "KEY INSIGHTS</p>",
    unsafe_allow_html=True,
)

xgb_mape      = metrics.loc["XGBoost", "mape"]        if "XGBoost" in metrics.index else None
xgb_temp_mape = metrics.loc["XGBoost + Temp", "mape"] if "XGBoost + Temp" in metrics.index else None

ins1, ins2, ins3 = st.columns(3)

with ins1:
    if xgb_mape and xgb_temp_mape:
        further = (xgb_mape - xgb_temp_mape) / xgb_mape * 100
        body = (
            f"Adding temperature cut MAPE from {xgb_mape:.1f}% to {xgb_temp_mape:.1f}% "
            f"(a further {further:.0f}% reduction). "
            f"Air conditioning load on a hot day can swing demand by 1,500-2,000 MW."
        )
    else:
        body = "Adding temperature as a feature significantly reduces MAPE."
    st.success(f"**Temperature matters**\n\n{body}")

with ins2:
    st.warning(
        "**ARIMA beats naive at its own resolution**\n\n"
        "ARIMA and SARIMA were fit on daily data - their 30-min scores are penalised by "
        "intra-day variation a flat daily forecast cannot represent. At daily granularity "
        "ARIMA achieves 12.3% MAPE, outperforming the naive baseline. A fair apples-to-apples "
        "comparison requires the daily metrics table."
    )

with ins3:
    st.info(
        "**Public holidays are the clearest remaining error**\n\n"
        "Christmas (25 Dec) and Boxing Day (26 Dec) fall mid-week in 2025 but demand "
        "collapses to Sunday-equivalent levels. All models over-predict by 1,000-2,000 MW "
        "on these days - directly addressable with a public holiday indicator feature."
    )

st.divider()
st.caption("Data: AEMO NSW1 Price and Demand | Temperature: Open-Meteo (ERA5)")
