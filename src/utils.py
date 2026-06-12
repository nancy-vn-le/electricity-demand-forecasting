"""Shared utilities for the AEMO demand forecasting project.

Metrics and plot helpers used across all model sections of the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HALF_HOURS_PER_DAY = 48       # 30-min intervals in one day
HALF_HOURS_PER_WEEK = 336     # 30-min intervals in one week (48 * 7)
TEST_WEEKS = 4                 # Number of weeks held out as test set
TEST_PERIODS = HALF_HOURS_PER_WEEK * TEST_WEEKS  # 1344 points

# XGBoost lag feature configuration
LAG_PERIODS = [
    HALF_HOURS_PER_DAY,           # 24 hours ago
    HALF_HOURS_PER_DAY * 2,       # 48 hours ago
    HALF_HOURS_PER_WEEK,          # 1 week ago
    HALF_HOURS_PER_WEEK * 2,      # 2 weeks ago
]
ROLLING_WINDOW = HALF_HOURS_PER_DAY  # 24-hour rolling statistics

# Plot styling
FIGSIZE_WIDE = (14, 4)
FIGSIZE_MEDIUM = (12, 4)
FIGSIZE_TALL = (14, 8)
COLOUR_TRAIN = "#2196F3"      # blue
COLOUR_TEST = "#FF9800"       # orange
COLOUR_NAIVE = "#9E9E9E"      # grey
COLOUR_ARIMA = "#E91E63"      # pink
COLOUR_SARIMA = "#9C27B0"     # purple
COLOUR_HW = "#4CAF50"         # green
COLOUR_XGB = "#F44336"        # red


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    actual : pd.Series
        Observed demand values (MW).
    predicted : pd.Series
        Model forecast values (MW).

    Returns
    -------
    float
        RMSE in the same units as the input (MW).
    """
    return float(np.sqrt(np.mean((actual.values - predicted.values) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : pd.Series
        Observed demand values (MW). Must not contain zeros.
    predicted : pd.Series
        Model forecast values (MW).

    Returns
    -------
    float
        MAPE as a percentage (e.g., 3.5 means 3.5%).

    Notes
    -----
    MAPE is undefined when actual == 0. Electricity demand is always
    positive so this is not an issue for this dataset.
    """
    return float(np.mean(np.abs((actual.values - predicted.values) / actual.values)) * 100)


def print_metrics(model_name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    """Calculate and print RMSE and MAPE for a model.

    Parameters
    ----------
    model_name : str
        Display name for the model (shown in output).
    actual : pd.Series
        Observed demand values.
    predicted : pd.Series
        Model forecast values.

    Returns
    -------
    dict
        Dictionary with keys 'model', 'rmse', 'mape'.
    """
    r = rmse(actual, predicted)
    m = mape(actual, predicted)
    print(f"{model_name:30s}  RMSE: {r:7.1f} MW   MAPE: {m:.2f}%")
    return {"model": model_name, "rmse": r, "mape": m}


# ---------------------------------------------------------------------------
# Feature Engineering (for XGBoost)
# ---------------------------------------------------------------------------

def make_lag_features(series: pd.Series) -> pd.DataFrame:
    """Create lag and calendar features for supervised ML forecasting.

    Features are created with .shift() so each row at time t only uses
    data from t-k (no data leakage into the future).

    Parameters
    ----------
    series : pd.Series
        Half-hourly demand series with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Feature matrix including lag columns, rolling statistics,
        and calendar variables. Rows with NaN (from lagging) are dropped.
    """
    df = pd.DataFrame({"demand": series})

    # Lag features — past demand values as predictors
    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = series.shift(lag)

    # Rolling statistics — capture recent demand level and volatility
    # NOTE: shift(1) ensures the window does not include the current period
    df[f"rolling_mean_{ROLLING_WINDOW}"] = (
        series.shift(1).rolling(ROLLING_WINDOW).mean()
    )
    df[f"rolling_std_{ROLLING_WINDOW}"] = (
        series.shift(1).rolling(ROLLING_WINDOW).std()
    )

    # Calendar features — capture systematic daily/weekly/seasonal patterns
    df["hour"] = series.index.hour
    df["dayofweek"] = series.index.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = series.index.month
    df["quarter"] = series.index.quarter
    df["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
    df["halfhour"] = series.index.hour * 2 + series.index.minute // 30

    # Drop rows where lag features are NaN (beginning of series)
    return df.dropna()


# ---------------------------------------------------------------------------
# Plot Helpers
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    actual: pd.Series,
    forecasts: dict,
    title: str = "Forecast vs Actual",
    zoom_weeks: int = 2,
) -> None:
    """Plot model forecasts against actual demand over the test period.

    Parameters
    ----------
    actual : pd.Series
        Actual demand over the test period.
    forecasts : dict
        Mapping of model name to forecast Series (same index as actual).
    title : str
        Chart title.
    zoom_weeks : int
        Number of weeks to show in the zoomed panel (default 2).
    """
    colour_map = {
        "Naive Baseline": COLOUR_NAIVE,
        "ARIMA": COLOUR_ARIMA,
        "SARIMA": COLOUR_SARIMA,
        "Holt-Winters": COLOUR_HW,
        "XGBoost": COLOUR_XGB,
    }

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL)

    # Full test period
    ax = axes[0]
    ax.plot(actual.index, actual.values, color="black", lw=1.5, label="Actual", zorder=5)
    for name, forecast in forecasts.items():
        ax.plot(forecast.index, forecast.values,
                color=colour_map.get(name, "steelblue"), lw=1, alpha=0.8, label=name)
    ax.set_title(f"{title} — Full Test Period ({TEST_WEEKS} weeks)")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Zoomed panel — first N weeks
    zoom_end = actual.index[0] + pd.Timedelta(weeks=zoom_weeks)
    ax = axes[1]
    mask = actual.index <= zoom_end
    ax.plot(actual.index[mask], actual.values[mask],
            color="black", lw=1.5, label="Actual", zorder=5)
    for name, forecast in forecasts.items():
        fmask = forecast.index <= zoom_end
        ax.plot(forecast.index[fmask], forecast.values[fmask],
                color=colour_map.get(name, "steelblue"), lw=1, alpha=0.8, label=name)
    ax.set_title(f"Zoomed — First {zoom_weeks} Weeks of Test Period")
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    plt.tight_layout()
    plt.savefig("../outputs/figures/forecast_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
