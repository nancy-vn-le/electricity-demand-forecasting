"""Download AEMO aggregated price and demand data for NSW.

AEMO publishes one CSV file per month at a predictable public URL. Each file
contains 30-minute settlement period data with TOTALDEMAND (MW) and RRP
(regional reference price, $/MWh). No authentication or ZIP extraction needed.

Data source:
    https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_NSW1.csv

Usage:
    python src/download_aemo.py

Output:
    data/raw/nsw1_price_and_demand.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION = "NSW1"
START_YEAR = 2019
END_YEAR = 2025  # inclusive
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "nsw1_price_and_demand.csv"

# One CSV per month — no ZIP, no authentication
AEMO_URL_TEMPLATE = (
    "https://aemo.com.au/aemo/data/nem/priceanddemand/"
    "PRICE_AND_DEMAND_{year}{month:02d}_NSW1.csv"
)

# AEMO publishes NEM times in local time (AEST/AEDT for NSW).
# We localise to Australia/Sydney so DST transitions are handled correctly:
#   - 'ambiguous=infer'         handles the April fall-back (hour repeats)
#   - 'nonexistent=shift_forward' handles the October spring-forward (hour skipped)
TZ_NSW = "Australia/Sydney"

# Columns present in AEMO price-and-demand CSV files
# NOTE: the column is named REGION (not REGIONID) in these files
EXPECTED_COLUMNS = {"SETTLEMENTDATE", "REGION", "TOTALDEMAND", "RRP"}


def download_month(session: requests.Session, year: int, month: int) -> pd.DataFrame | None:
    """Download and parse one month of AEMO price and demand data.

    Parameters
    ----------
    session : requests.Session
        Reused HTTP session (faster than opening a new connection each call).
    year : int
        Year to download.
    month : int
        Month number (1–12).

    Returns
    -------
    pd.DataFrame or None
        Raw DataFrame with string SETTLEMENTDATE column, or None on failure.
    """
    url = AEMO_URL_TEMPLATE.format(year=year, month=month)
    print(f"  {year}-{month:02d}  {url.split('/')[-1]} ... ", end="", flush=True)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"HTTP {response.status_code} — skipping")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"network error ({exc}) — skipping")
        return None

    try:
        df = pd.read_csv(pd.io.common.StringIO(response.text))
    except Exception as exc:
        print(f"parse error ({exc}) — skipping")
        return None

    # Normalise column names — some files use lowercase or extra whitespace
    df.columns = df.columns.str.strip().str.upper()

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        print(f"missing columns {missing} — skipping")
        return None

    df = df[list(EXPECTED_COLUMNS)].copy()
    df = df[df["REGION"] == REGION]

    if df.empty:
        print(f"no {REGION} rows — skipping")
        return None

    print(f"{len(df):,} rows")
    return df


def parse_and_combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate monthly frames, parse datetimes, and set the index.

    Parameters
    ----------
    frames : list of pd.DataFrame
        Raw monthly DataFrames from download_month().

    Returns
    -------
    pd.DataFrame
        Combined DataFrame indexed by a timezone-aware DatetimeIndex at
        30-minute resolution, sorted chronologically, with TOTALDEMAND
        and RRP columns.

    Notes
    -----
    AEMO switched from 30-minute to 5-minute settlement intervals in
    October 2021 (Five Minute Settlement reform). Files from Oct 2021
    onwards contain 5-minute rows; we resample to 30-minute means so the
    full series has a consistent frequency throughout.
    """
    combined = pd.concat(frames, ignore_index=True)

    # Convert to numeric before any aggregation
    combined["TOTALDEMAND"] = pd.to_numeric(combined["TOTALDEMAND"], errors="coerce")
    combined["RRP"] = pd.to_numeric(combined["RRP"], errors="coerce")

    # AEMO date format: "YYYY/MM/DD HH:MM:SS"
    combined["SETTLEMENTDATE"] = pd.to_datetime(
        combined["SETTLEMENTDATE"],
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    n_unparsed = combined["SETTLEMENTDATE"].isna().sum()
    if n_unparsed > 0:
        print(f"WARN: {n_unparsed} rows had unparseable dates and were dropped.")
    combined = combined.dropna(subset=["SETTLEMENTDATE"])

    # Attach NSW timezone — localise, not convert (timestamps are already local).
    # DST fall-back in April creates one ambiguous hour where the same wall-clock
    # time appears twice. We mark those as NaT and drop the 1–2 affected rows
    # rather than guessing which occurrence is pre/post transition.
    combined["SETTLEMENTDATE"] = combined["SETTLEMENTDATE"].dt.tz_localize(
        TZ_NSW,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    n_ambiguous = combined["SETTLEMENTDATE"].isna().sum()
    if n_ambiguous > 0:
        print(f"INFO: Dropped {n_ambiguous} rows at DST transition boundaries.")
    combined = combined.dropna(subset=["SETTLEMENTDATE"])

    combined = (
        combined
        .set_index("SETTLEMENTDATE")
        .sort_index()[["TOTALDEMAND", "RRP"]]
    )

    # Drop exact duplicate timestamps (can occur at month boundaries)
    combined = combined[~combined.index.duplicated(keep="first")]

    # Resample to 30-minute means to normalise the pre/post Five Minute
    # Settlement eras into one consistent frequency.
    combined = combined.resample("30min").mean()

    # Drop any empty 30-min bins produced by resampling (none expected)
    combined = combined.dropna(how="all")

    return combined


def check_interval_regularity(df: pd.DataFrame) -> None:
    """Print a warning for any gap larger than 90 minutes in the index."""
    diffs = df.index.to_series().diff().dropna()
    large_gaps = diffs[diffs > pd.Timedelta("90min")]
    if large_gaps.empty:
        print("Interval check: all gaps ≤ 90 min — looks clean.")
    else:
        print(f"WARN: {len(large_gaps)} gaps > 90 min (expected around DST transitions):")
        for ts, gap in large_gaps.head(5).items():
            print(f"  {ts}  →  gap = {gap}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_months = (END_YEAR - START_YEAR + 1) * 12
    print(f"Downloading AEMO {REGION} price and demand data")
    print(f"Period : {START_YEAR}-01 to {END_YEAR}-12  ({total_months} files)")
    print(f"Output : {OUTPUT_FILE}\n")

    frames = []
    t_start = time.time()

    # Reuse one HTTP session — avoids re-establishing TLS for each request
    with requests.Session() as session:
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"Year {year}:")
            for month in range(1, 13):
                df = download_month(session, year, month)
                if df is not None:
                    frames.append(df)

    if not frames:
        print("\nNo data downloaded. Check your internet connection.")
        sys.exit(1)

    print(f"\nCombining {len(frames)} monthly files ...")
    combined = parse_and_combine(frames)

    check_interval_regularity(combined)

    combined.to_csv(OUTPUT_FILE)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"Saved         : {OUTPUT_FILE}")
    print(f"Rows          : {len(combined):,}")
    print(f"Date range    : {combined.index.min()}  →  {combined.index.max()}")
    print(f"Demand range  : {combined['TOTALDEMAND'].min():.0f} – {combined['TOTALDEMAND'].max():.0f} MW")
    print(f"RRP range     : ${combined['RRP'].min():.2f} – ${combined['RRP'].max():.2f} /MWh")
    print(f"Elapsed       : {elapsed:.0f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
