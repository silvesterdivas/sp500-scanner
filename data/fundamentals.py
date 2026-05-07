"""Fundamentals via yfinance .info — P/E, growth, ROE, etc."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FUND_PATH = CACHE_DIR / "fundamentals.parquet"
FUND_REFRESH_HOURS = 24

FIELDS = [
    "trailingPE",
    "forwardPE",
    "pegRatio",
    "priceToBook",
    "priceToSalesTrailing12Months",
    "enterpriseToEbitda",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "grossMargins",
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
    "marketCap",
    "dividendYield",
    "beta",
    "sector",
    "industry",
]


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age = (time.time() - path.stat().st_mtime) / 3600
    return age > max_age_hours


def _fetch_one(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        row = {"ticker": ticker}
        for f in FIELDS:
            row[f] = info.get(f)
        # FCF yield = freeCashflow / marketCap
        fcf = row.get("freeCashflow")
        mcap = row.get("marketCap")
        row["fcf_yield"] = (fcf / mcap) if (fcf and mcap and mcap > 0) else None
        return row
    except Exception:
        return None


def fetch_fundamentals(
    tickers: list[str],
    force_refresh: bool = False,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch fundamentals for all tickers, with thread parallelism + caching."""
    if not force_refresh and not _is_stale(FUND_PATH, FUND_REFRESH_HOURS):
        return pd.read_parquet(FUND_PATH)

    print(f"Fetching fundamentals for {len(tickers)} tickers...")
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r:
                rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tickers)} done")

    df = pd.DataFrame(rows)
    df.to_parquet(FUND_PATH, index=False)
    print(f"Cached fundamentals for {len(df)} tickers")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.universe import get_tickers

    tickers = get_tickers()[:20]
    df = fetch_fundamentals(tickers, force_refresh=True)
    print(df[["ticker", "trailingPE", "returnOnEquity", "revenueGrowth", "sector"]].head())
