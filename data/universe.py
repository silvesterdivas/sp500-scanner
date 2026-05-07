"""Fetch and cache S&P 500 constituents from Wikipedia."""
from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
UNIVERSE_PATH = CACHE_DIR / "sp500_universe.parquet"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (compatible; sp500-agent/1.0; research dashboard)"
REFRESH_DAYS = 7


def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > max_age_days * 86400


def load_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame of S&P 500 constituents.

    Columns: symbol, security, gics_sector, gics_sub_industry, headquarters,
             date_added, cik, founded.
    """
    if not force_refresh and not _is_stale(UNIVERSE_PATH, REFRESH_DAYS):
        return pd.read_parquet(UNIVERSE_PATH)

    # Wikipedia rejects urllib's default UA; fetch with requests + UA.
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    # First table is the constituents
    df = tables[0].copy()
    df.columns = [
        c.lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    # yfinance uses dashes (BRK-B) where Wikipedia uses dots (BRK.B)
    df["symbol"] = df["symbol"].astype(str).str.replace(".", "-", regex=False)
    df.to_parquet(UNIVERSE_PATH, index=False)
    return df


def get_tickers(force_refresh: bool = False) -> list[str]:
    """Return just the list of tickers."""
    return load_universe(force_refresh)["symbol"].tolist()


if __name__ == "__main__":
    df = load_universe(force_refresh=True)
    print(f"Loaded {len(df)} S&P 500 constituents")
    print(df.head())
    print(f"Cached to: {UNIVERSE_PATH}")
