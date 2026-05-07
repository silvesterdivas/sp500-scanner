"""Bulk price + technicals download via yfinance, cached to Parquet.

Resilient to transient failures with exponential backoff. Splits big universes
into chunks to stay under Yahoo's per-request limits.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
PRICES_PATH = CACHE_DIR / "prices.parquet"
TECHNICALS_PATH = CACHE_DIR / "technicals.parquet"
PRICE_REFRESH_HOURS = 4

CHUNK_SIZE = 50  # tickers per yfinance call
MAX_RETRIES = 3
BACKOFF_SECONDS = 5


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def _download_chunk(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Download a chunk of tickers with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(
                tickers=tickers, start=start, end=end, interval="1d",
                group_by="ticker", auto_adjust=False, threads=True, progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("Empty response")
            return raw
        except Exception as e:
            wait = BACKOFF_SECONDS * (2 ** attempt)
            print(f"  Chunk failed (attempt {attempt+1}/{MAX_RETRIES}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    print(f"  Chunk failed after {MAX_RETRIES} retries; skipping.")
    return None


def download_prices(
    tickers: list[str],
    years: int = 3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV. Returns long-form DataFrame."""
    if not force_refresh and not _is_stale(PRICES_PATH, PRICE_REFRESH_HOURS):
        return pd.read_parquet(PRICES_PATH)

    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    print(f"Downloading {len(tickers)} tickers, {years}y history (chunks of {CHUNK_SIZE})...")
    rows = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        print(f"  [{i+1}-{i+len(chunk)}] / {len(tickers)}")
        raw = _download_chunk(chunk, start_s, end_s)
        if raw is None:
            continue
        for t in chunk:
            try:
                if t in raw.columns.get_level_values(0):
                    sub = raw[t].copy()
                else:
                    sub = None
            except Exception:
                sub = None
            if sub is None or sub.empty or sub["Close"].isna().all():
                continue
            sub = sub.dropna(subset=["Close"]).reset_index()
            sub.columns = [c.lower().replace(" ", "_") if isinstance(c, str) else c for c in sub.columns]
            sub["ticker"] = t
            rows.append(sub)

        # gentle pacing between chunks to avoid rate limits
        if i + CHUNK_SIZE < len(tickers):
            time.sleep(1)

    if not rows:
        # Fallback: try cached version if it exists
        if PRICES_PATH.exists():
            print("Download failed; loading from cache.")
            return pd.read_parquet(PRICES_PATH)
        raise RuntimeError(
            "No price data downloaded. Yahoo may be rate-limiting your IP. "
            "Wait a few minutes and retry, or run from a different network."
        )

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Keep only the columns we need
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_parquet(PRICES_PATH, index=False)
    print(f"Cached {len(df):,} price rows for {df['ticker'].nunique()} tickers")
    return df


def compute_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker technical indicators on the latest day."""
    out = []
    for t, g in prices.groupby("ticker"):
        if len(g) < 200:
            continue
        g = g.sort_values("date").copy()
        close = g["close"]
        vol = g["volume"]

        rsi = RSIIndicator(close=close, window=14).rsi()
        macd_obj = MACD(close=close)

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]

        ret_5 = close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 6 else np.nan
        ret_20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 21 else np.nan
        ret_60 = close.iloc[-1] / close.iloc[-61] - 1 if len(close) > 61 else np.nan

        vol_mean_20 = vol.tail(20).mean()
        vol_std_20 = vol.tail(20).std()
        vol_z = (vol.iloc[-1] - vol_mean_20) / vol_std_20 if vol_std_20 > 0 else 0

        out.append(
            {
                "ticker": t,
                "last_close": float(close.iloc[-1]),
                "sma_20": float(sma_20) if not pd.isna(sma_20) else float(close.iloc[-1]),
                "sma_50": float(sma_50) if not pd.isna(sma_50) else float(close.iloc[-1]),
                "sma_200": float(sma_200) if not pd.isna(sma_200) else float(close.iloc[-1]),
                "rsi_14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                "macd": float(macd_obj.macd().iloc[-1]) if not pd.isna(macd_obj.macd().iloc[-1]) else 0,
                "macd_signal": float(macd_obj.macd_signal().iloc[-1]) if not pd.isna(macd_obj.macd_signal().iloc[-1]) else 0,
                "return_5d": float(ret_5) if not pd.isna(ret_5) else 0,
                "return_20d": float(ret_20) if not pd.isna(ret_20) else 0,
                "return_60d": float(ret_60) if not pd.isna(ret_60) else 0,
                "vol_z_20d": float(vol_z),
                "dist_to_sma_200_pct": float((close.iloc[-1] - sma_200) / sma_200) if sma_200 else 0,
                "above_sma_200": bool(close.iloc[-1] > sma_200) if sma_200 else False,
            }
        )

    df = pd.DataFrame(out)
    df.to_parquet(TECHNICALS_PATH, index=False)
    return df


def load_or_refresh(tickers: list[str], force_refresh: bool = False):
    prices = download_prices(tickers, force_refresh=force_refresh)
    if force_refresh or _is_stale(TECHNICALS_PATH, PRICE_REFRESH_HOURS):
        technicals = compute_technicals(prices)
    else:
        technicals = pd.read_parquet(TECHNICALS_PATH)
    return prices, technicals


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.universe import get_tickers

    tickers = get_tickers()[:30]
    prices, tech = load_or_refresh(tickers, force_refresh=True)
    print(f"Prices: {len(prices)} rows for {prices['ticker'].nunique()} tickers")
    print(tech.head())
