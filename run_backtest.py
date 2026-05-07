"""One-time backtest runner. Caches results for the dashboard's Backtest tab.

Usage:
    python run_backtest.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from data.prices import download_prices
from data.universe import get_tickers
from backtest.runner import run_backtest, save_backtest


def fetch_spy_prices(years: int = 3) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)
    df = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), auto_adjust=False, progress=False)
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df[["date", "close"]]


def main():
    print("Loading universe + prices...")
    tickers = get_tickers()
    prices = download_prices(tickers, years=3)
    spy = fetch_spy_prices(years=3)

    print("\n=== Running SHORT-TERM backtest (weekly rebal., 5-day hold) ===")
    short_result = run_backtest(
        prices, spy, horizon="short",
        rebalance_days=5, hold_days=5, top_n=10,
        start=(datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d"),
    )
    print("Metrics:", short_result["metrics"])
    save_backtest(short_result, "short")

    print("\n=== Running LONG-TERM backtest (quarterly rebal., 90-day hold) ===")
    long_result = run_backtest(
        prices, spy, horizon="long",
        rebalance_days=63, hold_days=63, top_n=10,
        start=(datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d"),
    )
    print("Metrics:", long_result["metrics"])
    save_backtest(long_result, "long")

    print("\nDone. Restart the Streamlit app to see results in the Backtest tab.")


if __name__ == "__main__":
    main()
