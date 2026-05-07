"""Generate synthetic S&P 500-like data for end-to-end pipeline testing.

Use when external APIs (Yahoo, Finnhub) are unavailable / rate-limited.
Generates realistic-ish prices, fundamentals, and news so we can verify the
scoring engine, backtest framework, and dashboard render correctly.

Run: python scripts/synth_data.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.universe import get_tickers, load_universe
from data.prices import compute_technicals, PRICES_PATH, TECHNICALS_PATH
from data.fundamentals import FUND_PATH, FIELDS
from data.news import NEWS_PATH

rng = np.random.default_rng(42)


def gen_prices(tickers: list[str], years: int = 3) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)
    days = pd.bdate_range(start, end)
    rows = []
    for t in tickers:
        # Random starting price
        p0 = rng.uniform(20, 400)
        # Random drift + volatility per ticker
        mu = rng.normal(0.0003, 0.0002)  # daily expected return ~0.03%
        sigma = rng.uniform(0.012, 0.030)
        rets = rng.normal(mu, sigma, size=len(days))
        prices = p0 * np.exp(np.cumsum(rets))
        # Generate OHLC around close
        opens = prices * (1 + rng.normal(0, 0.003, size=len(days)))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.005, size=len(days))))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.005, size=len(days))))
        # Volume with surges
        base_vol = rng.uniform(1e6, 5e7)
        vols = base_vol * np.exp(rng.normal(0, 0.3, size=len(days)))
        df = pd.DataFrame({
            "date": days, "ticker": t,
            "open": opens, "high": highs, "low": lows, "close": prices,
            "adj_close": prices, "volume": vols.astype(int),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def gen_fundamentals(tickers: list[str], universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sector_map = dict(zip(universe["symbol"], universe["gics_sector"]))
    for t in tickers:
        row = {"ticker": t}
        for f in FIELDS:
            if f == "sector":
                row[f] = sector_map.get(t, "Unknown")
            elif f == "industry":
                row[f] = "—"
            elif f == "trailingPE":
                row[f] = float(rng.uniform(8, 60))
            elif f == "forwardPE":
                row[f] = float(rng.uniform(8, 50))
            elif f == "pegRatio":
                row[f] = float(rng.uniform(0.5, 4))
            elif f == "priceToBook":
                row[f] = float(rng.uniform(0.5, 15))
            elif f == "priceToSalesTrailing12Months":
                row[f] = float(rng.uniform(0.5, 20))
            elif f == "enterpriseToEbitda":
                row[f] = float(rng.uniform(5, 40))
            elif f == "returnOnEquity":
                row[f] = float(rng.normal(0.18, 0.15))
            elif f == "returnOnAssets":
                row[f] = float(rng.normal(0.08, 0.06))
            elif f == "profitMargins":
                row[f] = float(rng.normal(0.12, 0.10))
            elif f == "operatingMargins":
                row[f] = float(rng.normal(0.18, 0.12))
            elif f == "grossMargins":
                row[f] = float(rng.uniform(0.20, 0.70))
            elif f == "revenueGrowth":
                row[f] = float(rng.normal(0.08, 0.12))
            elif f == "earningsGrowth":
                row[f] = float(rng.normal(0.10, 0.20))
            elif f == "earningsQuarterlyGrowth":
                row[f] = float(rng.normal(0.10, 0.30))
            elif f == "debtToEquity":
                row[f] = float(rng.uniform(20, 250))
            elif f == "currentRatio":
                row[f] = float(rng.uniform(0.8, 4))
            elif f == "freeCashflow":
                row[f] = float(rng.uniform(1e8, 5e10))
            elif f == "marketCap":
                row[f] = float(rng.uniform(5e9, 3e12))
            elif f == "dividendYield":
                row[f] = float(rng.uniform(0, 0.05))
            elif f == "beta":
                row[f] = float(rng.uniform(0.5, 1.8))
        # FCF yield
        fcf = row.get("freeCashflow")
        mcap = row.get("marketCap")
        row["fcf_yield"] = (fcf / mcap) if (fcf and mcap and mcap > 0) else None
        rows.append(row)
    return pd.DataFrame(rows)


HEADLINES_POS = [
    "{} reports record quarterly earnings, beats expectations",
    "{} announces major partnership; stock surges",
    "Analysts upgrade {} to Buy on strong outlook",
    "{} unveils new product line, market reacts positively",
    "{} raises guidance, signals robust growth ahead",
]
HEADLINES_NEG = [
    "{} misses earnings; shares slide on weak guidance",
    "Investigation into {} weighs on shares",
    "{} faces lawsuit over recent business practices",
    "Analysts downgrade {} on margin concerns",
    "{} announces layoffs amid restructuring",
]
HEADLINES_NEU = [
    "{} CEO speaks at industry conference",
    "{} reports quarterly results in line with estimates",
    "{} continues expansion in international markets",
]


def gen_news(tickers: list[str]) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    end = datetime.now()
    rows = []
    for t in tickers:
        n = rng.integers(2, 8)
        for _ in range(n):
            kind = rng.choice(["pos", "neg", "neu"], p=[0.4, 0.2, 0.4])
            templates = {"pos": HEADLINES_POS, "neg": HEADLINES_NEG, "neu": HEADLINES_NEU}[kind]
            headline = rng.choice(templates).format(t)
            sentiment = analyzer.polarity_scores(headline)["compound"]
            dt = end - timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
            rows.append({
                "ticker": t, "datetime": dt, "headline": headline,
                "source": "Synthetic Press", "url": f"https://example.com/{t}",
                "sentiment": sentiment,
            })
    return pd.DataFrame(rows)


def main():
    universe = load_universe()
    tickers = universe["symbol"].tolist()
    print(f"Generating synthetic data for {len(tickers)} tickers...")

    print("  Prices...")
    prices = gen_prices(tickers, years=3)
    prices.to_parquet(PRICES_PATH, index=False)
    print(f"    {len(prices):,} rows -> {PRICES_PATH}")

    print("  Technicals...")
    tech = compute_technicals(prices)
    print(f"    {len(tech)} rows -> {TECHNICALS_PATH}")

    print("  Fundamentals...")
    fund = gen_fundamentals(tickers, universe)
    fund.to_parquet(FUND_PATH, index=False)
    print(f"    {len(fund)} rows -> {FUND_PATH}")

    print("  News + sentiment...")
    news = gen_news(tickers)
    news.to_parquet(NEWS_PATH, index=False)
    print(f"    {len(news)} rows -> {NEWS_PATH}")

    print("\n✅ Synthetic dataset ready.")


if __name__ == "__main__":
    main()
