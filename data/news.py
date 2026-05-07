"""News headlines + VADER sentiment, sourced from Finnhub free tier."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
NEWS_PATH = CACHE_DIR / "news.parquet"
NEWS_REFRESH_HOURS = 6

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

_analyzer = SentimentIntensityAnalyzer()


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age = (time.time() - path.stat().st_mtime) / 3600
    return age > max_age_hours


def _fetch_company_news(ticker: str, days: int = 7) -> list[dict]:
    """Hit Finnhub /company-news for the last `days` days."""
    if not FINNHUB_KEY:
        return []
    end = datetime.now().date()
    start = end - timedelta(days=days)
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": FINNHUB_KEY,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return []
        items = r.json() or []
        return items[:10]  # cap per ticker
    except Exception:
        return []


def _score_headline(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    if not text:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])


def fetch_news(
    tickers: list[str],
    force_refresh: bool = False,
    max_workers: int = 8,
    days: int = 7,
) -> pd.DataFrame:
    """Fetch headlines + VADER scores. Returns long-form DataFrame.

    Columns: ticker, datetime, headline, source, url, sentiment.
    Aggregates can be computed via aggregate_sentiment() below.
    """
    if not FINNHUB_KEY:
        print("WARNING: FINNHUB_API_KEY not set; returning empty news data")
        return pd.DataFrame(columns=["ticker", "datetime", "headline", "source", "url", "sentiment"])

    if not force_refresh and not _is_stale(NEWS_PATH, NEWS_REFRESH_HOURS):
        return pd.read_parquet(NEWS_PATH)

    print(f"Fetching news for {len(tickers)} tickers (Finnhub, {days}d window)...")
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_company_news, t, days): t for t in tickers}
        for i, fut in enumerate(as_completed(futures)):
            t = futures[fut]
            items = fut.result() or []
            for it in items:
                headline = it.get("headline", "")
                rows.append(
                    {
                        "ticker": t,
                        "datetime": pd.to_datetime(it.get("datetime", 0), unit="s"),
                        "headline": headline,
                        "source": it.get("source", ""),
                        "url": it.get("url", ""),
                        "sentiment": _score_headline(headline),
                    }
                )
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tickers)} done")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(NEWS_PATH, index=False)
    print(f"Cached {len(df)} news items")
    return df


def aggregate_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker aggregate: avg_sentiment, n_articles, frac_positive."""
    if news.empty:
        return pd.DataFrame(columns=["ticker", "avg_sentiment", "n_articles", "frac_positive"])
    g = news.groupby("ticker")
    out = pd.DataFrame(
        {
            "avg_sentiment": g["sentiment"].mean(),
            "n_articles": g["sentiment"].size(),
            "frac_positive": g["sentiment"].apply(lambda s: (s > 0.05).mean()),
        }
    ).reset_index()
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.universe import get_tickers

    tickers = get_tickers()[:5]
    news = fetch_news(tickers, force_refresh=True)
    print(news.head())
    print(aggregate_sentiment(news))
