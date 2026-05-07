"""Composite scoring: short-term (momentum/news) and long-term (value/quality)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _z(s: pd.Series) -> pd.Series:
    """Robust z-score (clips outliers to +/- 3 sigma)."""
    s = pd.to_numeric(s, errors="coerce")
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - mean) / std).clip(-3, 3).fillna(0)


# ---------- SHORT TERM ----------

SHORT_TERM_WEIGHTS = {
    "momentum": 0.35,    # 5d + 20d return
    "rsi_signal": 0.15,  # rsi 50-70 sweet spot, > 70 overbought
    "vol_surge": 0.15,   # volume z-score
    "news_sentiment": 0.25,
    "trend": 0.10,       # above 50-day SMA
}


def short_term_factors(technicals: pd.DataFrame, sentiment_agg: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scored short-term factors per ticker."""
    df = technicals.copy()
    if not sentiment_agg.empty:
        df = df.merge(sentiment_agg, on="ticker", how="left")
    df["avg_sentiment"] = df.get("avg_sentiment", pd.Series(0)).fillna(0)
    df["n_articles"] = df.get("n_articles", pd.Series(0)).fillna(0)

    # Momentum: blend of 5d (50%) + 20d (50%)
    momentum_raw = 0.5 * df["return_5d"] + 0.5 * df["return_20d"]
    df["f_momentum"] = _z(momentum_raw)

    # RSI signal: peak score around 60 (strong but not overbought), penalize > 75 and < 30
    def rsi_score(r):
        if pd.isna(r):
            return 0
        # Bell curve centered at 60
        return -((r - 60) ** 2) / 400  # peaks at 60, falls off
    df["f_rsi"] = _z(df["rsi_14"].apply(rsi_score))

    # Volume surge
    df["f_vol"] = _z(df["vol_z_20d"])

    # News sentiment (weighted by article count to penalize 1-article scores)
    sentiment_w = df["avg_sentiment"] * np.log1p(df["n_articles"])
    df["f_sentiment"] = _z(sentiment_w)

    # Trend (above 50-day SMA, weighted by distance)
    df["pct_above_sma_50"] = (df["last_close"] - df["sma_50"]) / df["sma_50"]
    df["f_trend"] = _z(df["pct_above_sma_50"])

    return df


def short_term_score(technicals: pd.DataFrame, sentiment_agg: pd.DataFrame) -> pd.DataFrame:
    df = short_term_factors(technicals, sentiment_agg)
    w = SHORT_TERM_WEIGHTS
    df["score"] = (
        w["momentum"] * df["f_momentum"]
        + w["rsi_signal"] * df["f_rsi"]
        + w["vol_surge"] * df["f_vol"]
        + w["news_sentiment"] * df["f_sentiment"]
        + w["trend"] * df["f_trend"]
    )
    return df.sort_values("score", ascending=False)


# ---------- LONG TERM ----------

LONG_TERM_WEIGHTS = {
    "valuation": 0.25,
    "growth": 0.25,
    "quality": 0.25,
    "balance_sheet": 0.10,
    "trend_200d": 0.15,
}


def long_term_factors(technicals: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scored long-term factors per ticker."""
    df = technicals.merge(fundamentals, on="ticker", how="inner")

    # Valuation: lower is better (cheaper). Use forwardPE preferred, fall back to trailingPE.
    df["pe"] = df["forwardPE"].fillna(df["trailingPE"])
    # Cap absurd PEs (negatives or > 100) to neutralize
    df["pe_clean"] = df["pe"].where((df["pe"] > 0) & (df["pe"] < 100), np.nan)
    df["f_valuation"] = _z(-df["pe_clean"])  # negate so cheaper -> higher z

    # Growth: revenue + earnings growth
    df["growth_blend"] = 0.5 * df["revenueGrowth"].fillna(0) + 0.5 * df["earningsGrowth"].fillna(0)
    df["f_growth"] = _z(df["growth_blend"])

    # Quality: ROE * profit margin
    df["quality_blend"] = (
        0.5 * df["returnOnEquity"].fillna(0)
        + 0.3 * df["profitMargins"].fillna(0)
        + 0.2 * df["operatingMargins"].fillna(0)
    )
    df["f_quality"] = _z(df["quality_blend"])

    # Balance sheet: low debt, healthy current ratio
    df["debt_inv"] = -df["debtToEquity"].fillna(df["debtToEquity"].median())
    df["f_balance"] = _z(df["debt_inv"])

    # Long-term trend (200d)
    df["f_trend_200"] = _z(df["dist_to_sma_200_pct"])

    return df


def long_term_score(technicals: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = long_term_factors(technicals, fundamentals)
    w = LONG_TERM_WEIGHTS
    df["score"] = (
        w["valuation"] * df["f_valuation"]
        + w["growth"] * df["f_growth"]
        + w["quality"] * df["f_quality"]
        + w["balance_sheet"] * df["f_balance"]
        + w["trend_200d"] * df["f_trend_200"]
    )
    return df.sort_values("score", ascending=False)


# ---------- TOP-N HELPERS ----------

def top_n_short(technicals: pd.DataFrame, sentiment_agg: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df = short_term_score(technicals, sentiment_agg)
    return df.head(n).reset_index(drop=True)


def top_n_long(technicals: pd.DataFrame, fundamentals: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df = long_term_score(technicals, fundamentals)
    return df.head(n).reset_index(drop=True)
