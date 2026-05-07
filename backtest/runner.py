"""Walk-forward backtest framework. Uses only price-derived signals
(point-in-time fundamentals/news aren't available on free tier).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------- helpers ----------

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - s.mean()) / std).clip(-3, 3).fillna(0)


def _technicals_at(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Compute technicals using only data up to `asof`."""
    df = prices[prices["date"] <= asof].copy()
    rows = []
    for t, g in df.groupby("ticker"):
        if len(g) < 200:
            continue
        g = g.sort_values("date")
        close = g["close"]
        vol = g["volume"]
        if len(close) < 21:
            continue
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        ret_5 = close.iloc[-1] / close.iloc[-6] - 1
        ret_20 = close.iloc[-1] / close.iloc[-21] - 1
        ret_60 = close.iloc[-1] / close.iloc[-61] - 1 if len(close) > 61 else np.nan
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss > 0 else np.inf
        rsi = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100
        vol_z = (vol.iloc[-1] - vol.tail(20).mean()) / vol.tail(20).std() if vol.tail(20).std() > 0 else 0
        rows.append({
            "ticker": t,
            "close": float(close.iloc[-1]),
            "sma_50": float(sma_50),
            "sma_200": float(sma_200),
            "rsi_14": float(rsi),
            "return_5d": float(ret_5),
            "return_20d": float(ret_20),
            "return_60d": float(ret_60) if not pd.isna(ret_60) else 0,
            "vol_z_20d": float(vol_z),
            "dist_sma_200": float((close.iloc[-1] - sma_200) / sma_200) if sma_200 else 0,
        })
    return pd.DataFrame(rows)


def _short_term_score_no_news(tech: pd.DataFrame) -> pd.Series:
    """Re-weighted short-term score without news (since point-in-time news isn't free)."""
    momentum = 0.5 * tech["return_5d"] + 0.5 * tech["return_20d"]
    rsi = tech["rsi_14"].apply(lambda r: -((r - 60) ** 2) / 400)
    pct_above_50 = (tech["close"] - tech["sma_50"]) / tech["sma_50"]
    return (
        0.50 * _z(momentum)
        + 0.20 * _z(rsi)
        + 0.15 * _z(tech["vol_z_20d"])
        + 0.15 * _z(pct_above_50)
    )


def _long_term_score_price_only(tech: pd.DataFrame) -> pd.Series:
    """Long-term score using only price/trend factors for backtest."""
    return (
        0.50 * _z(tech["return_60d"])
        + 0.30 * _z(tech["dist_sma_200"])
        + 0.20 * _z(-((tech["rsi_14"] - 50).abs()))  # prefer healthy mid-range RSI
    )


# ---------- core backtest ----------

def run_backtest(
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
    horizon: str = "short",  # "short" or "long"
    rebalance_days: int = 7,  # short: weekly; long: 90
    hold_days: int = 5,
    top_n: int = 10,
    start: str | None = None,
) -> dict:
    """Walk-forward simulator.

    For each rebalance date:
      - Score the universe using only data <= that date
      - "Buy" equal-weighted top N
      - Hold for hold_days, capture forward return
      - Compare to SPY return over the same period
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    spy_prices = spy_prices.copy()
    spy_prices["date"] = pd.to_datetime(spy_prices["date"])
    spy_close_by_date = dict(zip(spy_prices["date"], spy_prices["close"]))

    all_dates = sorted(prices["date"].unique())
    if start:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]

    # Need at least 200 days of history before first rebalance
    rebalance_dates = []
    for i, d in enumerate(all_dates):
        if i % rebalance_days == 0:
            rebalance_dates.append(d)

    trades = []
    equity = 1.0
    bench_equity = 1.0
    equity_curve = []

    score_fn = _short_term_score_no_news if horizon == "short" else _long_term_score_price_only

    for asof in rebalance_dates:
        # Need a future date hold_days ahead to compute return
        future_date = asof + pd.Timedelta(days=hold_days * 1.5)  # approx, market days
        future_prices = prices[prices["date"] > asof]
        if future_prices.empty:
            break

        tech = _technicals_at(prices, asof)
        if len(tech) < 50:
            continue
        tech["score"] = score_fn(tech)
        picks = tech.nlargest(top_n, "score")["ticker"].tolist()

        # forward return: hold_days market-days ahead
        rets = []
        for t in picks:
            tp = prices[(prices["ticker"] == t) & (prices["date"] >= asof)].sort_values("date")
            if len(tp) < hold_days + 1:
                continue
            entry = tp.iloc[0]["close"]
            exit_ = tp.iloc[min(hold_days, len(tp) - 1)]["close"]
            if entry > 0:
                rets.append(exit_ / entry - 1)
        if not rets:
            continue
        port_ret = float(np.mean(rets))

        # SPY benchmark over same period
        spy_today = spy_prices[spy_prices["date"] >= asof].sort_values("date")
        if len(spy_today) < hold_days + 1:
            break
        spy_entry = spy_today.iloc[0]["close"]
        spy_exit = spy_today.iloc[min(hold_days, len(spy_today) - 1)]["close"]
        spy_ret = spy_exit / spy_entry - 1

        equity *= 1 + port_ret
        bench_equity *= 1 + spy_ret
        trades.append({
            "date": asof,
            "picks": ",".join(picks[:5]) + ("..." if len(picks) > 5 else ""),
            "n_picks": len(picks),
            "portfolio_ret": port_ret,
            "spy_ret": spy_ret,
            "alpha": port_ret - spy_ret,
        })
        equity_curve.append({
            "date": asof,
            "portfolio": equity,
            "spy": bench_equity,
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve)

    # Metrics
    if not trades_df.empty:
        port_rets = trades_df["portfolio_ret"]
        spy_rets = trades_df["spy_ret"]
        # Annualize: roughly 252 trading days / hold_days periods per year
        periods_per_year = 252 / hold_days
        sharpe = (port_rets.mean() / port_rets.std()) * np.sqrt(periods_per_year) if port_rets.std() > 0 else 0
        spy_sharpe = (spy_rets.mean() / spy_rets.std()) * np.sqrt(periods_per_year) if spy_rets.std() > 0 else 0
        # Max drawdown on portfolio equity
        peak = curve_df["portfolio"].cummax()
        dd = (curve_df["portfolio"] - peak) / peak
        max_dd = float(dd.min())
        hit_rate = float((trades_df["alpha"] > 0).mean())

        metrics = {
            "horizon": horizon,
            "n_periods": len(trades_df),
            "total_return": float(equity - 1),
            "spy_total_return": float(bench_equity - 1),
            "alpha_total": float(equity - bench_equity),
            "avg_period_return": float(port_rets.mean()),
            "avg_spy_return": float(spy_rets.mean()),
            "avg_alpha": float(trades_df["alpha"].mean()),
            "sharpe": float(sharpe),
            "spy_sharpe": float(spy_sharpe),
            "max_drawdown": max_dd,
            "hit_rate_vs_spy": hit_rate,
        }
    else:
        metrics = {"horizon": horizon, "n_periods": 0}

    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity_curve": curve_df,
    }


def save_backtest(result: dict, name: str) -> None:
    base = CACHE_DIR / f"backtest_{name}"
    if not result["equity_curve"].empty:
        result["equity_curve"].to_parquet(f"{base}_curve.parquet", index=False)
    if not result["trades"].empty:
        result["trades"].to_parquet(f"{base}_trades.parquet", index=False)
    pd.DataFrame([result["metrics"]]).to_parquet(f"{base}_metrics.parquet", index=False)


def load_backtest(name: str) -> dict | None:
    base = CACHE_DIR / f"backtest_{name}"
    metrics_p = Path(f"{base}_metrics.parquet")
    if not metrics_p.exists():
        return None
    metrics = pd.read_parquet(metrics_p).iloc[0].to_dict()
    curve = pd.read_parquet(f"{base}_curve.parquet") if Path(f"{base}_curve.parquet").exists() else pd.DataFrame()
    trades = pd.read_parquet(f"{base}_trades.parquet") if Path(f"{base}_trades.parquet").exists() else pd.DataFrame()
    return {"metrics": metrics, "equity_curve": curve, "trades": trades}
