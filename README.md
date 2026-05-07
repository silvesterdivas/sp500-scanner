# S&P 500 Top 10 Scanner

A Python + Streamlit agent that scans all S&P 500 constituents every ~15 minutes and ranks them on two horizons:

- **Short-term (1-4 weeks)** — momentum, RSI, volume, news sentiment, trend.
- **Long-term (12+ months)** — valuation, growth, quality, balance sheet, 200-day trend.

Free-tier data only: **yfinance** (prices + fundamentals), **Finnhub** (news), **VADER** (sentiment).

> ⚠️ Informational signal aggregation. **Not financial advice.** Do your own research.

---

## Quick start

```bash
# 1. Install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional but recommended) Get a free Finnhub API key for news sentiment.
#    https://finnhub.io/register  →  60 req/min on free tier
cp .env.example .env
# edit .env and set FINNHUB_API_KEY=...

# 3. Run the dashboard
streamlit run app.py
```

First load downloads ~500 tickers × 3 years of price history (a few minutes).
Subsequent loads use the local Parquet cache.

## Run the backtest (optional)

```bash
python run_backtest.py
```

This produces equity curves vs. SPY for both horizons; results show up in the **Backtest** tab.

---

## Project layout

```
sp500_agent/
├── app.py                  # Streamlit dashboard (entrypoint)
├── run_backtest.py         # One-shot backtest runner
├── requirements.txt
├── .env.example
├── data/
│   ├── universe.py         # S&P 500 constituents from Wikipedia
│   ├── prices.py           # yfinance + technical indicators
│   ├── fundamentals.py     # yfinance .info → P/E, ROE, growth...
│   └── news.py             # Finnhub /company-news + VADER sentiment
├── signals/
│   └── scoring.py          # Short- and long-term composite scores
├── backtest/
│   └── runner.py           # Walk-forward backtest framework
└── cache/                  # Parquet cache (auto-created)
```

## How scoring works

Each factor is **z-scored within the S&P 500** (so values are relative to the cross-section), clipped at ±3σ to neutralize outliers, then weighted-summed.

Short-term weights: momentum 35%, RSI 15%, volume 15%, news sentiment 25%, trend 10%.
Long-term weights: valuation 25%, growth 25%, quality 25%, balance sheet 10%, trend 15%.

You can tune weights in `signals/scoring.py`.

## How the backtest works

Walk-forward simulation:
1. At each rebalance date, score the universe using **only data available at that date**.
2. Equal-weight the top 10.
3. Hold for the horizon (5 days short / ~63 days long).
4. Capture forward return; compare to SPY over the same period.

Limitations: only price-derived signals are used (point-in-time fundamentals and news aren't on free APIs), so this is a conservative test of the technical/momentum components.

## Deploy

Streamlit Community Cloud: push this folder to GitHub, point Streamlit Cloud at `app.py`, and set `FINNHUB_API_KEY` in the secrets panel.

## Architecture

Three layers:

1. **Ingestion** (`data/`) — pulls from yfinance + Finnhub, caches to Parquet so we don't re-hit free APIs unnecessarily.
2. **Scoring** (`signals/`) — z-scored composite factors, two weighting schemes for two horizons.
3. **Dashboard** (`app.py`) — Streamlit tabs: Top 10, Drill-down, Heatmap, Backtest, About.
