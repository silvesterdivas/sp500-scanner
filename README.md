# 📈 S&P 500 Scanner

A free, open-source dashboard that ranks every S&P 500 stock on **two horizons** and surfaces the top 10 picks for each:

- **🚀 Short-term (1–4 weeks)** — momentum, RSI, volume, news sentiment, trend.
- **💎 Long-term (12+ months)** — valuation, growth, quality, balance sheet, 200-day trend.

Plus a sector heatmap, per-ticker drill-down with charts and headlines, and a walk-forward backtest vs SPY.

> ⚠️ **Informational signal aggregation. Not financial advice.** Do your own research and consult a licensed advisor before trading.

---

## ✨ What's inside

| Tab | What you get |
| --- | --- |
| 🏆 **Top 10** | Two ranked tables, one per horizon, with score gradients and sector tags |
| 🔍 **Drill-down** | Per-ticker candlestick chart (1y), SMA 50/200 overlays, technicals + fundamentals + recent headlines |
| 📊 **Heatmap** | S&P 500 sector treemap, sized by market cap, colored by 20-day return |
| ⏪ **Backtest** | Walk-forward equity curves vs SPY for both horizons (Sharpe, drawdown, hit rate) |
| ℹ️ **About** | Methodology, weights, limitations |

---

## 🚀 Quick start

You'll need **Python 3.10+** and ~5 minutes for the first data download.

```bash
# 1. Clone
git clone https://github.com/silvesterdivas/sp500-scanner.git
cd sp500-scanner

# 2. Install dependencies (in a virtualenv)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

That's it. Streamlit will open a browser tab at `http://localhost:8501`.

> First load downloads ~500 tickers × 3 years of price history (a few minutes). Subsequent loads use the local Parquet cache and start in seconds.

---

## 🔑 Do I need an API key?

**No — the app works out of the box.** Yahoo Finance (the default source for prices and fundamentals) doesn't require a key.

For news headlines + sentiment, you can optionally add a **free** Finnhub key:

```bash
cp .env.example .env
# then edit .env and paste your key
```

Get one in 30 seconds at [finnhub.io/register](https://finnhub.io/register) (free tier = 60 requests/min).

| Feature | Without key | With Finnhub key |
| --- | --- | --- |
| Prices, charts, technicals | ✅ | ✅ |
| Fundamentals (P/E, ROE, growth, etc.) | ✅ | ✅ |
| Heatmap | ✅ | ✅ |
| Backtest | ✅ | ✅ |
| **News headlines (drill-down)** | — | ✅ |
| **Sentiment in short-term score** (25% weight) | — | ✅ |

Without the key, the short-term composite scores on momentum, RSI, volume, and trend only — still useful, just lighter on signal.

---

## 📂 Project layout

```
sp500-scanner/
├── app.py                # Streamlit dashboard (entrypoint)
├── run_backtest.py       # One-shot backtest runner
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml       # Theme
├── data/
│   ├── universe.py       # S&P 500 constituents from Wikipedia
│   ├── prices.py         # yfinance + technical indicators
│   ├── fundamentals.py   # yfinance .info → P/E, ROE, growth...
│   └── news.py           # Finnhub /company-news + VADER sentiment
├── signals/
│   └── scoring.py        # Short- and long-term composite scores
├── backtest/
│   └── runner.py         # Walk-forward backtest framework
└── cache/                # Parquet cache (auto-created, gitignored)
```

---

## 🧮 How scoring works

Every factor is **z-scored within the S&P 500** (so values are relative to the cross-section), clipped at ±3σ to neutralize outliers, then weighted-summed. Higher = more attractive.

**Short-term composite** (1–4 weeks):

| Factor | Weight |
| --- | --- |
| Momentum (5d + 20d return) | 35% |
| RSI signal (peaks at 60) | 15% |
| Volume surge (z-score vs 20d) | 15% |
| News sentiment (article-count weighted) | 25% |
| Trend (vs 50-day SMA) | 10% |

**Long-term composite** (12+ months):

| Factor | Weight |
| --- | --- |
| Valuation (forward P/E, lower better) | 25% |
| Growth (revenue + earnings growth) | 25% |
| Quality (ROE + margins) | 25% |
| Balance sheet (debt/equity) | 10% |
| Long-term trend (vs 200-day SMA) | 15% |

Tweak the weights in [`signals/scoring.py`](signals/scoring.py).

---

## ⏪ Run the backtest

```bash
python run_backtest.py
```

Walk-forward simulation:

1. At each rebalance date, score the universe using **only data available at that date**.
2. Equal-weight the top 10.
3. Hold for the horizon (5 days short / ~63 days long).
4. Capture forward return; compare to SPY.

Results show up automatically in the **Backtest** tab.

> Limitation: backtest uses only price-derived signals (point-in-time fundamentals and news aren't free), so it's a conservative test of the technical/momentum components.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Fork this repo to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect GitHub, point at `app.py`.
3. (Optional) Add `FINNHUB_API_KEY` to the **Secrets** panel for news.
4. Done — you'll get a public URL.

---

## 🛟 Troubleshooting

<details>
<summary><b>"No price data downloaded" / Yahoo rate-limiting</b></summary>

Yahoo Finance occasionally rate-limits aggressive scraping. Wait 5–10 minutes and try again, or run from a different network. The app will fall back to the local cache if a fresh fetch fails.
</details>

<details>
<summary><b>The Top 10 tables look blank/uncolored</b></summary>

Make sure you're on the latest dependencies — colored score gradients need `matplotlib`:
```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><b>"News disabled" pill in the sidebar</b></summary>

You don't have a Finnhub key set. The rest of the app still works — see the [API key table](#-do-i-need-an-api-key) above.
</details>

<details>
<summary><b>First run is slow</b></summary>

Yes — it downloads 3 years of daily prices for ~500 tickers and ~500 fundamental records. Subsequent runs hit the local Parquet cache in `cache/` and load in seconds. Click **🔄 Refresh data** in the sidebar to force a re-fetch.
</details>

<details>
<summary><b>Want to remove matplotlib?</b></summary>

It's only used for the colored gradient bars on the Top-10 tables. If you don't want the dependency, replace `Styler.background_gradient(...)` calls in `app.py` with simple `Styler.format(...)` and remove the matplotlib line from `requirements.txt`.
</details>

---

## 🏗️ Architecture

Three layers:

1. **Ingestion** (`data/`) — pulls from yfinance + Finnhub, caches to Parquet so we don't re-hit free APIs unnecessarily.
2. **Scoring** (`signals/`) — z-scored composite factors, two weighting schemes for two horizons.
3. **Dashboard** (`app.py`) — Streamlit tabs: Top 10, Drill-down, Heatmap, Backtest, About.

---

## ⚖️ Disclaimer

This dashboard is **informational signal aggregation**, not financial advice. Past performance (backtests included) does not predict future returns. Do not make investment decisions based solely on these scores. Consult a licensed financial advisor before trading.

---

## 📝 License

MIT — do whatever you want, but keep the disclaimer attached.
