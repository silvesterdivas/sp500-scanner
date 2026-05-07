"""Streamlit dashboard: top 10 short-term and long-term S&P 500 picks."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from data.fundamentals import fetch_fundamentals
from data.news import aggregate_sentiment, fetch_news
from data.prices import compute_technicals, download_prices
from data.universe import get_tickers, load_universe
from signals.scoring import (LONG_TERM_WEIGHTS, SHORT_TERM_WEIGHTS,
                             long_term_score, short_term_score)
from backtest.runner import load_backtest

st.set_page_config(
    page_title="S&P 500 Top 10 Scanner",
    page_icon="📈",
    layout="wide",
)

# ---------- caching wrappers ----------

@st.cache_data(ttl=900)  # 15 min
def cached_universe():
    return load_universe()


@st.cache_data(ttl=900)
def cached_data(force_refresh: bool):
    """Load all data with one call so the cache key is stable."""
    universe = load_universe()
    tickers = universe["symbol"].tolist()
    prices = download_prices(tickers, force_refresh=force_refresh)
    technicals = compute_technicals(prices)
    fundamentals = fetch_fundamentals(tickers, force_refresh=force_refresh)
    news = fetch_news(tickers, force_refresh=force_refresh)
    sentiment = aggregate_sentiment(news)
    return universe, prices, technicals, fundamentals, news, sentiment


# ---------- sidebar ----------

st.sidebar.title("📈 S&P 500 Scanner")
st.sidebar.caption("Free-tier APIs · 15-min delayed · Not financial advice")

force_refresh = st.sidebar.button("🔄 Force refresh data", help="Re-download from Yahoo + Finnhub")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 15 min", value=False)

if auto_refresh:
    st_autorefresh(interval=15 * 60 * 1000, key="auto_refresh")

st.sidebar.markdown("---")
st.sidebar.markdown("### Methodology")
with st.sidebar.expander("Short-term weights"):
    for k, v in SHORT_TERM_WEIGHTS.items():
        st.write(f"**{k}**: {v:.0%}")
with st.sidebar.expander("Long-term weights"):
    for k, v in LONG_TERM_WEIGHTS.items():
        st.write(f"**{k}**: {v:.0%}")

# ---------- load data ----------

with st.spinner("Loading market data..."):
    try:
        universe, prices, technicals, fundamentals, news, sentiment = cached_data(force_refresh)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"**Last refreshed:** {last_refresh}")
st.sidebar.markdown(f"**Universe:** {len(universe)} tickers")
st.sidebar.markdown(f"**Price rows:** {len(prices):,}")
if not news.empty:
    st.sidebar.markdown(f"**News articles:** {len(news):,}")
else:
    st.sidebar.warning("No news data — set FINNHUB_API_KEY in `.env`")

# ---------- compute rankings ----------

short = short_term_score(technicals, sentiment)
long_ = long_term_score(technicals, fundamentals)

# ---------- header ----------

st.title("S&P 500 Top 10 Stock Scanner")
st.caption(
    "Combines technicals, fundamentals, and news sentiment to surface short-term momentum "
    "and long-term value/quality picks. Data is 15-min delayed. **For research only — not financial advice.**"
)

tabs = st.tabs(["🏆 Top 10", "🔍 Drill-down", "📊 Heatmap", "⏪ Backtest", "ℹ️ About"])

# ---------- TAB 1: TOP 10 ----------

with tabs[0]:
    col1, col2 = st.columns(2)

    def _rank_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        d = df.head(n).copy()
        d["#"] = range(1, len(d) + 1)
        return d

    with col1:
        st.subheader("🚀 Top 10 — Short-term (1-4 weeks)")
        st.caption("Ranked by momentum, RSI, volume, news sentiment, trend.")
        s_top = _rank_table(short, 10)
        s_top_view = s_top.merge(universe[["symbol", "security", "gics_sector"]],
                                 left_on="ticker", right_on="symbol", how="left")
        st.dataframe(
            s_top_view[[
                "#", "ticker", "security", "gics_sector", "score",
                "return_5d", "return_20d", "rsi_14", "avg_sentiment", "last_close"
            ]].rename(columns={
                "security": "Name", "gics_sector": "Sector", "score": "Score",
                "return_5d": "5d %", "return_20d": "20d %", "rsi_14": "RSI",
                "avg_sentiment": "Sentiment", "last_close": "Price",
            }).style.format({
                "Score": "{:.2f}", "5d %": "{:.1%}", "20d %": "{:.1%}",
                "RSI": "{:.0f}", "Sentiment": "{:+.2f}", "Price": "${:.2f}",
            }),
            width="stretch", hide_index=True,
        )

    with col2:
        st.subheader("💎 Top 10 — Long-term (12+ months)")
        st.caption("Ranked by valuation, growth, quality, balance sheet, 200d trend.")
        l_top = _rank_table(long_, 10)
        l_top_view = l_top.merge(universe[["symbol", "security", "gics_sector"]],
                                 left_on="ticker", right_on="symbol", how="left")
        st.dataframe(
            l_top_view[[
                "#", "ticker", "security", "gics_sector", "score",
                "pe", "revenueGrowth", "returnOnEquity", "dist_to_sma_200_pct", "last_close"
            ]].rename(columns={
                "security": "Name", "gics_sector": "Sector", "score": "Score",
                "pe": "P/E", "revenueGrowth": "Rev. growth",
                "returnOnEquity": "ROE", "dist_to_sma_200_pct": "vs 200d", "last_close": "Price",
            }).style.format({
                "Score": "{:.2f}", "P/E": "{:.1f}", "Rev. growth": "{:.1%}",
                "ROE": "{:.1%}", "vs 200d": "{:.1%}", "Price": "${:.2f}",
            }),
            width="stretch", hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Composite-score formula:** each factor is z-scored within the S&P 500, then weighted-summed. Higher = more attractive on that axis.")

# ---------- TAB 2: DRILL-DOWN ----------

with tabs[1]:
    st.subheader("Per-ticker detail")
    options = sorted(technicals["ticker"].unique())
    default_idx = 0
    if not short.empty:
        default_idx = options.index(short.iloc[0]["ticker"]) if short.iloc[0]["ticker"] in options else 0
    pick = st.selectbox("Choose a ticker", options, index=default_idx)

    info_row = universe[universe["symbol"] == pick]
    if not info_row.empty:
        st.markdown(f"### {pick} — {info_row.iloc[0]['security']}")
        st.caption(f"{info_row.iloc[0]['gics_sector']} · {info_row.iloc[0]['gics_sub_industry']}")

    # Price chart
    p = prices[prices["ticker"] == pick].sort_values("date")
    p_recent = p.tail(252)  # 1y

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=p_recent["date"], open=p_recent["open"], high=p_recent["high"],
        low=p_recent["low"], close=p_recent["close"], name=pick,
    ))
    if len(p_recent) >= 50:
        fig.add_trace(go.Scatter(
            x=p_recent["date"], y=p_recent["close"].rolling(50).mean(),
            line=dict(color="orange", width=1), name="SMA 50",
        ))
    if len(p_recent) >= 200:
        fig.add_trace(go.Scatter(
            x=p_recent["date"], y=p_recent["close"].rolling(200).mean(),
            line=dict(color="purple", width=1), name="SMA 200",
        ))
    fig.update_layout(
        height=420, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, width="stretch")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Technicals**")
        t_row = technicals[technicals["ticker"] == pick]
        if not t_row.empty:
            r = t_row.iloc[0]
            st.metric("Last close", f"${r['last_close']:.2f}")
            st.metric("RSI(14)", f"{r['rsi_14']:.1f}")
            st.metric("5d return", f"{r['return_5d']:+.2%}")
            st.metric("20d return", f"{r['return_20d']:+.2%}")
            st.metric("vs 200-day SMA", f"{r['dist_to_sma_200_pct']:+.1%}")

    with col_b:
        st.markdown("**Fundamentals**")
        f_row = fundamentals[fundamentals["ticker"] == pick]
        if not f_row.empty:
            r = f_row.iloc[0]
            def show(label, key, fmt="{:.2f}"):
                v = r.get(key)
                if v is None or pd.isna(v):
                    st.metric(label, "—")
                else:
                    st.metric(label, fmt.format(v))
            show("Trailing P/E", "trailingPE", "{:.1f}")
            show("Forward P/E", "forwardPE", "{:.1f}")
            show("Revenue growth", "revenueGrowth", "{:+.1%}")
            show("ROE", "returnOnEquity", "{:.1%}")
            show("Profit margin", "profitMargins", "{:.1%}")
            show("Debt/Equity", "debtToEquity", "{:.1f}")

    with col_c:
        st.markdown("**Recent headlines**")
        n = news[news["ticker"] == pick].sort_values("datetime", ascending=False).head(5)
        if n.empty:
            st.write("_No recent headlines (or Finnhub key missing)._")
        else:
            for _, art in n.iterrows():
                emoji = "🟢" if art["sentiment"] > 0.1 else ("🔴" if art["sentiment"] < -0.1 else "⚪️")
                st.markdown(
                    f"{emoji} **[{art['headline']}]({art['url']})** "
                    f"<br><small>{art['source']} · {art['datetime'].strftime('%Y-%m-%d')} · "
                    f"sentiment {art['sentiment']:+.2f}</small>",
                    unsafe_allow_html=True,
                )

# ---------- TAB 3: HEATMAP ----------

with tabs[2]:
    st.subheader("S&P 500 — sector heatmap")
    st.caption("Treemap sized by market cap, colored by 20-day return.")

    df = technicals.merge(fundamentals[["ticker", "marketCap", "sector"]], on="ticker", how="left")
    df = df.merge(universe[["symbol", "security", "gics_sector"]], left_on="ticker", right_on="symbol", how="left")
    df["sector"] = df["gics_sector"].fillna(df["sector"]).fillna("Unknown")
    df["marketCap"] = df["marketCap"].fillna(1e9)
    df["return_pct"] = df["return_20d"] * 100

    fig = px.treemap(
        df,
        path=[px.Constant("S&P 500"), "sector", "ticker"],
        values="marketCap",
        color="return_pct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        range_color=(-15, 15),
        hover_data={"security": True, "return_pct": ":.2f", "marketCap": ":,.0f"},
    )
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# ---------- TAB 4: BACKTEST ----------

with tabs[3]:
    st.subheader("Strategy backtest (price-only signals)")
    st.caption(
        "Walk-forward simulation. At each rebalance date, we score the universe using only data "
        "available then, equal-weight the top 10, hold for the horizon, and compare to SPY. "
        "Note: backtest uses **price-derived signals only** — point-in-time fundamentals and news "
        "aren't available on free-tier APIs."
    )

    short_bt = load_backtest("short")
    long_bt = load_backtest("long")

    if short_bt is None and long_bt is None:
        st.info("No backtest results yet. Run `python run_backtest.py` from the project root.")
    else:
        col_s, col_l = st.columns(2)

        def _show_metrics(bt, label):
            m = bt["metrics"]
            st.markdown(f"### {label}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total return", f"{m['total_return']:+.1%}", delta=f"{m['alpha_total']:+.1%} vs SPY")
            mc2.metric("Sharpe", f"{m['sharpe']:.2f}", delta=f"{m['sharpe'] - m['spy_sharpe']:+.2f} vs SPY")
            mc3.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            st.metric("Hit rate vs SPY", f"{m['hit_rate_vs_spy']:.0%}", help="% of periods strategy beat SPY")

            curve = bt["equity_curve"]
            if not curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curve["date"], y=curve["portfolio"], name="Strategy", line=dict(color="#2E86AB")))
                fig.add_trace(go.Scatter(x=curve["date"], y=curve["spy"], name="SPY", line=dict(color="gray", dash="dash")))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
                st.plotly_chart(fig, width="stretch")

        with col_s:
            if short_bt:
                _show_metrics(short_bt, "Short-term (weekly rebal., 5-day hold)")
        with col_l:
            if long_bt:
                _show_metrics(long_bt, "Long-term (quarterly rebal., 90-day hold)")

# ---------- TAB 5: ABOUT ----------

with tabs[4]:
    st.markdown("""
    ### Methodology

    **Universe.** S&P 500 constituents pulled from Wikipedia (refreshed weekly).

    **Data sources (all free-tier).**
    - **Prices + technicals**: yfinance (Yahoo Finance, 15-min delayed).
    - **Fundamentals**: yfinance `.info` (P/E, growth, ROE, margins, etc.).
    - **News + sentiment**: Finnhub `/company-news` + VADER for headline scoring.

    **Short-term composite (1-4 weeks).** Z-scored within S&P 500, then weighted:
    - Momentum (5d + 20d return) — 35%
    - RSI signal (peaks at 60, penalizes oversold/overbought) — 15%
    - Volume surge (z-score vs 20-day) — 15%
    - News sentiment (article-count-weighted) — 25%
    - Trend (vs 50-day SMA) — 10%

    **Long-term composite (12+ months).** Z-scored within S&P 500, then weighted:
    - Valuation (forward P/E, lower is better) — 25%
    - Growth (revenue + earnings growth) — 25%
    - Quality (ROE + margins) — 25%
    - Balance sheet (debt/equity) — 10%
    - Long-term trend (vs 200-day SMA) — 15%

    **Backtest.** Walk-forward, equal-weighted top 10, comparing to SPY.
    Uses only price-derived signals (point-in-time fundamentals/news aren't free).

    ### Limitations
    - 15-minute delay on prices (free-tier reality).
    - yfinance is unofficial — Yahoo can change endpoints; cache mitigates short outages.
    - Fundamentals are *current*, not point-in-time, so the long-term backtest understates real-world look-ahead.
    - Sentiment is rule-based (VADER) on headlines only — fast and free, but less nuanced than LLM scoring.

    ### Disclaimer
    This dashboard is **informational signal aggregation**, not financial advice.
    Do not make investment decisions based solely on these scores.
    Consult a licensed financial advisor before trading.
    """)
