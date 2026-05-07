"""Streamlit dashboard: top 10 short-term and long-term S&P 500 picks."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from data.fundamentals import fetch_fundamentals
from data.news import aggregate_sentiment, fetch_news
from data.prices import compute_technicals, download_prices
from data.universe import load_universe
from signals.scoring import (LONG_TERM_WEIGHTS, SHORT_TERM_WEIGHTS,
                             long_term_score, short_term_score)
from backtest.runner import load_backtest

load_dotenv()
HAS_NEWS_KEY = bool(os.getenv("FINNHUB_API_KEY", "").strip())

st.set_page_config(
    page_title="S&P 500 Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- custom CSS ----------

st.markdown(
    """
<style>
:root {
  --accent: #22d3ee;
  --accent-soft: rgba(34, 211, 238, 0.12);
  --good: #4ade80;
  --bad:  #f87171;
  --warn: #fbbf24;
  --muted: rgba(229, 231, 235, 0.6);
  --line:  rgba(255, 255, 255, 0.07);
  --card:  rgba(255, 255, 255, 0.025);
}

/* Layout */
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }

/* Hero */
.hero {
  background: linear-gradient(135deg, rgba(34,211,238,0.10) 0%, rgba(99,102,241,0.07) 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 22px 26px;
  margin-bottom: 18px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 14px;
}
.hero-left { flex: 1; min-width: 280px; }
.hero-title {
  font-size: 1.85rem; font-weight: 750; letter-spacing: -0.02em;
  margin: 0 0 4px 0; line-height: 1.15;
}
.hero-sub {
  color: var(--muted); font-size: 0.95rem; max-width: 720px;
  margin: 0;
}
.hero-right { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
.pill {
  font-size: 0.78rem; padding: 5px 11px; border-radius: 999px;
  font-weight: 600; letter-spacing: 0.01em; white-space: nowrap;
  border: 1px solid transparent;
}
.pill-good { background: rgba(74,222,128,0.10); color: var(--good); border-color: rgba(74,222,128,0.25); }
.pill-warn { background: rgba(251,191,36,0.10); color: var(--warn); border-color: rgba(251,191,36,0.25); }
.pill-muted { background: rgba(255,255,255,0.04); color: var(--muted); border-color: var(--line); }

/* KPI metrics */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
  font-size: 0.72rem !important; opacity: 0.65;
  text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}
[data-testid="stMetricValue"] {
  font-size: 1.55rem !important; font-weight: 700;
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px; border-bottom: 1px solid var(--line);
  margin-top: 8px;
}
.stTabs [data-baseweb="tab"] {
  height: 42px; padding: 0 18px; border-radius: 8px 8px 0 0;
  font-weight: 500;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* Dataframe — tabular numerics, slimmer rows */
[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
[data-testid="stDataFrame"] table { font-size: 0.88rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #080c12; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stMarkdown h3 { font-size: 0.85rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.06em; }

/* Section subtitle */
.section-title {
  font-size: 1.05rem; font-weight: 650; margin: 4px 0 2px 0;
  display: flex; align-items: center; gap: 8px;
}
.section-sub { font-size: 0.85rem; color: var(--muted); margin: 0 0 10px 0; }

/* Headline cards */
.headline {
  border: 1px solid var(--line); border-radius: 10px;
  padding: 11px 13px; margin-bottom: 9px; background: var(--card);
  transition: border-color 120ms ease;
}
.headline:hover { border-color: rgba(34,211,238,0.35); }
.headline a { color: #e5e7eb; text-decoration: none; font-weight: 500; line-height: 1.35; }
.headline a:hover { color: var(--accent); }
.headline-meta { font-size: 0.74rem; color: var(--muted); margin-top: 5px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.dot-pos { color: var(--good); }
.dot-neg { color: var(--bad); }
.dot-neu { color: var(--muted); }

/* Empty state */
.empty {
  border: 1px dashed rgba(255,255,255,0.13);
  border-radius: 12px; padding: 18px 22px;
  background: rgba(255,255,255,0.012);
}
.empty h4 { margin: 0 0 4px 0; font-size: 0.98rem; }
.empty p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.5; }
.empty a { color: var(--accent); text-decoration: none; font-weight: 500; }
.empty a:hover { text-decoration: underline; }

/* Ticker chip */
.tckr {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600; color: var(--accent);
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- caching wrappers ----------

@st.cache_data(ttl=900)
def cached_data(force_refresh: bool):
    universe = load_universe()
    tickers = universe["symbol"].tolist()
    prices = download_prices(tickers, force_refresh=force_refresh)
    technicals = compute_technicals(prices)
    fundamentals = fetch_fundamentals(tickers, force_refresh=force_refresh)
    news = fetch_news(tickers, force_refresh=force_refresh)
    sentiment = aggregate_sentiment(news)
    return universe, prices, technicals, fundamentals, news, sentiment


# ---------- sidebar ----------

with st.sidebar:
    st.markdown("## 📈 S&P 500 Scanner")
    st.caption("Free-tier APIs · 15-min delayed")

    force_refresh = st.button("🔄 Refresh data", help="Re-download from Yahoo + Finnhub", use_container_width=True)
    auto_refresh = st.toggle("Auto-refresh (15 min)", value=False)

    if auto_refresh:
        st_autorefresh(interval=15 * 60 * 1000, key="auto_refresh")

    st.markdown("---")
    st.markdown("### Status")
    if HAS_NEWS_KEY:
        st.markdown('<span class="pill pill-good">● News connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-warn">● News disabled</span>', unsafe_allow_html=True)
        st.caption("Add `FINNHUB_API_KEY` to `.env` to enable headlines + sentiment. [Get a free key →](https://finnhub.io/register)")

    st.markdown("---")
    st.markdown("### Methodology")
    with st.expander("Short-term weights"):
        for k, v in SHORT_TERM_WEIGHTS.items():
            st.write(f"**{k}** — {v:.0%}")
    with st.expander("Long-term weights"):
        for k, v in LONG_TERM_WEIGHTS.items():
            st.write(f"**{k}** — {v:.0%}")

# ---------- load data ----------

with st.spinner("Loading market data..."):
    try:
        universe, prices, technicals, fundamentals, news, sentiment = cached_data(force_refresh)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

last_refresh = datetime.now().strftime("%H:%M")

# ---------- compute rankings ----------

short = short_term_score(technicals, sentiment)
long_ = long_term_score(technicals, fundamentals)

# ---------- hero ----------

if HAS_NEWS_KEY and not news.empty:
    pill_html = '<span class="pill pill-good">● Live · all signals</span>'
elif HAS_NEWS_KEY:
    pill_html = '<span class="pill pill-warn">● Live · news pending</span>'
else:
    pill_html = '<span class="pill pill-warn">● Limited · no news key</span>'

st.markdown(
    f"""
<div class="hero">
  <div class="hero-left">
    <h1 class="hero-title">S&P 500 Top 10 Scanner</h1>
    <p class="hero-sub">
      Combines technicals, fundamentals, and news sentiment to surface short-term momentum
      and long-term value/quality picks. <strong>For research only — not financial advice.</strong>
    </p>
  </div>
  <div class="hero-right">
    {pill_html}
    <span class="pill pill-muted">Updated {last_refresh}</span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

# ---------- KPI strip ----------

top_short_ticker = short.iloc[0]["ticker"] if not short.empty else "—"
top_short_score = float(short.iloc[0]["score"]) if not short.empty else 0
top_long_ticker = long_.iloc[0]["ticker"] if not long_.empty else "—"
top_long_score = float(long_.iloc[0]["score"]) if not long_.empty else 0
breadth = float(technicals["above_sma_200"].mean()) if "above_sma_200" in technicals.columns else 0
avg_20d = float(technicals["return_20d"].mean()) if not technicals.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Universe", f"{len(technicals):,}", help="Tickers with sufficient history")
k2.metric("Top short-term", top_short_ticker, delta=f"score {top_short_score:+.2f}")
k3.metric("Top long-term", top_long_ticker, delta=f"score {top_long_score:+.2f}")
k4.metric("Breadth (>200d SMA)", f"{breadth:.0%}", delta=f"avg 20d {avg_20d:+.1%}")

# ---------- tabs ----------

tabs = st.tabs(["🏆  Top 10", "🔍  Drill-down", "📊  Heatmap", "⏪  Backtest", "ℹ️  About"])

# ===================== TAB 1: TOP 10 =====================

with tabs[0]:
    if not HAS_NEWS_KEY:
        st.markdown(
            """
<div class="empty" style="margin-bottom:14px;">
  <h4>📰 Running without news sentiment</h4>
  <p>The short-term composite normally blends 25% news sentiment. Without a Finnhub key,
  it's scoring on momentum, RSI, volume, and trend only — still useful, but lighter on signal.
  <a href="https://finnhub.io/register">Grab a free key</a> and drop it in <code>.env</code> to unlock the full mix.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    def _rank_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        d = df.head(n).copy()
        d["#"] = range(1, len(d) + 1)
        return d

    def _style_short(df_view: pd.DataFrame):
        return (
            df_view.style
            .format({
                "Score": "{:.2f}", "5d": "{:+.1%}", "20d": "{:+.1%}",
                "RSI": "{:.0f}", "Sent.": "{:+.2f}", "Price": "${:,.2f}",
            })
            .background_gradient(subset=["Score"], cmap="Blues", vmin=-1, vmax=2.5)
            .background_gradient(subset=["5d", "20d"], cmap="RdYlGn", vmin=-0.15, vmax=0.15)
        )

    def _style_long(df_view: pd.DataFrame):
        return (
            df_view.style
            .format({
                "Score": "{:.2f}", "P/E": "{:.1f}", "Rev. growth": "{:+.1%}",
                "ROE": "{:.1%}", "vs 200d": "{:+.1%}", "Price": "${:,.2f}",
            }, na_rep="—")
            .background_gradient(subset=["Score"], cmap="Purples", vmin=-1, vmax=2.5)
            .background_gradient(subset=["vs 200d"], cmap="RdYlGn", vmin=-0.30, vmax=0.30)
        )

    with col1:
        st.markdown('<p class="section-title">🚀 Short-term · 1-4 weeks</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Ranked by momentum, RSI, volume, news sentiment, trend.</p>', unsafe_allow_html=True)
        s_top = _rank_table(short, 10)
        s_top_view = s_top.merge(
            universe[["symbol", "security", "gics_sector"]],
            left_on="ticker", right_on="symbol", how="left",
        )
        view = s_top_view[[
            "#", "ticker", "security", "gics_sector", "score",
            "return_5d", "return_20d", "rsi_14", "avg_sentiment", "last_close",
        ]].rename(columns={
            "ticker": "Ticker", "security": "Name", "gics_sector": "Sector",
            "score": "Score", "return_5d": "5d", "return_20d": "20d",
            "rsi_14": "RSI", "avg_sentiment": "Sent.", "last_close": "Price",
        })
        st.dataframe(_style_short(view), width="stretch", hide_index=True, height=420)

    with col2:
        st.markdown('<p class="section-title">💎 Long-term · 12+ months</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Ranked by valuation, growth, quality, balance sheet, 200d trend.</p>', unsafe_allow_html=True)
        l_top = _rank_table(long_, 10)
        l_top_view = l_top.merge(
            universe[["symbol", "security", "gics_sector"]],
            left_on="ticker", right_on="symbol", how="left",
        )
        view = l_top_view[[
            "#", "ticker", "security", "gics_sector", "score",
            "pe", "revenueGrowth", "returnOnEquity", "dist_to_sma_200_pct", "last_close",
        ]].rename(columns={
            "ticker": "Ticker", "security": "Name", "gics_sector": "Sector",
            "score": "Score", "pe": "P/E", "revenueGrowth": "Rev. growth",
            "returnOnEquity": "ROE", "dist_to_sma_200_pct": "vs 200d", "last_close": "Price",
        })
        st.dataframe(_style_long(view), width="stretch", hide_index=True, height=420)

    st.caption(
        "Each factor is z-scored within the S&P 500, clipped at ±3σ, then weighted-summed. "
        "Higher score = more attractive on that horizon."
    )

# ===================== TAB 2: DRILL-DOWN =====================

with tabs[1]:
    options = sorted(technicals["ticker"].unique())
    default_idx = 0
    if not short.empty and short.iloc[0]["ticker"] in options:
        default_idx = options.index(short.iloc[0]["ticker"])

    pick = st.selectbox("Choose a ticker", options, index=default_idx, label_visibility="collapsed")

    info_row = universe[universe["symbol"] == pick]
    name = info_row.iloc[0]["security"] if not info_row.empty else pick
    sector = info_row.iloc[0]["gics_sector"] if not info_row.empty else ""
    sub_industry = info_row.iloc[0]["gics_sub_industry"] if not info_row.empty else ""

    t_row = technicals[technicals["ticker"] == pick]
    last_close = float(t_row.iloc[0]["last_close"]) if not t_row.empty else 0
    ret_20d = float(t_row.iloc[0]["return_20d"]) if not t_row.empty else 0

    st.markdown(
        f"""
<div style="display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin: 4px 0 14px 0;">
  <span class="tckr" style="font-size:1.6rem;">{pick}</span>
  <span style="font-size:1.05rem; font-weight:500;">{name}</span>
  <span class="pill pill-muted">{sector}</span>
  <span style="margin-left:auto; font-variant-numeric: tabular-nums; font-weight:600; font-size:1.1rem;">
    ${last_close:,.2f}
    <span style="color: {'var(--good)' if ret_20d >= 0 else 'var(--bad)'}; font-size:0.9rem; margin-left:6px;">
      {ret_20d:+.2%} · 20d
    </span>
  </span>
</div>
        """,
        unsafe_allow_html=True,
    )
    if sub_industry:
        st.caption(sub_industry)

    # Price chart
    p = prices[prices["ticker"] == pick].sort_values("date")
    p_recent = p.tail(252)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=p_recent["date"], open=p_recent["open"], high=p_recent["high"],
        low=p_recent["low"], close=p_recent["close"], name=pick,
        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
    ))
    if len(p_recent) >= 50:
        fig.add_trace(go.Scatter(
            x=p_recent["date"], y=p_recent["close"].rolling(50).mean(),
            line=dict(color="#fbbf24", width=1.4), name="SMA 50",
        ))
    if len(p_recent) >= 200:
        fig.add_trace(go.Scatter(
            x=p_recent["date"], y=p_recent["close"].rolling(200).mean(),
            line=dict(color="#a78bfa", width=1.4), name="SMA 200",
        ))
    fig.update_layout(
        height=440, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, width="stretch")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<p class="section-title">📐 Technicals</p>', unsafe_allow_html=True)
        if not t_row.empty:
            r = t_row.iloc[0]
            st.metric("RSI(14)", f"{r['rsi_14']:.1f}")
            st.metric("5-day return", f"{r['return_5d']:+.2%}")
            st.metric("20-day return", f"{r['return_20d']:+.2%}")
            st.metric("vs 200-day SMA", f"{r['dist_to_sma_200_pct']:+.1%}")

    with col_b:
        st.markdown('<p class="section-title">📊 Fundamentals</p>', unsafe_allow_html=True)
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
        else:
            st.markdown('<div class="empty"><p>No fundamentals available for this ticker.</p></div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<p class="section-title">📰 Recent headlines</p>', unsafe_allow_html=True)
        n = news[news["ticker"] == pick].sort_values("datetime", ascending=False).head(5) if not news.empty else news
        if n is None or n.empty:
            if HAS_NEWS_KEY:
                st.markdown('<div class="empty"><p>No recent headlines for this ticker.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="empty"><h4>News disabled</h4>'
                    '<p>Set <code>FINNHUB_API_KEY</code> in <code>.env</code> to enable.<br>'
                    '<a href="https://finnhub.io/register">Get a free key →</a></p></div>',
                    unsafe_allow_html=True,
                )
        else:
            for _, art in n.iterrows():
                if art["sentiment"] > 0.1:
                    dot, klass = "●", "dot-pos"
                elif art["sentiment"] < -0.1:
                    dot, klass = "●", "dot-neg"
                else:
                    dot, klass = "●", "dot-neu"
                date_str = art["datetime"].strftime("%b %d") if pd.notna(art["datetime"]) else ""
                source = art.get("source", "") or ""
                st.markdown(
                    f"""<div class="headline">
  <a href="{art['url']}" target="_blank" rel="noopener">{art['headline']}</a>
  <div class="headline-meta">
    <span class="{klass}">{dot}</span>
    <span>{source}</span>
    <span>·</span>
    <span>{date_str}</span>
    <span>·</span>
    <span>sent {art['sentiment']:+.2f}</span>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

# ===================== TAB 3: HEATMAP =====================

with tabs[2]:
    st.markdown('<p class="section-title">S&P 500 sector heatmap</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Treemap sized by market cap, colored by 20-day return.</p>', unsafe_allow_html=True)

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
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        coloraxis_colorbar=dict(title="20d %", thickness=12, len=0.5),
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0f17", width=1)))
    st.plotly_chart(fig, width="stretch")

# ===================== TAB 4: BACKTEST =====================

with tabs[3]:
    st.markdown('<p class="section-title">Strategy backtest</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Walk-forward simulation. Top 10 equal-weight at each rebalance, '
        'held for the horizon, vs SPY. Uses <strong>price-derived signals only</strong> — '
        'point-in-time fundamentals/news aren\'t free-tier.</p>',
        unsafe_allow_html=True,
    )

    short_bt = load_backtest("short")
    long_bt = load_backtest("long")

    if short_bt is None and long_bt is None:
        st.markdown(
            '<div class="empty"><h4>No backtest results yet</h4>'
            '<p>Run <code>python run_backtest.py</code> from the project root, then revisit this tab.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        col_s, col_l = st.columns(2)

        def _show_metrics(bt, label):
            m = bt["metrics"]
            st.markdown(f'<p class="section-title">{label}</p>', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total return", f"{m['total_return']:+.1%}", delta=f"{m['alpha_total']:+.1%} vs SPY")
            mc2.metric("Sharpe", f"{m['sharpe']:.2f}", delta=f"{m['sharpe'] - m['spy_sharpe']:+.2f} vs SPY")
            mc3.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            st.metric("Hit rate vs SPY", f"{m['hit_rate_vs_spy']:.0%}", help="% of periods strategy beat SPY")

            curve = bt["equity_curve"]
            if not curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve["date"], y=curve["portfolio"],
                    name="Strategy", line=dict(color="#22d3ee", width=2),
                    fill="tozeroy", fillcolor="rgba(34,211,238,0.08)",
                ))
                fig.add_trace(go.Scatter(
                    x=curve["date"], y=curve["spy"],
                    name="SPY", line=dict(color="rgba(229,231,235,0.55)", dash="dash", width=1.5),
                ))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig, width="stretch")

        with col_s:
            if short_bt:
                _show_metrics(short_bt, "🚀 Short-term · weekly rebal., 5-day hold")
        with col_l:
            if long_bt:
                _show_metrics(long_bt, "💎 Long-term · quarterly rebal., 90-day hold")

# ===================== TAB 5: ABOUT =====================

with tabs[4]:
    st.markdown(
        """
### Methodology

**Universe.** S&P 500 constituents pulled from Wikipedia (refreshed weekly).

**Data sources (all free-tier).**
- **Prices + technicals** — yfinance (Yahoo Finance, 15-min delayed).
- **Fundamentals** — yfinance `.info` (P/E, growth, ROE, margins, etc.).
- **News + sentiment** — Finnhub `/company-news` + VADER for headline scoring.

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
        """
    )
