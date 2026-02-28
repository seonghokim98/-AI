"""
app.py - 주식 매매 신호 시각화 대시보드
Streamlit 기반 웹앱 (로컬 실행)

실행:
    streamlit run app.py
"""
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
import data_provider as dp
import market_filter as mf
import pattern_engine as pe

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="주식 매매 신호 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 데이터 로딩 (캐시 5분) ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_market_status() -> dict:
    status = mf.analyze_market()
    return {
        "regime": status.regime.value,
        "kospi_above_ma": status.kospi_above_ma,
        "kosdaq_above_ma": status.kosdaq_above_ma,
        "vix_ok": status.vix_ok,
        "foreign_buy_streak": status.foreign_buy_streak,
        "detail": status.detail,
    }


@st.cache_data(ttl=300)
def load_watchlist_signals() -> list[dict]:
    """감시 종목 전체 스캔 — 현재가 + 패턴 감지"""
    results = []
    for ticker in config.WATCHLIST:
        name = config.TICKER_NAME.get(ticker, ticker)
        df = dp.get_ohlcv(ticker)

        if df is None or df.empty:
            results.append({
                "종목명": name, "티커": ticker,
                "현재가": None, "전일대비": None,
                "패턴": "-", "신뢰도": None,
                "진입가": None, "저항선": None, "지지선": None,
                "rsi": None, "패턴상세": None,
            })
            continue

        current_price = float(df["Close"].iloc[-1])
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100

        pattern = pe.detect_patterns(df)

        results.append({
            "종목명": name,
            "티커": ticker,
            "현재가": current_price,
            "전일대비": change_pct,
            "패턴": pattern.pattern.value if pattern else "-",
            "신뢰도": float(pattern.confidence) if pattern else None,
            "진입가": float(pattern.entry_price) if pattern else None,
            "저항선": float(pattern.resistance_level) if pattern else None,
            "지지선": float(pattern.support_level) if pattern else None,
            "rsi": float(pattern.rsi) if pattern else None,
            "패턴상세": pattern.detail if pattern else None,
        })
    return results


@st.cache_data(ttl=300)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    return dp.get_ohlcv(ticker, period=period)


# ─── 차트 생성 ────────────────────────────────────────────────────────────────

def make_chart(df: pd.DataFrame, ticker: str, signal: dict) -> go.Figure:
    name = config.TICKER_NAME.get(ticker, ticker)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{name} ({ticker})", "거래량", "RSI"),
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="가격",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # 이동평균선
    ma_styles = [("MA5", "#ff9800", 1.2), ("MA20", "#42a5f5", 2), ("MA60", "#ab47bc", 2)]
    for col_name, color, width in ma_styles:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                name=col_name,
                line=dict(color=color, width=width),
                opacity=0.85,
            ), row=1, col=1)

    # 볼린저 밴드
    if "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"],
            name="BB 상단", line=dict(color="gray", width=1, dash="dot"),
            opacity=0.5, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"],
            name="BB 하단", line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.07)",
            opacity=0.5, showlegend=False,
        ), row=1, col=1)

    # 패턴 수평선
    if signal["패턴"] != "-":
        if signal.get("진입가"):
            fig.add_hline(
                y=signal["진입가"], line_color="#00e5ff", line_dash="dash", line_width=1.5,
                annotation_text=f"진입가 {signal['진입가']:,.0f}",
                annotation_font_color="#00e5ff",
                row=1, col=1,
            )
        if signal.get("지지선"):
            fig.add_hline(
                y=signal["지지선"], line_color="#ef5350", line_dash="dot", line_width=1.2,
                annotation_text=f"지지선 {signal['지지선']:,.0f}",
                annotation_font_color="#ef5350",
                row=1, col=1,
            )
        if signal.get("저항선"):
            fig.add_hline(
                y=signal["저항선"], line_color="#ffd54f", line_dash="dot", line_width=1.2,
                annotation_text=f"저항선 {signal['저항선']:,.0f}",
                annotation_font_color="#ffd54f",
                row=1, col=1,
            )

    # 거래량 바
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="거래량", marker_color=vol_colors, opacity=0.75,
    ), row=2, col=1)

    if "VOL_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VOL_MA20"],
            name="거래량 MA20", line=dict(color="#ff9800", width=1.5),
        ), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        rsi_colors = [
            "#ef5350" if v > 70 else ("#26a69a" if v < 30 else "#e91e63")
            for v in df["RSI"].fillna(50)
        ]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#e91e63", width=2),
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.08)", line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_color="#ef5350", line_dash="dash", line_width=1, row=3, col=1,
                      annotation_text="과매수 70")
        fig.add_hline(y=30, line_color="#26a69a", line_dash="dash", line_width=1, row=3, col=1,
                      annotation_text="과매도 30")

    fig.update_layout(
        height=700,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=80, t=40, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
    fig.update_yaxes(showgrid=True, gridcolor="#1e2130")

    return fig


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    auto_refresh = st.toggle("자동 새로고침 (5분)", value=False)
    chart_period = st.selectbox(
        "차트 기간", ["3mo", "6mo", "1y"], index=1,
        format_func=lambda x: {"3mo": "3개월", "6mo": "6개월", "1y": "1년"}[x],
    )
    st.markdown("---")
    st.markdown("**매매 원칙**")
    st.markdown(f"""
- 손절선 **{abs(config.STOP_LOSS_RATE)*100:.0f}%** (절대 원칙)
- 최소 손익비 **{config.MIN_REWARD_RISK_RATIO:.1f}:1**
- 신뢰도 기준 **60%**
- 최대 포지션 **{config.MAX_POSITIONS}개**
- 종목당 예산 **{config.BUDGET_PER_TRADE:,}원**
""")
    st.markdown("---")
    if st.button("🗑️ 캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.success("캐시 초기화 완료")


# ─── 헤더 ────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("📈 주식 매매 신호 대시보드")
with col_btn:
    st.write("")
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.caption(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 데이터: yfinance")

# ─── 시장 현황 ────────────────────────────────────────────────────────────────

st.subheader("🌍 시장 현황")
with st.spinner("시장 데이터 분석 중..."):
    market = load_market_status()

regime = market["regime"]
regime_emoji = {"강세장": "🟢", "중립": "🟡", "약세장": "🔴"}.get(regime, "⚪")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("시장 상태", f"{regime_emoji} {regime}")
with m2:
    v = "✅ MA 위" if market["kospi_above_ma"] else "❌ MA 아래"
    st.metric("코스피 20일선", v)
with m3:
    v = "✅ MA 위" if market["kosdaq_above_ma"] else "❌ MA 아래"
    st.metric("코스닥 20일선", v)
with m4:
    v = "✅ 정상" if market["vix_ok"] else "⚠️ 과열"
    st.metric("VIX", v)

if regime == "약세장":
    st.error("⚠️ 약세장 — 모든 매수 신호 차단 중입니다. 관망하세요.")
elif regime == "중립":
    st.warning("🟡 중립장 — 고품질 신호(신뢰도 높은 것)만 참고하세요.")
else:
    st.success("🟢 강세장 — 매수 신호 정상 작동 중입니다.")

with st.expander("시장 상세 보기"):
    st.caption(market["detail"])
    if market["foreign_buy_streak"] > 0:
        st.caption(f"외국인 {market['foreign_buy_streak']}일 연속 순매수")
    else:
        st.caption("외국인 수급: KIS API 미연동 (외국인 데이터 없음)")

st.divider()

# ─── 감시 종목 현황 ───────────────────────────────────────────────────────────

st.subheader("📋 감시 종목 현황")
with st.spinner("감시 종목 스캔 중... (최초 실행 시 1~2분 소요)"):
    signals = load_watchlist_signals()

signal_count = sum(1 for s in signals if s["패턴"] != "-")
st.caption(f"총 {len(signals)}개 종목 중 {signal_count}개 패턴 감지됨")

rows = []
for s in signals:
    price_str = f"{s['현재가']:,.0f}원" if s["현재가"] else "-"

    chg = s["전일대비"]
    if chg is not None:
        arrow = "▲" if chg >= 0 else "▼"
        chg_str = f"{arrow} {abs(chg):.2f}%"
    else:
        chg_str = "-"

    conf = s["신뢰도"]
    if conf is not None:
        filled = round(conf * 5)
        bar = "⬛" * filled + "⬜" * (5 - filled)
        conf_str = f"{bar} {conf*100:.0f}%"
    else:
        conf_str = "-"

    signal_icon = "📈 매수신호" if s["패턴"] != "-" else "-"

    rows.append({
        "종목명": s["종목명"],
        "현재가": price_str,
        "전일대비": chg_str,
        "감지 패턴": s["패턴"],
        "신뢰도": conf_str,
        "신호": signal_icon,
    })

df_table = pd.DataFrame(rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

st.divider()

# ─── 차트 분석 ────────────────────────────────────────────────────────────────

st.subheader("📊 차트 분석")

ticker_options = {
    f"{config.TICKER_NAME.get(t, t)} ({t})": t
    for t in config.WATCHLIST
}

# 패턴 감지 종목을 기본 선택
default_idx = 0
signal_tickers = [s["티커"] for s in signals if s["패턴"] != "-"]
if signal_tickers:
    labels = list(ticker_options.keys())
    tickers = list(ticker_options.values())
    if signal_tickers[0] in tickers:
        default_idx = tickers.index(signal_tickers[0])

selected_label = st.selectbox(
    "종목 선택",
    list(ticker_options.keys()),
    index=default_idx,
)
selected_ticker = ticker_options[selected_label]

selected_signal = next((s for s in signals if s["티커"] == selected_ticker), {})

if selected_signal.get("패턴") and selected_signal["패턴"] != "-":
    st.info(f"🔍 **{selected_signal['패턴']}** 패턴 감지 | {selected_signal.get('패턴상세', '')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        v = selected_signal.get("진입가")
        st.metric("추천 진입가", f"{v:,.0f}원" if v else "-")
    with c2:
        v = selected_signal.get("지지선")
        st.metric("지지선 (참고)", f"{v:,.0f}원" if v else "-")
    with c3:
        v = selected_signal.get("rsi")
        st.metric("RSI", f"{v:.1f}" if v else "-")

with st.spinner("차트 로딩 중..."):
    df_chart = load_ohlcv(selected_ticker, period=chart_period)

if df_chart is not None and not df_chart.empty:
    fig = make_chart(df_chart, selected_ticker, selected_signal or {})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"{selected_label} 차트 데이터를 불러올 수 없습니다.")

# ─── 자동 새로고침 ────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
