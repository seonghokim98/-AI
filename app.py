"""
app.py - 주식 매매 신호 시각화 대시보드
Streamlit 기반 웹앱 (로컬 실행)

실행:
    streamlit run app.py
"""
import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

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

@st.cache_data(ttl=60)
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


@st.cache_data(ttl=60)
def load_market_indices() -> dict:
    """주요 글로벌 지수 현재가 / 전일대비 (yfinance)"""
    index_map = {
        "코스피":      ("^KS11",    ""),
        "코스닥":      ("^KQ11",    ""),
        "S&P 500":    ("^GSPC",    "USD"),
        "나스닥 100":  ("^NDX",     "USD"),
        "닛케이 225":  ("^N225",    "JPY"),
        "항셍":        ("^HSI",     "HKD"),
        "달러인덱스":  ("DX-Y.NYB", ""),
        "금 선물":     ("GC=F",     "USD"),
        "WTI 원유":    ("CL=F",     "USD"),
        "VIX":         ("^VIX",     ""),
    }
    result = {}
    for name, (ticker, unit) in index_map.items():
        try:
            df = dp.get_index_data(ticker, period="5d")
            if df is not None and len(df) >= 2:
                cur  = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                chg  = (cur - prev) / prev * 100
                result[name] = {"price": cur, "change": chg, "unit": unit, "ticker": ticker}
        except Exception:
            pass
    return result


_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.6 Mobile/15E148 Safari/604.1"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    "Referer": "https://m.stock.naver.com/",
}

_NAVER_URLS = {
    "domestic": "https://m.stock.naver.com/domestic/index/KOSPI/total",
    "overseas": "https://m.stock.naver.com/worldstock/home/USA/discussion/ranking",
    "overview": "https://m.stock.naver.com/",
}


def _parse_next_data(key: str, nd: dict) -> list[str]:
    """Next.js __NEXT_DATA__ JSON 에서 뉴스/토론 문자열 목록 추출"""
    items = []
    try:
        props = nd.get("props", {}).get("pageProps", {})
        # 가능한 키 목록
        candidates = (
            props.get("discussions")
            or props.get("opinions")
            or props.get("news")
            or props.get("rankings")
            or props.get("articles")
            or []
        )
        for obj in candidates[:12]:
            text = (
                obj.get("title")
                or obj.get("headline")
                or obj.get("content", "")[:80]
            )
            text = text.strip()
            if 5 < len(text) < 200:
                stock = (
                    (obj.get("stock") or {}).get("name", "")
                    or obj.get("market", "")
                    or obj.get("source", "")
                )
                items.append(f"[{stock}] {text}" if stock else text)
    except Exception:
        pass
    return items


def _parse_html_fallback(soup) -> list[str]:
    """BeautifulSoup HTML fallback — 텍스트 블록 추출"""
    items = []
    if soup is None:
        return items
    for el in soup.find_all(["h3", "h4", "li", "span"], limit=60):
        text = el.get_text(strip=True)
        if 10 < len(text) < 160 and text not in items:
            items.append(text)
        if len(items) >= 10:
            break
    return items


@st.cache_data(ttl=600)
def load_naver_market_news() -> dict:
    """
    네이버 증권 모바일 페이지에서 이슈/뉴스/토론 크롤링 (10분 캐시).
    - Next.js __NEXT_DATA__ JSON 우선 파싱
    - 실패 시 HTML 직접 파싱 fallback
    - 전체 실패 시 error 필드에 메시지 반환
    """
    result = {k: {"items": [], "error": None} for k in _NAVER_URLS}

    if not _BS4_OK:
        for k in result:
            result[k]["error"] = "beautifulsoup4 패키지 미설치 (pip install beautifulsoup4)"
        return result

    for key, url in _NAVER_URLS.items():
        try:
            resp = requests.get(url, headers=_NAVER_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # ① Next.js 내장 JSON 파싱
            nd_tag = soup.find("script", {"id": "__NEXT_DATA__"})
            if nd_tag and nd_tag.string:
                items = _parse_next_data(key, json.loads(nd_tag.string))
                if items:
                    result[key]["items"] = items
                    continue

            # ② HTML 직접 파싱 fallback
            result[key]["items"] = _parse_html_fallback(soup)

        except requests.Timeout:
            result[key]["error"] = "연결 시간 초과 (10s)"
        except Exception as exc:
            result[key]["error"] = str(exc)[:80]

    return result


@st.cache_data(ttl=60)
def load_watchlist_signals() -> list[dict]:
    """감시 종목 전체 스캔 — 현재가 + 매수/매도 패턴 감지"""
    results = []
    for ticker in config.WATCHLIST:
        name = config.TICKER_NAME.get(ticker, ticker)
        df = dp.get_ohlcv(ticker)

        if df is None or df.empty:
            results.append({
                "종목명": name, "티커": ticker,
                "현재가": None, "전일대비": None,
                "패턴": "-", "신뢰도": None, "신호유형": None,
                "행동지침": None,
                "진입가": None, "저항선": None, "지지선": None,
                "rsi": None, "패턴상세": None,
            })
            continue

        current_price = float(df["Close"].iloc[-1])
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100

        # 매수 패턴 우선 탐지 후 매도 패턴 탐지 (매도가 있으면 우선 표시)
        buy_pattern = pe.detect_patterns(df)
        sell_pattern = pe.detect_sell_patterns(df)

        # 매도 패턴 우선 표시 (더 중요한 경고)
        pattern = sell_pattern if sell_pattern is not None else buy_pattern

        results.append({
            "종목명": name,
            "티커": ticker,
            "현재가": current_price,
            "전일대비": change_pct,
            "패턴": pattern.pattern.value if pattern else "-",
            "신뢰도": float(pattern.confidence) if pattern else None,
            "신호유형": pattern.signal_type if pattern else None,
            "행동지침": pattern.action if pattern else None,
            "진입가": float(pattern.entry_price) if pattern else None,
            "저항선": float(pattern.resistance_level) if pattern else None,
            "지지선": float(pattern.support_level) if pattern else None,
            "rsi": float(pattern.rsi) if pattern else None,
            "패턴상세": pattern.detail if pattern else None,
            # 매수/매도 패턴 각각 보관 (차트 표시용)
            "_buy_pattern": buy_pattern,
            "_sell_pattern": sell_pattern,
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

    # 패턴 수평선 (신호 유형에 따라 색상 차별화)
    sig_type = signal.get("신호유형", "BUY")
    entry_color = "#00e5ff" if sig_type == "BUY" else "#ff5252"   # 매수=청록, 매도=빨강
    if signal.get("패턴") and signal["패턴"] != "-":
        if signal.get("진입가"):
            label = "진입가" if sig_type == "BUY" else "현재가(매도)"
            fig.add_hline(
                y=signal["진입가"], line_color=entry_color, line_dash="dash", line_width=1.5,
                annotation_text=f"{label} {signal['진입가']:,.0f}",
                annotation_font_color=entry_color,
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
    auto_refresh = st.toggle("자동 새로고침", value=False)
    refresh_interval = st.selectbox(
        "새로고침 주기",
        [60, 120, 300],
        format_func=lambda x: {60: "1분", 120: "2분", 300: "5분"}[x],
        disabled=not auto_refresh,
    )
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
    st.markdown("**패턴 신호 범례**")
    st.markdown("""
📈 **사라 경고 (매수)**
- 상승비기형 → 폭등 대비 (100%)
- 깃발형 돌파 → 급하게 사 (80%)
- 역삼각형 돌파 → 급하게 사 (65%)

📉 **팔아라 경고 (매도)**
- 쌍봉 → 폭락 대비 (100%)
- 하락깃발 → 빨리 팔아 (80%)
- 하락 다이아몬드 → 천천히 매도 (65%)
- 박스권 → 건들지마 위험 (50%)
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

# ─── 감시 종목 스캔 (전체 페이지 공유) ───────────────────────────────────────

with st.spinner("감시 종목 스캔 중... (최초 실행 시 1~2분 소요)"):
    signals = load_watchlist_signals()

buy_signals  = [s for s in signals if s.get("신호유형") == "BUY"]
sell_signals = [s for s in signals if s.get("신호유형") == "SELL"]

# ─── 매수 신호 알림 ────────────────────────────────────────────────────────────

st.subheader("🚨 매수 신호 알림")

if buy_signals:
    for s in buy_signals:
        conf   = s.get("신뢰도") or 0
        filled = round(conf * 5)
        bar    = "⬛" * filled + "⬜" * (5 - filled)
        chg    = s["전일대비"]
        chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 2, 2.5])
            with c1:
                st.markdown(f"### {s['종목명']}")
                st.caption(f"`{s['티커']}` | {s.get('패턴', '-')}")
            with c2:
                price = s.get("진입가")
                st.metric("추천 진입가", f"{price:,.0f}원" if price else "-")
            with c3:
                st.metric("전일대비", chg_str)
            with c4:
                rsi = s.get("rsi")
                st.metric("RSI", f"{rsi:.1f}" if rsi else "-")
            with c5:
                st.metric("신뢰도", f"{bar} {conf*100:.0f}%")
                st.success(s.get("행동지침") or "📈 매수 신호")
else:
    st.info("현재 매수 신호 종목이 없습니다. 스캔 주기마다 자동으로 재확인합니다.")

if sell_signals:
    with st.expander(f"📉 매도 경고 종목 {len(sell_signals)}개 보기"):
        for s in sell_signals:
            chg = s["전일대비"]
            chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"
            st.warning(
                f"**{s['종목명']}** ({s['티커']}) | "
                f"{s.get('패턴', '-')} | {chg_str} | "
                f"{s.get('행동지침', '-')}"
            )

st.divider()

# ─── 국내/해외 증시 이슈 ──────────────────────────────────────────────────────

st.subheader("📰 국내/해외 증시 이슈")

with st.spinner("글로벌 지수 & 이슈 데이터 로딩 중..."):
    indices    = load_market_indices()
    naver_news = load_naver_market_news()

tab_kr, tab_us, tab_global = st.tabs(["🇰🇷 국내 증시", "🇺🇸 해외 증시", "🌐 증시 종합"])

# ── 국내 증시 탭 ──────────────────────────────────────────────────────────────
with tab_kr:
    kr_keys = ["코스피", "코스닥"]
    kr_cols = st.columns(len(kr_keys))
    for col, name in zip(kr_cols, kr_keys):
        d = indices.get(name)
        with col:
            if d:
                chg = d["change"]
                arrow = "▲" if chg >= 0 else "▼"
                color = "normal" if chg >= 0 else "inverse"
                st.metric(name, f"{d['price']:,.2f}", f"{arrow} {abs(chg):.2f}%",
                          delta_color=color)
            else:
                st.metric(name, "데이터 없음")

    st.markdown("---")
    st.markdown("**📌 국내 증시 이슈 & 토론** *(네이버 증권)*")

    domestic = naver_news.get("domestic", {})
    if domestic.get("items"):
        for i, item in enumerate(domestic["items"], 1):
            st.markdown(f"{i}. {item}")
    elif domestic.get("error"):
        st.caption(f"⚠️ 크롤링 실패: {domestic['error']}")
        st.info("페이지를 직접 확인하세요.")
    else:
        st.caption("뉴스/토론 항목을 불러오지 못했습니다. 아래 링크를 통해 확인하세요.")

    st.link_button(
        "🔗 네이버 KOSPI 상세 보기 →",
        "https://m.stock.naver.com/domestic/index/KOSPI/total",
        use_container_width=True,
    )

# ── 해외 증시 탭 ──────────────────────────────────────────────────────────────
with tab_us:
    us_keys = ["S&P 500", "나스닥 100", "닛케이 225", "항셍"]
    us_cols = st.columns(len(us_keys))
    for col, name in zip(us_cols, us_keys):
        d = indices.get(name)
        with col:
            if d:
                chg = d["change"]
                arrow = "▲" if chg >= 0 else "▼"
                color = "normal" if chg >= 0 else "inverse"
                price_str = f"{d['price']:,.2f}" + (f" {d['unit']}" if d["unit"] else "")
                st.metric(name, price_str, f"{arrow} {abs(chg):.2f}%",
                          delta_color=color)
            else:
                st.metric(name, "데이터 없음")

    st.markdown("---")
    st.markdown("**📌 미국 증시 이슈 & 토론** *(네이버 증권)*")

    overseas = naver_news.get("overseas", {})
    if overseas.get("items"):
        for i, item in enumerate(overseas["items"], 1):
            st.markdown(f"{i}. {item}")
    elif overseas.get("error"):
        st.caption(f"⚠️ 크롤링 실패: {overseas['error']}")
    else:
        st.caption("토론 항목을 불러오지 못했습니다. 아래 링크를 통해 확인하세요.")

    st.link_button(
        "🔗 네이버 미국 증시 이슈 보기 →",
        "https://m.stock.naver.com/worldstock/home/USA/discussion/ranking",
        use_container_width=True,
    )

# ── 증시 종합 탭 ──────────────────────────────────────────────────────────────
with tab_global:
    overview = naver_news.get("overview", {})
    if overview.get("items"):
        st.markdown("**📌 증시 종합 뉴스** *(네이버 증권)*")
        for i, item in enumerate(overview["items"], 1):
            st.markdown(f"{i}. {item}")
        st.markdown("---")

    # 전체 지수 테이블
    st.markdown("**🌐 글로벌 주요 지수 현황**")
    table_rows = []
    for name, d in indices.items():
        chg = d["change"]
        arrow = "▲" if chg >= 0 else "▼"
        badge = "🟢" if chg >= 0 else "🔴"
        table_rows.append({
            "지수":   name,
            "현재가": f"{d['price']:,.2f}" + (f" {d['unit']}" if d["unit"] else ""),
            "전일대비": f"{badge} {arrow} {abs(chg):.2f}%",
        })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.link_button(
        "🔗 네이버 증시 종합 보기 →",
        "https://m.stock.naver.com/",
        use_container_width=True,
    )

st.divider()

# ─── 감시 종목 현황 ───────────────────────────────────────────────────────────

st.subheader("📋 감시 종목 현황")

buy_count     = len(buy_signals)
sell_count    = len(sell_signals)
neutral_count = sum(1 for s in signals if s.get("신호유형") == "NEUTRAL")

col_b, col_s, col_n, col_t = st.columns(4)
with col_b:
    st.metric("📈 매수 신호", f"{buy_count}개")
with col_s:
    st.metric("📉 매도 경고", f"{sell_count}개")
with col_n:
    st.metric("⚠️ 중립/위험", f"{neutral_count}개")
with col_t:
    st.metric("🔍 전체 감시", f"{len(signals)}개")


def _build_rows(source: list[dict]) -> list[dict]:
    rows = []
    for s in source:
        price_str = f"{s['현재가']:,.0f}원" if s["현재가"] else "-"
        chg = s["전일대비"]
        chg_str = (f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%") if chg is not None else "-"
        conf = s["신뢰도"]
        if conf is not None:
            filled = round(conf * 5)
            conf_str = f"{'⬛' * filled}{'⬜' * (5 - filled)} {conf*100:.0f}%"
        else:
            conf_str = "-"
        sig = s.get("신호유형")
        signal_icon = {"BUY": "📈 매수신호", "SELL": "📉 매도경고", "NEUTRAL": "⚠️ 위험중립"}.get(sig, "-")
        rows.append({
            "종목명":    s["종목명"],
            "현재가":    price_str,
            "전일대비":  chg_str,
            "감지 패턴": s["패턴"],
            "신뢰도":    conf_str,
            "신호":      signal_icon,
            "행동 지침": s.get("행동지침") or "-",
        })
    return rows


tab_all, tab_buy, tab_sell = st.tabs([
    f"전체 종목 ({len(signals)})",
    f"📈 매수 신호만 ({buy_count})",
    f"📉 매도 경고만 ({sell_count})",
])

with tab_all:
    st.dataframe(pd.DataFrame(_build_rows(signals)), use_container_width=True, hide_index=True)

with tab_buy:
    if buy_signals:
        st.dataframe(pd.DataFrame(_build_rows(buy_signals)), use_container_width=True, hide_index=True)
    else:
        st.info("현재 매수 신호 종목이 없습니다.")

with tab_sell:
    if sell_signals:
        st.dataframe(pd.DataFrame(_build_rows(sell_signals)), use_container_width=True, hide_index=True)
    else:
        st.success("매도 경고 종목이 없습니다.")

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
    signal_type = selected_signal.get("신호유형", "BUY")
    action = selected_signal.get("행동지침", "")
    if signal_type == "SELL":
        st.error(f"📉 **{selected_signal['패턴']}** 매도 경고 | {action} | {selected_signal.get('패턴상세', '')}")
    elif signal_type == "NEUTRAL":
        st.warning(f"⚠️ **{selected_signal['패턴']}** 위험 중립 | {action} | {selected_signal.get('패턴상세', '')}")
    else:
        st.info(f"📈 **{selected_signal['패턴']}** 매수 신호 | {action} | {selected_signal.get('패턴상세', '')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = selected_signal.get("진입가")
        label = "현재가" if signal_type in ("SELL", "NEUTRAL") else "추천 진입가"
        st.metric(label, f"{v:,.0f}원" if v else "-")
    with c2:
        v = selected_signal.get("저항선")
        st.metric("저항선", f"{v:,.0f}원" if v else "-")
    with c3:
        v = selected_signal.get("지지선")
        st.metric("지지선", f"{v:,.0f}원" if v else "-")
    with c4:
        v = selected_signal.get("rsi")
        st.metric("RSI", f"{v:.1f}" if v else "-")

    # 매수/매도 패턴이 동시에 감지된 경우 경고 표시
    buy_p = selected_signal.get("_buy_pattern")
    sell_p = selected_signal.get("_sell_pattern")
    if buy_p and sell_p:
        st.warning(
            f"⚡ 매수/매도 패턴 충돌 감지 | "
            f"매도우선({sell_p.pattern.value}) vs 매수({buy_p.pattern.value}) — 관망 권장"
        )

with st.spinner("차트 로딩 중..."):
    df_chart = load_ohlcv(selected_ticker, period=chart_period)

if df_chart is not None and not df_chart.empty:
    fig = make_chart(df_chart, selected_ticker, selected_signal or {})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"{selected_label} 차트 데이터를 불러올 수 없습니다.")

# ─── 자동 새로고침 ────────────────────────────────────────────────────────────

if auto_refresh:
    now = time.time()
    if "next_refresh" not in st.session_state:
        st.session_state.next_refresh = now + refresh_interval
    remaining = int(st.session_state.next_refresh - now)
    if remaining <= 0:
        st.cache_data.clear()
        st.session_state.next_refresh = now + refresh_interval
        st.rerun()
    else:
        st.sidebar.caption(f"⏱ 다음 갱신: {remaining}초 후")
        time.sleep(1)
        st.rerun()
