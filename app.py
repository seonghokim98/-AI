"""
app.py - 주식 매매 신호 시각화 대시보드 (알고리즘 정의서 기반)
Streamlit 기반 웹앱

[알고리즘 반영]
  - 관심 종목 풀: PBR<1.0 AND PER<15 AND EPS>0 (기본적 분석)
  - 추격매수 차단: RSI>80 or MA20 이격도>15% → WAIT 표시
  - 4대 핵심 패턴: 이평선눌림목 / 지지저항전환 / 쌍바닥 / 깃발형
  - 진입 타점: MA20 ±2% + 거래량 50%↓ + 도지 → 지정가 MA20 매수
  - 절대 방어: 체결 즉시 -10% 스탑로스 자동 세팅

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
import valuation as vl
import trade_engine as te
import slack_bot as sb

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="주식 매매 신호 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 데이터 로딩 ──────────────────────────────────────────────────────────────

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


_INDEX_MAP = {
    "코스피":     ("^KS11",    ""),
    "코스닥":     ("^KQ11",    ""),
    "S&P 500":   ("^GSPC",    "USD"),
    "나스닥 100": ("^NDX",     "USD"),
    "닛케이 225": ("^N225",    "JPY"),
    "항셍":       ("^HSI",     "HKD"),
    "달러인덱스": ("DX-Y.NYB", ""),
    "금 선물":    ("GC=F",     "USD"),
    "WTI 원유":   ("CL=F",     "USD"),
    "VIX":        ("^VIX",     ""),
}


def _is_korean_market_open() -> bool:
    """한국 장중 여부 판단 (KST 09:00~15:30, 주말 제외)"""
    from datetime import timezone, timedelta
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


@st.cache_data(ttl=30)
def load_market_indices() -> dict:
    """지수 데이터 로드 — 네이버 금융 실시간 우선, yfinance 폴백 (TTL 30초)"""
    all_tickers = [v[0] for v in _INDEX_MAP.values()]

    # 1차: 네이버 금융 실시간 폴링 API
    naver_rt = dp.fetch_naver_realtime_indices(all_tickers)

    result = {}
    for name, (ticker, unit) in _INDEX_MAP.items():
        if ticker in naver_rt:
            d = naver_rt[ticker]
            result[name] = {
                "price":  d["price"],
                "change": d["change_pct"],
                "unit":   unit,
                "rt":     True,
            }
            continue

        # 2차: yfinance 폴백
        try:
            df = dp.get_index_data(ticker, period="5d")
            if df is not None and len(df) >= 2:
                cur  = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                chg  = (cur - prev) / prev * 100
                result[name] = {"price": cur, "change": chg, "unit": unit, "rt": False}
        except Exception:
            pass
    return result


_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    "Referer": "https://m.stock.naver.com/",
}
_NAVER_URLS = {
    "domestic": "https://m.stock.naver.com/domestic/index/KOSPI/total",
    "overseas": "https://m.stock.naver.com/worldstock/home/USA/discussion/ranking",
    "overview": "https://m.stock.naver.com/",
}


def _parse_next_data(key: str, nd: dict) -> list:
    items = []
    try:
        props = nd.get("props", {}).get("pageProps", {})
        candidates = (props.get("discussions") or props.get("opinions") or
                      props.get("news") or props.get("rankings") or
                      props.get("articles") or [])
        for obj in candidates[:12]:
            text = (obj.get("title") or obj.get("headline") or obj.get("content", "")[:80]).strip()
            if 5 < len(text) < 200:
                stock = ((obj.get("stock") or {}).get("name", "") or
                         obj.get("market", "") or obj.get("source", ""))
                items.append(f"[{stock}] {text}" if stock else text)
    except Exception:
        pass
    return items


def _parse_html_fallback(soup) -> list:
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
    result = {k: {"items": [], "error": None} for k in _NAVER_URLS}
    if not _BS4_OK:
        for k in result:
            result[k]["error"] = "beautifulsoup4 미설치 (pip install beautifulsoup4)"
        return result
    for key, url in _NAVER_URLS.items():
        try:
            resp = requests.get(url, headers=_NAVER_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            nd_tag = soup.find("script", {"id": "__NEXT_DATA__"})
            if nd_tag and nd_tag.string:
                items = _parse_next_data(key, json.loads(nd_tag.string))
                if items:
                    result[key]["items"] = items
                    continue
            result[key]["items"] = _parse_html_fallback(soup)
        except requests.Timeout:
            result[key]["error"] = "연결 시간 초과 (10s)"
        except Exception as exc:
            result[key]["error"] = str(exc)[:80]
    return result


@st.cache_data(ttl=300)
def load_watchlist_signals() -> list:
    """
    [알고리즘] 감시 종목 3단계 파이프라인:
    1단계: 밸류에이션 필터 (PBR<1.0, PER<15) → 관심 종목 풀 편입
    2단계: 추격매수 차단 (RSI>80 or 이격도>15%)
    3단계: 4대 패턴 + Find_Pullback_Entry → 타점 포착
    """
    # 배치 다운로드로 개별 20회 → 1회 요청으로 속도 개선
    dp.prefetch_all_ohlcv()

    results = []
    for ticker in config.WATCHLIST:
        name   = config.TICKER_NAME.get(ticker, ticker)
        market = config.TICKER_MARKET.get(ticker, "KOSPI")
        df     = dp.get_ohlcv(ticker)

        if df is None or df.empty:
            results.append(_empty_row(ticker, name, market))
            continue

        current_price = float(df["Close"].iloc[-1])
        prev_price    = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        change_pct    = (current_price - prev_price) / prev_price * 100

        # 1단계: 밸류에이션 필터
        val = vl.check_valuation(ticker)

        # 2단계: 추격매수 차단
        ob_status, rsi_val, disparity_val = pe.check_overbought(df)

        # 3단계: 패턴 탐지
        buy_pattern  = pe.detect_patterns(df)
        sell_pattern = pe.detect_sell_patterns(df)

        # 매매 계획 (타점 포착 시)
        trade_order = None
        if buy_pattern is not None:
            try:
                trade_order = te.calculate_trade(
                    ticker=ticker,
                    entry_price=buy_pattern.entry_price,
                    resistance_level=buy_pattern.resistance_level,
                )
            except Exception:
                pass

        pattern      = sell_pattern if sell_pattern is not None else buy_pattern
        pool_status  = _get_pool_status(val.in_pool, pattern, ob_status)

        results.append({
            "종목명":    name,
            "티커":      ticker,
            "시장":      market,
            "현재가":    current_price,
            "전일대비":  change_pct,
            "풀편입":    val.in_pool,
            "PER":       val.per,
            "PBR":       val.pbr,
            "PER_OK":    val.per_ok,
            "PBR_OK":    val.pbr_ok,
            "모멘텀":    val.momentum_bonus,
            "밸류상세":  val.detail,
            "과매수상태": ob_status,
            "RSI":       rsi_val,
            "이격도":    disparity_val,
            "패턴":      pattern.pattern.value if pattern else "-",
            "신뢰도":    float(pattern.confidence) if pattern else None,
            "신호유형":  pattern.signal_type if pattern else None,
            "행동지침":  pattern.action if pattern else None,
            "진입가":    float(pattern.entry_price) if pattern else None,
            "진입근거":  pattern.entry_reason if pattern else None,
            "저항선":    float(pattern.resistance_level) if pattern else None,
            "지지선":    float(pattern.support_level) if pattern else None,
            "패턴상세":  pattern.detail if pattern else None,
            "스탑로스가": trade_order.stop_loss_price if trade_order else None,
            "목표가1R":  trade_order.target_1r if trade_order else None,
            "수량":      trade_order.quantity if trade_order else None,
            "투입금액":  trade_order.budget_used if trade_order else None,
            "손익비":    trade_order.reward_risk if trade_order else None,
            "풀상태":    pool_status,
            "_buy_pattern":  buy_pattern,
            "_sell_pattern": sell_pattern,
        })
    return results


def _empty_row(ticker, name, market):
    return {
        "종목명": name, "티커": ticker, "시장": market,
        "현재가": None, "전일대비": None,
        "풀편입": False, "PER": None, "PBR": None,
        "PER_OK": False, "PBR_OK": False, "모멘텀": False, "밸류상세": "-",
        "과매수상태": "SAFE", "RSI": None, "이격도": None,
        "패턴": "-", "신뢰도": None, "신호유형": None,
        "행동지침": None, "진입가": None, "진입근거": None,
        "저항선": None, "지지선": None, "패턴상세": None,
        "스탑로스가": None, "목표가1R": None, "수량": None,
        "투입금액": None, "손익비": None,
        "풀상태": "데이터 없음",
        "_buy_pattern": None, "_sell_pattern": None,
    }


def _get_pool_status(in_pool: bool, pattern, ob_status: str) -> str:
    if pattern and pattern.signal_type == "BUY":
        return "🎯 타점 포착!" if in_pool else "🎯 타점 포착 (풀 미편입)"
    if pattern and pattern.signal_type == "SELL":
        return "⚠️ 매도 경고"
    if pattern and pattern.signal_type == "NEUTRAL":
        return "⚠️ 중립 위험"
    if ob_status == "WAIT":
        return "🚫 과매수 차단"
    if in_pool:
        return "🔍 관심 종목 풀 (타점 대기)"
    return "📋 모니터링 중"


@st.cache_data(ttl=300)
def load_ohlcv(ticker: str, period: str = "6mo"):
    return dp.get_ohlcv(ticker, period=period)


@st.cache_data(ttl=30)
def load_realtime_prices() -> dict:
    """
    감시 종목 한국 주식 실시간 현재가 (네이버 금융 폴링 API, TTL 30초).
    장중: 실시간 / 장외: 전일 종가 기준
    Returns: {ticker: {"price": float, "change_pct": float}}
    """
    code_to_ticker = {}
    for ticker in config.WATCHLIST:
        if ".KS" in ticker or ".KQ" in ticker:
            code = ticker.split(".")[0]
            code_to_ticker[code] = ticker

    raw = dp.fetch_naver_realtime_prices(list(code_to_ticker.keys()))
    return {code_to_ticker[code]: data for code, data in raw.items() if code in code_to_ticker}


@st.cache_data(ttl=300)
def load_heatmap_data() -> list:
    """감시 종목 등락률 데이터 (히트맵용)"""
    dp.prefetch_all_ohlcv()
    rows = []
    for ticker in config.WATCHLIST:
        name = config.TICKER_NAME.get(ticker, ticker)
        df = dp.get_ohlcv(ticker)
        if df is not None and len(df) >= 2:
            cur = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            rows.append({
                "name": name,
                "ticker": ticker,
                "change": round(chg, 2),
                "price": cur,
                "market": config.TICKER_MARKET.get(ticker, "KOSPI"),
            })
    return rows


@st.cache_data(ttl=300)
def load_fear_greed_index() -> dict:
    """
    공포/탐욕 지수 계산 (0 = 극도 공포, 100 = 극도 탐욕)

    구성 요소:
      - VIX 지수 (40%): VIX ≤ 12 → 탐욕, VIX ≥ 40 → 공포
      - KOSPI 20일 모멘텀 (30%): ±10% 범위를 0~100 매핑
      - KOSPI MA20 위치 (30%): MA 위=75, MA 아래=25
    """
    # VIX
    try:
        vix_df = dp.get_index_data("^VIX", "5d")
        vix = float(vix_df["Close"].iloc[-1]) if vix_df is not None and not vix_df.empty else 20.0
    except Exception:
        vix = 20.0
    vix_score = max(0.0, min(100.0, (40.0 - vix) / 28.0 * 100.0))

    # KOSPI 20일 모멘텀
    try:
        kospi = dp.get_index_data("^KS11", "3mo")
        if kospi is not None and len(kospi) >= 20:
            pct20 = (float(kospi["Close"].iloc[-1]) - float(kospi["Close"].iloc[-20])) / float(kospi["Close"].iloc[-20]) * 100
            momentum_score = max(0.0, min(100.0, 50.0 + pct20 * 5.0))
        else:
            momentum_score = 50.0
            pct20 = 0.0
    except Exception:
        momentum_score = 50.0
        pct20 = 0.0

    # KOSPI MA20 위치
    try:
        market_st = mf.analyze_market()
        ma_score = 75.0 if market_st.kospi_above_ma else 25.0
    except Exception:
        ma_score = 50.0

    score = int(max(0, min(100, round(0.4 * vix_score + 0.3 * momentum_score + 0.3 * ma_score))))

    if score <= 20:
        label, color, emoji = "극도 공포", "#c62828", "😱"
    elif score <= 40:
        label, color, emoji = "공포", "#ef5350", "😨"
    elif score <= 60:
        label, color, emoji = "중립", "#ffa726", "😐"
    elif score <= 80:
        label, color, emoji = "탐욕", "#26a69a", "😊"
    else:
        label, color, emoji = "극도 탐욕", "#00695c", "🤑"

    return {
        "score": score,
        "label": label,
        "color": color,
        "emoji": emoji,
        "vix": round(vix, 2),
        "vix_score": round(vix_score, 1),
        "momentum_pct": round(pct20, 2),
        "momentum_score": round(momentum_score, 1),
        "ma_score": ma_score,
    }


# ─── 차트 생성 ────────────────────────────────────────────────────────────────

def make_chart(df, ticker: str, signal: dict) -> go.Figure:
    name = config.TICKER_NAME.get(ticker, ticker)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{name} ({ticker})", "거래량", "RSI"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="가격",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    for col_name, color, width in [("MA5", "#ff9800", 1.2), ("MA20", "#42a5f5", 2.5), ("MA60", "#ab47bc", 2)]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=col_name,
                line=dict(color=color, width=width), opacity=0.9,
            ), row=1, col=1)

    if "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"], name="BB 상단",
            line=dict(color="gray", width=1, dash="dot"), opacity=0.5, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"], name="BB 하단",
            line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.07)",
            opacity=0.5, showlegend=False,
        ), row=1, col=1)

    sig_type = signal.get("신호유형", "BUY")
    if signal.get("패턴") and signal["패턴"] != "-":
        if signal.get("진입가"):
            color = "#00e5ff" if sig_type == "BUY" else "#ff5252"
            lbl   = "지정가(MA20)" if sig_type == "BUY" else "현재가"
            fig.add_hline(y=signal["진입가"], line_color=color, line_dash="dash", line_width=2,
                          annotation_text=f"{lbl} {signal['진입가']:,.0f}",
                          annotation_font_color=color, row=1, col=1)
        if signal.get("스탑로스가"):
            fig.add_hline(y=signal["스탑로스가"], line_color="#ff1744", line_dash="dot", line_width=1.5,
                          annotation_text=f"손절(-10%) {signal['스탑로스가']:,.0f}",
                          annotation_font_color="#ff1744", row=1, col=1)
        if signal.get("목표가1R"):
            fig.add_hline(y=signal["목표가1R"], line_color="#69f0ae", line_dash="dot", line_width=1.5,
                          annotation_text=f"1차 목표 {signal['목표가1R']:,.0f}",
                          annotation_font_color="#69f0ae", row=1, col=1)
        if signal.get("저항선"):
            fig.add_hline(y=signal["저항선"], line_color="#ffd54f", line_dash="dot", line_width=1,
                          annotation_text=f"저항선 {signal['저항선']:,.0f}",
                          annotation_font_color="#ffd54f", row=1, col=1)

    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량",
                         marker_color=vol_colors, opacity=0.75), row=2, col=1)
    if "VOL_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VOL_MA20"], name="거래량 MA20",
                                 line=dict(color="#ff9800", width=1.5)), row=2, col=1)
        # 거래량 50% 기준선 (눌림목 진입 조건)
        vol_dry_line = df["VOL_MA20"].dropna().iloc[-1] * config.TECH.volume_dry_ratio if len(df["VOL_MA20"].dropna()) > 0 else 0
        if vol_dry_line > 0:
            fig.add_hline(y=vol_dry_line, line_color="#ff9800", line_dash="dot", line_width=1,
                          annotation_text=f"거래량 50% ({vol_dry_line:,.0f})",
                          annotation_font_color="#ff9800", row=2, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(color="#e91e63", width=2)), row=3, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.12)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.08)", line_width=0, row=3, col=1)
        fig.add_hline(y=80, line_color="#ef5350", line_dash="dash", line_width=1.5,
                      annotation_text="과매수 80 (매수 차단)", row=3, col=1)
        fig.add_hline(y=30, line_color="#26a69a", line_dash="dash", line_width=1,
                      annotation_text="과매도 30", row=3, col=1)

    fig.update_layout(
        height=720, template="plotly_dark", showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=90, t=40, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
    fig.update_yaxes(title_text="거래량",   row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
    fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
    return fig


def make_heatmap_chart(heatmap_rows: list) -> go.Figure:
    """감시 종목 등락률 트리맵 히트맵"""
    if not heatmap_rows:
        return None

    labels  = [f"{d['name']}<br>{d['change']:+.2f}%" for d in heatmap_rows]
    changes = [d["change"] for d in heatmap_rows]
    custom  = [[d["ticker"], f"{d['price']:,.0f}", f"{d['change']:+.2f}%"] for d in heatmap_rows]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(heatmap_rows),
        values=[1] * len(heatmap_rows),
        customdata=custom,
        marker=dict(
            colors=changes,
            colorscale=[
                [0.0,  "#c62828"],
                [0.35, "#ef9a9a"],
                [0.5,  "#37474f"],
                [0.65, "#80cbc4"],
                [1.0,  "#00695c"],
            ],
            cmid=0,
            showscale=True,
            colorbar=dict(title="등락률(%)", ticksuffix="%", len=0.85, thickness=14),
        ),
        textfont=dict(size=13, color="white"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "현재가: %{customdata[1]}원<br>"
            "등락률: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="#0e1117",
    )
    return fig


def make_fear_greed_gauge(fg: dict) -> go.Figure:
    """공포/탐욕 지수 게이지 차트"""
    score = fg["score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": fg["color"]}},
        title={"text": f"{fg['emoji']} {fg['label']}", "font": {"size": 16, "color": fg["color"]}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0", "20", "40", "60", "80", "100"],
                "tickfont": {"size": 10, "color": "white"},
            },
            "bar": {"color": fg["color"], "thickness": 0.28},
            "bgcolor": "#0e1117",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  20],  "color": "rgba(198, 40, 40, 0.35)"},
                {"range": [20, 40],  "color": "rgba(239, 83, 80, 0.28)"},
                {"range": [40, 60],  "color": "rgba(255, 167, 38, 0.22)"},
                {"range": [60, 80],  "color": "rgba(38, 166, 154, 0.28)"},
                {"range": [80, 100], "color": "rgba(0, 105, 92, 0.35)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=260,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    auto_refresh = st.toggle("자동 새로고침", value=False)
    refresh_interval = st.selectbox(
        "새로고침 주기", [60, 120, 300],
        format_func=lambda x: {60: "1분", 120: "2분", 300: "5분"}[x],
        disabled=not auto_refresh,
    )
    chart_period = st.selectbox(
        "차트 기간", ["3mo", "6mo", "1y"], index=1,
        format_func=lambda x: {"3mo": "3개월", "6mo": "6개월", "1y": "1년"}[x],
    )

    st.markdown("---")
    st.markdown("**[알고리즘] 매매 원칙**")
    st.markdown(f"""
**관심 종목 풀** (기본 분석)
- PBR **< {config.VALUATION.max_pbr}** (순자산 이하)
- PER **< {config.VALUATION.max_per:.0f}** (저평가 기준)
- EPS > 0 (흑자 기업만)
- 모멘텀: 자사주 소각/배당/밸류업

**추격매수 차단** (절대 원칙)
- RSI **> {config.TECH.rsi_overbought:.0f}** → 매수 비활성화
- MA20 이격도 **> {config.TECH.disparity_overbought:.0f}%** → 비활성화

**진입 타점** (Find_Pullback_Entry)
- MA20 ±{config.TECH.pullback_ma_tolerance*100:.0f}% 이내
- 거래량 < 평균 {config.TECH.volume_dry_ratio*100:.0f}%
- 도지 캔들 (몸통 < 범위 {config.TECH.doji_body_ratio*100:.0f}%)
- → **지정가 MA20 매수**

**절대 방어** (체결 즉시)
- 손절 **-{abs(config.STOP_LOSS_RATE)*100:.0f}%** (예외 없음)
- 최소 손익비 **{config.MIN_REWARD_RISK_RATIO:.1f}:1**
""")

    st.markdown("---")
    st.markdown("**4대 핵심 패턴**")
    st.markdown("""
📈 **매수 패턴 (눌림목 타점)**
- 이평선 눌림목 → MA20 되돌림
- 지지/저항 전환 → 구 저항→지지
- 쌍바닥 → W패턴 넥라인 후 눌림
- 깃발형 응축 → 거래량 수렴

📉 **매도 경고**
- 쌍봉 → 폭락 대비
- 하락깃발 → 빨리 팔아
- 하락 다이아몬드 → 단계적 매도
- 박스권 → 건들지마 위험
""")

    st.markdown("---")
    st.markdown("**📲 슬랙 알림 설정**")
    slack_ok = bool(config.SLACK_WEBHOOK_URL)
    if slack_ok:
        st.success("✅ 슬랙 웹훅 연결됨")
    else:
        st.error("❌ SLACK_WEBHOOK_URL 미설정")
        st.caption(".env 파일에 SLACK_WEBHOOK_URL을 추가하세요.")

    slack_enabled = st.toggle(
        "매수/매도 감지 시 슬랙 알림",
        value=slack_ok,
        disabled=not slack_ok,
        key="slack_alert_enabled",
    )

    if slack_ok and st.button("📤 슬랙 테스트 발송", use_container_width=True):
        if sb.send_test():
            st.success("테스트 메시지 발송 완료!")
        else:
            st.error("발송 실패 — 웹훅 주소를 확인하세요")

    if "slack_sent_log" in st.session_state and st.session_state.slack_sent_log:
        with st.expander(f"📋 발송 내역 ({len(st.session_state.slack_sent_log)}건)"):
            for log in reversed(st.session_state.slack_sent_log[-10:]):
                st.caption(log)

    st.markdown("---")
    if st.button("🗑️ 캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.success("캐시 초기화 완료")


# ─── 헤더 ────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("📈 주식 매매 신호 대시보드")
    st.caption("알고리즘 정의서 기반 — 팩트+차트 논리 / 감정 배제 / 뇌동매매 원천 차단")
with col_btn:
    st.write("")
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

_market_open = _is_korean_market_open()
_rt_status   = "🟢 장중 실시간" if _market_open else "🔴 장외 (전일 종가)"
st.caption(
    f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"지수·종목: 네이버 금융 폴링 API ({_rt_status}) | 패턴분석: yfinance"
)

# ─── 시장 현황 ────────────────────────────────────────────────────────────────

st.subheader("🌍 시장 현황")
with st.spinner("시장 데이터 분석 중..."):
    market = load_market_status()

regime       = market["regime"]
regime_emoji = {"강세장": "🟢", "중립": "🟡", "약세장": "🔴"}.get(regime, "⚪")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("시장 상태", f"{regime_emoji} {regime}")
with m2: st.metric("코스피 20일선", "✅ MA 위" if market["kospi_above_ma"] else "❌ MA 아래")
with m3: st.metric("코스닥 20일선", "✅ MA 위" if market["kosdaq_above_ma"] else "❌ MA 아래")
with m4: st.metric("VIX", "✅ 정상" if market["vix_ok"] else "⚠️ 과열")

if regime == "약세장":
    st.error("⚠️ 약세장 — 모든 매수 신호 차단 중. 관망하세요.")
elif regime == "중립":
    st.warning("🟡 중립장 — 고품질 신호(신뢰도 높은 것)만 참고하세요.")
else:
    st.success("🟢 강세장 — 매수 신호 정상 작동 중.")

with st.expander("시장 상세 보기"):
    st.caption(market["detail"])
    st.caption("외국인 수급: KIS API 미연동" if not market["foreign_buy_streak"]
               else f"외국인 {market['foreign_buy_streak']}일 연속 순매수")

st.divider()

# ─── 실시간 가격 사전 로드 (히트맵 + 감시종목 현황 공통 사용, 30초 TTL) ───────
_rt_prices = load_realtime_prices()

# ─── 히트맵 & 공포/탐욕 지수 ──────────────────────────────────────────────────

st.subheader("🗺️ 감시 종목 히트맵 & 공포/탐욕 지수")

_hm_col, _fg_col = st.columns([3, 1], gap="medium")

with _hm_col:
    st.caption("감시 종목 등락률 히트맵 (초록=상승 / 빨강=하락)")
    with st.spinner("히트맵 로딩 중..."):
        _hm_data = load_heatmap_data()
    # 실시간 가격 반영 (네이버 금융 폴링 API)
    if _rt_prices:
        _hm_data = [
            {**d,
             "price":  _rt_prices[d["ticker"]]["price"],
             "change": round(_rt_prices[d["ticker"]]["change_pct"], 2)}
            if d["ticker"] in _rt_prices else d
            for d in _hm_data
        ]
    _hm_fig = make_heatmap_chart(_hm_data)
    if _hm_fig:
        st.plotly_chart(_hm_fig, use_container_width=True)
    else:
        st.info("히트맵 데이터를 불러오지 못했습니다.")

with _fg_col:
    st.caption("공포/탐욕 지수 (VIX·모멘텀·MA 위치 기반)")
    with st.spinner("지수 계산 중..."):
        _fg = load_fear_greed_index()
    _fg_fig = make_fear_greed_gauge(_fg)
    if _fg_fig:
        st.plotly_chart(_fg_fig, use_container_width=True)
    with st.expander("구성 요소 보기"):
        st.caption(f"VIX: {_fg['vix']:.1f} → 점수 {_fg['vix_score']:.0f}/100 (비중 40%)")
        st.caption(f"KOSPI 20일 모멘텀: {_fg.get('momentum_pct', 0):+.2f}% → 점수 {_fg['momentum_score']:.0f}/100 (비중 30%)")
        ma_label = "MA 위 (강세)" if _fg["ma_score"] >= 50 else "MA 아래 (약세)"
        st.caption(f"KOSPI MA20 위치: {ma_label} → 점수 {_fg['ma_score']:.0f}/100 (비중 30%)")
        st.caption("⚠️ 참고용 보조 지표입니다. 단독 매매 근거로 사용 금지.")

st.divider()

# ─── 감시 종목 스캔 ───────────────────────────────────────────────────────────

with st.spinner("감시 종목 스캔 중... (최초 실행 시 1~2분 소요)"):
    signals = load_watchlist_signals()

# ─── 실시간 현재가 반영 (네이버 금융 폴링 API, 30초 TTL) ────────────────────
# _rt_prices 는 히트맵 섹션에서 이미 로드됨 (load_realtime_prices 캐시 공유)
if _rt_prices:
    _updated_signals = []
    for _s in signals:
        _rt = _rt_prices.get(_s["티커"])
        if _rt:
            _s = dict(_s)
            _s["현재가"]   = _rt["price"]
            _s["전일대비"] = _rt["change_pct"]
        _updated_signals.append(_s)
    signals = _updated_signals

# ─── 슬랙 자동 알림 ────────────────────────────────────────────────────────

def _send_slack_alerts(signals: list) -> None:
    """
    매수/매도 신호 감지 시 슬랙으로 자동 알림.
    session_state로 중복 발송을 방지한다 (당일 동일 종목+패턴은 1회만 발송).
    """
    if not st.session_state.get("slack_alert_enabled", False):
        return
    if not config.SLACK_WEBHOOK_URL:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if "slack_sent_keys" not in st.session_state:
        st.session_state.slack_sent_keys = set()
    if "slack_sent_log" not in st.session_state:
        st.session_state.slack_sent_log = []

    # 날짜가 바뀌면 발송 이력 초기화
    if st.session_state.get("slack_sent_date") != today:
        st.session_state.slack_sent_keys = set()
        st.session_state.slack_sent_date = today

    for s in signals:
        ticker   = s["티커"]
        pattern  = s.get("패턴", "-")
        sig_type = s.get("신호유형")

        if sig_type not in ("BUY", "SELL", "NEUTRAL"):
            continue

        key = f"{today}_{ticker}_{pattern}_{sig_type}"
        if key in st.session_state.slack_sent_keys:
            continue

        sent = False
        if sig_type == "BUY":
            sent = sb.send_dashboard_buy_alert(s)
        elif sig_type in ("SELL", "NEUTRAL"):
            sent = sb.send_dashboard_sell_alert(s)

        if sent:
            st.session_state.slack_sent_keys.add(key)
            log_time = datetime.now().strftime("%H:%M")
            icon = "🎯" if sig_type == "BUY" else "📉"
            st.session_state.slack_sent_log.append(
                f"{log_time} {icon} {s['종목명']} ({pattern})"
            )

_send_slack_alerts(signals)

buy_signals     = [s for s in signals if s.get("신호유형") == "BUY"]
sell_signals    = [s for s in signals if s.get("신호유형") == "SELL"]
blocked_signals = [s for s in signals if s.get("과매수상태") == "WAIT"]

# ─── 매수 타점 알림 ────────────────────────────────────────────────────────────

st.subheader("🎯 매수 타점 포착 알림")
st.caption(
    "4대 패턴 Setup + Find_Pullback_Entry (MA20 ±2% & 거래량 50%↓ & 도지) 동시 충족 종목"
)

if buy_signals:
    for s in buy_signals:
        conf    = s.get("신뢰도") or 0
        filled  = round(conf * 5)
        bar     = "⬛" * filled + "⬜" * (5 - filled)
        chg     = s["전일대비"]
        chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 1.5, 2])
            with c1:
                st.markdown(f"### {s['종목명']}")
                st.caption(f"`{s['티커']}` | {s.get('시장','')} | {s.get('패턴', '-')}")
                pool_badge = "✅ 관심 풀 편입" if s.get("풀편입") else "⚠️ 풀 미편입"
                st.caption(pool_badge)
            with c2:
                price = s.get("진입가")
                st.metric("지정가 (MA20)", f"{price:,.0f}원" if price else "-")
                reason = s.get("진입근거")
                if reason:
                    st.caption(f"📌 {reason}")
            with c3:
                st.metric("전일대비", chg_str)
                rsi_v = s.get("RSI")
                st.metric("RSI", f"{rsi_v:.1f}" if rsi_v else "-")
            with c4:
                disp_v = s.get("이격도")
                st.metric("MA20 이격도", f"{disp_v:.1f}%" if disp_v is not None else "-")
                st.metric("신뢰도", f"{bar} {conf*100:.0f}%")
            with c5:
                stop = s.get("스탑로스가")
                t1   = s.get("목표가1R")
                rr   = s.get("손익비")
                st.metric("손절가 (-10%)", f"{stop:,.0f}원" if stop else "-")
                st.metric("1차 목표", f"{t1:,.0f}원" if t1 else "-")
                if rr:
                    st.caption(f"손익비 {rr:.1f}:1")

            per_str = (f"PER {s['PER']:.1f}{'✅' if s.get('PER_OK') else '❌'}"
                       if s.get("PER") and s["PER"] > 0 else "PER N/A")
            pbr_str = (f"PBR {s['PBR']:.2f}{'✅' if s.get('PBR_OK') else '❌'}"
                       if s.get("PBR") and s["PBR"] > 0 else "PBR N/A")
            mom_str = "모멘텀✅" if s.get("모멘텀") else ""
            st.caption(f"📊 {per_str} | {pbr_str}" + (f" | {mom_str}" if mom_str else ""))
            st.success(s.get("행동지침") or "📈 MA20 지정가 매수 검토")
else:
    st.info(
        "현재 타점 포착 종목 없음 — "
        "패턴 Setup과 눌림목 3조건(MA20±2% + 거래량50%↓ + 도지) 동시 충족 대기 중."
    )

if blocked_signals:
    with st.expander(f"🚫 추격매수 차단 {len(blocked_signals)}개 (RSI>80 or 이격도>15%)"):
        for s in blocked_signals:
            rsi_v  = s.get("RSI") or 0
            disp_v = s.get("이격도") or 0
            reasons = []
            if rsi_v > config.TECH.rsi_overbought:
                reasons.append(f"RSI {rsi_v:.1f}>80")
            if disp_v > config.TECH.disparity_overbought:
                reasons.append(f"이격도 {disp_v:.1f}%>15%")
            st.warning(f"**{s['종목명']}** ({s['티커']}) | 차단: {', '.join(reasons)} | 관망")

if sell_signals:
    with st.expander(f"📉 매도 경고 {len(sell_signals)}개"):
        for s in sell_signals:
            chg = s["전일대비"]
            chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"
            st.error(f"**{s['종목명']}** ({s['티커']}) | {s.get('패턴','-')} | {chg_str} | {s.get('행동지침','-')}")

st.divider()

# ─── 국내/해외 증시 이슈 ──────────────────────────────────────────────────────

st.subheader("📰 국내/해외 증시 이슈")
with st.spinner("글로벌 지수 & 이슈 로딩 중..."):
    indices    = load_market_indices()
    naver_news = load_naver_market_news()

tab_kr, tab_us, tab_global = st.tabs(["🇰🇷 국내 증시", "🇺🇸 해외 증시", "🌐 증시 종합"])

with tab_kr:
    kr_cols = st.columns(2)
    for col, name in zip(kr_cols, ["코스피", "코스닥"]):
        d = indices.get(name)
        with col:
            if d:
                chg = d["change"]
                rt_badge = " 🟢" if d.get("rt") else " 🔴"
                st.metric(
                    name + rt_badge,
                    f"{d['price']:,.2f}",
                    f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
                    delta_color="normal" if chg >= 0 else "inverse",
                    help="🟢 실시간 (네이버 금융 폴링 API)" if d.get("rt") else "🔴 지연 데이터 (yfinance)",
                )
            else:
                st.metric(name, "데이터 없음")
    st.markdown("---")
    st.markdown("**📌 국내 증시 이슈** *(네이버 증권)*")
    domestic = naver_news.get("domestic", {})
    if domestic.get("items"):
        for i, item in enumerate(domestic["items"], 1):
            st.markdown(f"{i}. {item}")
    elif domestic.get("error"):
        st.caption(f"⚠️ 크롤링 실패: {domestic['error']}")
    else:
        st.caption("뉴스 항목을 불러오지 못했습니다.")
    st.link_button("🔗 네이버 KOSPI →", "https://m.stock.naver.com/domestic/index/KOSPI/total", use_container_width=True)

with tab_us:
    us_cols = st.columns(4)
    for col, name in zip(us_cols, ["S&P 500", "나스닥 100", "닛케이 225", "항셍"]):
        d = indices.get(name)
        with col:
            if d:
                chg = d["change"]
                rt_badge = " 🟢" if d.get("rt") else " 🔴"
                price_str = f"{d['price']:,.2f}" + (f" {d['unit']}" if d["unit"] else "")
                st.metric(
                    name + rt_badge, price_str,
                    f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
                    delta_color="normal" if chg >= 0 else "inverse",
                    help="🟢 실시간 (네이버 금융 폴링 API)" if d.get("rt") else "🔴 지연 데이터 (yfinance)",
                )
            else:
                st.metric(name, "데이터 없음")
    st.markdown("---")
    st.markdown("**📌 미국 증시 이슈** *(네이버 증권)*")
    overseas = naver_news.get("overseas", {})
    if overseas.get("items"):
        for i, item in enumerate(overseas["items"], 1):
            st.markdown(f"{i}. {item}")
    elif overseas.get("error"):
        st.caption(f"⚠️ 크롤링 실패: {overseas['error']}")
    else:
        st.caption("항목을 불러오지 못했습니다.")
    st.link_button("🔗 네이버 미국 증시 →", "https://m.stock.naver.com/worldstock/home/USA/discussion/ranking", use_container_width=True)

with tab_global:
    overview = naver_news.get("overview", {})
    if overview.get("items"):
        st.markdown("**📌 증시 종합 뉴스**")
        for i, item in enumerate(overview["items"], 1):
            st.markdown(f"{i}. {item}")
        st.markdown("---")
    st.markdown("**🌐 글로벌 지수 현황**")
    table_rows = []
    for iname, d in indices.items():
        chg = d["change"]
        src = "실시간" if d.get("rt") else "지연"
        table_rows.append({
            "지수":    iname,
            "현재가":  f"{d['price']:,.2f}" + (f" {d['unit']}" if d["unit"] else ""),
            "전일대비": f"{'🟢' if chg >= 0 else '🔴'} {'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
            "출처":    src,
        })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.link_button("🔗 네이버 증시 종합 →", "https://m.stock.naver.com/", use_container_width=True)

st.divider()

# ─── 감시 종목 현황 ───────────────────────────────────────────────────────────

st.subheader("📋 감시 종목 현황")

buy_count     = len(buy_signals)
sell_count    = len(sell_signals)
pool_count    = sum(1 for s in signals if s.get("풀편입"))
blocked_count = len(blocked_signals)

col_b, col_s, col_p, col_w = st.columns(4)
with col_b: st.metric("🎯 타점 포착", f"{buy_count}개")
with col_s: st.metric("📉 매도 경고", f"{sell_count}개")
with col_p: st.metric("🔍 관심 종목 풀", f"{pool_count}개")
with col_w: st.metric("🚫 과매수 차단", f"{blocked_count}개")


def _build_rows(source: list) -> list:
    rows = []
    for s in source:
        price_str = f"{s['현재가']:,.0f}원" if s["현재가"] else "-"
        chg = s["전일대비"]
        chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"
        conf = s["신뢰도"]
        conf_str = (f"{'⬛' * round(conf * 5)}{'⬜' * (5 - round(conf * 5))} {conf*100:.0f}%"
                    if conf is not None else "-")
        entry_str = f"{s['진입가']:,.0f}원" if s.get("진입가") else "-"
        stop_str  = f"{s['스탑로스가']:,.0f}원" if s.get("스탑로스가") else "-"
        per_v = s.get("PER")
        pbr_v = s.get("PBR")
        val_str = " ".join(filter(None, [
            f"PE:{per_v:.0f}{'✅' if s.get('PER_OK') else '❌'}" if per_v and per_v > 0 else None,
            f"PB:{pbr_v:.1f}{'✅' if s.get('PBR_OK') else '❌'}" if pbr_v and pbr_v > 0 else None,
        ])) or "-"
        sig_icon = {"BUY": "🎯 타점", "SELL": "📉 매도", "NEUTRAL": "⚠️ 중립"}.get(s.get("신호유형"), "-")
        ob_str = "🚫" if s.get("과매수상태") == "WAIT" else ""
        rows.append({
            "시장":          s.get("시장", "-"),
            "종목명":        s["종목명"],
            "현재가":        price_str,
            "전일대비":      chg_str,
            "밸류에이션":    val_str,
            "풀편입":        "✅" if s.get("풀편입") else "-",
            "추격차단":      ob_str or "-",
            "감지 패턴":     s["패턴"],
            "신뢰도":        conf_str,
            "지정가(MA20)":  entry_str,
            "손절가(-10%)":  stop_str,
            "신호":          sig_icon,
            "풀 상태":       s.get("풀상태", "-"),
        })
    return rows


def _mkt(source: list, mkt: str) -> list:
    return [s for s in source if s.get("시장") == mkt]


kospi_sigs  = _mkt(signals, "KOSPI")
kosdaq_sigs = _mkt(signals, "KOSDAQ")
etf_sigs    = _mkt(signals, "ETF")

tab_buy, tab_k, tab_kq, tab_etf, tab_all = st.tabs([
    f"🎯 타점 포착 ({buy_count})",
    f"🏛️ KOSPI ({len(kospi_sigs)})",
    f"📊 KOSDAQ ({len(kosdaq_sigs)})",
    f"📦 ETF ({len(etf_sigs)})",
    f"📋 전체 ({len(signals)})",
])

with tab_buy:
    if buy_signals:
        st.dataframe(pd.DataFrame(_build_rows(buy_signals)), use_container_width=True, hide_index=True)
    else:
        st.info("타점 포착 종목 없음.")

with tab_k:
    st.dataframe(pd.DataFrame(_build_rows(kospi_sigs)), use_container_width=True, hide_index=True)

with tab_kq:
    st.dataframe(pd.DataFrame(_build_rows(kosdaq_sigs)), use_container_width=True, hide_index=True)

with tab_etf:
    st.dataframe(pd.DataFrame(_build_rows(etf_sigs)), use_container_width=True, hide_index=True)

with tab_all:
    st.dataframe(pd.DataFrame(_build_rows(signals)), use_container_width=True, hide_index=True)

st.divider()

# ─── 차트 분석 ────────────────────────────────────────────────────────────────

st.subheader("📊 차트 분석")

ticker_options = {f"{config.TICKER_NAME.get(t, t)} ({t})": t for t in config.WATCHLIST}

default_idx    = 0
signal_tickers = [s["티커"] for s in signals if s["패턴"] != "-"]
if signal_tickers:
    labels  = list(ticker_options.keys())
    tickers = list(ticker_options.values())
    if signal_tickers[0] in tickers:
        default_idx = tickers.index(signal_tickers[0])

selected_label  = st.selectbox("종목 선택", list(ticker_options.keys()), index=default_idx)
selected_ticker = ticker_options[selected_label]
selected_signal = next((s for s in signals if s["티커"] == selected_ticker), {})

if selected_signal.get("패턴") and selected_signal["패턴"] != "-":
    sig_type = selected_signal.get("신호유형", "BUY")
    action   = selected_signal.get("행동지침", "")

    if sig_type == "SELL":
        st.error(f"📉 **{selected_signal['패턴']}** | {action} | {selected_signal.get('패턴상세','')}")
    elif sig_type == "NEUTRAL":
        st.warning(f"⚠️ **{selected_signal['패턴']}** | {action}")
    else:
        st.info(f"🎯 **{selected_signal['패턴']}** | {action} | {selected_signal.get('패턴상세','')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = selected_signal.get("진입가")
        lbl = "지정가 (MA20)" if sig_type == "BUY" else "현재가"
        st.metric(lbl, f"{v:,.0f}원" if v else "-")
        reason = selected_signal.get("진입근거")
        if reason:
            st.caption(f"📌 {reason}")
    with c2:
        stop = selected_signal.get("스탑로스가")
        st.metric("손절가 (-10%)", f"{stop:,.0f}원" if stop else "-")
        t1 = selected_signal.get("목표가1R")
        if t1:
            st.caption(f"1차 목표: {t1:,.0f}원")
    with c3:
        rsi_v = selected_signal.get("RSI")
        st.metric("RSI", f"{rsi_v:.1f}" if rsi_v else "-")
        disp_v = selected_signal.get("이격도")
        if disp_v is not None:
            color = "🔴" if abs(disp_v) > config.TECH.disparity_overbought else "🟢"
            st.caption(f"{color} MA20 이격도: {disp_v:.1f}%")
    with c4:
        pool_ok = selected_signal.get("풀편입")
        st.metric("관심 종목 풀", "✅ 편입" if pool_ok else "❌ 미편입")
        per_v = selected_signal.get("PER")
        pbr_v = selected_signal.get("PBR")
        if per_v and pbr_v:
            st.caption(f"PER {per_v:.1f} | PBR {pbr_v:.2f}")

    ob = selected_signal.get("과매수상태")
    if ob == "WAIT":
        rsi_v  = selected_signal.get("RSI") or 0
        disp_v = selected_signal.get("이격도") or 0
        st.error(f"🚫 **추격매수 차단** — RSI {rsi_v:.1f} / 이격도 {disp_v:.1f}% "
                 f"(기준 RSI>{config.TECH.rsi_overbought:.0f} or 이격도>{config.TECH.disparity_overbought:.0f}%)")

    buy_p  = selected_signal.get("_buy_pattern")
    sell_p = selected_signal.get("_sell_pattern")
    if buy_p and sell_p:
        st.warning(f"⚡ 충돌 감지 — 매도({sell_p.pattern.value}) vs 매수({buy_p.pattern.value}) → 관망 권장")

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
