"""
slack_bot.py - 슬랙 알림 모듈
매수 신호 / 관망 신호 / 에러 알림을 슬랙 Webhook으로 발송한다.
자동 매수는 절대 없음. 알림만 발송.
"""
import logging
from datetime import datetime
from typing import Optional

import requests

import config
from market_filter import MarketStatus, MarketRegime
from pattern_engine import PatternResult
from risk import RiskResult
from valuation import ValuationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 대시보드 연동 알림 (app.py → 슬랙 실시간 알림)
# ---------------------------------------------------------------------------

def send_dashboard_buy_alert(signal: dict) -> bool:
    """
    대시보드 매수 타점 포착 시 슬랙 알림.
    app.py의 load_watchlist_signals() 결과 dict를 직접 받는다.
    """
    name    = signal.get("종목명", signal.get("티커", ""))
    ticker  = signal.get("티커", "")
    market  = signal.get("시장", "")
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")

    pattern   = signal.get("패턴", "-")
    conf      = signal.get("신뢰도") or 0
    conf_bar  = _confidence_emoji(conf)
    action    = signal.get("행동지침") or "MA20 지정가 매수 검토"
    reason    = signal.get("진입근거") or ""

    entry    = signal.get("진입가")
    stop     = signal.get("스탑로스가")
    target   = signal.get("목표가1R")
    rr       = signal.get("손익비")

    rsi_v    = signal.get("RSI")
    disp_v   = signal.get("이격도")
    chg      = signal.get("전일대비")
    chg_str  = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"

    per_v    = signal.get("PER")
    pbr_v    = signal.get("PBR")
    per_ok   = "✅" if signal.get("PER_OK") else "❌"
    pbr_ok   = "✅" if signal.get("PBR_OK") else "❌"
    pool_ok  = "✅ 관심 풀 편입" if signal.get("풀편입") else "⚠️ 풀 미편입"
    pool_st  = signal.get("풀상태", "")

    entry_str  = f"{entry:,.0f}원" if entry else "-"
    stop_str   = f"{stop:,.0f}원"  if stop  else "-"
    target_str = f"{target:,.0f}원" if target else "-"
    rr_str     = f"{rr:.1f}:1" if rr else "-"
    rsi_str    = f"{rsi_v:.1f}" if rsi_v else "-"
    disp_str   = f"{disp_v:.1f}%" if disp_v is not None else "-"
    per_str    = f"PER {per_v:.1f}{per_ok}" if per_v and per_v > 0 else "PER N/A"
    pbr_str    = f"PBR {pbr_v:.2f}{pbr_ok}" if pbr_v and pbr_v > 0 else "PBR N/A"

    msg = f"""🎯 *[매수 타점 포착]* — {now}

*종목:* {name} (`{ticker}`) | {market}
*패턴:* {pattern}
*신뢰도:* {conf_bar} ({conf*100:.0f}%)
*상태:* {pool_ok} | {pool_st}

*─── 진입 정보 ───*
• 지정가 (MA20): *{entry_str}*
• 손절가 (-10%): `{stop_str}` ← 절대 원칙
• 1차 목표가: {target_str}
• 손익비: {rr_str}

*─── 기술 지표 ───*
• RSI: {rsi_str}
• MA20 이격도: {disp_str}
• 전일대비: {chg_str}

*─── 밸류에이션 ───*
• {per_str} | {pbr_str}

*─── 행동 지침 ───*
`{action}`""" + (f"\n📌 {reason}" if reason else "") + """

> ⚠️ 자동 매수 없음. 반드시 직접 판단 후 주문하세요."""

    return _send(msg)


def send_dashboard_sell_alert(signal: dict) -> bool:
    """
    대시보드 매도 경고 감지 시 슬랙 알림.
    app.py의 load_watchlist_signals() 결과 dict를 직접 받는다.
    """
    name    = signal.get("종목명", signal.get("티커", ""))
    ticker  = signal.get("티커", "")
    market  = signal.get("시장", "")
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")

    pattern  = signal.get("패턴", "-")
    sig_type = signal.get("신호유형", "SELL")
    conf     = signal.get("신뢰도") or 0
    conf_bar = _confidence_emoji(conf)
    action   = signal.get("행동지침") or "매도/관망 검토"

    entry   = signal.get("진입가")
    resist  = signal.get("저항선")
    support = signal.get("지지선")
    rsi_v   = signal.get("RSI")
    chg     = signal.get("전일대비")
    chg_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "-"
    detail  = signal.get("패턴상세") or ""

    icon  = "📉" if sig_type == "SELL" else "⚠️"
    title = "매도 경고 발생" if sig_type == "SELL" else "위험 중립 경고"

    entry_str   = f"{entry:,.0f}원"   if entry   else "-"
    resist_str  = f"{resist:,.0f}원"  if resist  else "-"
    support_str = f"{support:,.0f}원" if support else "-"
    rsi_str     = f"{rsi_v:.1f}"      if rsi_v   else "-"

    msg = f"""{icon} *[{title}]* — {now}

*종목:* {name} (`{ticker}`) | {market}
*패턴:* {pattern}
*신뢰도:* {conf_bar} ({conf*100:.0f}%)
*행동 지침:* `{action}`

*─── 현재 상황 ───*
• 현재가: *{entry_str}*
• 저항선: {resist_str}
• 지지선: {support_str}
• RSI: {rsi_str}
• 전일대비: {chg_str}""" + (f"\n\n*─── 패턴 상세 ───*\n{detail}" if detail else "") + """

> ⚠️ 보유 중이라면 손절선 및 리스크를 재확인하세요."""

    return _send(msg)


# ---------------------------------------------------------------------------
# 메인 발송 함수 (main.py 스케줄러용)
# ---------------------------------------------------------------------------
def send_buy_signal(
    ticker: str,
    pattern: PatternResult,
    valuation: ValuationResult,
    risk: RiskResult,
    market: MarketStatus,
) -> bool:
    """
    매수 신호 슬랙 알림 발송 (main.py 스케줄러 파이프라인용).
    모든 필터 통과 조건을 이 함수에서 최종 확인한다.
    """
    name = config.TICKER_NAME.get(ticker, ticker)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    confidence_bar = _confidence_emoji(pattern.confidence)

    msg = f"""📈 *[매수 신호 발생]* — {now}

*종목:* {name} (`{ticker}`)
*패턴:* {pattern.pattern.value}
*신뢰도:* {confidence_bar} ({pattern.confidence*100:.0f}%)

*─── 진입 정보 ───*
• 지정가 매수: *{risk.entry_price:,.0f}원*
• 손절가 (-10%): `{risk.stop_loss_price:,.0f}원` ← 절대 원칙
• 1차 목표가: {risk.target_price_1r:,.0f}원 (손익비 {risk.reward_risk_ratio:.1f}:1)
• 2차 목표가: {risk.target_price_2r:,.0f}원

*─── 밸류에이션 ───*
{valuation.detail}

*─── 시장 상황 ───*
• 시장 트렌드: *{market.regime.value}*
• {market.detail}

*─── 패턴 상세 ───*
{pattern.detail}

> ⚠️ 이 메시지는 자동 매수가 아닌 *진입 검토 신호*입니다.
> 반드시 직접 판단 후 주문하세요."""

    return _send(msg)


def send_sell_signal(
    ticker: str,
    pattern: PatternResult,
    market: MarketStatus,
) -> bool:
    """
    매도/경고 패턴 슬랙 알림 발송 (이미지 기준 팔아라 경고).
    signal_type에 따라 아이콘과 메시지 톤을 차별화한다.
    """
    name = config.TICKER_NAME.get(ticker, ticker)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    confidence_bar = _confidence_emoji(pattern.confidence)

    if pattern.signal_type == "SELL":
        icon = "📉"
        title = "매도 경고 발생"
    else:
        icon = "⚠️"
        title = "위험 중립 경고"

    action = pattern.action or "매도/관망 검토"

    msg = f"""{icon} *[{title}]* — {now}

*종목:* {name} (`{ticker}`)
*패턴:* {pattern.pattern.value}
*신뢰도:* {confidence_bar} ({pattern.confidence*100:.0f}%)
*행동 지침:* `{action}`

*─── 현재 상황 ───*
• 현재가: *{pattern.entry_price:,.0f}원*
• 저항선: {pattern.resistance_level:,.0f}원
• 지지선: {pattern.support_level:,.0f}원
• RSI: {pattern.rsi:.1f}

*─── 시장 상황 ───*
• 시장 트렌드: *{market.regime.value}*
• {market.detail}

*─── 패턴 상세 ───*
{pattern.detail}

> ⚠️ 이 메시지는 *매도 검토 신호*입니다.
> 보유 중이라면 손절선 및 리스크를 재확인하세요."""

    return _send(msg)


def send_watchlist_signal(
    ticker: str,
    pattern: PatternResult,
    risk: RiskResult,
) -> bool:
    """
    관심 종목 감시 알림 (밸류에이션 미통과 종목 포함, 참고용)
    """
    name = config.TICKER_NAME.get(ticker, ticker)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = f"""👀 *[관심 종목 패턴 감지]* — {now}

*종목:* {name} (`{ticker}`)
*패턴:* {pattern.pattern.value} (신뢰도 {pattern.confidence*100:.0f}%)
*현재가:* {risk.entry_price:,.0f}원
*손절가:* {risk.stop_loss_price:,.0f}원

{pattern.detail}

> 밸류에이션 필터 미통과 — 참고만 할 것"""
    return _send(msg)


def send_market_alert(market: MarketStatus) -> bool:
    """
    시장 상태 경보 (약세장 진입 시 전송)
    """
    if market.regime == MarketRegime.BEAR:
        icon = "🔴"
        title = "약세장 진입 — 전종목 관망"
    elif market.regime == MarketRegime.NEUTRAL:
        icon = "🟡"
        title = "시장 중립 — 선별적 진입"
    else:
        icon = "🟢"
        title = "강세장 — 정상 감시"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"""{icon} *[시장 트렌드 업데이트]* — {now}

*상태:* {market.regime.value} — {title}
{market.detail}"""
    return _send(msg)


def send_waiting_signal(reason: str) -> bool:
    """
    조건 미충족 시 관망 알림 (과도한 발송 방지 — 시장 상태 변화 시에만 호출)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"""⚠️ *[관망]* — {now}

현재 조건 미충족 → *기다려!*
사유: {reason}"""
    return _send(msg)


def send_error(context: str, error: Exception) -> bool:
    """시스템 에러 알림"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"""🚨 *[시스템 에러]* — {now}

*위치:* {context}
*에러:* `{type(error).__name__}: {error}`"""
    return _send(msg)


def send_daily_summary(signals_count: int, tickers_scanned: int, market: MarketStatus) -> bool:
    """
    일일 운영 요약 (장 마감 후 전송)
    """
    now = datetime.now().strftime("%Y-%m-%d")
    msg = f"""📊 *[일일 운영 요약]* — {now}

• 스캔 종목 수: {tickers_scanned}개
• 발생 신호 수: {signals_count}개
• 최종 시장 상태: {market.regime.value}
• {market.detail}

> 내일도 원칙을 지키자."""
    return _send(msg)


# ---------------------------------------------------------------------------
# 내부 전송
# ---------------------------------------------------------------------------
def _send(text: str) -> bool:
    url = config.SLACK_WEBHOOK_URL
    if not url:
        logger.warning("SLACK_WEBHOOK_URL not set; message not sent:\n%s", text)
        return False

    try:
        resp = requests.post(
            url,
            json={"text": text, "mrkdwn": True},
            timeout=10,
        )
        resp.raise_for_status()
        logger.debug("Slack message sent successfully")
        return True
    except requests.RequestException as exc:
        logger.error("Failed to send Slack message: %s", exc)
        return False


def send_test() -> bool:
    """슬랙 연결 테스트 메시지 발송"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"""✅ *[테스트 알림]* — {now}

슬랙 웹훅 연결 성공!
주식 매매 신호 시스템이 정상적으로 슬랙에 메시지를 보낼 수 있습니다.

• 감시 종목: {len(config.WATCHLIST)}개
• 스캔 주기: {config.SCAN_INTERVAL_MINUTES}분

> 이 메시지가 보이면 설정이 완료된 것입니다."""
    return _send(msg)


def _confidence_emoji(confidence: float) -> str:
    filled = round(confidence * 5)
    return "⬛" * filled + "⬜" * (5 - filled)
