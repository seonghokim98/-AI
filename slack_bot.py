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
# 메인 발송 함수
# ---------------------------------------------------------------------------
def send_buy_signal(
    ticker: str,
    pattern: PatternResult,
    valuation: ValuationResult,
    risk: RiskResult,
    market: MarketStatus,
) -> bool:
    """
    매수 신호 슬랙 알림 발송.
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

*─── 수량 & 예산 ───*
• 예산 {config.BUDGET_PER_TRADE:,}원 기준 수량: *{risk.quantity}주*
• 투입금액: {risk.total_cost:,.0f}원
• 최대 손실: -{risk.max_loss:,.0f}원

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


def _confidence_emoji(confidence: float) -> str:
    filled = round(confidence * 5)
    return "⬛" * filled + "⬜" * (5 - filled)
