"""
trade_engine.py - 매매 실행 엔진 (알고리즘 정의서 Execute_Trade 구현)

[알고리즘 정의서] 매수 실행 및 예산 관리 (How to buy)

  FUNCTION Execute_Trade(Stock, User_Budget, Target_Entry_Price):
    // 1. 지정가 매수 가능 최대 수량 계산
    Max_Shares = FLOOR(User_Budget / Target_Entry_Price)
    // 2. 지정가 주문만 전송 (시장가 절대 금지)
    Order = Send_Limit_Order(BUY, Price=Target_Entry_Price, Qty=Max_Shares)
    // 3. 체결 즉시 -10% 스탑로스 자동 세팅
    Stop_Loss_Price = Target_Entry_Price * 0.90
    Send_Stop_Loss_Order(SELL, Price=Stop_Loss_Price, Qty=Max_Shares)

핵심 원칙:
  - 시장가(Market Order) 기능 완전 삭제 — 지정가(Limit Order)만 허용
  - 매수 체결 즉시 -10% 절대 방어 스탑로스 세팅 (예외 없음)
  - 예산 맞춤형 포지션 사이징: FLOOR(예산 ÷ 진입가)
  - 손절 도달 시: 시장 상황/뉴스 무관하게 예외 없이 전량 매도

현재 상태: 슬랙 알림 전용 (KIS API 연동 시 실주문 활성화)
"""
import logging
from dataclasses import dataclass
from typing import Optional

import config
import pattern_engine as pe
import risk as rm

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    ticker: str
    name: str
    order_type: str          # "LIMIT_BUY" | "STOP_LOSS"
    price: float             # 지정가 (시장가 절대 금지)
    quantity: int            # 수량 = FLOOR(예산 ÷ 진입가)
    budget_used: float       # 실제 투입 예산
    stop_loss_price: float   # -10% 절대 손절가
    stop_loss_amount: float  # 최대 손실 금액
    target_1r: float         # 1차 목표가 (손익비 2:1)
    target_2r: float         # 2차 목표가 (손익비 3:1)
    reward_risk: float       # 실제 손익비
    is_viable: bool          # 예산 내 최소 1주 이상 가능 여부
    detail: str


def calculate_trade(
    ticker: str,
    entry_price: float,
    resistance_level: float,
    budget: Optional[int] = None,
) -> TradeOrder:
    """
    [알고리즘] Execute_Trade 계산부:
    지정가 매수 가능 수량 + -10% 스탑로스 자동 계산.

    Args:
        ticker: 종목 코드
        entry_price: MA20 기반 지정가 매수 단가
        resistance_level: 1차 목표가 계산 기준 저항선
        budget: 투입 예산 (None이면 config.BUDGET_PER_TRADE 사용)
    """
    if budget is None:
        budget = config.BUDGET_PER_TRADE

    name = config.TICKER_NAME.get(ticker, ticker)

    # [알고리즘] Max_Shares = FLOOR(User_Budget / Target_Entry_Price)
    quantity = int(budget // entry_price)
    is_viable = quantity >= 1
    if not is_viable:
        logger.warning("Budget too small: %s budget=%d entry=%.0f → 0 shares",
                       name, budget, entry_price)
        quantity = 0

    # [알고리즘] Stop_Loss_Price = Target_Entry_Price * 0.90 (절대 원칙)
    stop_loss_price = round(entry_price * (1 + config.STOP_LOSS_RATE), 2)
    stop_loss_per_share = entry_price - stop_loss_price
    stop_loss_amount = round(stop_loss_per_share * quantity, 2)

    budget_used = round(entry_price * quantity, 2)

    # 1차 목표가: 저항선 vs 손익비 2:1 중 높은 값
    min_target_1r = entry_price + (stop_loss_per_share * config.MIN_REWARD_RISK_RATIO)
    target_1r = max(resistance_level, min_target_1r) if resistance_level > entry_price else min_target_1r
    target_2r = entry_price + (stop_loss_per_share * 3.0)

    actual_gain = target_1r - entry_price
    reward_risk = round(actual_gain / stop_loss_per_share, 2) if stop_loss_per_share > 0 else 0

    detail = _build_order_detail(
        name, entry_price, stop_loss_price, target_1r, target_2r,
        quantity, budget_used, stop_loss_amount, reward_risk, budget,
    )

    return TradeOrder(
        ticker=ticker,
        name=name,
        order_type="LIMIT_BUY",
        price=entry_price,
        quantity=quantity,
        budget_used=budget_used,
        stop_loss_price=stop_loss_price,
        stop_loss_amount=stop_loss_amount,
        target_1r=round(target_1r, 2),
        target_2r=round(target_2r, 2),
        reward_risk=reward_risk,
        is_viable=is_viable,
        detail=detail,
    )


def execute_trade(ticker: str, pattern: pe.PatternResult, budget: Optional[int] = None) -> TradeOrder:
    """
    [알고리즘] Execute_Trade 전체 흐름:
    1. 지정가 계산 (MA20 기반)
    2. 수량 계산 (예산 ÷ 진입가, 소수 버림)
    3. 슬랙 알림 (실매매는 KIS API 연동 후 활성화)
    4. 체결 즉시 -10% 스탑로스 계산 반환
    """
    if pattern is None or pattern.entry_price <= 0:
        raise ValueError(f"Invalid pattern for {ticker}")

    order = calculate_trade(
        ticker=ticker,
        entry_price=pattern.entry_price,
        resistance_level=pattern.resistance_level,
        budget=budget,
    )

    if not order.is_viable:
        logger.warning("Trade not viable for %s: %s", ticker, order.detail)
        return order

    logger.info(
        "[LIMIT ORDER] %s | 진입가: %.0f원 | %d주 | 손절: %.0f원 | 목표: %.0f원",
        order.name, order.price, order.quantity,
        order.stop_loss_price, order.target_1r,
    )

    # KIS API 미연동 상태 → 로그만 출력, 실주문 없음
    # TODO: KIS API 연동 후 아래 주석 해제
    # _send_limit_order_kis(order)
    # _send_stop_loss_kis(order)

    return order


def check_stop_loss_hit(current_price: float, stop_loss_price: float) -> bool:
    """
    [알고리즘] 스탑로스 도달 여부 확인.
    도달 시 시장 상황/뉴스와 무관하게 예외 없이 전량 매도.
    """
    if current_price <= stop_loss_price:
        logger.critical(
            "STOP-LOSS HIT: current=%.0f <= stop=%.0f → 예외 없이 즉시 전량 매도",
            current_price, stop_loss_price,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# KIS API 연동 스텁 (추후 활성화)
# ---------------------------------------------------------------------------

def _send_limit_order_kis(order: TradeOrder) -> Optional[str]:
    """
    [미연동] 한국투자증권 KIS REST API 지정가 매수 주문.
    KIS_APP_KEY 설정 후 활성화.

    절대 원칙: 시장가(Market Order) 절대 사용 금지.
    """
    if not config.KIS_APP_KEY:
        logger.debug("KIS API not configured; limit order skipped")
        return None
    # TODO: KIS API 지정가 주문 구현
    # POST https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/trading/order-cash
    # ord_dvsn = "00"  # 지정가 (01=시장가 — 절대 사용 금지)
    logger.warning("KIS API stub: LIMIT BUY %s %d주 @ %.0f원",
                   order.ticker, order.quantity, order.price)
    return None


def _send_stop_loss_kis(order: TradeOrder) -> Optional[str]:
    """
    [미연동] 체결 즉시 -10% 스탑로스 자동 세팅.
    체결 후 예외 없이 즉시 실행.
    """
    if not config.KIS_APP_KEY:
        return None
    logger.warning("KIS API stub: STOP-LOSS SELL %s %d주 @ %.0f원 (-10%%)",
                   order.ticker, order.quantity, order.stop_loss_price)
    return None


# ---------------------------------------------------------------------------
# 주문 상세 메시지
# ---------------------------------------------------------------------------

def _build_order_detail(name, entry, stop, t1, t2, qty, cost, max_loss, rr, budget) -> str:
    lines = [
        f"[{name}] 지정가 매수 계획",
        f"진입가 (MA20): {entry:,.0f}원  ← 지정가 주문 (시장가 절대 금지)",
        f"수량: {qty}주  ← FLOOR({budget:,}원 ÷ {entry:,.0f}원)",
        f"투입금액: {cost:,.0f}원",
        f"────────────────────────────────",
        f"스탑로스: {stop:,.0f}원 (-10%, 체결 즉시 자동 세팅, 예외 없음)",
        f"최대 손실: -{max_loss:,.0f}원",
        f"1차 목표: {t1:,.0f}원  |  2차 목표: {t2:,.0f}원",
        f"손익비: {rr:.1f}:1",
    ]
    return "\n".join(lines)
