"""
risk.py - 리스크 관리 모듈 (매매 원칙 절대 원칙)
  1. -10% 손절가 자동 계산 (절대 원칙, 예외 없음)
  2. 손익비 계산 (최소 2:1 기준)
  3. 예산 기반 수량 계산
  4. 계좌 드로다운 경보
"""
import logging
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    entry_price: float
    stop_loss_price: float      # -10% 절대 손절가
    stop_loss_rate: float       # 실제 손절 비율 (보통 -0.10)
    target_price_1r: float      # 1차 목표가 (손익비 2:1 기준)
    target_price_2r: float      # 2차 목표가 (손익비 3:1 기준)
    quantity: int               # 예산 기반 수량
    total_cost: float           # 실제 투입 금액
    max_loss: float             # 최대 손실 금액 (수량 × 손절폭)
    reward_risk_ratio: float    # 실제 손익비
    is_risk_ok: bool            # 손익비 기준 충족 여부
    detail: str


def calculate_risk(
    entry_price: float,
    resistance_level: float,
    budget: Optional[int] = None,
) -> RiskResult:
    """
    진입가와 저항선을 기반으로 리스크/리워드를 계산한다.

    Args:
        entry_price:      진입 예정 가격
        resistance_level: 저항선 (1차 목표가 계산 기준)
        budget:           투입 예산 (None이면 config.BUDGET_PER_TRADE 사용)

    Returns:
        RiskResult 객체
    """
    if budget is None:
        budget = config.BUDGET_PER_TRADE

    # 절대 손절가: -10% (매매 원칙, 예외 없음)
    stop_rate = config.STOP_LOSS_RATE   # -0.10
    stop_loss_price = round(entry_price * (1 + stop_rate), 2)
    stop_loss_amt = entry_price - stop_loss_price   # 주당 손실액 (양수)

    # 목표가 계산 (저항선 기반 + 손익비 보완)
    # 1차: 저항선이 손익비 2:1 이상이면 저항선 사용, 아니면 강제 계산
    resistance_gain = resistance_level - entry_price if resistance_level > entry_price else 0
    min_target_1r = entry_price + (stop_loss_amt * config.MIN_REWARD_RISK_RATIO)  # 최소 2:1

    target_1r = max(resistance_level, min_target_1r) if resistance_level > entry_price else min_target_1r
    target_2r = entry_price + (stop_loss_amt * 3.0)   # 3:1 목표

    # 실제 손익비 (1차 목표 기준)
    actual_gain = target_1r - entry_price
    rr_ratio = actual_gain / stop_loss_amt if stop_loss_amt > 0 else 0

    # 수량 계산 (예산 ÷ 진입가, 소수 버림)
    quantity = int(budget // entry_price)
    if quantity < 1:
        quantity = 1

    total_cost = round(entry_price * quantity, 2)
    max_loss = round(stop_loss_amt * quantity, 2)

    is_risk_ok = rr_ratio >= config.MIN_REWARD_RISK_RATIO

    detail = _build_detail(entry_price, stop_loss_price, target_1r, target_2r,
                           quantity, total_cost, max_loss, rr_ratio, is_risk_ok)

    if not is_risk_ok:
        logger.warning("Risk/Reward %.2f:1 below minimum %.1f:1 for entry=%.2f",
                       rr_ratio, config.MIN_REWARD_RISK_RATIO, entry_price)

    return RiskResult(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        stop_loss_rate=stop_rate,
        target_price_1r=round(target_1r, 2),
        target_price_2r=round(target_2r, 2),
        quantity=quantity,
        total_cost=total_cost,
        max_loss=max_loss,
        reward_risk_ratio=round(rr_ratio, 2),
        is_risk_ok=is_risk_ok,
        detail=detail,
    )


def check_account_drawdown(current_value: float, initial_value: float) -> bool:
    """
    계좌 드로다운이 임계값 초과 시 True 반환 → 신호 중단.
    """
    if initial_value <= 0:
        return False
    drawdown = (current_value - initial_value) / initial_value
    exceeded = drawdown <= config.MAX_DRAWDOWN_HALT
    if exceeded:
        logger.critical("Account drawdown %.1f%% exceeded halt threshold %.1f%%",
                        drawdown * 100, config.MAX_DRAWDOWN_HALT * 100)
    return exceeded


def _build_detail(entry, stop, t1, t2, qty, cost, max_loss, rr, is_ok) -> str:
    lines = [
        f"진입가: {entry:,.0f}원",
        f"손절가: {stop:,.0f}원 ({config.STOP_LOSS_RATE*100:.0f}%, 절대 원칙)",
        f"1차 목표: {t1:,.0f}원 | 2차 목표: {t2:,.0f}원",
        f"수량: {qty}주 | 투입금액: {cost:,.0f}원",
        f"최대 손실: -{max_loss:,.0f}원 | 손익비: {rr:.1f}:1 {'✅' if is_ok else '❌ (신호 억제)'}",
    ]
    return "\n".join(lines)
