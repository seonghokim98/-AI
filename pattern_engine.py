"""
pattern_engine.py - 기술적 패턴 인식 엔진 (매매 원칙 핵심)
허용 패턴:
  1. 골든크로스 (5일선 > 20일선 전환)
  2. 쌍바닥 (Double Bottom)
  3. 깃발형 돌파 (Flag Breakout)
  4. 돌파 + 눌림목 확인 ← 핵심 패턴
  5. 지지/저항 전환 (Support-Resistance Flip)

조건 미충족 시 항상 None 반환 → 슬랙에 "관망" 전송
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

cfg = config.TECH


class PatternType(Enum):
    GOLDEN_CROSS = "골든크로스"
    DOUBLE_BOTTOM = "쌍바닥"
    FLAG_BREAKOUT = "깃발형 돌파"
    BREAKOUT_PULLBACK = "돌파 + 눌림목 확인"
    SR_FLIP = "지지/저항 전환"


@dataclass
class PatternResult:
    pattern: PatternType
    confidence: float          # 0~1 신뢰도
    entry_price: float         # 추천 진입가 (지정가)
    resistance_level: float    # 돌파 기준 저항선
    support_level: float       # 지지선 (손절 보조)
    volume_ok: bool            # 거래량 충족 여부
    rsi: float                 # 현재 RSI
    detail: str                # 슬랙 전송용 설명


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def detect_patterns(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    4가지 패턴을 우선순위 순서로 검사하여 첫 번째 충족 패턴을 반환.
    아무 패턴도 없으면 None 반환 → 관망.

    우선순위:
      1. 돌파+눌림목 (가장 정교, 승률 높음)
      2. 깃발형 돌파
      3. 쌍바닥
      4. 골든크로스
      5. 지지/저항 전환
    """
    if df is None or len(df) < cfg.ma_long:
        return None

    checks = [
        _check_breakout_pullback,
        _check_flag_breakout,
        _check_double_bottom,
        _check_golden_cross,
        _check_sr_flip,
    ]
    for check_fn in checks:
        result = check_fn(df)
        if result is not None:
            logger.info("Pattern detected: %s (confidence=%.2f)",
                        result.pattern.value, result.confidence)
            return result

    return None


# ---------------------------------------------------------------------------
# 1. 돌파 + 눌림목 확인 (핵심 패턴)
# ---------------------------------------------------------------------------
def _check_breakout_pullback(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    조건:
      1. N봉 전 저항선 돌파 (거래량 급증 동반)
      2. 이후 저항선 부근으로 되돌림 (눌림목)
      3. 현재 봉이 저항선 근처에서 지지 확인 (RSI 과매도 아님)
    """
    n = cfg.breakout_window
    tol = cfg.pullback_tolerance
    vol_ratio = cfg.volume_surge_ratio

    if len(df) < n + 5:
        return None

    # 돌파 기준: 최근 n봉 최고가
    recent = df.iloc[-(n + 5):-3]         # 돌파 탐색 구간
    resistance = recent["High"].max()

    # 돌파봉 탐색 (현재 -3 ~ -5봉 사이)
    breakout_bar = None
    for i in range(-5, -2):
        bar = df.iloc[i]
        prev_bar_vol_ma = df["VOL_MA20"].iloc[i]
        if (bar["Close"] > resistance and
                bar["Volume"] > prev_bar_vol_ma * vol_ratio):
            breakout_bar = i
            break

    if breakout_bar is None:
        return None

    # 현재봉: 저항선 부근으로 되돌림(눌림목) 확인
    current = df.iloc[-1]
    pullback_zone_upper = resistance * (1 + tol)
    pullback_zone_lower = resistance * (1 - tol)

    in_pullback = pullback_zone_lower <= current["Close"] <= pullback_zone_upper

    if not in_pullback:
        return None

    # RSI가 너무 낮으면(과매도) 급락일 수 있으므로 제외
    rsi = float(df["RSI"].iloc[-1])
    if rsi < cfg.rsi_oversold:
        logger.debug("Breakout+pullback RSI too low: %.1f", rsi)
        return None

    # 거래량 확인 (눌림목 구간은 거래량 감소가 정상)
    vol_ok = current["Volume"] < df["VOL_MA20"].iloc[-1] * 1.2

    entry_price = float(current["Close"])
    support = float(df[f"MA{cfg.ma_mid}"].iloc[-1])

    confidence = _compute_confidence(
        rsi=rsi, vol_ok=vol_ok, in_zone=in_pullback, base=0.75
    )

    return PatternResult(
        pattern=PatternType.BREAKOUT_PULLBACK,
        confidence=confidence,
        entry_price=round(entry_price, 2),
        resistance_level=round(resistance, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        detail=(f"저항선 {resistance:,.0f}원 돌파 후 눌림목 재확인 | "
                f"RSI {rsi:.1f} | 진입가 {entry_price:,.0f}원"),
    )


# ---------------------------------------------------------------------------
# 2. 깃발형 돌파 (Flag Breakout)
# ---------------------------------------------------------------------------
def _check_flag_breakout(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    조건:
      1. 강한 상승 깃대 존재 (5% 이상 급등)
      2. 이후 수렴 구간 (깃발, 5~15봉)
      3. 깃발 상단 돌파
    """
    min_pole = cfg.flag_pole_min_ratio
    consol = cfg.flag_consolidation_bars
    max_retrace = cfg.flag_max_retracement

    if len(df) < consol + 10:
        return None

    # 깃대 탐색: 최근 10~25봉 구간에서 강한 단기 상승
    search_end = len(df) - consol - 1
    search_start = max(0, search_end - 15)

    pole_start = None
    pole_end = None
    pole_low = None
    pole_high = None

    for i in range(search_start, search_end):
        segment_ret = (df["Close"].iloc[i + 3] - df["Close"].iloc[i]) / df["Close"].iloc[i]
        if segment_ret >= min_pole:
            pole_start = i
            pole_end = i + 3
            pole_low = float(df["Close"].iloc[i])
            pole_high = float(df["Close"].iloc[i + 3])
            break

    if pole_start is None:
        return None

    # 깃발 구간: 깃대 이후 수렴
    flag_bars = df.iloc[pole_end: pole_end + consol + 5]
    if len(flag_bars) < consol:
        return None

    flag_high = float(flag_bars["High"].max())
    flag_low = float(flag_bars["Low"].min())

    # 되돌림 비율 확인
    pole_height = pole_high - pole_low
    retrace = (pole_high - flag_low) / pole_height
    if retrace > max_retrace:
        return None

    # 현재봉이 깃발 상단 돌파인지 확인
    current_close = float(df["Close"].iloc[-1])
    if current_close <= flag_high:
        return None

    # 거래량 증가 확인
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1]) * cfg.volume_surge_ratio
    rsi = float(df["RSI"].iloc[-1])
    support = float(df[f"MA{cfg.ma_mid}"].iloc[-1])

    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.70)

    return PatternResult(
        pattern=PatternType.FLAG_BREAKOUT,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(flag_high, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        detail=(f"깃대 상승 {pole_low:,.0f}→{pole_high:,.0f}원 ({min_pole*100:.0f}%↑) | "
                f"깃발 수렴 후 {flag_high:,.0f}원 돌파 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 3. 쌍바닥 (Double Bottom)
# ---------------------------------------------------------------------------
def _check_double_bottom(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    조건:
      1. 최근 N봉 내 두 개의 저점이 비슷한 가격대 (허용 오차 3%)
      2. 두 저점 사이에 반등 구간 존재
      3. 현재 가격이 두 저점을 잇는 넥라인(반등 고점) 돌파 또는 근접
    """
    window = cfg.double_bottom_window
    tol = cfg.double_bottom_tolerance

    if len(df) < window + 5:
        return None

    recent = df["Close"].iloc[-window:].reset_index(drop=True)

    # 첫 번째 저점
    idx1 = int(recent.idxmin())
    val1 = float(recent.iloc[idx1])

    # 두 번째 저점: idx1 이후 최소 5봉 이상 떨어진 곳
    if idx1 + 5 >= len(recent):
        return None
    second_half = recent.iloc[idx1 + 5:]
    if second_half.empty:
        return None
    idx2 = int(second_half.idxmin())   # idxmin()은 이미 절댓값 라벨 반환
    val2 = float(recent.iloc[idx2])

    # 두 저점의 가격 차이 허용 오차 확인
    if abs(val1 - val2) / val1 > tol:
        return None

    # 넥라인: 두 저점 사이 최고가
    neckline = float(recent.iloc[idx1:idx2].max())

    # 현재 가격이 넥라인 근처 또는 돌파 여부
    current_close = float(df["Close"].iloc[-1])
    near_neckline = current_close >= neckline * 0.98

    if not near_neckline:
        return None

    rsi = float(df["RSI"].iloc[-1])
    support = (val1 + val2) / 2
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])

    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.65)

    return PatternResult(
        pattern=PatternType.DOUBLE_BOTTOM,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(neckline, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        detail=(f"쌍바닥 저점1={val1:,.0f}원 / 저점2={val2:,.0f}원 | "
                f"넥라인 {neckline:,.0f}원 근접/돌파 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 4. 골든크로스 (5일선 > 20일선 전환)
# ---------------------------------------------------------------------------
def _check_golden_cross(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    조건:
      1. 전일: MA5 < MA20
      2. 당일: MA5 > MA20 (골든크로스 발생)
      3. 거래량 평균 이상
    """
    ma5 = df[f"MA{cfg.ma_short}"]
    ma20 = df[f"MA{cfg.ma_mid}"]

    if ma5.iloc[-2] >= ma20.iloc[-2]:
        return None
    if ma5.iloc[-1] <= ma20.iloc[-1]:
        return None

    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    current_close = float(df["Close"].iloc[-1])
    resistance = float(df["High"].rolling(cfg.breakout_window).max().iloc[-1])
    support = float(ma20.iloc[-1])

    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.60)

    return PatternResult(
        pattern=PatternType.GOLDEN_CROSS,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(resistance, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        detail=(f"MA{cfg.ma_short}({ma5.iloc[-1]:,.0f}) > MA{cfg.ma_mid}({ma20.iloc[-1]:,.0f}) "
                f"골든크로스 발생 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 5. 지지/저항 전환 (SR Flip)
# ---------------------------------------------------------------------------
def _check_sr_flip(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    조건:
      1. 과거 저항선(고점 밀집 구간)이 이전에 돌파됨
      2. 현재 그 구간으로 되돌아와 지지 확인 (저항→지지 전환)
    """
    n = cfg.breakout_window
    if len(df) < n * 2:
        return None

    # 과거 저항 구간: n*2 ~ n봉 전 구간의 최고가
    old_range = df["High"].iloc[-(n * 2):-n]
    resistance_level = float(old_range.max())

    # 그 이후 구간에서 저항 돌파 확인
    after_break = df["Close"].iloc[-n:-1]
    was_broken = bool((after_break > resistance_level).any())

    if not was_broken:
        return None

    # 현재 가격이 구 저항선 ±3% 이내
    current_close = float(df["Close"].iloc[-1])
    in_zone = resistance_level * (1 - cfg.pullback_tolerance) <= current_close <= resistance_level * (1 + cfg.pullback_tolerance)

    if not in_zone:
        return None

    rsi = float(df["RSI"].iloc[-1])
    if rsi < cfg.rsi_oversold:
        return None

    vol_ok = float(df["Volume"].iloc[-1]) < float(df["VOL_MA20"].iloc[-1]) * 1.3
    support = resistance_level * 0.97

    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.68)

    return PatternResult(
        pattern=PatternType.SR_FLIP,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(resistance_level, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        detail=(f"구 저항선 {resistance_level:,.0f}원 → 지지 전환 확인 | "
                f"현재가 {current_close:,.0f}원 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 신뢰도 계산 보조
# ---------------------------------------------------------------------------
def _compute_confidence(rsi: float, vol_ok: bool, in_zone: bool, base: float) -> float:
    score = base
    if vol_ok:
        score += 0.10
    if 45 <= rsi <= 65:       # 이상적인 RSI 구간
        score += 0.10
    elif rsi > 70:             # 과매수 감점
        score -= 0.15
    if not in_zone:
        score -= 0.20
    return round(min(max(score, 0.0), 1.0), 2)
