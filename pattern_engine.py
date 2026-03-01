"""
pattern_engine.py - 기술적 패턴 인식 엔진 (매매 원칙 핵심)
매수 패턴 (사라 경고):
  1. 골든크로스 (MA50/200 교차)
  2. 쌍바닥 (Double Bottom)
  3. 깃발형 돌파 (Flag Breakout) — 80% 신뢰도
  4. 돌파 + 눌림목 확인 ← 핵심 패턴
  5. 지지/저항 전환 (Support-Resistance Flip)
  6. 하락쐐기형 돌파 (Falling Wedge)
  7. 역삼각형 돌파 — 65% 신뢰도, "급하게 사"
  8. 상승비기형 (Ascending Triangle) — 100% 신뢰도, "폭등 대비"

매도 패턴 (팔아라 경고):
  A. 쌍봉 (Double Top) — 100% 신뢰도, "폭락 대비"
  B. 하락깃발 (Bearish Flag) — 80% 신뢰도, "빨리 팔아"
  C. 하락 다이아몬드 — 65% 신뢰도, "천천히 내려간다"
  D. 박스권 — 50% 중립, "건들지마 위험"

조건 미충족 시 항상 None 반환 → 슬랙에 "관망" 전송
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

cfg = config.TECH


class PatternType(Enum):
    # ── 매수 패턴 (사라 경고) ──────────────────────────────────────────────
    GOLDEN_CROSS = "골든크로스 (MA50/200)"
    DOUBLE_BOTTOM = "쌍바닥"
    FLAG_BREAKOUT = "깃발형 돌파"
    BREAKOUT_PULLBACK = "돌파 + 눌림목 확인"
    SR_FLIP = "지지/저항 전환"
    FALLING_WEDGE = "하락쐐기형 돌파"
    INVERSE_TRIANGLE = "역삼각형 돌파"          # 65% 매수, "급하게 사"
    ASCENDING_TRIANGLE = "상승비기형"            # 100% 매수, "폭등 대비"
    # ── 매도/경고 패턴 (팔아라 경고) ─────────────────────────────────────
    DOUBLE_TOP = "쌍봉"                          # 100% 매도, "폭락 대비"
    BEARISH_FLAG = "하락깃발"                    # 80% 매도, "빨리 팔아"
    BEARISH_DIAMOND = "하락 다이아몬드"          # 65% 매도, "천천히 내려간다"
    BOX_RANGE = "박스권"                         # 50% 중립, "건들지마 위험"


@dataclass
class PatternResult:
    pattern: PatternType
    confidence: float          # 0~1 신뢰도
    entry_price: float         # 추천 진입가 (지정가) / 현재가
    resistance_level: float    # 돌파 기준 저항선
    support_level: float       # 지지선 (손절 보조)
    volume_ok: bool            # 거래량 충족 여부
    rsi: float                 # 현재 RSI
    detail: str                # 슬랙 전송용 설명
    signal_type: str = "BUY"   # "BUY" | "SELL" | "NEUTRAL"
    action: str = ""           # 행동 지침 (예: "급하게 사", "빨리 팔아")
    entry_reason: str = ""     # 추천 진입가 근거 (예: "MA20 지지 확인", "돌파선+0.5%")


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
        _check_ascending_triangle,   # 상승비기형 — 100% 매수, 최우선
        _check_breakout_pullback,
        _check_flag_breakout,
        _check_falling_wedge,
        _check_inverse_triangle,     # 역삼각형 — 65% 매수
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


def detect_sell_patterns(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    매도/경고 패턴 탐지 (이미지 기준 팔아라 경고).
    None 반환 시 → 관망 유지.

    우선순위:
      1. 쌍봉 (100% sell) — 폭락 대비 즉시 매도
      2. 하락깃발 (80% sell) — 빨리 팔아
      3. 하락 다이아몬드 (65% sell) — 천천히 내려간다
      4. 박스권 (50% neutral) — 건들지마 위험
    """
    if df is None or len(df) < cfg.ma_long:
        return None

    sell_checks = [
        _check_double_top,
        _check_bearish_flag,
        _check_bearish_diamond,
        _check_box_range,
    ]
    for check_fn in sell_checks:
        result = check_fn(df)
        if result is not None:
            logger.info("Sell pattern detected: %s (confidence=%.2f)",
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

    current_close = float(current["Close"])
    support = float(df[f"MA{cfg.ma_mid}"].iloc[-1])

    confidence = _compute_confidence(
        rsi=rsi, vol_ok=vol_ok, in_zone=in_pullback, base=0.75
    )
    entry_price, entry_reason = _smart_entry(df, current_close, resistance, "pullback")

    return PatternResult(
        pattern=PatternType.BREAKOUT_PULLBACK,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(resistance, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        entry_reason=entry_reason,
        detail=(f"저항선 {resistance:,.0f}원 돌파 후 눌림목 재확인 | "
                f"RSI {rsi:.1f} | 추천 진입 {entry_reason}"),
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
    entry_price, entry_reason = _smart_entry(df, current_close, flag_high, "breakout")

    return PatternResult(
        pattern=PatternType.FLAG_BREAKOUT,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(flag_high, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        entry_reason=entry_reason,
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
    entry_price, entry_reason = _smart_entry(df, current_close, neckline, "reversal")

    return PatternResult(
        pattern=PatternType.DOUBLE_BOTTOM,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(neckline, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        entry_reason=entry_reason,
        detail=(f"쌍바닥 저점1={val1:,.0f}원 / 저점2={val2:,.0f}원 | "
                f"넥라인 {neckline:,.0f}원 근접/돌파 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 4. 골든크로스 (MA50 > MA200 교차 — 이미지 기준)
# ---------------------------------------------------------------------------
def _check_golden_cross(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: MA50이 MA200을 상향 돌파 (골든크로스)
    조건:
      1. 최근 3봉 이내에 MA50 < MA200 → MA50 > MA200 전환 발생
      2. 거래량 평균 이상
    """
    fast_key = f"MA{cfg.ma_gc_fast}"   # MA50
    slow_key = f"MA{cfg.ma_gc_slow}"   # MA200

    if fast_key not in df.columns or slow_key not in df.columns:
        return None

    ma_fast = df[fast_key]
    ma_slow = df[slow_key]

    # NaN 제거 후 최소 2봉 필요
    valid_fast = ma_fast.dropna()
    valid_slow = ma_slow.dropna()
    if len(valid_fast) < 2 or len(valid_slow) < 2:
        return None

    # 최근 3봉 이내 골든크로스 발생 여부 확인
    crossed = False
    lookback = min(3, len(valid_fast) - 1)
    for i in range(-lookback, 0):
        if (ma_fast.iloc[i - 1] < ma_slow.iloc[i - 1] and
                ma_fast.iloc[i] > ma_slow.iloc[i]):
            crossed = True
            break
    if not crossed:
        return None

    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    current_close = float(df["Close"].iloc[-1])
    resistance = float(df["High"].rolling(cfg.breakout_window).max().iloc[-1])
    support = float(ma_slow.iloc[-1])   # MA200이 주요 지지선

    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.65)
    confidence = round(min(1.0, confidence + _candle_body_score(df)), 2)
    entry_price, entry_reason = _smart_entry(df, current_close, support, "reversal")

    return PatternResult(
        pattern=PatternType.GOLDEN_CROSS,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(resistance, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        entry_reason=entry_reason,
        detail=(f"MA{cfg.ma_gc_fast}({ma_fast.iloc[-1]:,.0f}) > "
                f"MA{cfg.ma_gc_slow}({ma_slow.iloc[-1]:,.0f}) "
                f"골든크로스 | RSI {rsi:.1f}"),
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
    entry_price, entry_reason = _smart_entry(df, current_close, resistance_level, "pullback")

    return PatternResult(
        pattern=PatternType.SR_FLIP,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(resistance_level, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        entry_reason=entry_reason,
        detail=(f"구 저항선 {resistance_level:,.0f}원 → 지지 전환 확인 | "
                f"현재가 {current_close:,.0f}원 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 6. 하락쐐기형 돌파 (Falling Wedge — 이미지 기준)
# ---------------------------------------------------------------------------
def _check_falling_wedge(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 하락쐐기형 돌파 — 상승 반전 패턴
    조건:
      1. 상단 추세선: 고점이 점점 낮아짐 (하강)
      2. 하단 추세선: 저점도 낮아지지만 상단보다 완만 (수렴)
      3. 수렴 구간 중 거래량 감소
      4. 상단 추세선 돌파 시 거래량 급증
    """
    window = cfg.wedge_window  # 25봉
    if len(df) < window + 5:
        return None

    recent = df.iloc[-(window + 5):-1]   # 마지막봉 제외한 수렴 구간

    # 로컬 고점 탐색
    highs = []
    for i in range(2, len(recent) - 2):
        if (recent["High"].iloc[i] > recent["High"].iloc[i - 1] and
                recent["High"].iloc[i] > recent["High"].iloc[i + 1]):
            highs.append((i, float(recent["High"].iloc[i])))

    # 로컬 저점 탐색
    lows = []
    for i in range(2, len(recent) - 2):
        if (recent["Low"].iloc[i] < recent["Low"].iloc[i - 1] and
                recent["Low"].iloc[i] < recent["Low"].iloc[i + 1]):
            lows.append((i, float(recent["Low"].iloc[i])))

    if len(highs) < 2 or len(lows) < 2:
        return None

    # 상단 추세선: 첫 고점 → 마지막 고점 (하강해야 함)
    h1_idx, h1_val = highs[0]
    h2_idx, h2_val = highs[-1]
    if h2_idx <= h1_idx or h2_val >= h1_val:
        return None
    upper_slope = (h2_val - h1_val) / (h2_idx - h1_idx)

    # 하단 추세선: 첫 저점 → 마지막 저점 (하강해야 하지만 상단보다 완만)
    l1_idx, l1_val = lows[0]
    l2_idx, l2_val = lows[-1]
    if l2_idx <= l1_idx or l2_val >= l1_val:
        return None
    lower_slope = (l2_val - l1_val) / (l2_idx - l1_idx)

    # 수렴 조건: 상단이 하단보다 가파르게 하락 (두 선이 좁혀짐)
    if upper_slope >= lower_slope:
        return None

    # 현재봉 위치에서의 상단 추세선 값
    bars_since_h1 = (window + 5) - 1 - h1_idx
    upper_at_now = h1_val + upper_slope * bars_since_h1

    # 현재 종가가 상단 추세선을 돌파했는지 확인
    current_close = float(df["Close"].iloc[-1])
    if current_close < upper_at_now * 0.99:
        return None

    # 거래량: 수렴 구간 감소 + 현재봉 급증
    half = len(recent) // 2
    early_vol = float(recent["Volume"].iloc[:half].mean())
    late_vol = float(recent["Volume"].iloc[half:].mean())
    vol_decreasing = late_vol <= early_vol * 1.15

    current_vol = float(df["Volume"].iloc[-1])
    vol_ma = float(df["VOL_MA20"].iloc[-1])
    vol_surge = current_vol > vol_ma * cfg.volume_surge_ratio

    if not (vol_decreasing and vol_surge):
        return None

    rsi = float(df["RSI"].iloc[-1])
    if rsi < cfg.rsi_oversold:
        return None

    support = float(df[f"MA{cfg.ma_mid}"].iloc[-1])
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_surge, in_zone=True, base=0.70)
    confidence = round(min(1.0, confidence + _candle_body_score(df)), 2)
    entry_price, entry_reason = _smart_entry(df, current_close, upper_at_now, "breakout")

    return PatternResult(
        pattern=PatternType.FALLING_WEDGE,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(upper_at_now, 2),
        support_level=round(support, 2),
        volume_ok=vol_surge,
        rsi=rsi,
        entry_reason=entry_reason,
        detail=(f"하락쐐기형 상단추세선({upper_at_now:,.0f}원) 돌파 | "
                f"거래량 수렴→급증 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 7. 역삼각형 돌파 (Inverse Triangle — 이미지 기준 65% 매수)
# ---------------------------------------------------------------------------
def _check_inverse_triangle(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 역삼각형 돌파 — 65% 매수, "급하게 사"
    조건:
      1. 하락하는 고점 (저항선이 점차 낮아짐)
      2. 수평/소폭 상승하는 저점 (지지선이 유지됨)
      3. 현재봉이 하락 저항선 상향 돌파 + 거래량 급증
    """
    window = cfg.wedge_window
    if len(df) < window + 5:
        return None

    recent = df.iloc[-(window + 5):-1]

    highs = []
    for i in range(2, len(recent) - 2):
        if (recent["High"].iloc[i] > recent["High"].iloc[i - 1] and
                recent["High"].iloc[i] > recent["High"].iloc[i + 1]):
            highs.append((i, float(recent["High"].iloc[i])))

    lows = []
    for i in range(2, len(recent) - 2):
        if (recent["Low"].iloc[i] < recent["Low"].iloc[i - 1] and
                recent["Low"].iloc[i] < recent["Low"].iloc[i + 1]):
            lows.append((i, float(recent["Low"].iloc[i])))

    if len(highs) < 2 or len(lows) < 2:
        return None

    h1_idx, h1_val = highs[0]
    h2_idx, h2_val = highs[-1]
    if h2_idx <= h1_idx or h2_val >= h1_val:  # 고점은 하락해야 함
        return None
    upper_slope = (h2_val - h1_val) / (h2_idx - h1_idx)

    l1_idx, l1_val = lows[0]
    l2_idx, l2_val = lows[-1]
    if l2_idx <= l1_idx:
        return None
    lower_slope = (l2_val - l1_val) / (l2_idx - l1_idx)

    # 역삼각형: 저점은 수평 또는 상승 (하락쐐기형과 구분)
    if lower_slope < -0.002 * l1_val:
        return None

    bars_since_h1 = (window + 5) - 1 - h1_idx
    upper_at_now = h1_val + upper_slope * bars_since_h1

    current_close = float(df["Close"].iloc[-1])
    if current_close < upper_at_now * 0.99:
        return None

    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1]) * cfg.volume_surge_ratio
    rsi = float(df["RSI"].iloc[-1])
    if rsi < cfg.rsi_oversold:
        return None

    support = float(df[f"MA{cfg.ma_mid}"].iloc[-1])
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.60)
    entry_price, entry_reason = _smart_entry(df, current_close, upper_at_now, "breakout")

    return PatternResult(
        pattern=PatternType.INVERSE_TRIANGLE,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(upper_at_now, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="BUY",
        action="🟢 급하게 사",
        entry_reason=entry_reason,
        detail=(f"역삼각형 하락저항선({upper_at_now:,.0f}원) 돌파 | "
                f"저점 지지 확인 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 8. 상승비기형 (Ascending Triangle — 이미지 기준 100% 매수)
# ---------------------------------------------------------------------------
def _check_ascending_triangle(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 상승비기형 — 100% 매수, "폭등에 대비해"
    조건:
      1. 수평 저항선 (고점이 비슷한 수준에서 반복, ±3%)
      2. 상승하는 지지선 (저점이 점점 높아짐)
      3. 수렴 후 저항선 상향 돌파 + 거래량 급증
    """
    window = cfg.wedge_window
    if len(df) < window + 5:
        return None

    recent = df.iloc[-(window + 5):-1]

    highs = []
    for i in range(2, len(recent) - 2):
        if (recent["High"].iloc[i] > recent["High"].iloc[i - 1] and
                recent["High"].iloc[i] > recent["High"].iloc[i + 1]):
            highs.append((i, float(recent["High"].iloc[i])))

    lows = []
    for i in range(2, len(recent) - 2):
        if (recent["Low"].iloc[i] < recent["Low"].iloc[i - 1] and
                recent["Low"].iloc[i] < recent["Low"].iloc[i + 1]):
            lows.append((i, float(recent["Low"].iloc[i])))

    if len(highs) < 2 or len(lows) < 2:
        return None

    h1_idx, h1_val = highs[0]
    h2_idx, h2_val = highs[-1]
    if h2_idx <= h1_idx:
        return None

    l1_idx, l1_val = lows[0]
    l2_idx, l2_val = lows[-1]
    if l2_idx <= l1_idx:
        return None
    lower_slope = (l2_val - l1_val) / (l2_idx - l1_idx)

    # 상승삼각형: 고점은 수평(±3%), 저점은 상승
    high_range_ratio = abs(h2_val - h1_val) / h1_val
    if high_range_ratio > 0.03:
        return None
    if lower_slope <= 0:  # 저점이 반드시 상승해야 함
        return None

    flat_resistance = (h1_val + h2_val) / 2
    current_close = float(df["Close"].iloc[-1])
    if current_close < flat_resistance * 0.99:
        return None

    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1]) * cfg.volume_surge_ratio
    if not vol_ok:
        return None

    rsi = float(df["RSI"].iloc[-1])
    support = l2_val
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, in_zone=True, base=0.88)
    confidence = round(min(1.0, confidence + _candle_body_score(df)), 2)
    entry_price, entry_reason = _smart_entry(df, current_close, flat_resistance, "breakout")

    return PatternResult(
        pattern=PatternType.ASCENDING_TRIANGLE,
        confidence=confidence,
        entry_price=entry_price,
        resistance_level=round(flat_resistance, 2),
        support_level=round(support, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="BUY",
        action="🟢 폭등 대비 매수",
        entry_reason=entry_reason,
        detail=(f"상승비기형 수평저항({flat_resistance:,.0f}원) 돌파 | "
                f"저점 상승 추세 | 거래량 급증 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# A. 쌍봉 (Double Top — 이미지 기준 100% 매도)
# ---------------------------------------------------------------------------
def _check_double_top(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 쌍봉 — 100% 매도, "폭락에 대비해"
    조건:
      1. 최근 N봉 내 두 개의 고점이 비슷한 가격대 (허용 오차 3%)
      2. 두 고점 사이에 하락 구간 존재 (넥라인)
      3. 현재 가격이 넥라인 근처이거나 하향 돌파
    """
    window = cfg.double_bottom_window  # 30봉
    tol = cfg.double_bottom_tolerance  # 3%

    if len(df) < window + 5:
        return None

    recent_high = df["High"].iloc[-window:].reset_index(drop=True)
    recent_low = df["Low"].iloc[-window:].reset_index(drop=True)

    idx1 = int(recent_high.idxmax())
    val1 = float(recent_high.iloc[idx1])

    if idx1 + 5 >= len(recent_high):
        return None
    second_half = recent_high.iloc[idx1 + 5:]
    if second_half.empty:
        return None
    idx2 = int(second_half.idxmax())
    val2 = float(recent_high.iloc[idx2])

    if abs(val1 - val2) / val1 > tol:
        return None

    neckline_section = recent_low.iloc[idx1:idx2]
    if neckline_section.empty:
        return None
    neckline = float(neckline_section.min())

    current_close = float(df["Close"].iloc[-1])
    if current_close > neckline * 1.02:
        return None

    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    resistance = (val1 + val2) / 2

    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.88)

    return PatternResult(
        pattern=PatternType.DOUBLE_TOP,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(resistance, 2),
        support_level=round(neckline * 0.95, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="SELL",
        action="⚠️ 폭락 대비 즉시 매도 검토",
        detail=(f"쌍봉 고점1={val1:,.0f}원 / 고점2={val2:,.0f}원 | "
                f"넥라인 {neckline:,.0f}원 하향 돌파 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# B. 하락깃발 (Bearish Flag — 이미지 기준 80% 매도)
# ---------------------------------------------------------------------------
def _check_bearish_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 하락깃발 — 80% 매도, "빨리 팔아"
    조건:
      1. 강한 하락 깃대 (5% 이상 급락)
      2. 이후 짧은 반등 수렴 구간 (깃발)
      3. 깃발 하단 지지 이탈
    """
    min_pole = cfg.flag_pole_min_ratio
    consol = cfg.flag_consolidation_bars
    max_retrace = cfg.flag_max_retracement

    if len(df) < consol + 10:
        return None

    search_end = len(df) - consol - 1
    search_start = max(0, search_end - 15)

    pole_start = None
    pole_end = None
    pole_high = None
    pole_low = None

    for i in range(search_start, search_end):
        segment_ret = (df["Close"].iloc[i] - df["Close"].iloc[i + 3]) / df["Close"].iloc[i]
        if segment_ret >= min_pole:
            pole_start = i
            pole_end = i + 3
            pole_high = float(df["Close"].iloc[i])
            pole_low = float(df["Close"].iloc[i + 3])
            break

    if pole_start is None:
        return None

    flag_bars = df.iloc[pole_end: pole_end + consol + 5]
    if len(flag_bars) < consol:
        return None

    flag_high = float(flag_bars["High"].max())
    flag_low = float(flag_bars["Low"].min())

    pole_height = pole_high - pole_low
    retrace = (flag_high - pole_low) / pole_height
    if retrace > max_retrace:
        return None

    current_close = float(df["Close"].iloc[-1])
    if current_close >= flag_low:
        return None

    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1]) * cfg.volume_surge_ratio

    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.75)

    return PatternResult(
        pattern=PatternType.BEARISH_FLAG,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(flag_high, 2),
        support_level=round(pole_low * 0.97, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="SELL",
        action="🔴 빨리 팔아",
        detail=(f"하락깃대 {pole_high:,.0f}→{pole_low:,.0f}원 ({min_pole*100:.0f}%↓) | "
                f"깃발 반등 후 {flag_low:,.0f}원 이탈 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# C. 하락 다이아몬드 (Bearish Diamond — 이미지 기준 65% 매도)
# ---------------------------------------------------------------------------
def _check_bearish_diamond(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 하락 다이아몬드 — 65% 매도, "천천히 내려간다"
    조건:
      1. 가격 변동폭이 먼저 확대된 후 수렴 (다이아몬드 형태)
      2. 고점이 먼저 상승 후 하락
      3. 현재 종가가 수렴 구간 저점 이하로 돌파
    """
    window = 30
    if len(df) < window + 5:
        return None

    recent = df.iloc[-(window + 5):-1]
    half = len(recent) // 2

    early_range = float(recent["High"].iloc[:half].max()) - float(recent["Low"].iloc[:half].min())
    late_range = float(recent["High"].iloc[half:].max()) - float(recent["Low"].iloc[half:].min())

    if early_range <= 0 or late_range >= early_range * 0.85:
        return None

    early_high = float(recent["High"].iloc[:half].max())
    late_high = float(recent["High"].iloc[half:].max())
    if late_high >= early_high:
        return None

    current_close = float(df["Close"].iloc[-1])
    late_low = float(recent["Low"].iloc[half:].min())
    if current_close >= late_low:
        return None

    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])

    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.60)

    return PatternResult(
        pattern=PatternType.BEARISH_DIAMOND,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(late_high, 2),
        support_level=round(late_low * 0.95, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="SELL",
        action="🟠 천천히 내려간다 — 단계적 매도",
        detail=(f"다이아몬드 변동폭 확대→수렴 | "
                f"후반 저점 {late_low:,.0f}원 이탈 | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# D. 박스권 (Box Range — 이미지 기준 50% 중립/위험)
# ---------------------------------------------------------------------------
def _check_box_range(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    이미지 기준: 박스권 — 50% 중립, "건들지마 위험"
    조건:
      1. 최근 N봉 동안 고점/저점이 일정 범위 안에서 반복
      2. 이동평균 기울기가 거의 0 (방향성 없음)
    """
    window = 20
    if len(df) < window + 5:
        return None

    recent = df.iloc[-window:]
    box_high = float(recent["High"].max())
    box_low = float(recent["Low"].min())

    if box_low <= 0:
        return None

    box_range_ratio = (box_high - box_low) / box_low
    if not (0.04 <= box_range_ratio <= 0.15):
        return None

    ma20 = df[f"MA{cfg.ma_mid}"].dropna()
    if len(ma20) < window:
        return None
    ma_slope = (float(ma20.iloc[-1]) - float(ma20.iloc[-window])) / float(ma20.iloc[-window])
    if abs(ma_slope) > 0.04:
        return None

    current_close = float(df["Close"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    vol_ok = float(df["Volume"].iloc[-1]) < float(df["VOL_MA20"].iloc[-1]) * 1.2

    return PatternResult(
        pattern=PatternType.BOX_RANGE,
        confidence=0.45,
        entry_price=round(current_close, 2),
        resistance_level=round(box_high, 2),
        support_level=round(box_low, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        signal_type="NEUTRAL",
        action="⚠️ 건들지마 위험 — 관망",
        detail=(f"박스권 상단={box_high:,.0f}원 / 하단={box_low:,.0f}원 | "
                f"변동폭 {box_range_ratio*100:.1f}% | RSI {rsi:.1f}"),
    )


# ---------------------------------------------------------------------------
# 스마트 진입가 계산
# ---------------------------------------------------------------------------
def _smart_entry(
    df: pd.DataFrame,
    current_close: float,
    key_level: float,
    mode: str,
) -> tuple[float, str]:
    """
    패턴 유형과 기술 지표를 고려한 스마트 진입가 계산.
    Returns: (entry_price, reason_string)

    mode:
      "breakout" - 저항 돌파 직후  → 돌파선+0.5%, 과열 시 MA5 눌림목 대기
      "pullback" - 눌림목 진입     → MA20 또는 구 저항선(지지 전환) 기준
      "reversal" - 반전 패턴       → BB 하단 또는 MA20 근처
    """
    ma20_col = f"MA{cfg.ma_mid}"    # MA20
    ma5_col  = f"MA{cfg.ma_short}"  # MA5

    ma20 = (
        float(df[ma20_col].iloc[-1])
        if ma20_col in df.columns and pd.notna(df[ma20_col].iloc[-1])
        else None
    )
    ma5 = (
        float(df[ma5_col].iloc[-1])
        if ma5_col in df.columns and pd.notna(df[ma5_col].iloc[-1])
        else None
    )
    bb_lower = (
        float(df["BB_LOWER"].iloc[-1])
        if "BB_LOWER" in df.columns and pd.notna(df["BB_LOWER"].iloc[-1])
        else None
    )

    if mode == "breakout":
        base = key_level * 1.005
        # 현재가가 돌파선 대비 2% 이상 올랐다면 MA5 눌림목 대기 권장
        if current_close > key_level * 1.02 and ma5 and ma5 > key_level:
            return round(ma5), f"MA5 눌림목 {ma5:,.0f}원 대기 (단기 과열)"
        return round(base), f"돌파선+0.5% {round(base):,.0f}원"

    elif mode == "pullback":
        # 눌림목 진입: MA20이 핵심 기준
        if ma20:
            if ma20 >= key_level * 0.97:
                return round(ma20), f"MA20 {round(ma20):,.0f}원 (지지 확인)"
            return round(key_level), f"지지선 {round(key_level):,.0f}원 (구 저항→지지 전환)"
        return round(current_close), "현재가 진입 (MA 미산출)"

    elif mode == "reversal":
        # 반전: BB 하단 → MA20 순서로 보수적 진입가 산정
        if bb_lower and bb_lower > key_level * 0.95 and bb_lower < current_close:
            return round(bb_lower * 1.01), f"BB 하단 {round(bb_lower):,.0f}원 근처"
        if ma20 and ma20 < current_close:
            return round(ma20), f"MA20 {round(ma20):,.0f}원 근처"
        return round(current_close), "현재가 진입"

    return round(current_close), "현재가 기준"


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


def _compute_sell_confidence(rsi: float, vol_ok: bool, base: float) -> float:
    """매도 패턴 신뢰도 — RSI 과매수가 매도 신호 확인"""
    score = base
    if vol_ok:
        score += 0.05
    if rsi > 70:   # 과매수 = 매도 신호 확인
        score += 0.08
    elif rsi < 35:  # 과매도 = 이미 많이 빠진 상태 → 신호 약화
        score -= 0.12
    return round(min(max(score, 0.0), 1.0), 2)


def _candle_body_score(df: pd.DataFrame, n: int = 3) -> float:
    """최근 n봉의 양봉 비율을 0~0.05 범위로 반환 (신뢰도 보조)"""
    recent = df.tail(n)
    bull_count = int((recent["Close"] > recent["Open"]).sum())
    return round((bull_count / n) * 0.05, 3)
