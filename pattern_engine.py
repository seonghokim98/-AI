"""
pattern_engine.py - 기술적 패턴 인식 엔진 (매매 원칙 핵심)

[알고리즘 정의서] 진입 타점 포착 (기술적 분석: When to buy)

──────────────────────────────────────────────────────────────────────────
[추격매수 차단 로직] — Check_Overbought
  RSI > 80 (과매수) 또는 MA20 이격도 > 15% → 매수 트리거 비활성화

[4대 핵심 패턴] — 모든 패턴의 제1조건: 돌파 후 되돌림(눌림목) 확인
  1. 이평선 눌림목 (MA Pullback)
     : 장대양봉 형성 후, 주가가 MA20 부근까지 거래량을 줄이며 하락하는 '되돌림'
  2. 지지/저항 전환 (SR Flip)
     : 강력한 저항선을 대량 거래량으로 돌파 후, 해당 가격대까지 다시 하락(조정)
  3. 쌍바닥 (Double Bottom)
     : 우측 저점이 좌측 저점보다 높은 W자 패턴, 넥라인 돌파 후 눌림 구간
  4. 깃발형 응축 (Flag Pattern)
     : 급등 후 좁은 박스권에서 거래량이 마르며 에너지를 응축하는 구간

[진입 타점 원칙] — Find_Pullback_Entry (모든 매수 공통)
  3가지 ALL 충족 시 MA20(지정가 매수 단가) 반환:
    1. Support_Test: ABS(현재가 - MA20) / MA20 < 2%
    2. Volume_Dry:   오늘 거래량 < 20일 평균 × 50%
    3. Doji_Brake:   캔들 몸통 < 전체 범위 × 30%
──────────────────────────────────────────────────────────────────────────

[매도/경고 패턴] — 팔아라 경고
  A. 쌍봉 (Double Top)       — 폭락 대비 즉시 매도
  B. 하락깃발 (Bearish Flag) — 빨리 팔아
  C. 하락 다이아몬드          — 천천히 내려간다
  D. 박스권                   — 건들지마 위험
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

cfg = config.TECH


class PatternType(Enum):
    # ── 매수 패턴 (4대 핵심) ──────────────────────────────────────────────
    MA_PULLBACK   = "이평선 눌림목"    # 장대양봉 → MA20 되돌림
    SR_FLIP       = "지지/저항 전환"   # 저항→지지 전환 눌림목
    DOUBLE_BOTTOM = "쌍바닥"           # W패턴 넥라인 돌파 후 눌림목
    FLAG_BREAKOUT = "깃발형 응축"      # 급등 후 거래량 수렴 응축
    # ── 매도/경고 패턴 ────────────────────────────────────────────────────
    DOUBLE_TOP      = "쌍봉"
    BEARISH_FLAG    = "하락깃발"
    BEARISH_DIAMOND = "하락 다이아몬드"
    BOX_RANGE       = "박스권"


@dataclass
class PatternResult:
    pattern: PatternType
    confidence: float           # 0~1 신뢰도
    entry_price: float          # MA20 지정가 매수 단가
    resistance_level: float     # 돌파 기준 저항선
    support_level: float        # 지지선 (손절 보조)
    volume_ok: bool             # 거래량 충족 여부
    rsi: float                  # 현재 RSI
    disparity: float            # MA20 이격도 (%)
    detail: str                 # 슬랙 전송용 설명
    signal_type: str = "BUY"    # "BUY" | "SELL" | "NEUTRAL"
    action: str = ""            # 행동 지침
    entry_reason: str = ""      # 진입 근거


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def check_overbought(df: pd.DataFrame) -> tuple:
    """
    [알고리즘] Check_Overbought 구현
    Returns: (status, rsi, disparity)
      "WAIT" → 추격매수 금지 | "SAFE" → 진입 가능
    """
    if df is None or len(df) < cfg.ma_mid:
        return "SAFE", 50.0, 0.0

    ma20_col = f"MA{cfg.ma_mid}"
    current = float(df["Close"].iloc[-1])
    ma20 = (float(df[ma20_col].iloc[-1])
            if ma20_col in df.columns and pd.notna(df[ma20_col].iloc[-1])
            else current)
    rsi = (float(df["RSI"].iloc[-1])
           if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1])
           else 50.0)

    disparity = ((current - ma20) / ma20) * 100 if ma20 > 0 else 0.0

    if rsi > cfg.rsi_overbought or disparity > cfg.disparity_overbought:
        return "WAIT", rsi, disparity
    return "SAFE", rsi, disparity


def detect_patterns(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    [알고리즘] 4대 핵심 패턴 탐지 파이프라인.

    Step 1: Check_Overbought — RSI>80 or 이격도>15% → 즉시 None (관망)
    Step 2: 4대 패턴 순서 탐지 — 각 패턴 내부에서 Find_Pullback_Entry 적용
    """
    if df is None or len(df) < cfg.ma_long:
        return None

    # [알고리즘] Check_Overbought — 추격매수 절대 차단
    status, rsi, disparity = check_overbought(df)
    if status == "WAIT":
        logger.debug("Overbought BLOCKED: RSI=%.1f, Disparity=%.1f%%", rsi, disparity)
        return None

    for check_fn in [
        _check_ma_pullback,    # 1. 이평선 눌림목 (핵심)
        _check_sr_flip,        # 2. 지지/저항 전환
        _check_double_bottom,  # 3. 쌍바닥
        _check_flag_breakout,  # 4. 깃발형 응축
    ]:
        result = check_fn(df)
        if result is not None:
            logger.info("Pattern: %s conf=%.2f entry=%.0f RSI=%.1f disp=%.1f%%",
                        result.pattern.value, result.confidence,
                        result.entry_price, result.rsi, result.disparity)
            return result

    return None


def detect_sell_patterns(df: pd.DataFrame) -> Optional[PatternResult]:
    """매도/경고 패턴 탐지"""
    if df is None or len(df) < cfg.ma_long:
        return None

    for check_fn in [_check_double_top, _check_bearish_flag,
                     _check_bearish_diamond, _check_box_range]:
        result = check_fn(df)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# [핵심] Find_Pullback_Entry — 알고리즘 정의서 직접 구현
# ---------------------------------------------------------------------------

def _find_pullback_entry(df: pd.DataFrame) -> Optional[float]:
    """
    [알고리즘] Find_Pullback_Entry 구현
    3가지 조건 ALL 충족 시 MA20(지정가 매수 단가) 반환, 아니면 None.
    """
    ma20_col = f"MA{cfg.ma_mid}"
    if ma20_col not in df.columns:
        return None

    current = float(df["Close"].iloc[-1])
    ma20 = float(df[ma20_col].iloc[-1])
    if ma20 <= 0 or pd.isna(ma20):
        return None

    vol_today = float(df["Volume"].iloc[-1])
    vol_avg = float(df["VOL_MA20"].iloc[-1]) if "VOL_MA20" in df.columns else vol_today
    if vol_avg <= 0:
        return None

    open_p  = float(df["Open"].iloc[-1])
    close_p = float(df["Close"].iloc[-1])
    high_p  = float(df["High"].iloc[-1])
    low_p   = float(df["Low"].iloc[-1])

    # 조건 1: Support_Test — MA20 ±2% 이내 (지지선 테스트)
    support_test = abs(current - ma20) / ma20 < cfg.pullback_ma_tolerance

    # 조건 2: Volume_Dry — 거래량 급감 (세력 이탈 없음)
    volume_dry = vol_today < (vol_avg * cfg.volume_dry_ratio)

    # 조건 3: Doji_Brake — 도지형 캔들 또는 하락 멈춤 신호
    candle_body = abs(open_p - close_p)
    candle_tail = high_p - low_p
    doji_brake = (candle_tail > 0) and (candle_body < candle_tail * cfg.doji_body_ratio)

    logger.debug(
        "Pullback[MA20=%.0f curr=%.0f disp=%.1f%%]: support=%s vol_dry=%s doji=%s "
        "(vol_ratio=%.2f body_ratio=%.2f)",
        ma20, current, abs(current - ma20) / ma20 * 100,
        support_test, volume_dry, doji_brake,
        vol_today / vol_avg,
        candle_body / candle_tail if candle_tail > 0 else 0,
    )

    if support_test and volume_dry and doji_brake:
        return round(ma20)   # MA20을 지정가 매수 단가로 설정
    return None              # 타점 안 옴 → 대기


# ---------------------------------------------------------------------------
# 1. 이평선 눌림목 (MA Pullback)
# ---------------------------------------------------------------------------

def _check_ma_pullback(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    [알고리즘] 이평선 눌림목
    Setup: 최근 N봉 이내 장대양봉(몸통 3%↑) 발생 → 이후 MA20으로 조정 중
    Entry: Find_Pullback_Entry (MA20 ±2% + 거래량 50%↓ + 도지)
    """
    lookback = cfg.big_candle_lookback
    if len(df) < lookback + 5:
        return None

    # Setup: 최근 장대양봉 탐색
    big_bull_idx = None
    for i in range(-lookback, -1):
        bar = df.iloc[i]
        open_p  = float(bar["Open"])
        close_p = float(bar["Close"])
        if open_p <= 0:
            continue
        body_pct = (close_p - open_p) / open_p
        if body_pct >= cfg.big_candle_min_ratio:
            big_bull_idx = i
            break

    if big_bull_idx is None:
        return None

    ma20_col = f"MA{cfg.ma_mid}"
    if ma20_col not in df.columns:
        return None

    ma20 = float(df[ma20_col].iloc[-1])
    current = float(df["Close"].iloc[-1])
    bull_high = float(df["High"].iloc[big_bull_idx])

    # 현재가가 장대양봉 고점보다 낮아야 함 (조정 진행 중)
    if current >= bull_high:
        return None

    # [알고리즘] Find_Pullback_Entry
    entry_price = _find_pullback_entry(df)
    if entry_price is None:
        return None

    _, rsi, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) < float(df["VOL_MA20"].iloc[-1]) * cfg.volume_dry_ratio
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, base=0.75)

    return PatternResult(
        pattern=PatternType.MA_PULLBACK,
        confidence=confidence,
        entry_price=float(entry_price),
        resistance_level=round(bull_high, 2),
        support_level=round(ma20 * 0.97, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        disparity=disparity,
        entry_reason=f"MA20 {entry_price:,.0f}원 지지 확인 (지정가)",
        action="📈 MA20 눌림목 지정가 매수",
        detail=(
            f"장대양봉({cfg.big_candle_min_ratio*100:.0f}%↑) → MA20 눌림목 | "
            f"RSI {rsi:.1f} | 이격도 {disparity:.1f}% | 진입가 MA20={entry_price:,.0f}원"
        ),
    )


# ---------------------------------------------------------------------------
# 2. 지지/저항 전환 (SR Flip)
# ---------------------------------------------------------------------------

def _check_sr_flip(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    [알고리즘] 지지/저항 전환
    Setup: 과거 강력한 저항선을 대량 거래량으로 돌파 → 해당 가격대로 조정
    Entry: Find_Pullback_Entry (MA20이 저항선 근방에 있어야 함)
    """
    n = cfg.breakout_window
    if len(df) < n * 2 + 5:
        return None

    # 과거 저항 구간 최고가
    old_range = df["High"].iloc[-(n * 2):-n]
    resistance_level = float(old_range.max())

    # 대량 거래량 돌파 확인 (최근 n봉 이내)
    after_range = df.iloc[-n:-1]
    breakout_with_volume = False
    for i in range(len(after_range)):
        bar = after_range.iloc[i]
        vol_ma = (float(after_range["VOL_MA20"].iloc[i])
                  if "VOL_MA20" in after_range.columns and pd.notna(after_range["VOL_MA20"].iloc[i])
                  else 0)
        if (vol_ma > 0 and
                float(bar["Close"]) > resistance_level and
                float(bar["Volume"]) > vol_ma * cfg.volume_surge_ratio):
            breakout_with_volume = True
            break

    if not breakout_with_volume:
        return None

    # 현재 저항→지지 전환 구간 도달 확인 (±3%)
    current_close = float(df["Close"].iloc[-1])
    in_flip_zone = (resistance_level * (1 - cfg.pullback_tolerance) <=
                    current_close <=
                    resistance_level * (1 + cfg.pullback_tolerance))
    if not in_flip_zone:
        return None

    # [알고리즘] Find_Pullback_Entry
    entry_price = _find_pullback_entry(df)
    if entry_price is None:
        return None

    _, rsi, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) < float(df["VOL_MA20"].iloc[-1]) * cfg.volume_dry_ratio
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, base=0.72)

    return PatternResult(
        pattern=PatternType.SR_FLIP,
        confidence=confidence,
        entry_price=float(entry_price),
        resistance_level=round(resistance_level, 2),
        support_level=round(resistance_level * 0.97, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        disparity=disparity,
        entry_reason=f"구 저항→지지 전환 | MA20={entry_price:,.0f}원 지정가",
        action="📈 저항→지지 전환 구간 지정가 매수",
        detail=(
            f"구 저항선 {resistance_level:,.0f}원 → 지지 전환 | "
            f"대량 거래량 돌파 확인 | RSI {rsi:.1f} | 이격도 {disparity:.1f}%"
        ),
    )


# ---------------------------------------------------------------------------
# 3. 쌍바닥 (Double Bottom)
# ---------------------------------------------------------------------------

def _check_double_bottom(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    [알고리즘] 쌍바닥 — 우측 저점이 좌측보다 높은 W자
    Setup: W패턴 + 넥라인 돌파 또는 근접
    Entry: Find_Pullback_Entry (넥라인 돌파 후 MA20 눌림목)
    """
    window = cfg.double_bottom_window
    tol = cfg.double_bottom_tolerance

    if len(df) < window + 5:
        return None

    recent = df["Close"].iloc[-window:].reset_index(drop=True)

    idx1 = int(recent.idxmin())
    val1 = float(recent.iloc[idx1])

    if idx1 + 5 >= len(recent):
        return None
    second_half = recent.iloc[idx1 + 5:]
    if second_half.empty:
        return None
    idx2_rel = int(second_half.idxmin() - idx1 - 5)
    idx2 = idx1 + 5 + idx2_rel
    if idx2 >= len(recent):
        idx2 = len(recent) - 1
    val2 = float(recent.iloc[idx2])

    # [알고리즘] 우측 저점이 좌측보다 높아야 함
    if val2 <= val1 * (1 - tol):
        return None
    if abs(val1 - val2) / val1 > tol:
        return None

    # 넥라인: 두 저점 사이 최고가
    neckline = float(recent.iloc[idx1:idx2].max()) if idx1 < idx2 else val1 * 1.05

    current_close = float(df["Close"].iloc[-1])
    if current_close < neckline * 0.97:
        return None

    # [알고리즘] Find_Pullback_Entry
    entry_price = _find_pullback_entry(df)
    if entry_price is None:
        return None

    _, rsi, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    confidence = _compute_confidence(rsi=rsi, vol_ok=vol_ok, base=0.68)

    return PatternResult(
        pattern=PatternType.DOUBLE_BOTTOM,
        confidence=confidence,
        entry_price=float(entry_price),
        resistance_level=round(neckline, 2),
        support_level=round((val1 + val2) / 2, 2),
        volume_ok=vol_ok,
        rsi=rsi,
        disparity=disparity,
        entry_reason=f"W패턴 넥라인={neckline:,.0f}원 돌파 후 MA20 눌림목",
        action="📈 쌍바닥 W패턴 MA20 지정가 매수",
        detail=(
            f"쌍바닥 좌={val1:,.0f}원 / 우={val2:,.0f}원 (우측↑) | "
            f"넥라인 {neckline:,.0f}원 돌파 후 MA20 눌림목 | RSI {rsi:.1f}"
        ),
    )


# ---------------------------------------------------------------------------
# 4. 깃발형 응축 (Flag Pattern)
# ---------------------------------------------------------------------------

def _check_flag_breakout(df: pd.DataFrame) -> Optional[PatternResult]:
    """
    [알고리즘] 깃발형 응축
    Setup: 급등(깃대) 후 좁은 박스권에서 거래량이 마르며 에너지 응축
    Entry: Find_Pullback_Entry (응축 중 MA20 근방 + 거래량 극감 + 도지)
    """
    min_pole = cfg.flag_pole_min_ratio
    consol = cfg.flag_consolidation_bars
    max_retrace = cfg.flag_max_retracement

    if len(df) < consol + 20:
        return None

    search_end   = len(df) - consol - 1
    search_start = max(0, search_end - 20)

    pole_low  = None
    pole_high = None
    pole_end  = None

    for i in range(search_start, search_end):
        if i + 3 >= len(df):
            break
        segment_ret = (df["Close"].iloc[i + 3] - df["Close"].iloc[i]) / df["Close"].iloc[i]
        if segment_ret >= min_pole:
            pole_low  = float(df["Close"].iloc[i])
            pole_high = float(df["Close"].iloc[i + 3])
            pole_end  = i + 3
            break

    if pole_low is None or pole_end is None:
        return None

    flag_start = pole_end
    flag_end   = min(flag_start + consol + 5, len(df) - 1)
    if flag_end - flag_start < consol:
        return None

    flag_bars  = df.iloc[flag_start:flag_end]
    flag_high  = float(flag_bars["High"].max())
    flag_low   = float(flag_bars["Low"].min())

    pole_height = pole_high - pole_low
    if pole_height <= 0:
        return None
    retrace = (pole_high - flag_low) / pole_height
    if retrace > max_retrace:
        return None

    # 깃발 구간 거래량 수렴 확인 (응축의 핵심)
    flag_vol_avg = float(flag_bars["Volume"].mean())
    vol_ma_at_flag = (float(df["VOL_MA20"].iloc[flag_start])
                      if "VOL_MA20" in df.columns and pd.notna(df["VOL_MA20"].iloc[flag_start])
                      else flag_vol_avg * 1.2)
    vol_contracting = flag_vol_avg < vol_ma_at_flag * 0.8
    if not vol_contracting:
        return None

    # 현재 가격이 깃발 구간 내에 있는지 확인 (응축 중)
    current_close = float(df["Close"].iloc[-1])
    in_flag = flag_low * 0.98 <= current_close <= flag_high * 1.02
    if not in_flag:
        return None

    # [알고리즘] Find_Pullback_Entry
    entry_price = _find_pullback_entry(df)
    if entry_price is None:
        return None

    _, rsi, disparity = check_overbought(df)
    confidence = _compute_confidence(rsi=rsi, vol_ok=True, base=0.70)

    return PatternResult(
        pattern=PatternType.FLAG_BREAKOUT,
        confidence=confidence,
        entry_price=float(entry_price),
        resistance_level=round(flag_high, 2),
        support_level=round(flag_low, 2),
        volume_ok=True,
        rsi=rsi,
        disparity=disparity,
        entry_reason=f"깃발 응축 중 MA20={entry_price:,.0f}원 지정가 선매수",
        action="📈 깃발 응축 MA20 지정가 선매수",
        detail=(
            f"깃대 {pole_low:,.0f}→{pole_high:,.0f}원 ({min_pole*100:.0f}%↑) | "
            f"깃발 거래량 수렴 응축 중 | RSI {rsi:.1f} | 이격도 {disparity:.1f}%"
        ),
    )


# ---------------------------------------------------------------------------
# 매도/경고 패턴 (팔아라 경고)
# ---------------------------------------------------------------------------

def _check_double_top(df: pd.DataFrame) -> Optional[PatternResult]:
    """쌍봉 — 폭락 대비 즉시 매도"""
    window = cfg.double_bottom_window
    tol = cfg.double_bottom_tolerance
    if len(df) < window + 5:
        return None

    recent_high = df["High"].iloc[-window:].reset_index(drop=True)
    recent_low  = df["Low"].iloc[-window:].reset_index(drop=True)

    idx1 = int(recent_high.idxmax())
    val1 = float(recent_high.iloc[idx1])
    if idx1 + 5 >= len(recent_high):
        return None
    second_half = recent_high.iloc[idx1 + 5:]
    if second_half.empty:
        return None
    idx2_rel = int(second_half.idxmax() - idx1 - 5)
    idx2 = idx1 + 5 + idx2_rel
    if idx2 >= len(recent_high):
        return None
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

    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 50.0
    _, _, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.88)

    return PatternResult(
        pattern=PatternType.DOUBLE_TOP,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round((val1 + val2) / 2, 2),
        support_level=round(neckline * 0.95, 2),
        volume_ok=vol_ok, rsi=rsi, disparity=disparity,
        signal_type="SELL",
        action="⚠️ 폭락 대비 즉시 매도 검토",
        detail=f"쌍봉 고점1={val1:,.0f}원/고점2={val2:,.0f}원 | 넥라인 {neckline:,.0f}원 하향 | RSI {rsi:.1f}",
    )


def _check_bearish_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """하락깃발 — 빨리 팔아"""
    min_pole = cfg.flag_pole_min_ratio
    consol = cfg.flag_consolidation_bars
    max_retrace = cfg.flag_max_retracement

    if len(df) < consol + 10:
        return None

    search_end   = len(df) - consol - 1
    search_start = max(0, search_end - 15)

    pole_high = None
    pole_low  = None
    pole_end  = None

    for i in range(search_start, search_end):
        if i + 3 >= len(df):
            break
        segment_ret = (df["Close"].iloc[i] - df["Close"].iloc[i + 3]) / df["Close"].iloc[i]
        if segment_ret >= min_pole:
            pole_high = float(df["Close"].iloc[i])
            pole_low  = float(df["Close"].iloc[i + 3])
            pole_end  = i + 3
            break

    if pole_high is None or pole_end is None:
        return None

    flag_bars = df.iloc[pole_end: pole_end + consol + 5]
    if len(flag_bars) < consol:
        return None

    flag_high = float(flag_bars["High"].max())
    flag_low  = float(flag_bars["Low"].min())
    pole_height = pole_high - pole_low
    if pole_height <= 0:
        return None
    if (flag_high - pole_low) / pole_height > max_retrace:
        return None

    current_close = float(df["Close"].iloc[-1])
    if current_close >= flag_low:
        return None

    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 50.0
    _, _, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1]) * cfg.volume_surge_ratio
    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.75)

    return PatternResult(
        pattern=PatternType.BEARISH_FLAG,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(flag_high, 2),
        support_level=round(pole_low * 0.97, 2),
        volume_ok=vol_ok, rsi=rsi, disparity=disparity,
        signal_type="SELL",
        action="🔴 빨리 팔아",
        detail=f"하락깃대 {pole_high:,.0f}→{pole_low:,.0f}원 | 깃발 {flag_low:,.0f}원 이탈 | RSI {rsi:.1f}",
    )


def _check_bearish_diamond(df: pd.DataFrame) -> Optional[PatternResult]:
    """하락 다이아몬드 — 천천히 내려간다"""
    window = 30
    if len(df) < window + 5:
        return None

    recent = df.iloc[-(window + 5):-1]
    half   = len(recent) // 2

    early_range = float(recent["High"].iloc[:half].max()) - float(recent["Low"].iloc[:half].min())
    late_range  = float(recent["High"].iloc[half:].max()) - float(recent["Low"].iloc[half:].min())

    if early_range <= 0 or late_range >= early_range * 0.85:
        return None
    if float(recent["High"].iloc[half:].max()) >= float(recent["High"].iloc[:half].max()):
        return None

    current_close = float(df["Close"].iloc[-1])
    late_low = float(recent["Low"].iloc[half:].min())
    if current_close >= late_low:
        return None

    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 50.0
    _, _, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) > float(df["VOL_MA20"].iloc[-1])
    confidence = _compute_sell_confidence(rsi=rsi, vol_ok=vol_ok, base=0.60)

    return PatternResult(
        pattern=PatternType.BEARISH_DIAMOND,
        confidence=confidence,
        entry_price=round(current_close, 2),
        resistance_level=round(float(recent["High"].iloc[half:].max()), 2),
        support_level=round(late_low * 0.95, 2),
        volume_ok=vol_ok, rsi=rsi, disparity=disparity,
        signal_type="SELL",
        action="🟠 천천히 내려간다 — 단계적 매도",
        detail=f"다이아몬드 변동폭 확대→수렴 | 후반 저점 {late_low:,.0f}원 이탈 | RSI {rsi:.1f}",
    )


def _check_box_range(df: pd.DataFrame) -> Optional[PatternResult]:
    """박스권 — 건들지마 위험"""
    window = 20
    if len(df) < window + 5:
        return None

    recent   = df.iloc[-window:]
    box_high = float(recent["High"].max())
    box_low  = float(recent["Low"].min())

    if box_low <= 0:
        return None

    box_range_ratio = (box_high - box_low) / box_low
    if not (0.04 <= box_range_ratio <= 0.15):
        return None

    ma20_col = f"MA{cfg.ma_mid}"
    ma20 = df[ma20_col].dropna()
    if len(ma20) < window:
        return None
    ma_slope = (float(ma20.iloc[-1]) - float(ma20.iloc[-window])) / float(ma20.iloc[-window])
    if abs(ma_slope) > 0.04:
        return None

    current_close = float(df["Close"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 50.0
    _, _, disparity = check_overbought(df)
    vol_ok = float(df["Volume"].iloc[-1]) < float(df["VOL_MA20"].iloc[-1]) * 1.2

    return PatternResult(
        pattern=PatternType.BOX_RANGE,
        confidence=0.45,
        entry_price=round(current_close, 2),
        resistance_level=round(box_high, 2),
        support_level=round(box_low, 2),
        volume_ok=vol_ok, rsi=rsi, disparity=disparity,
        signal_type="NEUTRAL",
        action="⚠️ 건들지마 위험 — 관망",
        detail=f"박스권 상단={box_high:,.0f}원/하단={box_low:,.0f}원 | 변동폭 {box_range_ratio*100:.1f}% | RSI {rsi:.1f}",
    )


# ---------------------------------------------------------------------------
# 신뢰도 계산
# ---------------------------------------------------------------------------

def _compute_confidence(rsi: float, vol_ok: bool, base: float) -> float:
    score = base
    if vol_ok:
        score += 0.08
    if 40 <= rsi <= 60:
        score += 0.10     # 이상적 RSI 구간 (눌림목)
    elif rsi > 70:
        score -= 0.15     # 과매수 감점
    return round(min(max(score, 0.0), 1.0), 2)


def _compute_sell_confidence(rsi: float, vol_ok: bool, base: float) -> float:
    score = base
    if vol_ok:
        score += 0.05
    if rsi > 70:
        score += 0.08     # 과매수 = 매도 신호 확인
    elif rsi < 35:
        score -= 0.12     # 과매도 = 신호 약화
    return round(min(max(score, 0.0), 1.0), 2)
