"""
market_filter.py - 시장 트렌드 필터 (매매 원칙 1번)
시장 전체가 하락 추세일 때는 모든 매수 신호를 차단한다.
"시장 안 좋으면 → 무조건 기다려!(관망)"
"""
import logging
from dataclasses import dataclass
from enum import Enum

import config
import data_provider as dp

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "강세장"      # 매수 신호 허용
    NEUTRAL = "중립"    # 선별적 허용 (고품질 신호만)
    BEAR = "약세장"     # 모든 신호 차단


@dataclass
class MarketStatus:
    regime: MarketRegime
    kospi_above_ma: bool
    kosdaq_above_ma: bool
    vix_ok: bool
    foreign_buy_streak: int
    detail: str           # 슬랙 전송용 상세 메시지


def analyze_market() -> MarketStatus:
    """
    코스피, 코스닥, VIX, 외국인 수급을 종합하여 시장 상태를 판단한다.

    Returns:
        MarketStatus 객체
    """
    cfg = config.MARKET

    kospi_ok = _check_index_trend("^KS11", cfg.kospi_ma_period)
    kosdaq_ok = _check_index_trend("^KQ11", cfg.kosdaq_ma_period)
    vix_ok = _check_vix(config.MARKET.vix_threshold)
    foreign_streak = _check_foreign_net_buy(cfg.foreign_buy_streak)

    # 판단 로직 -------------------------------------------------------
    # 강세 조건: 코스피 20일선 위 + VIX 정상
    # 중립 조건: 코스피 or 코스닥 중 하나만 양호
    # 약세 조건: 코스피 20일선 아래 or VIX 과열

    if not kospi_ok and not vix_ok:
        regime = MarketRegime.BEAR
    elif kospi_ok and vix_ok:
        regime = MarketRegime.BULL if (kosdaq_ok or foreign_streak >= cfg.foreign_buy_streak) else MarketRegime.NEUTRAL
    else:
        regime = MarketRegime.NEUTRAL

    detail = _build_detail(kospi_ok, kosdaq_ok, vix_ok, foreign_streak)
    logger.info("Market regime: %s | %s", regime.value, detail)

    return MarketStatus(
        regime=regime,
        kospi_above_ma=kospi_ok,
        kosdaq_above_ma=kosdaq_ok,
        vix_ok=vix_ok,
        foreign_buy_streak=foreign_streak,
        detail=detail,
    )


def is_market_ok(status: MarketStatus) -> bool:
    """
    신호 발생 허용 여부 반환.
    약세장(BEAR)이면 무조건 False.
    """
    return status.regime != MarketRegime.BEAR


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _check_index_trend(ticker: str, ma_period: int) -> bool:
    """지수가 MA 위에 있으면 True"""
    df = dp.get_index_data(ticker)
    if df is None or len(df) < ma_period:
        logger.warning("Cannot check trend for %s; treating as bearish", ticker)
        return False

    ma = df["Close"].rolling(ma_period).mean()
    last_close = df["Close"].iloc[-1]
    last_ma = ma.iloc[-1]

    above = bool(last_close > last_ma)
    pct_from_ma = (last_close - last_ma) / last_ma * 100
    logger.debug("%s close=%.2f MA%d=%.2f (%.1f%%) above=%s",
                 ticker, last_close, ma_period, last_ma, pct_from_ma, above)
    return above


def _check_vix(threshold: float) -> bool:
    """
    VIX 지수가 임계값 미만이면 True (공포 없음 = 시장 양호).
    한국 시장은 VKOSPI가 적합하나 yfinance 미지원,
    ^VIX(미국 VIX)를 대리 지표로 사용.
    """
    df = dp.get_index_data("^VIX", period="1mo")
    if df is None or df.empty:
        logger.warning("VIX data unavailable; treating as OK")
        return True

    last_vix = float(df["Close"].iloc[-1])
    ok = last_vix < threshold
    logger.debug("VIX=%.2f threshold=%.1f ok=%s", last_vix, threshold, ok)
    return ok


def _check_foreign_net_buy(required_streak: int) -> int:
    """
    외국인 순매수 연속일 수 반환.
    KIS API 미연동 시 0 반환 (차단 요소가 아닌 보너스 조건으로 처리).
    """
    net_buys = dp.get_foreign_net_buy("^KS11", days=required_streak + 2)
    if net_buys is None:
        return 0

    streak = 0
    for val in reversed(net_buys):
        if val > 0:
            streak += 1
        else:
            break
    return streak


def _build_detail(kospi_ok: bool, kosdaq_ok: bool, vix_ok: bool, streak: int) -> str:
    parts = [
        f"코스피 20일선 {'위' if kospi_ok else '아래'}",
        f"코스닥 20일선 {'위' if kosdaq_ok else '아래'}",
        f"VIX {'정상' if vix_ok else '과열'}",
    ]
    if streak > 0:
        parts.append(f"외국인 {streak}일 연속 순매수")
    return " | ".join(parts)
