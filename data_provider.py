"""
data_provider.py - 시장 데이터 수집 모듈
yfinance 기반 OHLCV + 지표 데이터 제공
한국투자증권 REST API 연동 옵션 포함
"""
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 캐시 유효 시간 (초) - 동일 종목 반복 호출 방지
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_CACHE_TTL_SECONDS: int = 60


def _is_cache_valid(key: str) -> bool:
    if key not in _CACHE:
        return False
    cached_at, _ = _CACHE[key]
    return (datetime.now() - cached_at).seconds < _CACHE_TTL_SECONDS


def _get_cache(key: str) -> Optional[pd.DataFrame]:
    if _is_cache_valid(key):
        return _CACHE[key][1]
    return None


def _set_cache(key: str, df: pd.DataFrame) -> None:
    _CACHE[key] = (datetime.now(), df)


# ---------------------------------------------------------------------------
# OHLCV 데이터 수집
# ---------------------------------------------------------------------------
def get_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    종목 OHLCV 데이터를 반환한다.
    yfinance 사용, 한국 종목은 .KS / .KQ 접미사 필요.

    Args:
        ticker: 'AAPL', '005930.KS' 등
        period:  '1mo', '3mo', '6mo', '1y'
        interval: '1d', '1h', '5m'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        실패 시 None
    """
    cache_key = f"{ticker}_{period}_{interval}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        # 최신 yfinance는 단일 종목도 MultiIndex 컬럼으로 반환 → 첫 레벨만 사용
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        # 이동평균 및 거래량 지표 사전 계산
        df = _add_indicators(df)
        _set_cache(cache_key, df)
        return df

    except Exception as exc:
        logger.error("Failed to fetch OHLCV for %s: %s", ticker, exc)
        return None


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """공통 기술적 지표 컬럼 추가"""
    cfg = config.TECH

    # 이동평균
    df[f"MA{cfg.ma_short}"] = df["Close"].rolling(cfg.ma_short).mean()
    df[f"MA{cfg.ma_mid}"] = df["Close"].rolling(cfg.ma_mid).mean()
    df[f"MA{cfg.ma_long}"] = df["Close"].rolling(cfg.ma_long).mean()

    # 거래량 이동평균
    df["VOL_MA20"] = df["Volume"].rolling(cfg.ma_mid).mean()

    # RSI
    df["RSI"] = _rsi(df["Close"], cfg.rsi_period)

    # 볼린저 밴드
    bb_mid = df["Close"].rolling(cfg.bb_period).mean()
    bb_std = df["Close"].rolling(cfg.bb_period).std()
    df["BB_UPPER"] = bb_mid + cfg.bb_std * bb_std
    df["BB_LOWER"] = bb_mid - cfg.bb_std * bb_std
    df["BB_MID"] = bb_mid

    # ATR (손절가 계산 보조)
    df["ATR"] = _atr(df, 14)

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# 지수 데이터
# ---------------------------------------------------------------------------
def get_index_data(ticker: str = "^KS11", period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    코스피(^KS11), 코스닥(^KQ11), VIX(^VIX) 등 지수 데이터 반환
    """
    cache_key = f"index_{ticker}_{period}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return None

        # 최신 yfinance는 단일 종목도 MultiIndex 컬럼으로 반환 → 첫 레벨만 사용
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        _set_cache(cache_key, df)
        return df
    except Exception as exc:
        logger.error("Failed to fetch index %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# 종목 기본 정보 (PER, PBR 등 펀더멘털)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def get_fundamentals(ticker: str) -> dict:
    """
    yfinance를 통해 PER, PBR, EPS, 배당수익률 등 반환.
    장중에는 캐시되어 반복 호출 최소화.

    Returns:
        {'per': float, 'pbr': float, 'eps': float, 'market_cap': int, ...}
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "per": info.get("trailingPE") or info.get("forwardPE"),
            "pbr": info.get("priceToBook"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "shares_outstanding": info.get("sharesOutstanding"),
        }
    except Exception as exc:
        logger.error("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# 현재가 조회 (실시간 단순 버전)
# ---------------------------------------------------------------------------
def get_current_price(ticker: str) -> Optional[float]:
    try:
        data = yf.Ticker(ticker).fast_info
        price = data.last_price
        return float(price) if price else None
    except Exception as exc:
        logger.error("Failed to get current price for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# 외국인 순매수 데이터 (yfinance 미지원 → KIS API 스텁)
# ---------------------------------------------------------------------------
def get_foreign_net_buy(ticker: str, days: int = 5) -> Optional[list]:
    """
    외국인 일별 순매수 금액 리스트 반환 (최근 days일).
    yfinance에서는 제공하지 않으므로 한국투자증권 API 연동 필요.
    현재는 스텁(Stub)으로 None 반환.
    KIS API 연동 후 실제 데이터로 교체할 것.
    """
    if not config.KIS_APP_KEY:
        logger.debug("KIS API key not set; foreign net buy skipped")
        return None

    # TODO: KIS REST API 연동
    # endpoint = "https://openapi.koreainvestment.com:9443/..."
    return None


# ---------------------------------------------------------------------------
# 감시 종목 일괄 수집
# ---------------------------------------------------------------------------
def fetch_all_watchlist(period: str = "6mo") -> dict[str, pd.DataFrame]:
    """
    config.WATCHLIST 전체 종목의 OHLCV 데이터를 딕셔너리로 반환.
    """
    result = {}
    for ticker in config.WATCHLIST:
        df = get_ohlcv(ticker, period=period)
        if df is not None and len(df) >= config.TECH.ma_long:
            result[ticker] = df
        else:
            logger.warning("Skipping %s: insufficient data", ticker)
    return result
