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
import requests
import yfinance as yf

try:
    from bs4 import BeautifulSoup as _BS
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 캐시 유효 시간 (초) - 동일 종목 반복 호출 방지
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_CACHE_TTL_SECONDS: int = 300  # 5분 (자동새로고침 주기와 맞춤)


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
def get_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
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
# 네이버 금융 스크래핑 — 한국 종목 PER / PBR / EPS
# ---------------------------------------------------------------------------
_NAVER_FIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    "Referer": "https://finance.naver.com/",
}


def _parse_naver_num(el) -> Optional[float]:
    """네이버 금융 em 태그 → float 변환"""
    if el is None:
        return None
    text = el.get_text(strip=True).replace(",", "").replace("배", "").replace("원", "")
    try:
        v = float(text)
        return v if v != 0 else None
    except (ValueError, TypeError):
        return None


def _fetch_naver_fundamentals(ticker: str) -> dict:
    """
    네이버 금융 종목 메인 페이지에서 PER / PBR / EPS 스크래핑.
    한국 종목(.KS / .KQ)에만 적용.
    beautifulsoup4 미설치 시 빈 dict 반환.
    """
    if not _BS4_OK:
        return {}

    code = ticker.split(".")[0]
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        resp = requests.get(url, headers=_NAVER_FIN_HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = _BS(resp.text, "html.parser")

        per = _parse_naver_num(soup.select_one("em#_per"))
        eps = _parse_naver_num(soup.select_one("em#_eps"))
        pbr = _parse_naver_num(soup.select_one("em#_pbr"))

        if any(v is not None for v in [per, eps, pbr]):
            logger.debug("Naver fundamentals OK for %s: PER=%s PBR=%s EPS=%s", ticker, per, pbr, eps)
            return {"per": per, "pbr": pbr, "eps": eps}
        return {}
    except Exception as exc:
        logger.warning("Naver fundamentals scraping failed for %s: %s", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# 종목 기본 정보 (PER, PBR 등 펀더멘털)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def get_fundamentals(ticker: str) -> dict:
    """
    PER, PBR, EPS, 배당수익률 등 반환.
    한국 종목(.KS/.KQ): 네이버 금융 스크래핑 우선, 실패 시 yfinance 폴백.
    해외 종목: yfinance 사용.

    Returns:
        {'per': float, 'pbr': float, 'eps': float, 'market_cap': int, ...}
    """
    # ── 한국 종목 → 네이버 금융 우선 ──────────────────────────────────────
    if ".KS" in ticker or ".KQ" in ticker:
        naver = _fetch_naver_fundamentals(ticker)
        if naver.get("per") or naver.get("pbr") or naver.get("eps"):
            # 시가총액 등 추가 정보는 yfinance에서 보완
            try:
                info = yf.Ticker(ticker).info
                naver.setdefault("market_cap", info.get("marketCap"))
                naver.setdefault("sector", info.get("sector", ""))
                naver.setdefault("industry", info.get("industry", ""))
                naver.setdefault("dividend_yield", info.get("dividendYield"))
                naver.setdefault("52w_high", info.get("fiftyTwoWeekHigh"))
                naver.setdefault("52w_low", info.get("fiftyTwoWeekLow"))
                naver.setdefault("shares_outstanding", info.get("sharesOutstanding"))
            except Exception:
                pass
            return naver

    # ── 해외 종목 또는 네이버 실패 → yfinance ─────────────────────────────
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
# 네이버 금융 실시간 데이터 — 폴링 API (지연 없는 장중 실시간 시세)
# ---------------------------------------------------------------------------
_NAVER_POLLING_URL = "https://polling.finance.naver.com/api/realtime"
_NAVER_POLLING_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    "Referer": "https://finance.naver.com/",
    "X-Requested-With": "XMLHttpRequest",
}

# 네이버 모바일 API (폴링 API 실패 시 폴백)
_NAVER_MOBILE_BASE = "https://m.stock.naver.com/api"
_NAVER_MOBILE_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.6 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://m.stock.naver.com/",
}

# yfinance 티커 → 네이버 폴링 API 지수 코드 매핑
_YF_TO_NAVER_INDEX: dict = {
    "^KS11":    "KOSPI",
    "^KQ11":    "KOSDAQ",
    "^GSPC":    "SNP500",
    "^NDX":     "NASDAQ",
    "^N225":    "NIKKEI",
    "^HSI":     "HSI",
    "^VIX":     "VIX",
    "DX-Y.NYB": "DXY",
}


def _parse_polling_item(item: dict) -> Optional[dict]:
    """네이버 폴링 API 단건 응답 → {"price", "change", "change_pct"} 변환
    rf 필드: "2"=상승, "5"=하락, "3"=보합
    """
    try:
        price = float(str(item.get("nv", 0)).replace(",", ""))
        cv    = float(str(item.get("cv", 0)).replace(",", ""))
        cr    = float(str(item.get("cr", 0)).replace(",", ""))
        rf    = str(item.get("rf", "3"))
        if rf == "5":
            cv, cr = -abs(cv), -abs(cr)
        elif rf == "2":
            cv, cr = abs(cv), abs(cr)
        if price > 0:
            return {"price": price, "change": cv, "change_pct": cr}
    except (ValueError, TypeError):
        pass
    return None


def _parse_mobile_response(data: dict) -> Optional[dict]:
    """
    네이버 모바일 API 응답 파싱 (필드명이 버전마다 다를 수 있어 다중 시도).
    updown/fluctuations: "2"=상승, "5"=하락, "1"=보합
    """
    try:
        price_raw = (data.get("closePrice") or data.get("currentPrice")
                     or data.get("stockEndPrice") or "")
        ratio_raw = (data.get("changesRatio") or data.get("fluctuationsRatio")
                     or data.get("changeRate") or "")
        updown    = str(data.get("updown") or data.get("fluctuations") or "1")

        if not price_raw or not ratio_raw:
            return None

        price = float(str(price_raw).replace(",", ""))
        ratio = float(str(ratio_raw).replace(",", "").replace("%", ""))

        if updown == "5":
            ratio = -abs(ratio)
        else:
            ratio = abs(ratio)

        if price > 0:
            return {"price": price, "change": 0.0, "change_pct": ratio}
    except (ValueError, TypeError):
        pass
    return None


def _fetch_naver_mobile_index(naver_code: str) -> Optional[dict]:
    """네이버 모바일 API로 단일 지수 조회 (폴링 API 실패 시 폴백)"""
    url = f"{_NAVER_MOBILE_BASE}/index/{naver_code}/price"
    try:
        resp = requests.get(url, headers=_NAVER_MOBILE_HDR, timeout=5)
        return _parse_mobile_response(resp.json())
    except Exception as exc:
        logger.warning("Naver mobile index API failed for %s: %s", naver_code, exc)
    return None


def _fetch_naver_mobile_stock(code: str) -> Optional[dict]:
    """네이버 모바일 API로 단일 종목 실시간 시세 조회 (폴링 API 실패 시 폴백)"""
    url = f"{_NAVER_MOBILE_BASE}/stock/{code}/price"
    try:
        resp = requests.get(url, headers=_NAVER_MOBILE_HDR, timeout=5)
        return _parse_mobile_response(resp.json())
    except Exception as exc:
        logger.warning("Naver mobile stock API failed for %s: %s", code, exc)
    return None


def fetch_naver_realtime_indices(yf_tickers: list) -> dict:
    """
    네이버 금융에서 여러 지수 실시간 데이터를 조회.
    1차: 폴링 API (배치) → 2차: 모바일 API (개별, 폴백)

    Args:
        yf_tickers: yfinance 티커 리스트 (예: ["^KS11", "^KQ11", "^GSPC"])
    Returns:
        {yf_ticker: {"price": float, "change": float, "change_pct": float}}
    """
    ticker_to_naver = {t: _YF_TO_NAVER_INDEX[t] for t in yf_tickers if t in _YF_TO_NAVER_INDEX}
    if not ticker_to_naver:
        return {}

    naver_to_yf = {v: k for k, v in ticker_to_naver.items()}
    # ":","," 를 URL-인코딩하지 않도록 raw URL로 직접 구성
    query = ",".join(f"SERVICE_INDEX:{nc}" for nc in ticker_to_naver.values())

    result = {}
    try:
        resp = requests.get(
            f"{_NAVER_POLLING_URL}?query={query}",   # params= 미사용 (인코딩 방지)
            headers=_NAVER_POLLING_HDR,
            timeout=6,
        )
        data = resp.json()
        for area in data.get("result", {}).get("areas", []):
            for item in area.get("datas", []):
                nv_cd = item.get("cd", "")
                yf_tk = naver_to_yf.get(nv_cd)
                if yf_tk is None:
                    continue
                parsed = _parse_polling_item(item)
                if parsed:
                    result[yf_tk] = parsed
    except Exception as exc:
        logger.warning("Naver polling index API failed: %s", exc)

    # 폴링 API에서 누락된 지수 → 모바일 API 폴백
    for naver_code, yf_tk in naver_to_yf.items():
        if yf_tk not in result:
            parsed = _fetch_naver_mobile_index(naver_code)
            if parsed:
                result[yf_tk] = parsed

    return result


def fetch_naver_realtime_prices(kr_codes: list) -> dict:
    """
    네이버 금융에서 한국 종목 실시간 시세 일괄 조회.
    1차: 폴링 API (배치) → 2차: 모바일 API (개별, 폴백)

    Args:
        kr_codes: 6자리 종목 코드 리스트 (예: ["005930", "000660"])
    Returns:
        {code: {"price": float, "change": float, "change_pct": float}}
    """
    if not kr_codes:
        return {}

    BATCH = 20
    result = {}
    for i in range(0, len(kr_codes), BATCH):
        batch = kr_codes[i: i + BATCH]
        # ":","," 를 URL-인코딩하지 않도록 raw URL로 직접 구성
        query = ",".join(f"SERVICE_ITEM:{c}" for c in batch)
        try:
            resp = requests.get(
                f"{_NAVER_POLLING_URL}?query={query}",   # params= 미사용 (인코딩 방지)
                headers=_NAVER_POLLING_HDR,
                timeout=8,
            )
            data = resp.json()
            for area in data.get("result", {}).get("areas", []):
                for item in area.get("datas", []):
                    code = item.get("cd", "")
                    if code not in batch:
                        continue
                    parsed = _parse_polling_item(item)
                    if parsed:
                        result[code] = parsed
        except Exception as exc:
            logger.warning("Naver polling stock API failed (batch %d): %s", i, exc)

    # 폴링 API에서 누락된 종목 → 모바일 API 폴백
    for code in kr_codes:
        if code not in result:
            parsed = _fetch_naver_mobile_stock(code)
            if parsed:
                result[code] = parsed

    return result


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
# 감시 종목 배치 사전 다운로드 (속도 개선)
# ---------------------------------------------------------------------------
def prefetch_all_ohlcv(period: str = "1y", interval: str = "1d") -> None:
    """
    config.WATCHLIST 전체 종목을 단일 배치 요청으로 다운로드하여 캐시를 채운다.
    개별 get_ohlcv() 20회 호출 → 1회 배치 요청으로 대체, 속도 대폭 개선.
    이미 캐시가 유효한 종목은 건너뛴다.
    """
    tickers = config.WATCHLIST

    # 캐시 미만료 종목은 스킵
    tickers_to_fetch = [
        t for t in tickers if not _is_cache_valid(f"{t}_{period}_{interval}")
    ]
    if not tickers_to_fetch:
        return

    try:
        raw = yf.download(
            tickers_to_fetch,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
        if raw is None or raw.empty:
            return

        for ticker in tickers_to_fetch:
            cache_key = f"{ticker}_{period}_{interval}"
            try:
                if len(tickers_to_fetch) == 1:
                    df_t = raw.copy()
                else:
                    df_t = raw[ticker].copy()

                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = df_t.columns.get_level_values(0)

                df_t = df_t[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if not df_t.empty and len(df_t) >= config.TECH.ma_long:
                    df_t = _add_indicators(df_t)
                    _set_cache(cache_key, df_t)
            except Exception as exc:
                logger.warning("Batch prefetch failed for %s: %s", ticker, exc)

    except Exception as exc:
        logger.error("Batch OHLCV download failed: %s", exc)


# ---------------------------------------------------------------------------
# 감시 종목 일괄 수집
# ---------------------------------------------------------------------------
def fetch_all_watchlist(period: str = "1y") -> dict[str, pd.DataFrame]:
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
