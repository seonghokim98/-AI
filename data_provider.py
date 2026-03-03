"""
data_provider.py - 시장 데이터 수집 모듈
네이버 증권 모바일 API 기반 (실시간, 지연 없음)
yfinance는 네이버 미지원 데이터의 폴백으로만 사용
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

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
# 네이버 모바일 API — OHLCV 헬퍼 (실시간, 지연 없음)
# ---------------------------------------------------------------------------

def _period_to_count(period: str, default: int = 260) -> int:
    """period 문자열 → 필요 봉 수 변환 (거래일 기준)"""
    return {"1mo": 33, "3mo": 70, "6mo": 135, "1y": 265, "2y": 530}.get(period, default)


def _parse_naver_rows(values: list, price_key: str = "closePrice") -> list:
    """
    네이버 API tradingValues 배열 → (date, open, high, low, close, volume) 파싱.
    종목은 closePrice, 지수는 closeIndex 사용.
    """
    rows = []
    open_key  = "openIndex"  if "Index" in price_key else "openPrice"
    high_key  = "highIndex"  if "Index" in price_key else "highPrice"
    low_key   = "lowIndex"   if "Index" in price_key else "lowPrice"

    for item in values:
        try:
            date_str = str(item.get("localDate", ""))
            if len(date_str) != 8:
                continue
            dt = datetime.strptime(date_str, "%Y%m%d")

            def _fv(*keys):
                for k in keys:
                    v = item.get(k)
                    if v is not None:
                        try:
                            return float(str(v).replace(",", ""))
                        except (ValueError, TypeError):
                            pass
                return 0.0

            close_p = _fv(price_key, "closePrice", "closeIndex")
            open_p  = _fv(open_key,  "openPrice",  "openIndex")
            high_p  = _fv(high_key,  "highPrice",  "highIndex")
            low_p   = _fv(low_key,   "lowPrice",   "lowIndex")
            vol     = _fv("accumulatedTradingVolume", "tradeVolume")

            if close_p > 0:
                rows.append((dt, open_p or close_p, high_p or close_p,
                              low_p or close_p, close_p, vol))
        except (ValueError, TypeError):
            continue
    return rows


def _rows_to_df(rows: list) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


def _fetch_naver_stock_ohlcv(code: str, count: int = 265) -> Optional[pd.DataFrame]:
    """네이버 모바일 API — 한국 종목 일봉 OHLCV 반환"""
    url = f"{_NAVER_MOBILE_BASE}/stock/{code}/sise/day"
    try:
        resp = requests.get(
            url,
            params={"timeframe": "day", "count": count, "requestType": 0},
            headers=_NAVER_MOBILE_HDR,
            timeout=12,
        )
        data = resp.json()
        values = data.get("tradingValues") or data.get("priceValues") or []
        rows = _parse_naver_rows(values, price_key="closePrice")
        df = _rows_to_df(rows)
        if df is not None:
            logger.debug("Naver stock OHLCV OK [%s]: %d bars", code, len(df))
        return df
    except Exception as exc:
        logger.warning("Naver stock OHLCV failed [%s]: %s", code, exc)
        return None


def _fetch_naver_index_ohlcv(naver_code: str, count: int = 70) -> Optional[pd.DataFrame]:
    """네이버 모바일 API — 지수 일봉 OHLCV (KOSPI, KOSDAQ, VIX 등)"""
    url = f"{_NAVER_MOBILE_BASE}/index/{naver_code}/sise/day"
    try:
        resp = requests.get(
            url,
            params={"timeframe": "day", "count": count},
            headers=_NAVER_MOBILE_HDR,
            timeout=10,
        )
        data = resp.json()
        values = data.get("tradingValues") or []
        rows = _parse_naver_rows(values, price_key="closeIndex")
        df = _rows_to_df(rows)
        if df is not None:
            logger.debug("Naver index OHLCV OK [%s]: %d bars", naver_code, len(df))
        return df
    except Exception as exc:
        logger.warning("Naver index OHLCV failed [%s]: %s", naver_code, exc)
        return None


# ---------------------------------------------------------------------------
# OHLCV 데이터 수집 (네이버 우선 → yfinance 폴백)
# ---------------------------------------------------------------------------
def get_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    종목 OHLCV 데이터 반환.
    1차: 네이버 모바일 API (실시간, 지연 없음)
    2차: yfinance 폴백 (네이버 실패 시)

    Args:
        ticker: '005930.KS', 'AAPL' 등
        period: '1mo', '3mo', '6mo', '1y'
        interval: '1d' (일봉만 네이버 지원)
    """
    cache_key = f"{ticker}_{period}_{interval}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    df = None

    # ── 1차: 네이버 모바일 API (한국 종목 + 일봉) ───────────────────────
    if (".KS" in ticker or ".KQ" in ticker) and interval == "1d":
        code  = ticker.split(".")[0]
        count = _period_to_count(period)
        df    = _fetch_naver_stock_ohlcv(code, count=count)
        if df is not None:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            if len(df) >= config.TECH.ma_long:
                df = _add_indicators(df)
                _set_cache(cache_key, df)
                return df
        df = None  # 데이터 부족 → yfinance 폴백

    # ── 2차: yfinance 폴백 ─────────────────────────────────────────────
    if not _YF_OK:
        logger.error("yfinance not available and Naver failed for %s", ticker)
        return None
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
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
# 지수 데이터 (네이버 우선 → yfinance 폴백)
# ---------------------------------------------------------------------------
def get_index_data(ticker: str = "^KS11", period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    코스피(^KS11), 코스닥(^KQ11), VIX(^VIX) 등 지수 데이터 반환.
    1차: 네이버 모바일 API (실시간)
    2차: yfinance 폴백
    """
    cache_key = f"index_{ticker}_{period}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    # ── 1차: 네이버 모바일 API ──────────────────────────────────────────
    naver_code = _YF_TO_NAVER_INDEX.get(ticker)
    if naver_code:
        count = _period_to_count(period, default=70)
        df = _fetch_naver_index_ohlcv(naver_code, count=count)
        if df is not None and not df.empty:
            _set_cache(cache_key, df)
            return df

    # ── 2차: yfinance 폴백 ─────────────────────────────────────────────
    if not _YF_OK:
        return None
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return None

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
def _fetch_naver_stock_basic(code: str) -> dict:
    """
    네이버 모바일 API — 종목 기본정보 (시가총액, 52주 고저, 배당수익률 등).
    PER/PBR/EPS는 _fetch_naver_fundamentals() 에서 가져오므로 여기선 보완 데이터만.
    """
    url = f"{_NAVER_MOBILE_BASE}/stock/{code}/basic"
    try:
        resp = requests.get(url, headers=_NAVER_MOBILE_HDR, timeout=8)
        data = resp.json()

        def _fv(k):
            v = data.get(k)
            if v is None:
                return None
            try:
                return float(str(v).replace(",", "").replace("%", ""))
            except (ValueError, TypeError):
                return None

        return {
            "market_cap":        _fv("marketValue"),
            "dividend_yield":    _fv("dividendRate"),
            "52w_high":          _fv("fiftyTwoWeekHigh"),
            "52w_low":           _fv("fiftyTwoWeekLow"),
            "shares_outstanding": _fv("totalIssueAmount"),
        }
    except Exception as exc:
        logger.debug("Naver stock basic failed for %s: %s", code, exc)
        return {}


@lru_cache(maxsize=64)
def get_fundamentals(ticker: str) -> dict:
    """
    PER, PBR, EPS, 배당수익률 등 반환.
    한국 종목: 네이버 금융 스크래핑 + 모바일 API 우선, yfinance 폴백.
    해외 종목: yfinance 사용.

    Returns:
        {'per': float, 'pbr': float, 'eps': float, 'market_cap': int, ...}
    """
    # ── 한국 종목 → 네이버 우선 ───────────────────────────────────────────
    if ".KS" in ticker or ".KQ" in ticker:
        code  = ticker.split(".")[0]
        naver = _fetch_naver_fundamentals(ticker)   # PER / PBR / EPS

        # 시가총액 등 보완 데이터는 네이버 모바일 API에서
        basic = _fetch_naver_stock_basic(code)
        for k, v in basic.items():
            if v is not None:
                naver.setdefault(k, v)

        if naver.get("per") or naver.get("pbr") or naver.get("eps"):
            naver.setdefault("sector", config.TICKER_INDUSTRY.get(ticker, ""))
            return naver

    # ── yfinance 폴백 ──────────────────────────────────────────────────────
    if not _YF_OK:
        return {}
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
# 현재가 조회 (네이버 실시간 → yfinance 폴백)
# ---------------------------------------------------------------------------
def get_current_price(ticker: str) -> Optional[float]:
    """
    현재가 조회.
    1차: 네이버 폴링 API (실시간, 지연 없음)
    2차: yfinance 폴백
    """
    # 한국 종목 → 네이버 실시간
    if ".KS" in ticker or ".KQ" in ticker:
        code = ticker.split(".")[0]
        try:
            prices = fetch_naver_realtime_prices([code])
            if code in prices:
                return float(prices[code]["price"])
        except Exception as exc:
            logger.debug("Naver price failed for %s: %s", ticker, exc)

    # yfinance 폴백
    if not _YF_OK:
        return None
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
# 외국인 수급 데이터 — 네이버 금융 스크래핑
# ---------------------------------------------------------------------------
_FOREIGN_CACHE: dict = {}
_FOREIGN_CACHE_TTL: int = 600  # 10분 (외인 데이터는 변동 느림)


def _parse_frgn_num(td) -> Optional[float]:
    """외국인 순매수량 td → float (+ = 매수, - = 매도, 0 = 보합)"""
    text = td.get_text(strip=True).replace(",", "").replace("+", "")
    if not text or text == "-":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _parse_frgn_ratio(td) -> Optional[float]:
    """외국인 지분율 td → float (유효 범위 0~100)"""
    text = td.get_text(strip=True).replace(",", "").replace("%", "")
    if not text or text == "-":
        return None
    try:
        v = float(text)
        return v if 0 < v < 100 else None
    except ValueError:
        return None


def get_foreign_flow(ticker: str, days: int = 10) -> dict:
    """
    [알고리즘] Check_Foreigner_Flow 구현 — 외국인 순매수 + 지분율 추세 조회.

    네이버 금융 외국인 현황 페이지(frgn.naver)에서 스크래핑.
    한국 종목(.KS/.KQ)만 지원. ETF·해외 종목은 UNKNOWN 반환.

    Returns:
        {
            'net_buy_5d':       float,  # 5일 누적 순매수량 (+ = 매수, - = 매도)
            'ownership_trend':  str,    # 'UP' | 'DOWN' | 'FLAT'
            'ownership_ratio':  float,  # 최신 지분율 (%)
            'signal':           str,    # 'STRONG_BUY_SIGNAL' | 'WEAK_SIGNAL' | 'UNKNOWN'
        }
    """
    # ETF 및 해외 종목은 외인 수급 분석 제외
    if ".KS" not in ticker and ".KQ" not in ticker:
        return {"signal": "UNKNOWN"}
    if config.TICKER_MARKET.get(ticker, "") == "ETF":
        return {"signal": "UNKNOWN"}
    if not _BS4_OK:
        return {"signal": "UNKNOWN"}

    code = ticker.split(".")[0]
    cache_key = f"frgn_{code}"

    # 캐시 확인
    if cache_key in _FOREIGN_CACHE:
        cached_at, cached_data = _FOREIGN_CACHE[cache_key]
        if (datetime.now() - cached_at).seconds < _FOREIGN_CACHE_TTL:
            return cached_data

    try:
        import re
        date_re = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")

        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        resp = requests.get(url, headers=_NAVER_FIN_HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = _BS(resp.text, "html.parser")

        table = soup.select_one("table.type2")
        if table is None:
            logger.warning("Foreign flow table not found for %s", ticker)
            return {"signal": "UNKNOWN"}

        net_buys: list = []
        ownership_ratios: list = []

        for row in table.select("tr"):
            tds = row.select("td")
            if len(tds) < 5:
                continue

            date_text = tds[0].get_text(strip=True)
            if not date_re.match(date_text):
                continue

            # 외국인 순매수량: td[3] 우선, 실패 시 td[2] 시도
            nb = _parse_frgn_num(tds[3])
            if nb is None:
                nb = _parse_frgn_num(tds[2])
            net_buys.append(nb if nb is not None else 0.0)

            # 외국인 지분율: 뒤에서부터 0~100 범위 값 탐색
            ratio = None
            for idx in (5, 6, -1, -2, -3):
                try:
                    v = _parse_frgn_ratio(tds[idx])
                    if v is not None:
                        ratio = v
                        break
                except IndexError:
                    continue
            if ratio is not None:
                ownership_ratios.append(ratio)

            if len(net_buys) >= days:
                break

        if not net_buys:
            logger.warning("No foreign flow data parsed for %s", ticker)
            return {"signal": "UNKNOWN"}

        # [알고리즘] Foreign_Net_Buy_5D — 최근 5일 누적 순매수
        net_buy_5d = sum(net_buys[:5])

        # [알고리즘] Foreign_Ownership_Trend — 10일 지분율 추세
        ownership_trend = "FLAT"
        if len(ownership_ratios) >= 6:
            recent = sum(ownership_ratios[:5]) / 5
            older_slice = ownership_ratios[5:min(10, len(ownership_ratios))]
            older = sum(older_slice) / len(older_slice)
            if recent > older + 0.1:
                ownership_trend = "UP"
            elif recent < older - 0.1:
                ownership_trend = "DOWN"
        elif len(ownership_ratios) >= 2:
            if ownership_ratios[0] > ownership_ratios[-1] + 0.1:
                ownership_trend = "UP"
            elif ownership_ratios[0] < ownership_ratios[-1] - 0.1:
                ownership_trend = "DOWN"

        # [알고리즘] Check_Foreigner_Flow 신호 결정
        if net_buy_5d > 0 and ownership_trend == "UP":
            signal = "STRONG_BUY_SIGNAL"
        else:
            signal = "WEAK_SIGNAL"

        result = {
            "net_buy_5d":      net_buy_5d,
            "ownership_trend": ownership_trend,
            "ownership_ratio": ownership_ratios[0] if ownership_ratios else None,
            "signal":          signal,
        }
        _FOREIGN_CACHE[cache_key] = (datetime.now(), result)
        logger.info(
            "Foreign flow [%s]: 5D_NB=%+.0f trend=%s → %s",
            ticker, net_buy_5d, ownership_trend, signal,
        )
        return result

    except Exception as exc:
        logger.warning("Foreign flow fetch failed for %s: %s", ticker, exc)
        return {"signal": "UNKNOWN"}


# ---------------------------------------------------------------------------
# 감시 종목 사전 다운로드 (네이버 병렬 요청 → yfinance 배치 폴백)
# ---------------------------------------------------------------------------
def prefetch_all_ohlcv(period: str = "1y", interval: str = "1d") -> None:
    """
    config.WATCHLIST 전체 종목 OHLCV를 병렬 네이버 요청으로 캐시에 채운다.
    이미 캐시가 유효한 종목은 건너뛴다.
    """
    tickers_to_fetch = [
        t for t in config.WATCHLIST
        if not _is_cache_valid(f"{t}_{period}_{interval}")
    ]
    if not tickers_to_fetch:
        return

    def _fetch_one(ticker: str) -> None:
        try:
            get_ohlcv(ticker, period=period, interval=interval)
        except Exception as exc:
            logger.warning("prefetch failed for %s: %s", ticker, exc)

    # 최대 8개 동시 요청 (네이버 서버 부하 방지)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers_to_fetch}
        for fut in as_completed(futures):
            pass  # 에러는 _fetch_one 내부에서 로깅

    logger.info("prefetch_all_ohlcv: %d tickers fetched", len(tickers_to_fetch))


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
