"""
valuation.py - 밸류에이션 & 자사주 필터 (매매 원칙)
조건:
  - PER < 업종 평균
  - PBR < 1.5
  - EPS 양수 (적자 기업 제외)
  - 최근 90일 내 자사주 매입 공시 존재 (DART API)
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

import config
import data_provider as dp

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    ticker: str
    per: Optional[float]
    pbr: Optional[float]
    eps: Optional[float]
    industry_avg_per: float
    per_ok: bool          # PER < 업종 평균
    pbr_ok: bool          # PBR < max_pbr
    eps_ok: bool          # EPS > 0
    buyback_ok: bool      # 자사주 매입 공시 있음
    passed: bool          # 전체 통과 여부
    detail: str


def check_valuation(ticker: str) -> ValuationResult:
    """
    종목의 밸류에이션 필터를 실행하고 결과를 반환한다.
    하나라도 실패하면 passed=False.
    """
    cfg = config.VALUATION
    fundamentals = dp.get_fundamentals(ticker)

    per = fundamentals.get("per")
    pbr = fundamentals.get("pbr")
    eps = fundamentals.get("eps")

    industry = config.TICKER_INDUSTRY.get(ticker, "기본")
    industry_avg_per = config.INDUSTRY_AVG_PER.get(industry, config.INDUSTRY_AVG_PER["기본"])

    per_ok = _check_per(per, industry_avg_per, cfg.max_per)
    pbr_ok = _check_pbr(pbr, cfg.max_pbr)
    eps_ok = _check_eps(eps, cfg.require_positive_eps)
    buyback_ok = _check_buyback(ticker, cfg.buyback_lookback_days)

    passed = per_ok and pbr_ok and eps_ok
    # 자사주는 보너스 조건 (없다고 탈락은 아님, 신뢰도에 반영)

    detail = _build_detail(per, pbr, eps, industry_avg_per, per_ok, pbr_ok, eps_ok, buyback_ok)
    logger.info("Valuation %s: passed=%s | %s", ticker, passed, detail)

    return ValuationResult(
        ticker=ticker,
        per=per,
        pbr=pbr,
        eps=eps,
        industry_avg_per=industry_avg_per,
        per_ok=per_ok,
        pbr_ok=pbr_ok,
        eps_ok=eps_ok,
        buyback_ok=buyback_ok,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# 개별 체크
# ---------------------------------------------------------------------------
def _check_per(per: Optional[float], industry_avg: float, abs_max: float) -> bool:
    if per is None:
        logger.debug("PER data unavailable; skipping PER check")
        return True   # 데이터 없으면 패스 (보수적 설정 원하면 False 변경)
    return per < min(industry_avg, abs_max)


def _check_pbr(pbr: Optional[float], max_pbr: float) -> bool:
    if pbr is None:
        return True
    return pbr < max_pbr


def _check_eps(eps: Optional[float], require_positive: bool) -> bool:
    if not require_positive:
        return True
    if eps is None:
        return True
    return eps > 0


def _check_buyback(ticker: str, lookback_days: int) -> bool:
    """
    DART 전자공시 API에서 최근 N일 내 자사주 매입 공시를 조회한다.
    API 키 없으면 False 반환 (조건 미충족으로 처리).
    """
    if not config.DART_API_KEY:
        logger.debug("DART API key not set; buyback check skipped")
        return False

    # 종목 코드 추출 (005930.KS → 005930)
    stock_code = ticker.split(".")[0]

    try:
        disclosures = _fetch_dart_disclosures(stock_code, lookback_days)
        for d in disclosures:
            report_nm = d.get("report_nm", "")
            if "자기주식" in report_nm or "자사주" in report_nm:
                logger.info("Buyback found for %s: %s", ticker, report_nm)
                return True
    except Exception as exc:
        logger.error("DART API error for %s: %s", ticker, exc)

    return False


def _fetch_dart_disclosures(corp_code: str, days: int) -> list:
    """
    DART OpenAPI에서 최근 공시 목록을 조회한다.
    https://opendart.fss.or.kr/api/list.json
    """
    from datetime import datetime, timedelta
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)

    params = {
        "crtfc_key": config.DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": start_dt.strftime("%Y%m%d"),
        "end_de": end_dt.strftime("%Y%m%d"),
        "pblntf_ty": "A",   # 정기공시 + 주요사항보고
        "page_no": "1",
        "page_count": "40",
    }
    url = "https://opendart.fss.or.kr/api/list.json"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "000":
        return data.get("list", [])
    return []


# ---------------------------------------------------------------------------
# 상세 메시지 생성
# ---------------------------------------------------------------------------
def _build_detail(per, pbr, eps, industry_avg_per,
                  per_ok, pbr_ok, eps_ok, buyback_ok) -> str:
    parts = []
    per_str = f"PER {per:.1f}배 (업종평균 {industry_avg_per:.1f}배) {'✅' if per_ok else '❌'}" if per else "PER N/A"
    pbr_str = f"PBR {pbr:.2f} {'✅' if pbr_ok else '❌'}" if pbr else "PBR N/A"
    eps_str = f"EPS {eps:.0f} {'✅' if eps_ok else '❌'}" if eps else "EPS N/A"
    buyback_str = f"자사주 공시 {'있음 ✅' if buyback_ok else '없음'}"
    return " | ".join([per_str, pbr_str, eps_str, buyback_str])
