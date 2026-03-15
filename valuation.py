"""
valuation.py - 밸류에이션 & 모멘텀 필터 (매매 원칙)

[알고리즘 정의서] 종목 선정 필터링 (기본적 분석: What to buy)
  - 가치 평가지표: PBR < 1.0 AND PBR > 0 (순자산 이하 저평가)
                  PER < 15  AND PER > 0  (절대 저평가 기준)
  - 모멘텀 지표:  최근 90일 내 '자사주 소각/매입', '배당/고배당', '밸류업 프로그램'
                  공시 존재 시 가중치 부여 → 관심 종목 풀 우선 편입
  - 원칙:        뉴스/펀더멘탈은 '관심 종목 풀(Pool)' 구성 용도만.
                  좋은 뉴스에 시장가로 따라붙는 로직은 절대 금지.
"""
import logging
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
    per_ok: bool          # PER: 0 < PER < 15
    pbr_ok: bool          # PBR: 0 < PBR < 1.0
    eps_ok: bool          # EPS > 0 (적자 기업 제외)
    buyback_ok: bool      # 모멘텀 키워드 공시 있음 (가중치 부여)
    passed: bool          # 관심 종목 풀 편입 여부 (per_ok AND pbr_ok AND eps_ok)
    in_pool: bool         # passed와 동일 (명시적 가독성용)
    momentum_bonus: bool  # 자사주/배당/밸류업 공시 여부 (보너스 가중치)
    detail: str


def check_valuation(ticker: str) -> ValuationResult:
    """
    [알고리즘] Filter_Watchlist 구현:
      Condition_1 = PBR < 1.0 AND PBR > 0
      Condition_2 = PER < 15  AND PER > 0
      Condition_3 = 모멘텀 공시 확인 (자사주 소각/매입, 배당, 밸류업)

      IF Condition_1 AND Condition_2: 관심 종목 풀 편입
      IF Condition_3: 우선순위 가중치 부여
    """
    cfg = config.VALUATION
    fundamentals = dp.get_fundamentals(ticker)

    per = fundamentals.get("per")
    pbr = fundamentals.get("pbr")
    eps = fundamentals.get("eps")

    per_ok = _check_per(per, cfg.max_per)
    pbr_ok = _check_pbr(pbr, cfg.max_pbr)
    eps_ok = _check_eps(eps, cfg.require_positive_eps)
    buyback_ok = _check_momentum(ticker, cfg.buyback_lookback_days, cfg.momentum_keywords)

    # 관심 종목 풀 편입 기준: PBR + PER + EPS 동시 충족
    passed = per_ok and pbr_ok and eps_ok

    detail = _build_detail(per, pbr, eps, per_ok, pbr_ok, eps_ok, buyback_ok)
    logger.info("Valuation %s: pool=%s momentum=%s | %s", ticker, passed, buyback_ok, detail)

    return ValuationResult(
        ticker=ticker,
        per=per,
        pbr=pbr,
        eps=eps,
        per_ok=per_ok,
        pbr_ok=pbr_ok,
        eps_ok=eps_ok,
        buyback_ok=buyback_ok,
        passed=passed,
        in_pool=passed,
        momentum_bonus=buyback_ok,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# 개별 체크 — [알고리즘] 조건 구현
# ---------------------------------------------------------------------------

def _check_per(per: Optional[float], max_per: float) -> bool:
    """
    [알고리즘] Condition_2: PER < 15 AND PER > 0
    - 음수 PER (적자): 제외
    - PER = 0: 데이터 오류로 제외
    - PER > 15: 고평가로 제외
    """
    if per is None:
        return True   # 데이터 없으면 통과 (보수적 운용 원하면 False)
    if per <= 0:
        return False  # 음수/0: 적자 또는 오류
    return per < max_per


def _check_pbr(pbr: Optional[float], max_pbr: float) -> bool:
    """
    [알고리즘] Condition_1: PBR < 1.0 AND PBR > 0
    - PBR < 1.0: 시장가 < 순자산가치 → 저평가 우량주
    - PBR <= 0: 데이터 오류로 제외
    """
    if pbr is None:
        return True
    if pbr <= 0:
        return False
    return pbr < max_pbr


def _check_eps(eps: Optional[float], require_positive: bool) -> bool:
    """EPS > 0: 적자 기업 완전 제외"""
    if not require_positive:
        return True
    if eps is None:
        return True
    return eps > 0


def _check_momentum(ticker: str, lookback_days: int, keywords: tuple) -> bool:
    """
    [알고리즘] Condition_3: 모멘텀 공시 확인
    DART 전자공시 API에서 최근 N일 내 키워드 공시 조회.
    - 자사주 소각/매입, 배당/고배당, 밸류업 프로그램
    - API 키 없으면 False (보너스 없음, 풀 편입은 PBR/PER 기준으로 가능)
    """
    if not config.DART_API_KEY:
        logger.debug("DART API key not set; momentum check skipped")
        return False

    stock_code = ticker.split(".")[0]
    try:
        disclosures = _fetch_dart_disclosures(stock_code, lookback_days)
        for d in disclosures:
            report_nm = d.get("report_nm", "")
            for kw in keywords:
                if kw in report_nm:
                    logger.info("Momentum '%s' found for %s: %s", kw, ticker, report_nm)
                    return True
    except Exception as exc:
        logger.error("DART API error for %s: %s", ticker, exc)

    return False


def _fetch_dart_disclosures(corp_code: str, days: int) -> list:
    """DART OpenAPI 공시 목록 조회"""
    from datetime import datetime, timedelta
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)

    params = {
        "crtfc_key": config.DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": start_dt.strftime("%Y%m%d"),
        "end_de": end_dt.strftime("%Y%m%d"),
        "pblntf_ty": "A",
        "page_no": "1",
        "page_count": "40",
    }
    resp = requests.get("https://opendart.fss.or.kr/api/list.json",
                        params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", []) if data.get("status") == "000" else []


# ---------------------------------------------------------------------------
# 상세 메시지 생성
# ---------------------------------------------------------------------------

def _build_detail(per, pbr, eps, per_ok, pbr_ok, eps_ok, momentum_ok) -> str:
    per_str = (f"PER {per:.1f} {'✅ <15' if per_ok else '❌ 고평가'}"
               if per and per > 0 else "PER N/A")
    pbr_str = (f"PBR {pbr:.2f} {'✅ <1.0' if pbr_ok else '❌ 고평가'}"
               if pbr and pbr > 0 else "PBR N/A")
    eps_str = (f"EPS {eps:.0f} {'✅' if eps_ok else '❌ 적자'}"
               if eps else "EPS N/A")
    momentum_str = f"모멘텀 {'있음 ✅' if momentum_ok else '없음'}"
    return " | ".join([per_str, pbr_str, eps_str, momentum_str])
