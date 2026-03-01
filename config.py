"""
config.py - 시스템 전체 설정 파일
API 키, 예산, 감시 종목 등 환경변수 기반 설정 관리
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 슬랙
# ---------------------------------------------------------------------------
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#trading-signals")

# ---------------------------------------------------------------------------
# 한국투자증권 REST API (Open API)
# 실제 사용 시 환경변수로 주입
# ---------------------------------------------------------------------------
KIS_APP_KEY: str = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET: str = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO: str = os.getenv("KIS_ACCOUNT_NO", "")      # 계좌번호
KIS_IS_REAL: bool = os.getenv("KIS_IS_REAL", "false").lower() == "true"

# ---------------------------------------------------------------------------
# DART (전자공시) API
# https://opendart.fss.or.kr 에서 발급
# ---------------------------------------------------------------------------
DART_API_KEY: str = os.getenv("DART_API_KEY", "")

# ---------------------------------------------------------------------------
# 예산 설정
# ---------------------------------------------------------------------------
TOTAL_BUDGET: int = int(os.getenv("TOTAL_BUDGET", "100000"))   # 원 단위, 기본 10만원
MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))       # 동시 보유 최대 종목 수
BUDGET_PER_TRADE: int = TOTAL_BUDGET // MAX_POSITIONS           # 종목당 투입 예산

# ---------------------------------------------------------------------------
# 리스크 파라미터 (매매 원칙)
# ---------------------------------------------------------------------------
STOP_LOSS_RATE: float = -0.10          # -10% 손절 원칙 (절대 원칙)
MIN_REWARD_RISK_RATIO: float = 2.0     # 최소 손익비 2:1
MAX_DRAWDOWN_HALT: float = -0.20       # 계좌 -20% 시 신호 중단

# ---------------------------------------------------------------------------
# 기술적 분석 파라미터
# ---------------------------------------------------------------------------
@dataclass
class TechnicalConfig:
    # 이동평균선
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60

    # 골든크로스 판단 기간
    golden_cross_lookback: int = 2

    # 쌍바닥 판단
    double_bottom_window: int = 30      # 최근 N봉 내 탐색
    double_bottom_tolerance: float = 0.03  # 두 저점 가격 허용 오차 3%

    # 돌파 + 눌림목
    breakout_window: int = 20           # 저항선 계산 기간
    pullback_tolerance: float = 0.03    # 저항선 대비 허용 이탈 3%
    volume_surge_ratio: float = 1.5     # 돌파 시 평균 거래량 대비 1.5배 이상

    # 깃발형 돌파
    flag_pole_min_ratio: float = 0.05   # 깃대 최소 상승률 5%
    flag_consolidation_bars: int = 5    # 깃발 수렴 구간 최소 봉 수
    flag_max_retracement: float = 0.50  # 깃대 대비 최대 되돌림 50%

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 40.0          # 눌림목 확인 시 RSI 기준 (과도한 매도세 제외)

    # 볼린저 밴드
    bb_period: int = 20
    bb_std: float = 2.0

    # 골든크로스 이평선 (이미지 기준: 50선 / 200선 교차)
    ma_gc_fast: int = 50    # 단기선
    ma_gc_slow: int = 200   # 장기선

    # 하락쐐기형 탐색 구간
    wedge_window: int = 25


TECH = TechnicalConfig()

# ---------------------------------------------------------------------------
# 밸류에이션 필터 (매매 원칙)
# ---------------------------------------------------------------------------
@dataclass
class ValuationConfig:
    max_per: float = 30.0               # 업종 평균 대비가 우선이나, 절대 상한선
    max_pbr: float = 1.5
    require_positive_eps: bool = True   # 적자 기업 제외
    buyback_lookback_days: int = 90     # 최근 N일 내 자사주 공시 확인


VALUATION = ValuationConfig()

# ---------------------------------------------------------------------------
# 시장 필터 (매매 원칙: 시장 나쁘면 무조건 관망)
# ---------------------------------------------------------------------------
@dataclass
class MarketFilterConfig:
    kospi_ma_period: int = 20           # 코스피 20일선 기준
    kosdaq_ma_period: int = 20
    vix_threshold: float = 30.0         # VIX 30 초과 시 관망
    # 외국인 순매수 전환 판단: 최근 N일 연속 순매수
    foreign_buy_streak: int = 3


MARKET = MarketFilterConfig()

# ---------------------------------------------------------------------------
# 스케줄러
# ---------------------------------------------------------------------------
SCAN_INTERVAL_MINUTES: int = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 5            # 개장 5분 후부터 감시 시작
MARKET_CLOSE_HOUR: int = 15
MARKET_CLOSE_MINUTE: int = 20          # 장 마감 10분 전 종료

# ---------------------------------------------------------------------------
# 감시 종목 리스트 (yfinance 티커, 한국 종목은 .KS/.KQ 접미사)
# ---------------------------------------------------------------------------
WATCHLIST: List[str] = [
    # KOSPI 대형주
    "005930.KS",   # 삼성전자
    "000660.KS",   # SK하이닉스
    "005380.KS",   # 현대차
    "035420.KS",   # NAVER
    "051910.KS",   # LG화학
    "006400.KS",   # 삼성SDI
    "035720.KS",   # 카카오
    "207940.KS",   # 삼성바이오로직스
    "068270.KS",   # 셀트리온
    "028260.KS",   # 삼성물산
    # KOSDAQ 중형주
    "247540.KQ",   # 에코프로비엠
    "086520.KQ",   # 에코프로
    "091990.KQ",   # 셀트리온헬스케어
    "196170.KQ",   # 알테오젠
    "039030.KQ",   # 이오테크닉스
]

# 업종 평균 PER (수동 관리, 실제 운용 시 DART/FnGuide 연동 권장)
INDUSTRY_AVG_PER: dict = {
    "반도체": 18.0,
    "자동차": 10.0,
    "인터넷": 25.0,
    "화학": 12.0,
    "바이오": 35.0,
    "배터리": 22.0,
    "기본": 15.0,   # 업종 미분류 기본값
}

# 종목 → 업종 매핑 (확장 가능)
TICKER_INDUSTRY: dict = {
    "005930.KS": "반도체",
    "000660.KS": "반도체",
    "005380.KS": "자동차",
    "035420.KS": "인터넷",
    "051910.KS": "화학",
    "006400.KS": "배터리",
    "035720.KS": "인터넷",
    "207940.KS": "바이오",
    "068270.KS": "바이오",
    "028260.KS": "기본",
    "247540.KQ": "배터리",
    "086520.KQ": "배터리",
    "091990.KQ": "바이오",
    "196170.KQ": "바이오",
    "039030.KQ": "반도체",
}

# 종목 한글명 매핑
TICKER_NAME: dict = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "005380.KS": "현대차",
    "035420.KS": "NAVER",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035720.KS": "카카오",
    "207940.KS": "삼성바이오로직스",
    "068270.KS": "셀트리온",
    "028260.KS": "삼성물산",
    "247540.KQ": "에코프로비엠",
    "086520.KQ": "에코프로",
    "091990.KQ": "셀트리온헬스케어",
    "196170.KQ": "알테오젠",
    "039030.KQ": "이오테크닉스",
}
