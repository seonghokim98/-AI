"""
main.py - 메인 오케스트레이터 & 스케줄러
매매 원칙을 100% 강제하는 파이프라인:

  [실시간 데이터 수집]
       ↓
  [시장 트렌드 필터]   ← 약세장이면 무조건 관망
       ↓
  [기술적 패턴 엔진]   ← 허용 패턴 4종만
       ↓
  [밸류에이션 필터]    ← PER/PBR/EPS + 자사주
       ↓
  [리스크 계산]        ← -10% 손절, 손익비 2:1 이상
       ↓
  [슬랙 알림 발송]     ← 자동매매 없음, 알림만
"""
import logging
import sys
import time
from datetime import datetime, time as dtime

import schedule

import config
import data_provider as dp
import market_filter as mf
import pattern_engine as pe
import risk as rm
import slack_bot as sb
import valuation as vl

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_signal.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# 상태 추적
# ---------------------------------------------------------------------------
class SessionState:
    """현재 세션의 실행 상태를 추적한다."""
    def __init__(self):
        self.last_market_regime = None
        self.daily_signals: list = []
        self.scan_count: int = 0
        self.start_time = datetime.now()

    def reset_daily(self):
        self.daily_signals = []
        self.scan_count = 0


_state = SessionState()


# ---------------------------------------------------------------------------
# 핵심 파이프라인
# ---------------------------------------------------------------------------
def scan_market() -> None:
    """
    전체 매매 원칙 파이프라인 실행.
    5분마다 스케줄러에 의해 호출된다.
    """
    if not _is_market_hours():
        return

    _state.scan_count += 1
    logger.info("=== 스캔 #%d 시작 ===", _state.scan_count)

    # ── Step 1: 시장 트렌드 필터 ─────────────────────────────────────────
    try:
        market = mf.analyze_market()
    except Exception as exc:
        logger.error("Market filter failed: %s", exc)
        sb.send_error("market_filter.analyze_market", exc)
        return

    # 시장 상태가 바뀌었을 때만 알림 발송 (과도한 알림 방지)
    if market.regime != _state.last_market_regime:
        sb.send_market_alert(market)
        _state.last_market_regime = market.regime

    if not mf.is_market_ok(market):
        logger.info("Market BEAR — all signals blocked. Waiting.")
        return   # 약세장: 아무것도 하지 않는다

    # ── Step 2: 감시 종목 데이터 수집 ────────────────────────────────────
    logger.info("Fetching watchlist data (%d tickers)...", len(config.WATCHLIST))
    all_data = dp.fetch_all_watchlist()

    if not all_data:
        logger.warning("No valid data fetched for watchlist")
        return

    # ── Step 3~5: 종목별 파이프라인 ──────────────────────────────────────
    for ticker, df in all_data.items():
        try:
            _process_ticker(ticker, df, market)
        except Exception as exc:
            logger.error("Error processing %s: %s", ticker, exc)
            sb.send_error(f"process_ticker({ticker})", exc)

    logger.info("=== 스캔 #%d 완료 (%d종목) ===", _state.scan_count, len(all_data))


def _process_ticker(ticker: str, df, market: mf.MarketStatus) -> None:
    """단일 종목 파이프라인 처리 (매수 + 매도 패턴 통합)"""
    name = config.TICKER_NAME.get(ticker, ticker)

    # ── Step 3a: 매도/경고 패턴 탐지 (팔아라 경고 — 이미지 기준) ─────────
    sell_pattern = pe.detect_sell_patterns(df)
    if sell_pattern is not None and sell_pattern.confidence >= 0.45:
        logger.info("%s: Sell pattern=%s confidence=%.2f action=%s",
                    name, sell_pattern.pattern.value,
                    sell_pattern.confidence, sell_pattern.action)
        sb.send_sell_signal(ticker, sell_pattern, market)
        _state.daily_signals.append({
            "ticker": ticker,
            "pattern": sell_pattern.pattern.value,
            "signal_type": sell_pattern.signal_type,
            "entry": sell_pattern.entry_price,
            "action": sell_pattern.action,
            "time": datetime.now().isoformat(),
        })

    # ── Step 3b: 매수 패턴 탐지 (사라 경고 — 이미지 기준) ───────────────
    pattern = pe.detect_patterns(df)

    if pattern is None:
        logger.debug("%s: No buy pattern detected", name)
        return   # 패턴 없음 → 관망

    logger.info("%s: Buy pattern=%s confidence=%.2f",
                name, pattern.pattern.value, pattern.confidence)

    # 신뢰도 임계값 (60% 미만은 무시)
    if pattern.confidence < 0.60:
        logger.debug("%s: Pattern confidence too low (%.2f)", name, pattern.confidence)
        return

    # ── Step 4: 밸류에이션 필터 ──────────────────────────────────────────
    valuation = vl.check_valuation(ticker)

    # ── Step 5: 리스크 계산 ──────────────────────────────────────────────
    risk = rm.calculate_risk(
        entry_price=pattern.entry_price,
        resistance_level=pattern.resistance_level,
    )

    # 손익비 미달 → 신호 억제
    if not risk.is_risk_ok:
        logger.info("%s: Risk/Reward %.2f:1 too low — signal suppressed",
                    name, risk.reward_risk_ratio)
        return

    # ── Step 6: 슬랙 발송 ─────────────────────────────────────────────
    if valuation.passed:
        # 모든 조건 통과 → 정식 매수 신호
        sent = sb.send_buy_signal(ticker, pattern, valuation, risk, market)
        if sent:
            _state.daily_signals.append({
                "ticker": ticker,
                "pattern": pattern.pattern.value,
                "signal_type": "BUY",
                "entry": pattern.entry_price,
                "stop": risk.stop_loss_price,
                "action": pattern.action or "매수 검토",
                "time": datetime.now().isoformat(),
            })
            logger.info("BUY SIGNAL SENT: %s %s @ %.2f",
                        name, pattern.pattern.value, pattern.entry_price)
    else:
        # 패턴은 맞으나 밸류에이션 미통과 → 관심 종목 알림
        logger.info("%s: Pattern OK but valuation failed — watchlist alert", name)
        sb.send_watchlist_signal(ticker, pattern, risk)


# ---------------------------------------------------------------------------
# 장 시간 확인
# ---------------------------------------------------------------------------
def _is_market_hours() -> bool:
    """한국 주식 시장 운영 시간 (평일 09:05 ~ 15:20)"""
    now = datetime.now()
    if now.weekday() >= 5:   # 토, 일
        return False

    open_t = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    close_t = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    current_t = now.time()

    return open_t <= current_t <= close_t


def _is_close_to_market_close() -> bool:
    """장 마감 5분 이내"""
    now = datetime.now().time()
    close_t = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    close_minus5 = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE - 5)
    return close_minus5 <= now <= close_t


# ---------------------------------------------------------------------------
# 일일 마감 요약
# ---------------------------------------------------------------------------
def daily_close_summary() -> None:
    """장 마감 후 일일 요약 발송"""
    if not _state.daily_signals and _state.scan_count == 0:
        return

    try:
        market = mf.analyze_market()
    except Exception:
        market = None

    if market:
        sb.send_daily_summary(
            signals_count=len(_state.daily_signals),
            tickers_scanned=len(config.WATCHLIST),
            market=market,
        )

    logger.info("Daily summary: %d signals in %d scans",
                len(_state.daily_signals), _state.scan_count)
    _state.reset_daily()


# ---------------------------------------------------------------------------
# 스케줄러 설정
# ---------------------------------------------------------------------------
def setup_scheduler() -> None:
    # 5분마다 시장 스캔
    schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(scan_market)

    # 장 마감 후 일일 요약 (15:30)
    schedule.every().day.at("15:30").do(daily_close_summary)

    logger.info("Scheduler configured: scan every %d minutes", config.SCAN_INTERVAL_MINUTES)


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
def main() -> None:
    # --test 플래그: 슬랙 테스트 메시지 전송 후 종료
    if "--test" in sys.argv:
        if not config.SLACK_WEBHOOK_URL:
            logger.error("SLACK_WEBHOOK_URL 미설정 — .env 파일을 확인하세요")
            sys.exit(1)
        logger.info("슬랙 테스트 메시지 발송 중...")
        if sb.send_test():
            logger.info("테스트 메시지 전송 성공!")
        else:
            logger.error("테스트 메시지 전송 실패 — 웹훅 주소를 확인하세요")
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("주식 매매 신호 시스템 시작")
    logger.info("손절: %.0f%% | 최소 손익비: %.1f:1",
                config.STOP_LOSS_RATE * 100, config.MIN_REWARD_RISK_RATIO)
    logger.info("감시 종목: %d개", len(config.WATCHLIST))
    logger.info("=" * 60)

    # 환경 변수 체크
    if not config.SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL 미설정 — 슬랙 알림 비활성화")

    setup_scheduler()

    # 시작 시 즉시 1회 스캔
    scan_market()

    # 메인 루프
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
