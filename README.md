# 주식 매매 신호 시스템

매매 원칙을 100% 강제하는 **실시간 시장 분석 + 슬랙 자동 알림** 시스템.
자동매매가 아닌 **조건 충족 시에만 슬랙으로 진입 신호 발송** 구조.

---

## 아키텍처

```
[실시간 데이터 수집]   yfinance / KIS REST API
        ↓
[시장 트렌드 필터]     코스피 20일선 · 코스닥 · VIX · 외국인 수급
        ↓                ← 약세장이면 무조건 관망
[기술적 패턴 엔진]     돌파+눌림목 / 깃발형 / 쌍바닥 / 골든크로스 / SR전환
        ↓                ← 허용 패턴 5종만
[밸류에이션 필터]      PER < 업종평균 · PBR < 1.5 · EPS 양수 · 자사주 공시
        ↓
[리스크 계산]          -10% 손절가 · 손익비 2:1 · 수량 계산
        ↓
[슬랙 알림 발송]       진입가 · 손절가 · 목표가 · 수량 포함
```

---

## 매매 원칙 강제 목록

| 원칙 | 구현 위치 | 위반 시 |
|------|-----------|---------|
| 약세장 관망 | `market_filter.py` | 전 종목 신호 차단 |
| 허용 패턴만 진입 | `pattern_engine.py` | 신호 미발생 |
| PER < 업종 평균 | `valuation.py` | 관심 종목 알림만 |
| PBR < 1.5 | `valuation.py` | 관심 종목 알림만 |
| -10% 손절 절대 원칙 | `risk.py` | 항상 계산, 슬랙 전송 |
| 손익비 2:1 이상 | `risk.py` | 신호 억제 |

---

## 파일 구조

```
stock-alert-system/
├── main.py           # 스케줄러 + 오케스트레이터
├── config.py         # API 키, 예산, 종목 리스트
├── data_provider.py  # 데이터 수집 (yfinance, KIS API)
├── market_filter.py  # 시장 트렌드 분석
├── pattern_engine.py # 기술적 패턴 인식 엔진
├── valuation.py      # 밸류에이션 + DART 자사주 필터
├── risk.py           # 리스크/리워드 계산
├── slack_bot.py      # 슬랙 알림 모듈
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 빠른 시작

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에 슬랙 웹훅 URL 입력
```

### 2. 로컬 실행

```bash
pip install -r requirements.txt
python main.py
```

### 3. Docker 실행

```bash
docker build -t stock-alert .
docker run -d --env-file .env --name stock-alert stock-alert
```

---

## 슬랙 알림 예시

**매수 신호 발생:**
```
📈 [매수 신호 발생] — 2026-02-28 10:15

종목: 삼성전자 (005930.KS)
패턴: 돌파 + 눌림목 확인
신뢰도: ⬛⬛⬛⬛⬜ (80%)

─── 진입 정보 ───
• 지정가 매수: 72,500원
• 손절가 (-10%): 65,250원 ← 절대 원칙
• 1차 목표가: 79,750원 (손익비 2.0:1)
• 2차 목표가: 83,125원

─── 수량 & 예산 ───
• 예산 33,333원 기준 수량: 0주 → 1주
• 투입금액: 72,500원

─── 밸류에이션 ───
PER 11.2배 (업종평균 18.0배) ✅ | PBR 1.2 ✅ | EPS 4,820 ✅ | 자사주 공시 없음
```

**관망 시:**
```
⚠️ [관망] — 현재 조건 미충족 → 기다려!
사유: 코스피 20일선 아래 | VIX 과열
```

---

## API 연동

### 필수
- **Slack Webhook**: [슬랙 앱 설정](https://api.slack.com/messaging/webhooks)

### 선택
- **한국투자증권 REST API**: 외국인 수급 데이터
  → [KIS Developers](https://apiportal.koreainvestment.com)
- **DART 전자공시 API**: 자사주 매입 공시 확인
  → [DART Open API](https://opendart.fss.or.kr)

---

## 주의사항

- **자동매매 없음**: 이 시스템은 알림만 발송합니다. 실제 주문은 직접 판단 후 실행하세요.
- **과거 성과 무보장**: 기술적 패턴은 확률적 도구입니다. 손절 원칙을 반드시 준수하세요.
- **초기 운용 권장**: 최소 3개월 슬랙 알림 검증 후 규모를 늘리세요.
