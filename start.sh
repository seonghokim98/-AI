#!/bin/bash
# 주식 신호 시스템 - 한 번에 실행 스크립트
# 맥 / 리눅스용
#
# 사용법:
#   ./start.sh           → 슬랙 신호 스캐너 실행 (main.py)
#   ./start.sh dashboard → 대시보드 실행 (app.py, 브라우저 자동 오픈)

set -e

echo "======================================"
echo "  주식 신호 시스템"
echo "======================================"

# 1. 가상환경 생성 (없으면)
if [ ! -d "venv" ]; then
    echo "[1/3] 가상환경 만드는 중..."
    python3 -m venv venv
fi

# 2. 패키지 설치 (requirements.txt 기준 — 전체 설치)
echo "[2/3] 패키지 설치 중..."
source venv/bin/activate
pip install -q -r requirements.txt

# 3. .env 파일 확인
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        touch .env
    fi
    echo ""
    echo "⚠️  .env 파일을 열어서 SLACK_WEBHOOK_URL을 입력하세요."
    echo "    nano .env  또는  메모장으로 .env 파일 열기"
    exit 1
fi

# 4. 실행 모드 분기
MODE="${1:-scanner}"

if [ "$MODE" = "dashboard" ]; then
    echo "[3/3] 대시보드 실행! (브라우저 자동 오픈)"
    echo ""
    echo "  ▶ http://localhost:8501"
    echo "  종료하려면 Ctrl+C"
    echo ""
    streamlit run app.py --server.port 8501
else
    echo "[3/3] 슬랙 신호 스캐너 실행!"
    echo "  ▶ 대시보드 실행 원하면: ./start.sh dashboard"
    echo ""
    python main.py
fi
