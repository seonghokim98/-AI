#!/bin/bash
# 주식 신호 시스템 - 한 번에 실행 스크립트
# 맥 / 리눅스용

set -e

echo "======================================"
echo "  주식 신호 시스템 시작"
echo "======================================"

# 1. 가상환경 생성 (없으면)
if [ ! -d "venv" ]; then
    echo "[1/3] 가상환경 만드는 중..."
    python3 -m venv venv
fi

# 2. 패키지 설치
echo "[2/3] 패키지 설치 중..."
source venv/bin/activate
pip install -q yfinance "pandas==2.2.3" "numpy==1.26.4" "requests==2.32.3" "schedule==1.2.2" "python-dotenv==1.0.1" "colorlog==6.9.0"

# 3. .env 파일 확인
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  .env 파일을 열어서 SLACK_WEBHOOK_URL을 입력하세요."
    echo "    nano .env  또는  메모장으로 .env 파일 열기"
    exit 1
fi

# 4. 실행
echo "[3/3] 시스템 실행!"
echo ""
python main.py
