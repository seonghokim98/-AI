@echo off
REM 주식 신호 시스템 - 한 번에 실행 스크립트
REM 윈도우용 (.bat 파일 더블클릭으로 실행)

echo ======================================
echo   주식 신호 시스템 시작
echo ======================================

REM 1. 가상환경 생성
if not exist "venv" (
    echo [1/3] 가상환경 만드는 중...
    python -m venv venv
)

REM 2. 패키지 설치
echo [2/3] 패키지 설치 중...
call venv\Scripts\activate
pip install -q yfinance "pandas==2.2.3" "numpy==1.26.4" "requests==2.32.3" "schedule==1.2.2" "python-dotenv==1.0.1" "colorlog==6.9.0"

REM 3. .env 파일 확인
if not exist ".env" (
    copy .env.example .env
    echo.
    echo ⚠️  .env 파일을 메모장으로 열어서 SLACK_WEBHOOK_URL을 입력하세요.
    pause
    exit /b 1
)

REM 4. 실행
echo [3/3] 시스템 실행!
echo.
python main.py
pause
