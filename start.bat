@echo off
REM 주식 신호 시스템 - 한 번에 실행 스크립트
REM 윈도우용 (.bat 파일 더블클릭으로 실행)
REM
REM 사용법:
REM   start.bat            -> 슬랙 신호 스캐너 실행 (main.py)
REM   start.bat dashboard  -> 대시보드 실행 (app.py, 브라우저 자동 오픈)

echo ======================================
echo   주식 신호 시스템
echo ======================================

REM 1. 가상환경 생성
if not exist "venv" (
    echo [1/3] 가상환경 만드는 중...
    python -m venv venv
)

REM 2. 패키지 설치 (requirements.txt 기준)
echo [2/3] 패키지 설치 중...
call venv\Scripts\activate
pip install -q -r requirements.txt

REM 3. .env 파일 확인
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
    ) else (
        type nul > .env
    )
    echo.
    echo ^⚠️  .env 파일을 메모장으로 열어서 SLACK_WEBHOOK_URL을 입력하세요.
    pause
    exit /b 1
)

REM 4. 실행 모드 분기
set MODE=%1
if "%MODE%"=="" set MODE=scanner

if "%MODE%"=="dashboard" (
    echo [3/3] 대시보드 실행^^! ^(브라우저 자동 오픈^)
    echo.
    echo   ▶ http://localhost:8501
    echo   종료하려면 Ctrl+C
    echo.
    streamlit run app.py --server.port 8501
) else (
    echo [3/3] 슬랙 신호 스캐너 실행^^!
    echo   ▶ 대시보드 실행 원하면: start.bat dashboard
    echo.
    python main.py
    pause
)
