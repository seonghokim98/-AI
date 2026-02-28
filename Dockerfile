FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 한국 시간대 설정
ENV TZ=Asia/Seoul

# 의존성 설치 (캐시 레이어 분리)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY *.py .

# 환경변수는 외부에서 주입 (.env 또는 docker run -e)
# SLACK_WEBHOOK_URL, KIS_APP_KEY, KIS_APP_SECRET, DART_API_KEY 등

# 로그 디렉토리
RUN mkdir -p /app/logs

CMD ["python", "main.py"]
