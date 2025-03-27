# 베이스 이미지 선택 (Python 3.12-slim)
FROM python:3.12-slim

# 환경 변수 설정: 표준 출력 버퍼링 비활성화 등
ENV PYTHONUNBUFFERED=1

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# MongoDB 환경 변수 설정
ENV MONGO_HOST=$MONGO_HOST
ENV MONGO_PORT=$MONGO_PORT
ENV MONGO_USER=$MONGO_USER
ENV MONGO_PASSWORD=$MONGO_PASSWORD
ENV MONGO_DATABASE=$MONGO_DATABASE
ENV MONGO_COLLECTION=$MONGO_COLLECTION

# 터널링 비활성화 (명시적으로 설정)
ENV USE_SSH_TUNNEL=false

# libgomp 설치: LightGBM 등에서 필요한 OpenMP 라이브러리 설치
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Python 종속성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 전체 프로젝트 복사 (logging_config.json, .env, app/ 등 포함)
COPY . .

# /app/logs 디렉토리 생성
RUN mkdir -p /app/logs

# 볼륨 마운트 설정 (로컬의 storage 디렉토리를 컨테이너의 /app/storage 디렉토리와 매핑)
VOLUME ["/app/storage"]

# PYTHONPATH 설정 (필요 시)
ENV PYTHONPATH=/app

# 애플리케이션이 사용하는 포트 노출 (main.py에서 5000번 사용)
EXPOSE 5000

# 애플리케이션 실행
CMD ["python", "main.py"]