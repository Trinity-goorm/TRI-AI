# app/config/mongo_config.py

from dotenv import load_dotenv
from pathlib import Path
import os
import logging

# 로거 설정
logger = logging.getLogger("app.config.mongo")

# .env 파일 로드
load_dotenv()

# MongoDB 설정
MONGO_HOST = os.environ.get('MONGO_HOST')
MONGO_PORT = int(os.environ.get('MONGO_PORT', 27017))
MONGO_USER = os.environ.get('MONGO_USER')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')
MONGO_DATABASE = os.environ.get('MONGO_DATABASE')
MONGO_COLLECTION = os.environ.get('MONGO_COLLECTION', 'recsys_data')

# 필수 환경 변수 확인
if not all([MONGO_HOST, MONGO_USER, MONGO_PASSWORD, MONGO_DATABASE]):
    logger.warning("MongoDB 연결에 필요한 일부 환경 변수가 설정되지 않았습니다.")
    logger.warning("설정을 확인하거나 .env 파일을 추가해 주세요.")

# SSH 터널링 설정
USE_SSH_TUNNEL = os.environ.get('USE_SSH_TUNNEL', 'false').lower() == 'true'
SSH_HOST = os.environ.get('SSH_HOST')
SSH_PORT = int(os.environ.get('SSH_PORT', 22))
SSH_USER = os.environ.get('SSH_USER')
SSH_PASSWORD = os.environ.get('SSH_PASSWORD')
SSH_KEY_PATH = os.environ.get('SSH_KEY_PATH')

# SSH 터널링이 활성화된 경우 필수 환경 변수 확인
if USE_SSH_TUNNEL and not all([SSH_HOST, SSH_USER]) and not (SSH_PASSWORD or SSH_KEY_PATH):
    logger.warning("SSH 터널링이 활성화되었지만 필요한 환경 변수가 설정되지 않았습니다.")
    logger.warning("SSH_HOST, SSH_USER 및 SSH_PASSWORD 또는 SSH_KEY_PATH를 확인해주세요.")

# 동기화 설정
SYNC_INTERVAL_HOURS = float(os.environ.get('SYNC_INTERVAL_HOURS', 1.0))
SYNC_ON_STARTUP = os.environ.get('SYNC_ON_STARTUP', 'false').lower() == 'true'

# MongoDB 설정 로드 알림 (민감한 정보는 로깅하지 않음)
if MONGO_HOST:
    logger.info(f"MongoDB 설정 로드: {MONGO_HOST}:{MONGO_PORT}/{MONGO_DATABASE}")
if USE_SSH_TUNNEL and SSH_HOST:
    logger.info(f"SSH 터널링 활성화: {SSH_HOST}:{SSH_PORT}")
else:
    logger.info("SSH 터널링 비활성화")