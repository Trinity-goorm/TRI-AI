# app/config.py

from dotenv import load_dotenv
from pathlib import Path
import os
import logging
from flask import abort

# settings.py에서 하이퍼파라미터 값들을 가져옴
from app.setting import A_VALUE, B_VALUE, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT

# 로거 설정
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# Docker 환경 감지 및 BASE_DIR 설정
try:
    # Docker 환경인지 확인 (Linux의 /proc/1/cgroup 파일 존재 여부 활용)
    with open('/proc/1/cgroup', 'rt') as f:
        cgroup_content = f.read()
    if 'docker' in cgroup_content:
        BASE_DIR = Path("/app")
        logger.info("Docker 환경으로 감지되었습니다.")
    else:
        BASE_DIR = Path(__file__).resolve().parent.parent
        logger.info("로컬 환경으로 감지되었습니다.")
except FileNotFoundError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    logger.info("로컬 환경으로 간주합니다. BASE_DIR를 기본값으로 설정합니다.")
except Exception as e:
    logger.error(f"환경 감지 중 오류 발생: {e}")
    abort(500, description="환경 감지 중 오류 발생")

# JSON 데이터를 받을 디렉토리 (업로드 디렉토리)
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/input_json")

# 추천 결과(피드백)를 저장할 디렉토리
FEEDBACK_DIR = BASE_DIR / os.getenv("FEEDBACK_DIR", "storage/output_feedback_json")

# 디렉토리 존재 여부 확인 및 생성
try:
    for directory in [UPLOAD_DIR, FEEDBACK_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리가 준비되었습니다: {directory}")
except Exception as e:
    logger.error(f"디렉토리 생성 실패: {e}")
    abort(500, description="디렉토리 생성 실패")
