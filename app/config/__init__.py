from dotenv import load_dotenv
from pathlib import Path
import os
import logging

# settings.py에서 하이퍼파라미터 값들을 가져옴
try:
    from app.setting import A_VALUE, B_VALUE, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT
except ImportError:
    # 설정 값이 없을 경우 기본값 설정
    A_VALUE, B_VALUE = 1.0, 1.0
    REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT = 1.0, 1.0, 1.0
    
# 로거 설정
logger = logging.getLogger("app.config")

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
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        logger.info("로컬 환경으로 감지되었습니다.")
except FileNotFoundError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    logger.info("로컬 환경으로 간주합니다. BASE_DIR를 기본값으로 설정합니다.")
except Exception as e:
    logger.error(f"환경 감지 중 오류 발생: {e}")
    raise Exception("환경 감지 중 오류 발생")

# 환경 변수에서 설정을 로드하거나 기본값 사용
def get_env_path(env_var, default_path, create=True):
    path_str = os.environ.get(env_var)
    
    if path_str:
        path = Path(path_str)
        logger.info(f"{env_var} 환경 변수로부터 경로를 설정합니다: {path}")
    else:
        path = Path(BASE_DIR) / default_path
        logger.info(f"로컬 환경으로 간주합니다. {env_var}를 기본값으로 설정합니다.")
    
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리가 생성되었습니다: {path}")
    
    if path.exists():
        logger.info(f"디렉토리가 준비되었습니다: {path}")
    
    return path

# 저장소 기본 디렉토리
STORAGE_DIR = get_env_path('STORAGE_DIR', 'storage')

# 레스토랑 데이터 저장 디렉토리
RESTAURANTS_DIR = get_env_path('RESTAURANTS_DIR', 'storage/input_json/restaurants')

# 사용자 데이터 저장 디렉토리
USER_DIR = get_env_path('USER_DIR', 'storage/input_json/user')

# 피드백 JSON 파일 디렉토리 (추천 결과 저장)
FEEDBACK_DIR = get_env_path('FEEDBACK_DIR', 'storage/output_feedback_json')

# 추가 설정이 필요한 경우 여기에 정의