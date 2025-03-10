# config 모듈 초기화 파일
import os
import logging
from pathlib import Path

logger = logging.getLogger("app.config")

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

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

# 입력 JSON 파일 디렉토리 (RDS에서 가져온 데이터 저장)
UPLOAD_DIR = get_env_path('UPLOAD_DIR', 'storage/input_json')

# 피드백 JSON 파일 디렉토리 (추천 결과 저장)
FEEDBACK_DIR = get_env_path('FEEDBACK_DIR', 'storage/output_feedback_json')

# 추가 설정이 필요한 경우 여기에 정의