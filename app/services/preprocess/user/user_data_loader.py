import json
import os
import logging

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def user_load_data(data_source):
    """
    다양한 형태의 사용자 데이터를 로드하는 함수
    
    Args:
        data_source: 파일 경로, JSON 문자열, 또는 파이썬 객체(dict/list)
        
    Returns:
        list: 사용자 데이터 리스트
    """
    # 문자열인 경우
    if isinstance(data_source, str):
        # 파일 경로인지 확인
        if os.path.exists(data_source):
            logger.info(f"파일에서 사용자 데이터 로드 중: {data_source}")
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # JSON 문자열로 간주
            logger.info("JSON 문자열에서 사용자 데이터 파싱 중")
            try:
                data = json.loads(data_source)
            except json.JSONDecodeError:
                logger.error(f"유효하지 않은 JSON 문자열 또는 파일 경로: {data_source}")
                raise ValueError(f"유효한 JSON 문자열 또는 파일 경로가 아닙니다: {data_source}")
    else:
        # 이미 파이썬 객체인 경우
        logger.info("파이썬 객체에서 사용자 데이터 처리 중")
        data = data_source
    
    # 단일 사용자 데이터(dict)인 경우 리스트로 변환
    if isinstance(data, dict):
        logger.debug("단일 사용자 데이터를 리스트로 변환")
        data = [data]
    
    logger.info(f"총 {len(data)}명의 사용자 데이터 로드 완료")
    return data