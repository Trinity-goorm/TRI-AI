# app/services/preprocess/user/user_data_loader.py

import json
import os
import glob
import logging

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def get_latest_user_data_file(directory_path="storage/input_json/user/"):
    """
    지정된 디렉토리에서 가장 최근에 생성된 recsys_data 파일을 찾는 함수
    """
    logger.info(f"디렉토리에서 최신 사용자 데이터 파일 검색 중: {directory_path}")
    
    # recsys_data로 시작하는 모든 JSON 파일 검색
    file_pattern = os.path.join(directory_path, "recsys_data_*.json")
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        error_msg = f"지정된 디렉토리에 recsys_data JSON 파일이 없습니다: {directory_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # 수정 시간을 기준으로 가장 최근 파일 선택
    latest_file = max(json_files, key=os.path.getmtime)
    logger.info(f"최신 사용자 데이터 파일 발견: {latest_file}")
    
    return latest_file

def user_load_data(data_source):
    """
    다양한 형태의 사용자 데이터를 로드하는 함수
    
    Args:
        data_source: 파일 경로, JSON 문자열, 또는 파이썬 객체(dict/list)
        
    Returns:
        list: 사용자 데이터 리스트
    """
    # 디렉토리만 지정된 경우 최신 파일 자동 검색
    if isinstance(data_source, str) and os.path.isdir(data_source):
        data_source = get_latest_user_data_file(data_source)
    
    # 문자열인 경우 파일 로드 또는 JSON 파싱
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