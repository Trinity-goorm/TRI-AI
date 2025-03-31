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

# 다음 함수를 추가합니다
def restructure_user_data(data_source):
    """
    여러 형태의 사용자 데이터를 사용자 ID별로 통합하는 함수
    """
    logger.info("사용자 데이터를 ID별로 재구조화합니다.")
    user_data = {}
    
    # 사용자 기본 정보 추가
    if "user_data" in data_source and isinstance(data_source["user_data"], list):
        for item in data_source["user_data"]:
            if not isinstance(item, dict) or "user_id" not in item:
                continue
            user_id = item["user_id"]
            if user_id not in user_data:
                user_data[user_id] = {"user_info": {}}
            user_data[user_id]["user_info"] = item
    
    # 사용자 선호도 정보 추가
    if "user_preferences" in data_source and isinstance(data_source["user_preferences"], list):
        for item in data_source["user_preferences"]:
            if not isinstance(item, dict) or "user_id" not in item:
                continue
            user_id = item["user_id"]
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id]["preferences"] = item
    
    # 사용자 찜 정보 추가
    if "likes" in data_source and isinstance(data_source["likes"], list):
        likes_by_user = {}
        for item in data_source["likes"]:
            if not isinstance(item, dict) or "user_id" not in item:
                continue
            user_id = item["user_id"]
            if user_id not in likes_by_user:
                likes_by_user[user_id] = []
            likes_by_user[user_id].append(item)
        
        for user_id, likes in likes_by_user.items():
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id]["likes"] = likes
    
    # 사용자 예약 정보 추가
    if "reservations" in data_source and isinstance(data_source["reservations"], list):
        reservations_by_user = {}
        for item in data_source["reservations"]:
            if not isinstance(item, dict) or "user_id" not in item:
                continue
            user_id = item["user_id"]
            if user_id not in reservations_by_user:
                reservations_by_user[user_id] = []
            reservations_by_user[user_id].append(item)
        
        for user_id, reservations in reservations_by_user.items():
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id]["reservations"] = reservations
    
    # 딕셔너리를 리스트로 변환
    result = list(user_data.values())
    logger.info(f"재구조화 완료: {len(result)}명의 사용자 데이터")
    return result

# user_load_data 함수를 수정합니다
def user_load_data(data_source):
    """
    다양한 형태의 사용자 데이터를 로드하는 함수
    
    Args:
        data_source: 파일 경로, JSON 문자열, 또는 파이썬 객체
        
    Returns:
        list: 사용자 데이터 리스트
    """
    logger.info(f"데이터 소스 타입: {type(data_source)}")
    
    # 파일 경로 리스트인 경우 각 파일을 개별적으로 로드
    if isinstance(data_source, list) and all(isinstance(path, str) for path in data_source):
        logger.info(f"여러 파일에서 사용자 데이터 로드 중: {len(data_source)}개 파일")
        
        # 각 파일에서 데이터 수집
        combined_data = {
            "user_data": [],
            "user_preferences": [],
            "likes": [],
            "reservations": []
        }
        
        for file_path in data_source:
            if not os.path.exists(file_path):
                logger.warning(f"파일이 존재하지 않음: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # 파일 이름으로 데이터 유형 추정
                file_name = os.path.basename(file_path).lower()
                
                # 디버깅 정보
                logger.debug(f"파일 {file_name} 로드, 데이터 타입: {type(file_data)}")
                
                # 데이터 형식 및 파일명에 따라 적절한 카테고리에 추가
                if "user_data" in file_name:
                    combined_data["user_data"].extend(file_data if isinstance(file_data, list) else [file_data])
                elif "preferences" in file_name:
                    combined_data["user_preferences"].extend(file_data if isinstance(file_data, list) else [file_data])
                elif "like" in file_name:
                    combined_data["likes"].extend(file_data if isinstance(file_data, list) else [file_data])
                elif "reservation" in file_name:
                    combined_data["reservations"].extend(file_data if isinstance(file_data, list) else [file_data])
                elif "recsys" in file_name:
                    # recsys 파일은 이미 통합된 형식일 수 있으므로 구조 분석
                    if isinstance(file_data, dict):
                        # 각 키별로 처리
                        for key, value in file_data.items():
                            if key in combined_data and isinstance(value, list):
                                combined_data[key].extend(value)
                    elif isinstance(file_data, list):
                        # 사용자 데이터 구조를 분석하여 적절한 카테고리에 추가
                        for item in file_data:
                            if isinstance(item, dict):
                                if "preferences" in item:
                                    combined_data["user_preferences"].append(item["preferences"])
                                if "user_info" in item:
                                    combined_data["user_data"].append(item["user_info"])
                                if "likes" in item and isinstance(item["likes"], list):
                                    combined_data["likes"].extend(item["likes"])
                                if "reservations" in item and isinstance(item["reservations"], list):
                                    combined_data["reservations"].extend(item["reservations"])
            except Exception as e:
                logger.error(f"파일 {file_path} 로드 중 오류: {e}", exc_info=True)
        
        # 수집된 데이터 요약
        for key, items in combined_data.items():
            logger.info(f"{key}: {len(items)}개 항목 수집")
        
        # 수집된 데이터로 사용자별 구조화
        data = restructure_user_data(combined_data)
        return data
