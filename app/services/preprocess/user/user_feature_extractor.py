# app/services/preprocess/user/user_feature_extractor.py

import logging
import numpy as np
from typing import List, Dict, Any

# 로거 설정
logger = logging.getLogger(__name__)

def user_extract_basic_info(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 기본 정보를 추출하는 함수
    
    Args:
        user: 사용자 데이터 딕셔너리
    
    Returns:
        Dict: 추출된 사용자 기본 정보
    """
    result = {}
    
    # 로그에 데이터 구조 출력 (디버깅용)
    logger.debug(f"사용자 데이터 키: {list(user.keys())}")
    
    # 1. user_info에서 ID 찾기
    if "user_info" in user and isinstance(user["user_info"], dict) and "user_id" in user["user_info"]:
        result["user_id"] = user["user_info"]["user_id"]
        logger.debug(f"user_info에서 user_id {result['user_id']} 추출")
    
    # 2. preferences에서 ID 찾기
    elif "preferences" in user and isinstance(user["preferences"], dict) and "user_id" in user["preferences"]:
        result["user_id"] = user["preferences"]["user_id"]
        logger.debug(f"preferences에서 user_id {result['user_id']} 추출")
    
    # 3. reservations에서 ID 찾기
    elif "reservations" in user and isinstance(user["reservations"], list) and len(user["reservations"]) > 0:
        for res in user["reservations"]:
            if isinstance(res, dict) and "user_id" in res:
                result["user_id"] = res["user_id"]
                logger.debug(f"reservations에서 user_id {result['user_id']} 추출")
                break
    
    # 4. likes에서 ID 찾기
    elif "likes" in user and isinstance(user["likes"], list) and len(user["likes"]) > 0:
        for like in user["likes"]:
            if isinstance(like, dict) and "user_id" in like:
                result["user_id"] = like["user_id"]
                logger.debug(f"likes에서 user_id {result['user_id']} 추출")
                break
    
    # 5. 최상위 레벨에서 ID 찾기
    elif "user_id" in user:
        result["user_id"] = user["user_id"]
        logger.debug(f"최상위에서 user_id {result['user_id']} 추출")
    else:
        logger.warning(f"사용자 ID를 찾을 수 없습니다: {user.keys()}")
        return {}
    
    # 정수 또는 문자열 ID를 문자열로 통일
    result["user_id"] = str(result["user_id"])
    
    # 선호 가격 범위 (preferences에서)
    if "preferences" in user and isinstance(user["preferences"], dict):
        pref = user["preferences"]
        if "max_price" in pref:
            result["max_price"] = pref["max_price"]
        if "min_price" in pref:
            result["min_price"] = pref["min_price"]
        
        # 선호 카테고리 처리
        if "preferred_categories" in pref and isinstance(pref["preferred_categories"], list):
            # 카테고리 ID를 원-핫 인코딩으로 변환
            for cat_id in range(1, 13):  # 1부터 12까지
                cat_key = f"category_{cat_id}"
                result[cat_key] = 1 if cat_id in pref["preferred_categories"] else 0
        else:
            # 카테고리 데이터가 없는 경우 모두 0으로 설정
            for cat_id in range(1, 13):
                result[f"category_{cat_id}"] = 0
    else:
        # preferences 데이터가 없는 경우 기본값 설정
        result["max_price"] = 0
        # 카테고리 모두 0으로 설정
        for cat_id in range(1, 13):
            result[f"category_{cat_id}"] = 0
    
    return result

def user_extract_reservation_features(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 예약 관련 특성을 추출하는 함수
    
    Args:
        user: 사용자 데이터 딕셔너리
    
    Returns:
        Dict: 추출된 예약 관련 특성
    """
    result = {}
    
    if "reservations" not in user or not isinstance(user["reservations"], list):
        result["completed_reservations"] = 0
        result["reservation_completion_rate"] = 0.0
        return result
    
    # 전체 예약 수
    total_reservations = len(user["reservations"])
    result["total_reservations"] = total_reservations
    
    # 완료된 예약 수
    completed = sum(1 for res in user["reservations"] if res.get("status") == "COMPLETED")
    result["completed_reservations"] = completed
    
    # 예약 완료율
    result["reservation_completion_rate"] = round(completed / total_reservations, 2) if total_reservations > 0 else 0.0
    
    return result

def user_extract_like_features(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 찜 관련 특성을 추출하는 함수
    
    Args:
        user: 사용자 데이터 딕셔너리
    
    Returns:
        Dict: 추출된 찜 관련 특성
    """
    result = {}
    
    if "likes" not in user or not isinstance(user["likes"], list):
        result["total_likes"] = 0
        return result
    
    # 전체 찜 수
    result["total_likes"] = len(user["likes"])
    
    return result

def user_extract_features(user_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    사용자 데이터 리스트에서 필요한 특성을 추출하는 함수
    
    Args:
        user_data_list: 사용자 데이터 딕셔너리 목록
    
    Returns:
        List[Dict]: 추출된 특성의 딕셔너리 목록
    """
    result = []
    
    for idx, user in enumerate(user_data_list):
        try:
            # 기본 정보 추출
            user_features = user_extract_basic_info(user)
            
            # 사용자 ID가 없는 경우 건너뜀
            if not user_features:
                logger.warning(f"유효한 사용자 ID가 없는 데이터 건너뜀: {idx+1}번째 항목")
                continue
            
            # 예약 관련 특성 추출
            reservation_features = user_extract_reservation_features(user)
            user_features.update(reservation_features)
            
            # 찜 관련 특성 추출
            like_features = user_extract_like_features(user)
            user_features.update(like_features)
            
            # 복합 지표 계산
            # 1. 찜/예약 비율
            if user_features.get("completed_reservations", 0) > 0:
                user_features["like_to_reservation_ratio"] = round(
                    user_features.get("total_likes", 0) / user_features.get("completed_reservations", 1), 
                    2
                )
            else:
                # 예약이 없는 경우 기본값 설정
                user_features["like_to_reservation_ratio"] = 5.0 if user_features.get("total_likes", 0) > 0 else 0.0
            
            # 추출 결과가 유효한 경우 결과 리스트에 추가
            if user_features["user_id"]:
                result.append(user_features)
            
        except Exception as e:
            logger.error(f"사용자 특성 추출 중 오류 발생: {idx+1}번째 항목 - {str(e)}", exc_info=True)
            # 오류가 있어도 계속 진행
    
    logger.info(f"총 {len(result)}명의 사용자 특성 추출 완료")
    return result