# app/services/preprocess/user/user_feature_extractor.py

from .user_category_encoder import user_encode_categories
import logging

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def user_extract_features(user_data_list):
    """
    사용자 데이터 리스트에서 필요한 특성 추출
    
    Args:
        user_data_list: 사용자 데이터 리스트
        
    Returns:
        list: 추출된 특성 딕셔너리 리스트
    """
    logger.info(f"총 {len(user_data_list)}명의 사용자에 대한 특성 추출 시작")
    user_features_list = []
    
    for i, user in enumerate(user_data_list):
        # 기본 사용자 정보 추출
        features = user_extract_basic_info(user)
        
        # 가격 및 카테고리 정보 추출
        user_extract_preferences(user, features)
        
        # 예약 정보 추출
        user_extract_reservation_info(user, features)
        
        # 찜 정보 추출
        user_extract_like_info(user, features)
        
        # 파생 변수 계산
        user_calculate_derived_features(features)
        
        user_features_list.append(features)
        
        # 로깅 - 진행 상황 (100명마다 로그)
        if (i+1) % 100 == 0 or i+1 == len(user_data_list):
            logger.debug(f"사용자 특성 추출 진행 중: {i+1}/{len(user_data_list)}")
    
    logger.info(f"사용자 특성 추출 완료. 총 {len(user_features_list)}개의 특성 집합 생성")
    return user_features_list

def user_extract_basic_info(user):
    """사용자의 기본 정보 추출"""
    user_id = user["user_info"]["user_id"]
    logger.debug(f"사용자 ID {user_id}의 기본 정보 추출")
    
    return {
        "user_id": user_id,
        # 티켓 정보는 최종 변수 선택에서 제외
    }

def user_extract_preferences(user, features):
    """사용자의 가격 및 카테고리 선호도 정보 추출"""
    user_id = features["user_id"]
    
    if "preferences" in user:
        logger.debug(f"사용자 ID {user_id}의 선호도 정보 추출")
        
        # 가격 정보 - max_price만 사용
        features["max_price"] = user["preferences"]["max_price"]
        
        # 카테고리 정보
        preferred_categories = user["preferences"]["preferred_categories"]
        logger.debug(f"사용자 ID {user_id}의 선호 카테고리: {preferred_categories}")
        user_encode_categories(features, preferred_categories)
    else:
        logger.warning(f"사용자 ID {user_id}에 선호도 정보가 없습니다. 기본값 사용")
        features["max_price"] = 0
        user_encode_categories(features, [])

def user_extract_reservation_info(user, features):
    """사용자의 예약 정보 추출"""
    user_id = features["user_id"]
    
    if "reservations" in user:
        # 전체 예약 및 완료된 예약 수
        user_reservations = user["reservations"]
        completed_reservations = [r for r in user_reservations if r["status"] == "COMPLETED"]
        
        features["completed_reservations"] = len(completed_reservations)
        
        # 예약 완료율 계산 (선택적 변수)
        if len(user_reservations) > 0:
            features["reservation_completion_rate"] = len(completed_reservations) / len(user_reservations)
        else:
            features["reservation_completion_rate"] = 0.0
            
        logger.debug(f"사용자 ID {user_id}의 완료된 예약 수: {len(completed_reservations)}")
    else:
        logger.warning(f"사용자 ID {user_id}에 예약 정보가 없습니다. 기본값 사용")
        features["completed_reservations"] = 0
        features["reservation_completion_rate"] = 0.0

def user_extract_like_info(user, features):
    """사용자의 찜 정보 추출"""
    user_id = features["user_id"]
    
    if "likes" in user:
        features["total_likes"] = len(user["likes"])
        logger.debug(f"사용자 ID {user_id}의 찜 수: {features['total_likes']}")
    else:
        logger.warning(f"사용자 ID {user_id}에 찜 정보가 없습니다. 기본값 사용")
        features["total_likes"] = 0

def user_calculate_derived_features(features):
    """파생 변수 계산"""
    user_id = features["user_id"]
    
    # like_to_reservation_ratio 계산 (찜 대비 예약 비율)
    if features["completed_reservations"] > 0:
        features["like_to_reservation_ratio"] = features["total_likes"] / features["completed_reservations"]
    else:
        features["like_to_reservation_ratio"] = 5.0  # 예약 없이 찜만 있는 경우 최대값 설정
    
    logger.debug(f"사용자 ID {user_id}의 파생 변수 계산 완료")