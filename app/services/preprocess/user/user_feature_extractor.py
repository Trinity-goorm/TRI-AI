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
        "empty_ticket_count": user["user_info"]["empty_ticket_count"],
        "normal_ticket_count": user["user_info"]["normal_ticket_count"]
    }

def user_extract_preferences(user, features):
    """사용자의 가격 및 카테고리 선호도 정보 추출"""
    user_id = features["user_id"]
    
    if "preferences" in user:
        logger.debug(f"사용자 ID {user_id}의 선호도 정보 추출")
        # 가격 정보
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
        completed_reservations = [r for r in user["reservations"] if r["status"] == "COMPLETED"]
        features["completed_reservations"] = len(completed_reservations)
        logger.debug(f"사용자 ID {user_id}의 완료된 예약 수: {len(completed_reservations)}")
    else:
        logger.warning(f"사용자 ID {user_id}에 예약 정보가 없습니다. 기본값 사용")
        features["completed_reservations"] = 0

def user_extract_like_info(user, features):
    """사용자의 찜 정보 추출"""
    user_id = features["user_id"]
    
    if "likes" in user:
        features["total_likes"] = len(user["likes"])
        logger.debug(f"사용자 ID {user_id}의 찜 수: {features['total_likes']}")
    else:
        logger.warning(f"사용자 ID {user_id}에 찜 정보가 없습니다. 기본값 사용")
        features["total_likes"] = 0