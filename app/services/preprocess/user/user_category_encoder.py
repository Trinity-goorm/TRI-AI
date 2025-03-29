import logging

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def user_encode_categories(features, categories, category_count=12):
    """
    사용자 선호 카테고리 원-핫 인코딩
    
    Args:
        features: 특성 딕셔너리 (수정됨)
        categories: 카테고리 ID 리스트
        category_count: 총 카테고리 수
    """
    user_id = features.get("user_id", "알 수 없음")
    logger.debug(f"사용자 ID {user_id}의 카테고리 원-핫 인코딩 시작")
    
    # 모든 카테고리 초기화
    for i in range(1, category_count + 1):
        features[f"category_{i}"] = 0
    
    # 선호 카테고리 설정
    for category in categories:
        if 1 <= category <= category_count:
            features[f"category_{category}"] = 1
        else:
            logger.warning(f"사용자 ID {user_id}의 카테고리 ID {category}가 범위를 벗어납니다. (범위: 1-{category_count})")
    
    logger.debug(f"사용자 ID {user_id}의 카테고리 원-핫 인코딩 완료")