import logging
from .user_data_loader import user_load_data
from .user_feature_extractor import user_extract_features
from .user_data_processor import user_convert_to_dataframe, user_check_missing_features, user_save_to_csv
from .user_category_encoder import user_encode_categories

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def user_preprocess_data(data_source, save_path=None):
    """
    사용자 데이터 전처리 통합 함수
    
    Args:
        data_source: 파일 경로, JSON 문자열, 또는 파이썬 객체
        save_path: 저장할 CSV 파일 경로 (None이면 저장하지 않음)
        
    Returns:
        DataFrame: 전처리된 사용자 특성 데이터프레임
    """
    logger.info("사용자 데이터 전처리 시작")
    
    # 필수 특성 목록
    required_features = [
        "user_id", 
        "max_price", 
        "empty_ticket_count", 
        "normal_ticket_count",
        "total_likes", 
        "completed_reservations"
    ]
    
    # 1. 데이터 로드
    logger.info("1단계: 사용자 데이터 로드")
    user_data_list = user_load_data(data_source)
    
    # 2. 특성 추출
    logger.info("2단계: 사용자 특성 추출")
    user_features_list = user_extract_features(user_data_list)
    
    # 3. 데이터프레임으로 변환
    logger.info("3단계: 특성 데이터프레임 변환")
    user_features_df = user_convert_to_dataframe(user_features_list)
    
    # 4. 결측 특성 확인 및 처리
    logger.info("4단계: 결측 특성 확인 및 처리")
    user_features_df = user_check_missing_features(user_features_df, required_features)
    
    # 5. 결과 요약
    logger.info(f"사용자 데이터 전처리 완료: 총 {len(user_features_df)} 명의 사용자, {len(user_features_df.columns)} 개의 특성")
    logger.debug(f"추출된 특성: {', '.join(user_features_df.columns)}")
    
    # 6. 저장 (지정된 경우)
    if save_path:
        logger.info("5단계: 전처리된 데이터 저장")
        user_save_to_csv(user_features_df, save_path)
    
    return user_features_df