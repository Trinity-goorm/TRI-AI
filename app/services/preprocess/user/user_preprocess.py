# app/services/preprocess/user/user_preprocess.py

import logging
import pandas as pd  # 이 줄을 추가
import os
from .user_data_loader import user_load_data, get_latest_user_data_file
from .user_feature_extractor import user_extract_features
from .user_data_processor import user_convert_to_dataframe, user_check_missing_features, user_save_to_csv

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

    # 데이터 소스가 지정되지 않은 경우 기본 경로 사용
    if data_source is None:
        try:
            data_source = os.path.join("storage/input_json/user/")
            logger.info(f"기본 경로 사용: {data_source}")
        except Exception as e:
            logger.error(f"기본 경로 사용 중 오류: {str(e)}")
            raise

    # 필수 특성 목록
    required_features = [
        "user_id", 
        "max_price",
        "category_1", "category_2", "category_3", "category_4", 
        "category_5", "category_6", "category_7", "category_8", 
        "category_9", "category_10", "category_11", "category_12",
        "completed_reservations",
        "reservation_completion_rate",
        "total_likes", 
        "like_to_reservation_ratio"
    ]
    
    # 1. 데이터 로드
    logger.info("1단계: 사용자 데이터 로드")
    user_data_list = user_load_data(data_source)
    
    # 로드된 데이터 검증
    if not user_data_list:
        logger.warning("로드된 사용자 데이터가 없습니다. 빈 데이터프레임을 반환합니다.")
        empty_df = pd.DataFrame(columns=required_features)
        if save_path:
            user_save_to_csv(empty_df, save_path)
        return empty_df
    
    # 데이터 샘플 로깅
    if len(user_data_list) > 0:
        logger.debug(f"첫 번째 사용자 데이터 샘플: {user_data_list[0].keys()}")
    
    # 2. 특성 추출
    logger.info("2단계: 사용자 특성 추출")
    user_features_list = user_extract_features(user_data_list)
    
    # 특성 추출 결과 검증
    if not user_features_list:
        logger.warning("사용자 특성이 추출되지 않았습니다. 빈 데이터프레임을 반환합니다.")
        empty_df = pd.DataFrame(columns=required_features)
        if save_path:
            user_save_to_csv(empty_df, save_path)
        return empty_df
    
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