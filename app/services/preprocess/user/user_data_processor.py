# app/services/preprocess/user/user_data_processor.py

import pandas as pd
import logging

# 모듈 로거 설정
logger = logging.getLogger(__name__)

def user_convert_to_dataframe(features_list):
    """사용자 특성 리스트를 데이터프레임으로 변환"""
    logger.info(f"{len(features_list)}개의 사용자 특성을 데이터프레임으로 변환 중")
    
    try:
        df = pd.DataFrame(features_list)
        logger.debug(f"데이터프레임 생성 완료. 크기: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"데이터프레임 변환 중 오류 발생: {str(e)}")
        raise

def user_check_missing_features(df, required_features):
    """사용자 데이터의 필수 특성이 누락되었는지 확인하고 필요시 추가"""
    logger.info("필수 특성 누락 여부 확인 중")
    
    missing_columns = [col for col in required_features if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"다음 필수 열이 누락되었습니다: {', '.join(missing_columns)}")
        # 누락된 열을 0으로 채움
        for col in missing_columns:
            df[col] = 0
            logger.debug(f"누락된 열 '{col}'을 0으로 채움")
    else:
        logger.info("모든 필수 특성이 존재합니다.")
    
    return df

def user_save_to_csv(df, output_path="preprocessed_user_features.csv"):
    """사용자 데이터프레임을 CSV 파일로 저장"""
    logger.info(f"전처리된 사용자 데이터를 {output_path}에 저장 중")
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"데이터 저장 완료: {output_path} (행: {df.shape[0]}, 열: {df.shape[1]})")
    except Exception as e:
        logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
        raise