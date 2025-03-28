# app/services/preprocess/restaurant/preprocessor.py

import pandas as pd
from app.services.preprocess.restaurant.data_loader import load_restaurant_json_files
from app.services.preprocess.restaurant.convert_category import convert_category
from app.services.preprocess.restaurant.phone_format import format_phone
from app.services.preprocess.restaurant.convenience import normalize_convenience
from app.services.preprocess.restaurant.caution import normalize_caution
from app.services.preprocess.restaurant.operating_days import count_operating_days
from app.services.preprocess.restaurant.time_range import (
    extract_open_time, extract_close_time, convert_to_minutes, compute_duration
)
from app.services.preprocess.restaurant.encoding import select_final_columns
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # 기본 컬럼 타입 변환
        df['review'] = pd.to_numeric(df['review'], errors='coerce')
        logger.debug("review 데이터 변환이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error converting 'review' column: {e}", exc_info=True)
        raise e

    try:
        df['category_id'] = df['category_id'].apply(convert_category)
        logger.debug("category_id 변환이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error converting 'category_id': {e}", exc_info=True)
        raise e
    
    try:
        df['phone_number'] = df['phone_number'].apply(format_phone)
        logger.debug("phone_number 변환이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error formatting 'phone_number': {e}", exc_info=True)
    

    # 3. 편의시설 처리
    # 결측치는 빈 문자열("")로 처리 후 리스트 생성
    try:
        df['convenience_list'] = df['convenience'].fillna("").apply(lambda x: normalize_convenience(x)[0] if x != "" else [])
        mlb_conv = MultiLabelBinarizer()
        conv_encoded = mlb_conv.fit_transform(df['convenience_list'])
        conv_encoded_df = pd.DataFrame(conv_encoded,
                                    columns=[f"conv_{col}" for col in mlb_conv.classes_],
                                    index=df.index)
        logger.debug("convenience_list 변환이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error processing convenience data: {e}", exc_info=True)
        raise e

    
    # 4. 유의사항 처리
    try:
        df['caution_list'] = df['caution'].fillna("").apply(
            lambda x: normalize_caution(x)[0] if x != "" else []
        )
        mlb_caution = MultiLabelBinarizer()
        caution_encoded = mlb_caution.fit_transform(df['caution_list'])
        caution_encoded_df = pd.DataFrame(
            caution_encoded, columns=[f"caution_{col}" for col in mlb_caution.classes_], index=df.index
        )
        logger.debug("caution_list 변환이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error processing caution data: {e}", exc_info=True)
        caution_encoded_df = pd.DataFrame()
    
    # 5. expanded_days를 이용한 operating_days_count 계산
    try:
        df['operating_days_count'] = df['expanded_days'].apply(
            lambda x: count_operating_days(x) if pd.notna(x) else None
        )
        logger.debug("operating_days_count 계산이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error calculating operating_days_count: {e}", exc_info=True)
    
    # 6. time_range 처리
    try:
        df['open_time'] = df['time_range'].apply(extract_open_time)
        df['close_time'] = df['time_range'].apply(extract_close_time)
        df['open_minutes'] = df['open_time'].apply(convert_to_minutes)
        df['close_minutes'] = df['close_time'].apply(convert_to_minutes)
        df['duration'] = df.apply(
            lambda row: compute_duration(row['open_minutes'], row['close_minutes']), axis=1
        )
        df['duration_hours'] = df['duration'] / 60.0
        df['open_hour'] = df['open_time'].apply(lambda x: float(x.split(":")[0]) if x and ":" in x else None)
        df['close_hour'] = df['close_time'].apply(lambda x: None if x=="24:00" else (float(x.split(":")[0]) if x and ":" in x else None))
    except Exception as e:
        logger.error(f"Error processing time_range data: {e}", exc_info=True)
        raise e
    
    # 7. 인코딩된 편의시설, 유의사항 컬럼 병합
    try:
        df = pd.concat([df, conv_encoded_df, caution_encoded_df], axis=1)
        # 불필요한 리스트 컬럼 제거
        df.drop(columns=['convenience_list', 'caution_list'], inplace=True)
    except:
        logger.error(f"Error concatenating encoded columns: {e}", exc_info=True)
    
    # 8. 최종 출력할 컬럼만 선택 (추가로, 편의시설 및 유의사항 인코딩 컬럼 선택)
    try:
        conv_cols = [col for col in df.columns if col.startswith("conv_") and col != "conv_편의시설 정보 없음"]
        caution_cols = [col for col in df.columns if col.startswith("caution_") and col != "caution_유의사항 정보 없음"]
        df_final = select_final_columns(df, conv_cols, caution_cols)
        logger.debug("최종 출력할 컬럼 계산이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error selecting final columns: {e}", exc_info=True)
        df_final = df  # fallback
    
    return df_final
