# app/services/preprocessor.py

import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 전처리 작업을 수행하여 최종 DataFrame을 반환합니다.
    (여기서는 예시로 review 컬럼의 숫자 변환 및 일부 컬럼 선택)
    """
    df['review'] = pd.to_numeric(df['review'], errors='coerce')
    # (추가 전처리 작업: 카테고리 변환, 전화번호 포맷 등 필요 시 추가)
    final_columns = [
        "id", "name", "category_id", "score", "review", "address",
        "operating_hour", "expanded_days", "open_time", "close_time",
        "duration", "duration_hours", "time_range", "phone_number",
        "image_urls", "convenience", "caution", "is_deleted",
        "operating_days_count", "open_hour", "close_hour"
    ]
    return df[final_columns].copy()
