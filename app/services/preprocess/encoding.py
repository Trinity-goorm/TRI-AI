# app/services/preprocess/encoding.py

def select_final_columns(df, conv_cols, caution_cols):
    """최종 출력할 컬럼 목록을 구성하여 DataFrame에서 선택합니다."""
    final_columns = [
        "db_category_id", "restaurant_id", "name", "category_id", "score", "review", "address",
        "operating_hour", "expanded_days", "open_time", "close_time",
        "duration", "duration_hours", "time_range", "phone_number",
        "image_urls", "convenience", "caution", "is_deleted",
        "operating_days_count", "open_hour", "close_hour"
    ]
    final_columns += conv_cols + caution_cols
    return df[final_columns]
