# app/services/preprocess/restaurant/encoding.py

def select_final_columns(df, conv_cols, caution_cols):
    # 최종 컬럼 목록
    final_columns = [
        'restaurant_id', 'name', 'category_id', 'address', 
        'phone_number', 'score', 'review', 'operating_days_count',
        'duration_hours'  # time_range 대신 duration_hours 사용
    ]
    
    # db_category_id와 image_urls가 있는 경우에만 추가
    if 'db_category_id' in df.columns:
        final_columns.append('db_category_id')
    
    if 'image_urls' in df.columns:
        final_columns.append('image_urls')
    
    # 모든 convenience 컬럼 추가
    final_columns.extend(conv_cols)
    
    # 모든 caution 컬럼 추가
    final_columns.extend(caution_cols)
    
    # 데이터프레임에 있는 컬럼만 필터링
    available_columns = [col for col in final_columns if col in df.columns]
    
    return df[available_columns]