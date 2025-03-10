# SQL 쿼리 설정 파일
# 각 테이블에서 필요한 필드만 선택하는 쿼리 정의

# 식당 데이터 쿼리
RESTAURANT_QUERY = """
SELECT 
    r.id as restaurant_id,
    r.name,
    r.address,
    r.average_price,
    r.caution,
    r.convenience,
    r.expanded_days,
    r.is_deleted,
    r.operating_hour,
    r.phone_number,
    r.rating as score,
    r.review_count as review,
    r.time_range as duration_hours,
    rc.category_id
FROM restaurant r
JOIN restaurant_category rc ON r.id = rc.Restaurant_id
WHERE r.is_deleted = 0
"""

# 사용자 가격 범위 선호도 쿼리
USER_PREFERENCES_QUERY = """
SELECT 
    user_id,
    min_price,
    max_price
FROM user_preferences
WHERE deleted_at IS NULL
"""

# 사용자 카테고리 선호도 쿼리
USER_PREFERENCE_CATEGORIES_QUERY = """
SELECT 
    user_id,
    category_id
FROM user_preference_categories
WHERE deleted_at IS NULL
"""

# 찜 데이터 쿼리
LIKES_QUERY = """
SELECT 
    user_id,
    restaurant_id
FROM likes
WHERE deleted_at IS NULL
"""

# 예약 데이터 쿼리
RESERVATIONS_QUERY = """
SELECT 
    user_id,
    restaurant_id,
    status
FROM reservations
WHERE status = 'Complete'
AND deleted_at IS NULL
"""

# 필요에 따라 추가 쿼리를 정의할 수 있습니다.
# 예: 특정 기간 내 인기 식당 데이터, 최근 리뷰 데이터 등

# 쿼리 파라미터가 필요한 경우의 예시
def get_user_reservations_query(user_id):
    return f"""
    SELECT 
        user_id,
        restaurant_id,
        status
    FROM reservations
    WHERE user_id = {user_id}
    AND status = 'Complete'
    AND deleted_at IS NULL
    """