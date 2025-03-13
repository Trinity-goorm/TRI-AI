# SQL 쿼리 설정 파일
# 각 테이블에서 필요한 필드만 선택하는 쿼리 정의

# 식당 데이터 쿼리 (이미 정상 작동)
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
JOIN restaurant_category rc ON r.id = rc.restaurant_id
WHERE r.is_deleted = 0
"""

# 사용자 기본 정보 쿼리 (테이블명 수정)
USER_QUERY = """
SELECT 
    id as user_id, 
    sex, 
    status, 
    is_deleted,
    empty_ticket_count, 
    normal_ticket_count
FROM user
WHERE status = 'ACTIVE' AND is_deleted = 0
"""

# 사용자 가격 범위 선호도 쿼리 (테이블명 수정, deleted_at 조건 제거)
USER_PREFERENCES_QUERY = """
SELECT 
    user_id,
    min_price,
    max_price
FROM user_preference
"""

# 사용자 카테고리 선호도 쿼리 (테이블명 수정, deleted_at 조건 제거)
USER_PREFERENCE_CATEGORIES_QUERY = """
SELECT 
    user_preference_id,
    category_id
FROM user_preference_category
"""

# 찜 데이터 쿼리 (deleted_at 조건 제거)
LIKES_QUERY = """
SELECT 
    user_id,
    restaurant_id
FROM likes
"""

# 예약 데이터 쿼리 (테이블명 수정, deleted_at 조건 제거)
RESERVATIONS_QUERY = """
SELECT 
    user_id, 
    restaurant_id, 
    status,
    reservation_time_id, 
    reservation_date, 
    seat_type_id
FROM reservation
WHERE status = 'COMPLETED'
"""

# 파라미터가 있는 쿼리 함수도 수정
def get_user_reservations_query(user_id):
    """특정 사용자의 예약 데이터를 가져오는 쿼리"""
    return f"""
    SELECT 
        user_id,
        restaurant_id,
        status,
        reservation_time_id,
        reservation_date,
        seat_type_id
    FROM reservation
    WHERE user_id = {user_id}
    AND status = 'COMPLETED'
    """

def get_reservations_by_date_range_query(start_date, end_date):
    """특정 날짜 범위의 예약 데이터를 가져오는 쿼리"""
    return f"""
    SELECT 
        user_id,
        restaurant_id,
        status,
        reservation_time_id,
        reservation_date,
        seat_type_id
    FROM reservation
    WHERE reservation_date BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'COMPLETED'
    """