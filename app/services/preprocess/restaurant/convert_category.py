# app/services/preprocess/restaurant/convert_category.py

import logging

logger = logging.getLogger(__name__)

category_mapping = {
    "중식": 1,
    "일식집": 2,
    "브런치카페": 3,
    "파스타": 4,
    "이탈리안": 5,
    "이자카야": 6,
    "한식집": 7,
    "치킨": 8,
    "스테이크": 9,
    "고깃집": 10,
    "다이닝바": 11,
    "오마카세": 12
}

def convert_category(cat):
    """카테고리 값을 매핑 딕셔너리를 사용하여 변환합니다."""
    try:
        if cat in category_mapping:
            return category_mapping[cat]
        return int(cat)
    except Exception as e:
        # 예외 발생 시, 로깅 후 원래 값을 반환하거나, 사용자 정의 예외 발생
        logging.getLogger(__name__).error(f"Error converting category {cat}: {e}", exc_info=True)
        raise e
