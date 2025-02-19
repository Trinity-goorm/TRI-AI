# app/services/preprocess/convert_category.py

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
    if cat in category_mapping:
        return category_mapping[cat]
    try:
        return int(cat)
    except:
        return cat
