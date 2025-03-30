# app/services/preprocess/restaurant/caution.py
from collections import Counter

def normalize_caution(val):
    """유의사항 문자열을 리스트로 정규화합니다."""
    items = val.split(',')
    normalized_items = []
    counter = Counter()
    for item in items:
        normalized_item = item.strip()
        if normalized_item == "정보 없음":
            normalized_item = "유의사항 정보 없음"
        normalized_items.append(normalized_item)
        counter[normalized_item] += 1
    return normalized_items, counter
