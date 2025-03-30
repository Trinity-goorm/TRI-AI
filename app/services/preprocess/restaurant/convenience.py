# app/services/preprocess/restaurant/convenience.py

from collections import Counter

def normalize_convenience(val):
    """편의시설 문자열을 리스트로 정규화합니다."""
    items = val.split('\n')
    normalized_items = []
    counter = Counter()
    for item in items:
        normalized_item = item.strip()
        if normalized_item == "정보 없음":
            normalized_item = "편의시설 정보 없음"
        normalized_items.append(normalized_item)
        counter[normalized_item] += 1
    return normalized_items, counter
