# app/services/preprocess/operating_days.py

def count_operating_days(expanded_days):
    """expanded_days 문자열을 바탕으로 영업일 수를 계산합니다."""
    day_order = ["월", "화", "수", "목", "금", "토", "일"]
    if not isinstance(expanded_days, str) or expanded_days.strip() == "":
        return None
    expanded_days = expanded_days.strip()
    if "~" in expanded_days:
        parts = expanded_days.split("~")
        if len(parts) != 2:
            return None
        start = parts[0].strip()
        end = parts[1].strip()
        if start in day_order and end in day_order:
            start_idx = day_order.index(start)
            end_idx = day_order.index(end)
            if start_idx <= end_idx:
                return end_idx - start_idx + 1
            else:
                # wrap-around, 예: "토~화"
                return (7 - start_idx) + (end_idx + 1)
        else:
            return None
    else:
        days = [d.strip() for d in expanded_days.split(",") if d.strip() != ""]
        return len(days)
