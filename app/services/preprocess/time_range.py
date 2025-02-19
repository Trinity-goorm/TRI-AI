# app/services/preprocess/time_range.py

def extract_open_time(time_range):
    """시간 범위에서 open_time 추출"""
    if isinstance(time_range, str) and " ~ " in time_range:
        return time_range.split(" ~ ")[0]
    return None

def extract_close_time(time_range):
    """시간 범위에서 close_time 추출"""
    if isinstance(time_range, str) and " ~ " in time_range:
        return time_range.split(" ~ ")[1]
    return None

def convert_to_minutes(time_str):
    """HH:MM 형식의 시간을 분으로 변환 (24:00은 1440분)"""
    if not time_str:
        return None
    if time_str == "24:00":
        return 1440
    try:
        h, m = map(int, time_str.split(":"))
        return h * 60 + m
    except:
        return None

def compute_duration(open_minutes, close_minutes):
    """영업시간(분)을 계산"""
    if open_minutes is None or close_minutes is None:
        return None
    if close_minutes >= open_minutes:
        return close_minutes - open_minutes
    else:
        return (24 * 60 - open_minutes) + close_minutes
