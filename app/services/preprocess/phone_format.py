# app/services/preprocess/phone_format.py
import pandas as pd

def format_phone(num):
    """전화번호를 정수 변환 후 문자열로 반환합니다."""
    if pd.isna(num):
        return ""
    try:
        num_int = int(num)
        return str(num_int)
    except:
        return str(num)
