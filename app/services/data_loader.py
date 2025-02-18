# app/services/data_loader.py

import json
import pandas as pd

def load_data(json_path: str) -> pd.DataFrame:
    """
    주어진 JSON 파일 경로에서 데이터를 로드하여 DataFrame으로 반환합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)
