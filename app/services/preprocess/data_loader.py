# app/services/preprocess/data_loader.py
# 임시로 디텍토리 안에 있는 데이터 활용 - 추후 BE에서 데이터 받아와서 테스트 할것.
# 2차 고도화때. 

import os
import json
import glob
import pandas as pd

def load_and_merge_json_files(directory: str) -> pd.DataFrame:
    """
    주어진 디렉토리 내의 모든 <카테고리명>_restaurants_table.json 파일들을 읽어
    데이터를 합쳐 하나의 DataFrame으로 반환합니다.
    """
    # 디렉토리 내의 모든 JSON 파일 경로 찾기
    file_pattern = os.path.join(directory, "restaurants_table*.json")
    json_files = glob.glob(file_pattern)
    
    merged_data = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 각 파일의 데이터가 리스트 형태라면 병합, 그렇지 않으면 단일 데이터로 추가
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
    
    # 리스트를 DataFrame으로 변환하여 반환
    return pd.DataFrame(merged_data)

if __name__ == "__main__":
    directory_path = "data/crawling_2nd_data/json"
    df_merged = load_and_merge_json_files(directory_path)
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    print(df_merged.head())
