# app/services/preprocess/data_loader.py
# 임시로 디텍토리 안에 있는 데이터 활용 - 추후 BE에서 데이터 받아와서 테스트 할것.
# 2차 고도화때. 

import os
import json
import glob
import pandas as pd

def load_and_merge_json_files(directory: str) -> pd.DataFrame:
    """
    주어진 디렉터리 내의 모든 <카테고리명>_restaurants_table.json 파일들을 읽어
    데이터를 합쳐 하나의 DataFrame으로 반환합니다.
    """
    # 디렉터리 내의 모든 JSON 파일 경로 찾기 (예: 'korean_restaurants_table.json', 'japanese_restaurants_table.json' 등)
    file_pattern = os.path.join(directory, "*_restaurants_table.json")
    json_files = glob.glob(file_pattern)
    
    merged_data = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 각 파일의 데이터가 리스트 형태라면 합쳐줍니다.
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
    
    # 리스트를 DataFrame으로 변환
    return pd.DataFrame(merged_data)

# 사용 예시
directory_path = "data/crawling_2nd_data/json"
df_merged = load_and_merge_json_files(directory_path)

# 데이터 확인 (실제 컬럼명들을 출력해보면 전처리 시에 참고하기 좋습니다)
print("Merged DataFrame columns:", df_merged.columns.tolist())

# 이후 전처리 함수에 합쳐진 데이터를 전달하여 모델에 넣을 준비를 합니다.
# 예를 들어, preprocess_data() 함수가 있다면:
# df_processed = preprocess_data(df_merged)

