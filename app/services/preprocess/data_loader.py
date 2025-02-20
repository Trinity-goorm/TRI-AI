# app/services/preprocess/data_loader.py
# 임시로 디텍토리 안에 있는 데이터 활용 - 추후 BE에서 데이터 받아와서 테스트 할것.
# 2차 고도화때. 

import os
import json
import glob
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_merge_json_files(directory: str) -> pd.DataFrame:
    """
    주어진 디렉토리 내의 모든 <카테고리명>_restaurants_table.json 파일들을 읽어
    데이터를 합쳐 하나의 DataFrame으로 반환합니다.
    """
    # 디렉토리 내의 모든 JSON 파일 경로 찾기
    file_pattern = os.path.join(directory, "restaurants_table*.json")
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        logger.error(f"No JSON files found in directory: {directory}", exc_info=True)

    merged_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 데이터가 리스트인지 아닌지 확인
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}", exc_info=True)
            # 필요에 따라 계속 진행하거나 예외를 재발생할 수 있음.
            raise e
    
    try:
        df = pd.DataFrame(merged_data)
        logger.debug("json 데이터가 병합이 완료되었습니다.")
    except Exception as e:
        logger.error(f"Error converting merged data to DataFrame: {e}", exc_info=True)
        raise e  # 에러 발생 시 예외를 재발생시킵니다.
    
    return df
