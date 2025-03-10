import os
import json
import glob
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def load_restaurant_json_files(directory: str) -> pd.DataFrame:
    """
    주어진 디렉토리 내의 모든 <카테고리명>_restaurants_table.json 파일들을 읽어
    데이터를 합쳐 하나의 DataFrame으로 반환합니다.
    """
    # 디렉토리 내의 모든 JSON 파일 경로 찾기
    file_pattern = os.path.join(directory, "restaurants_table*.json")
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        logger.error(f"No restaurant JSON files found in directory: {directory}")
        raise FileNotFoundError(f"No restaurant JSON files found in directory: {directory}")

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
        logger.debug(f"식당 데이터 병합 완료: {len(df)}개 항목")
    except Exception as e:
        logger.error(f"Error converting restaurant data to DataFrame: {e}", exc_info=True)
        raise e  # 에러 발생 시 예외를 재발생시킵니다.
    
    return df

def load_user_json_files(directory: str) -> Dict[str, pd.DataFrame]:
    """
    사용자 관련 데이터 파일들을 로드하여 DataFrame 딕셔너리로 반환합니다.
    
    로드되는 데이터:
    - user_preferences: 사용자 가격 범위 선호도 (min_price, max_price)
    - user_preference_categories: 사용자 카테고리 선호도 (category_id)
    - likes: 사용자가 찜한 식당 정보 (restaurant_id)
    - reservations: 사용자 예약 정보 (restaurant_id, status)
    """
    try:
        dir_path = Path(directory)
        user_data_frames = {}
        
        # 1. 사용자 가격 범위 선호도 데이터
        pref_files = [f for f in dir_path.glob('user_preferences_*.json')]
        if pref_files:
            latest_file = max(pref_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_data_frames["user_preferences"] = pd.DataFrame(data)
            logger.info(f"사용자 가격 범위 선호도 데이터 로드 완료: {len(data)}개 항목")
        else:
            logger.warning("사용자 가격 범위 선호도 데이터 파일을 찾을 수 없습니다.")
        
        # 2. 사용자 카테고리 선호도 데이터
        cat_pref_files = [f for f in dir_path.glob('user_preference_categories_*.json')]
        if cat_pref_files:
            latest_file = max(cat_pref_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_data_frames["user_preference_categories"] = pd.DataFrame(data)
            logger.info(f"사용자 카테고리 선호도 데이터 로드 완료: {len(data)}개 항목")
        else:
            logger.warning("사용자 카테고리 선호도 데이터 파일을 찾을 수 없습니다.")
        
        # 3. 찜 데이터
        like_files = [f for f in dir_path.glob('likes_*.json')]
        if like_files:
            latest_file = max(like_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_data_frames["likes"] = pd.DataFrame(data)
            logger.info(f"찜 데이터 로드 완료: {len(data)}개 항목")
        else:
            logger.warning("찜 데이터 파일을 찾을 수 없습니다.")
        
        # 4. 예약 데이터
        reservation_files = [f for f in dir_path.glob('reservations_*.json')]
        if reservation_files:
            latest_file = max(reservation_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_data_frames["reservations"] = pd.DataFrame(data)
            logger.info(f"예약 데이터 로드 완료: {len(data)}개 항목")
            
            # 완료된 예약만 필터링
            if "reservations" in user_data_frames and "status" in user_data_frames["reservations"].columns:
                user_data_frames["reservations"] = user_data_frames["reservations"][
                    user_data_frames["reservations"]["status"] == "Complete"
                ]
                logger.info(f"완료된 예약만 필터링: {len(user_data_frames['reservations'])}개 항목")
        else:
            logger.warning("예약 데이터 파일을 찾을 수 없습니다.")
        
        # 사용자 관련 데이터 수 확인
        if not user_data_frames:
            logger.warning("사용자 관련 데이터가 없습니다. 기본 추천만 제공됩니다.")
        
        return user_data_frames
    
    except Exception as e:
        logger.error(f"사용자 데이터 로드 중 오류 발생: {str(e)}", exc_info=True)
        return {}

def get_latest_file(directory: Path, prefix: str) -> Path:
    """특정 접두사를 가진 파일 중 가장 최신 파일 반환"""
    files = list(directory.glob(f"{prefix}*.json"))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)