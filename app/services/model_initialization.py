import logging
import asyncio
from typing import Dict, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from app.config import UPLOAD_DIR
from app.services.preprocess.data_loader import load_restaurant_json_files, load_user_json_files
from app.services.preprocess.preprocessor import preprocess_data
from app.services.model_trainer import train_model

logger = logging.getLogger(__name__)

# 전역 변수로 모델 상태 관리
globals_dict = {}
is_initializing = False
last_initialization = None

async def initialize_model(force_reload=False):
    """모델 초기화 및 로딩 함수 (비동기 지원)"""
    global globals_dict, is_initializing, last_initialization
    
    # 이미 초기화 중이면 대기
    if is_initializing and not force_reload:
        logger.info("모델 초기화가 이미 진행 중입니다. 완료될 때까지 대기합니다.")
        while is_initializing:
            await asyncio.sleep(1)
        return globals_dict
    
    # 초기화 플래그 설정
    is_initializing = True
    
    try:
        logger.info("모델 초기화 시작")
        
        # ThreadPoolExecutor를 사용하여 데이터 로딩을 비동기로 실행
        with ThreadPoolExecutor() as executor:
            # 데이터 디렉토리
            directory = str(UPLOAD_DIR)
            
            # 1. 식당 데이터 로드
            future_restaurant = executor.submit(load_restaurant_json_files, directory)
            
            # 2. 사용자 관련 데이터 로드
            future_user_data = executor.submit(load_user_json_files, directory)
            
            # 결과 대기
            df_restaurant = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: future_restaurant.result()
            )
            
            user_data_frames = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: future_user_data.result()
            )
            
            # 3. 식당 데이터 전처리
            df_processed = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: preprocess_data(df_restaurant)
            )
            
            # 4. 모델 학습
            result_dict = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: train_model(df_processed)
            )
            
            # 5. 결과 저장
            globals_dict = result_dict
            
            # 6. 사용자 관련 데이터 추가
            globals_dict["user_data_frames"] = user_data_frames
            
            # 7. 모델 학습 완료 로깅
            logger.info("모델 초기화 완료")
            model_info = {
                "df_model_shape": globals_dict["df_model"].shape if "df_model" in globals_dict else None,
                "model_features": globals_dict.get("model_features"),
                "user_data_count": {k: len(v) for k, v in user_data_frames.items()}
            }
            logger.info(f"모델 정보: {model_info}")
            
            last_initialization = asyncio.get_event_loop().time()
            
    except Exception as e:
        logger.error(f"모델 초기화 중 오류: {str(e)}", exc_info=True)
    finally:
        is_initializing = False
    
    return globals_dict

def get_model():
    """모델 및 관련 데이터 가져오기 (동기 함수)"""
    if not globals_dict:
        logger.warning("모델이 초기화되지 않았습니다.")
    
    return globals_dict