import os
import time
import json
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.config import RESTAURANTS_DIR, USER_DIR, FEEDBACK_DIR
from app.schema.recommendation_schema import UserData, RecommendationItem, CATEGORY_MAPPING
from app.services.preprocess.restaurant.data_loader import load_restaurant_json_files, load_user_json_files
from app.services.preprocess.restaurant.preprocessor import preprocess_data
from app.services.model_trainer import train_model
from app.services.model_trainer.recommendation import generate_recommendations
from typing import List, Dict, Any

# 라우터 설정
logger = logging.getLogger("recommendation_api")
router = APIRouter()

# 글로벌 변수 초기화
globals_dict = {}

# 초기 데이터 로딩 및 모델 학습
def initialize_model():
    global globals_dict
    
    try:
        # 식당 데이터 로드
        df_raw = load_restaurant_json_files(str(RESTAURANTS_DIR))
        
        # 사용자 데이터 로드 (향후 사용)
        user_data_frames = load_user_json_files(str(USER_DIR))
        
        # 데이터 전처리
        df_final = preprocess_data(df_raw)
        
        # 모델 학습
        globals_dict = train_model(df_final)
        
        # 사용자 데이터 저장 (향후 개인화 추천에 사용)
        globals_dict["user_data_frames"] = user_data_frames
        
        logger.info("모델 초기화 성공")
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        globals_dict = {}

# 서버 시작 시 모델 초기화
initialize_model()

@router.post("",
             response_model=Dict[str, Any],  # 구체적인 응답 모델이 필요하다면 별도로 정의
             responses={
                 200: {"description": "Success"},
                 400: {"description": "Bad Request"},
                 500: {"description": "Internal Server Error"}
             })
async def recommend(user_data: UserData, background_tasks: BackgroundTasks):
    """사용자 데이터를 받아 추천 결과를 생성하고, 결과를 파일로 저장합니다."""
    try:
        user_id = user_data.user_id
        preferred_categories = user_data.preferred_categories
    
        if not user_id or not preferred_categories:
            raise HTTPException(status_code=400, detail="user_id와 preferred_categories를 입력해주세요.")
    
        # 선호 카테고리 이름을 해당 번호로 변환
        preferred_ids = [CATEGORY_MAPPING.get(cat) for cat in preferred_categories if cat in CATEGORY_MAPPING]
        
        if not preferred_ids:
            raise HTTPException(status_code=400, detail="유효한 선호 카테고리를 입력해주세요.")
        
        df_model = globals_dict.get("df_model")
        if df_model is None:
            raise HTTPException(status_code=500, detail="모델 데이터가 초기화되지 않았습니다.")
        
        filtered_df = df_model[df_model["category_id"].isin(preferred_ids)].copy()
        if filtered_df.empty:
            raise HTTPException(status_code=400, detail="해당 선호 카테고리에 해당하는 식당 데이터가 없습니다.")
        
        # 추천 결과 생성 (generate_recommendations는 JSON 문자열을 반환)
        result_json = generate_recommendations(
            filtered_df, 
            globals_dict["stacking_reg"], 
            globals_dict["model_features"], 
            user_id,
            globals_dict["scaler"]
        )

        # 백그라운드 작업으로 추천 결과 저장
        async def save_recommendation():
            try:
                timestamp = int(time.time())
                feedback_filename = f"recommendation_{user_id}_{timestamp}.json"
                feedback_filepath = os.path.join(str(FEEDBACK_DIR), feedback_filename)
                
                # 디렉토리는 이미 config에서 생성됨
                with open(feedback_filepath, "w", encoding="utf-8") as f:
                    f.write(result_json)
                logger.info(f"추천 결과가 {feedback_filepath}에 저장되었습니다.")
            except Exception as file_err:
                logger.error(f"추천 결과 저장 실패: {file_err}", exc_info=True)

        # 백그라운드 작업으로 추가
        background_tasks.add_task(save_recommendation)

        # JSON 문자열을 Python 객체로 변환하여 반환
        result_data = json.loads(result_json)
        return result_data
    
    except HTTPException:
        # 이미 생성된 HTTPException은 그대로 다시 발생시킴
        raise
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))