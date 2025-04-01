# app/router/recommendation_api.py

import os
import time
import json
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.config import RESTAURANTS_DIR, USER_DIR, FEEDBACK_DIR
from app.schema.recommendation_schema import UserData, RecommendationItem, CATEGORY_MAPPING
from app.services.preprocess.restaurant.data_loader import load_restaurant_json_files, load_user_json_files
from app.services.preprocess.restaurant.preprocessor import preprocess_data
from app.services.model_trainer import train_model
from app.services.model_trainer.recommendation import generate_recommendations
from app.services.preprocess.user.user_preprocess import user_preprocess_data  # 사용자 데이터 전처리 모듈 추가
from app.services.evaluation.evaluator import evaluate_recommendation_model
from typing import List, Dict, Any
from datetime import datetime

# 라우터 설정
logger = logging.getLogger("recommendation_api")
router = APIRouter()

# 글로벌 변수 초기화
globals_dict = {}
model_initializing = False  # 모델 초기화 상태를 추적하는 전역 변수
last_initialization_attempt = None  # 마지막 초기화 시도 시간

# 초기 데이터 로딩 및 모델 학습
def initialize_model(force=False):
    global globals_dict, model_initializing, last_initialization_attempt
    
    # 이미 초기화 중이면 중복 실행 방지
    if model_initializing and not force:
        logger.info("모델 초기화가 이미 진행 중입니다.")
        return False
    
    # 마지막 초기화 시도 기록
    current_time = datetime.now()
    last_initialization_attempt = current_time
    
    try:
        # 초기화 상태 설정
        model_initializing = True
        logger.info("모델 초기화 시작")
        
        # 데이터 디렉토리 및 파일 확인
        if not os.path.exists(str(RESTAURANTS_DIR)) or not os.path.exists(str(USER_DIR)):
            logger.error("필요한 데이터 디렉토리가 없습니다. 데이터 동기화가 완료되었는지 확인하세요.")
            model_initializing = False
            return False
            
        # 파일 존재 여부 확인
        restaurant_files = [f for f in os.listdir(str(RESTAURANTS_DIR)) if f.endswith('.json')]
        user_files = [f for f in os.listdir(str(USER_DIR)) if f.endswith('.json')]
        
        if not restaurant_files or not user_files:
            logger.error(f"데이터 파일이 충분하지 않습니다. 식당 파일: {len(restaurant_files)}개, 사용자 파일: {len(user_files)}개")
            model_initializing = False
            return False
        
        # 식당 데이터 로드
        df_raw = load_restaurant_json_files(str(RESTAURANTS_DIR))
        if df_raw.empty:
            logger.error("식당 데이터가 비어 있습니다.")
            model_initializing = False
            return False
            
        logger.info(f"식당 데이터 로드 완료: {len(df_raw)}개 식당")
        
        # 사용자 데이터 로드 및 전처리
        user_data_frames = load_user_json_files(str(USER_DIR))
        if not user_data_frames:
            logger.warning("사용자 데이터가 비어 있습니다. 기본 추천만 가능합니다.")
            # 사용자 데이터가 없어도 계속 진행 (기본 추천만 제공)
        else:
            logger.info(f"사용자 데이터 로드 완료: {len(user_data_frames)}개 파일")
        
        # 사용자 데이터 전처리 및 특성 추출 부분 수정
        try:
            # 전처리된 사용자 특성 파일 경로
            user_features_path = os.path.join(str(USER_DIR), "preprocessed_user_features.csv")
            
            # 이미 전처리된 파일이 있고 강제 초기화가 아니면 기존 파일 사용
            if os.path.exists(user_features_path) and not force:
                logger.info(f"기존 전처리된 사용자 특성 파일 로드: {user_features_path}")
                user_features_df = pd.read_csv(user_features_path)
            else:
                # 사용자 데이터 파일 경로 리스트 생성
                logger.info("사용자 데이터 전처리 시작")
                user_json_files = [
                    os.path.join(str(USER_DIR), f) for f in os.listdir(str(USER_DIR))
                    if f.endswith('.json') and not f.startswith('.')  # 숨김 파일 제외
                ]
                
                # 파일 경로 로깅
                logger.info(f"처리할 사용자 데이터 파일: {len(user_json_files)}개")
                for file in user_json_files[:5]:  # 첫 5개만 로깅
                    logger.debug(f"- {os.path.basename(file)}")
                
                # 파일 경로 리스트를 전달하여 전처리 수행
                user_features_df = user_preprocess_data(
                    user_json_files,  # 파일 경로 리스트 전달
                    save_path=user_features_path
                )
                logger.info(f"사용자 데이터 전처리 완료: {len(user_features_df)}명의 사용자 데이터")
            
            # 전역 변수에 사용자 특성 데이터 저장
            globals_dict["user_features_df"] = user_features_df
        except Exception as user_err:
            logger.error(f"사용자 데이터 전처리 중 오류 발생: {user_err}", exc_info=True)
            # 오류가 발생해도 계속 진행 (기본 추천은 가능하도록)
            globals_dict["user_features_df"] = None
        
        # 식당 데이터 전처리
        df_final = preprocess_data(df_raw)
        
        # 모델 학습
        model_dict = train_model(df_final)
        
        # 모델 관련 객체 저장
        globals_dict.update(model_dict)
        
        # 원본 사용자 데이터 저장 (필요시)
        globals_dict["user_data_frames"] = user_data_frames
        globals_dict["last_update"] = datetime.now()
        
        logger.info("모델 초기화 성공")
        # 초기화 완료 상태로 설정
        model_initializing = False
        return True
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}", exc_info=True)
        globals_dict = {}
        # 초기화 실패 상태로 설정
        model_initializing = False
        return False

# 모델 재초기화 엔드포인트 추가 (관리자용)
@router.post("/reload", response_model=Dict[str, str])
async def reload_model(force: bool = True):
    """모델을 강제로 다시 로드합니다. 관리자 전용 API입니다."""
    result = initialize_model(force=force)
    
    if result:
        return {"status": "success", "message": "모델 재초기화가 완료되었습니다."}
    else:
        raise HTTPException(
            status_code=503,
            detail="모델 재초기화에 실패했습니다. 로그를 확인하세요."
        )

# 모델 상태 확인 엔드포인트 추가 (상태 모니터링용)
@router.get("/status", response_model=Dict[str, Any])
async def check_model_status():
    """현재 모델의 초기화 상태를 확인합니다."""
    global globals_dict, model_initializing, last_initialization_attempt
    
    is_initialized = "stacking_reg" in globals_dict and "df_model" in globals_dict
    
    status = {
        "initialized": is_initialized,
        "initializing": model_initializing,
        "last_attempt": last_initialization_attempt.isoformat() if last_initialization_attempt else None,
        "last_update": globals_dict.get("last_update").isoformat() if globals_dict.get("last_update") else None
    }
    
    if is_initialized:
        # 기본 모델 통계 추가
        status.update({
            "restaurant_count": len(globals_dict.get("df_model", [])),
            "user_count": len(globals_dict.get("user_features_df", [])) if globals_dict.get("user_features_df") is not None else 0
        })
    
    return status

# 평가 지표 확인 엔드포인트 추가
@router.get("/evaluate", response_model=Dict[str, Any])
async def evaluate_model():
    """현재 추천 모델의 성능 지표를 계산합니다."""
    global globals_dict, model_initializing
    
    try:
        # 모델 초기화 상태 확인
        if not globals_dict or "stacking_reg" not in globals_dict or "df_model" not in globals_dict:
            raise HTTPException(
                status_code=503, 
                detail="모델이 초기화되지 않았습니다. 먼저 모델을 초기화하세요.",
                headers={"Retry-After": "30"}
            )
        
        # 모델 평가
        metrics = evaluate_recommendation_model(globals_dict)
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# recommend 함수 내부 수정
@router.post("",
             response_model=Dict[str, Any],
             responses={
                 200: {"description": "Success"},
                 400: {"description": "Bad Request"},
                 503: {"description": "Service Unavailable - Model not initialized yet"}
             })

async def recommend(user_data: UserData, background_tasks: BackgroundTasks):
    """사용자 데이터를 받아 개인화된 추천 결과를 생성하고, 결과를 파일로 저장합니다."""
    global model_initializing, globals_dict
    
    try:
        # 모델 초기화 상태 확인
        if not globals_dict or "stacking_reg" not in globals_dict or "df_model" not in globals_dict:
            # 모델이 초기화 중인지, 아니면 초기화에 실패했는지 구분
            if model_initializing:
                raise HTTPException(
                    status_code=503, 
                    detail="모델 초기화가 진행 중입니다. 잠시 후 다시 시도해주세요.",
                    headers={"Retry-After": "30"}  # 30초 후 재시도 권장
                )
            else:
                # 초기화 실패했다면 자동으로 다시 시도
                logger.info("모델이 초기화되지 않았습니다. 자동으로 초기화를 시도합니다.")
                initialize_result = initialize_model()
                
                if not initialize_result:
                    # 다시 시도해도 실패한 경우
                    raise HTTPException(
                        status_code=503, 
                        detail="모델 데이터가 초기화되지 않았습니다. 서버 관리자에게 문의하세요.",
                        headers={"Retry-After": "300"}  # 5분 후 재시도 권장
                    )
                # 초기화 성공했다면 계속 진행

        user_id = user_data.user_id
        preferred_categories = user_data.preferred_categories
    
        if not user_id or not preferred_categories:
            raise HTTPException(status_code=400, detail="user_id와 preferred_categories를 입력해주세요.")
    
        # 선호 카테고리 이름을 해당 번호로 변환
        preferred_ids = [CATEGORY_MAPPING.get(cat) for cat in preferred_categories if cat in CATEGORY_MAPPING]
        
        if not preferred_ids:
            raise HTTPException(status_code=400, detail="유효한 선호 카테고리를 입력해주세요.")
        
        df_model = globals_dict.get("df_model")
        
        # 사용자가 선호하는 카테고리의 식당만 필터링
        filtered_df = df_model[df_model["category_id"].isin(preferred_ids)].copy()
        if filtered_df.empty:
            raise HTTPException(status_code=400, detail="해당 선호 카테고리에 해당하는 식당 데이터가 없습니다.")
        
        # 전처리된 사용자 특성 데이터 가져오기
        user_features_df = globals_dict.get("user_features_df")
        
        # 추천 결과 생성 (개인화된 추천)
        result_json = generate_recommendations(
            filtered_df, 
            globals_dict["stacking_reg"], 
            globals_dict["model_features"], 
            user_id,
            globals_dict["scaler"],
            user_features=user_features_df  # 사용자 특성 데이터 전달 (없으면 None)
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
        logger.error(f"추천 API 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))