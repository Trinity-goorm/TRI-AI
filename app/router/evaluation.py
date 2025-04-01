# app/routers/evaluation.py

from fastapi import APIRouter, Depends, HTTPException
from app.schema.recommendation_schema import RecommendationEvaluationResponse
from app.services.evaluation.evaluator import evaluate_recommendation_model, evaluate_with_cross_validation
from app.services.evaluation.diversity_metrics import evaluate_recommendation_diversity
from app.services.model_trainer.hyperparameter_tuning import optimize_recommendation_parameters
from app.dependencies import get_globals_dict
from app.services.model_trainer import evaluate_model, optimize_recommendation_parameters 

# 각 알고리즘별 결과 평가
from app.services.model_trainer.recommenation.basic import generate_recommendations
from app.services.model_trainer.recommenation.hybrid import generate_hybrid_recommendations

import json
import logging
import numpy as np

router = APIRouter(
    prefix="/evaluate",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/basic", response_model=RecommendationEvaluationResponse)
async def evaluate_basic(globals_dict=Depends(get_globals_dict)):
    """기본 추천 시스템 평가 수행"""
    try:
        metrics = evaluate_recommendation_model(globals_dict)
        
        # 추가 다양성 지표 계산
        df_model = globals_dict.get("df_model")
        
        if "test_interactions" in globals_dict and df_model is not None:
            test_interactions = globals_dict.get("test_interactions")
            
            # 추천 결과를 사용자-식당 매핑 딕셔너리로 변환
            recommendations_dict = {}
            for user_id, metrics in globals_dict.get("recommendations_results", {}).items():
                recommendations_dict[user_id] = [r['restaurant_id'] for r in metrics.get('recommendations', [])]
            
            diversity_metrics = evaluate_recommendation_diversity(
                recommendations_dict, 
                user_history=test_interactions, 
                restaurant_data=df_model
            )
            
            # 기본 지표와 다양성 지표 병합
            metrics.update(diversity_metrics)
        
        return {"metrics": metrics, "status": "success"}
    
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cross-validation", response_model=RecommendationEvaluationResponse)
async def evaluate_with_cross_val(n_splits: int = 5, globals_dict=Depends(get_globals_dict)):
    """교차 검증 평가 수행"""
    try:
        metrics = evaluate_with_cross_validation(globals_dict, n_splits=n_splits)
        return {"metrics": metrics, "status": "success"}
    
    except Exception as e:
        logger.error(f"교차 검증 평가 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-params", response_model=dict)
async def optimize_parameters(n_trials: int = 30, timeout: int = 300, globals_dict=Depends(get_globals_dict)):
    """추천 시스템 파라미터 최적화"""
    try:
        df_model = globals_dict.get("df_model")
        user_features_df = globals_dict.get("user_features_df")
        model_features = globals_dict.get("model_features")
        
        if df_model is None or user_features_df is None or model_features is None:
            raise HTTPException(status_code=400, detail="필요한 데이터가 로드되지 않았습니다.")
        
        best_params = optimize_recommendation_parameters(
            df_model, 
            user_features_df, 
            model_features, 
            n_trials=n_trials,
            timeout=timeout
        )
        
        return {"best_parameters": best_params, "status": "success"}
    
    except Exception as e:
        logger.error(f"파라미터 최적화 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare-algorithms", response_model=dict)
async def compare_algorithms(globals_dict=Depends(get_globals_dict)):
    """서로 다른 추천 알고리즘 비교"""
    try:
        df_model = globals_dict.get("df_model")
        user_features_df = globals_dict.get("user_features_df")
        
        if df_model is None or user_features_df is None:
            raise HTTPException(status_code=400, detail="필요한 데이터가 로드되지 않았습니다.")
        
        # 샘플 사용자 선택
        sample_users = df_model['user_id'].sample(min(20, df_model['user_id'].nunique())).unique()
        
        basic_metrics = {}
        hybrid_metrics = {}
        
        for user_id in sample_users:
            # 기본 추천 알고리즘 평가
            stacking_reg = globals_dict.get("stacking_reg")
            scaler = globals_dict.get("scaler")
            model_features = globals_dict.get("model_features")
            
            if all([stacking_reg, scaler, model_features]):
                # 기본 추천 생성
                basic_result = json.loads(generate_recommendations(
                    df_model.copy(),
                    stacking_reg,
                    model_features,
                    user_id,
                    scaler,
                    user_features=user_features_df
                ))
                
                # 하이브리드 추천 생성
                hybrid_result = generate_hybrid_recommendations(
                    df_model.copy(),
                    df_model.copy(),
                    user_id,
                    n=15,
                    alpha=0.7
                )
                
                # 평가 지표 계산 및 저장 (여기서는 간소화를 위해 추천된 식당 수만 계산)
                basic_metrics[user_id] = {
                    'num_recommendations': len(basic_result.get('recommendations', [])),
                    'is_new_user': basic_result.get('is_new_user', True)
                }
                
                hybrid_metrics[user_id] = {
                    'num_recommendations': len(hybrid_result.get('recommendations', [])),
                    'is_new_user': hybrid_result.get('is_new_user', True)
                }
        
        # 결과 정리
        result = {
            'basic_algorithm': {
                'avg_recommendations': np.mean([m['num_recommendations'] for m in basic_metrics.values()]),
                'new_user_ratio': np.mean([m['is_new_user'] for m in basic_metrics.values()]),
                'coverage': len(basic_metrics) / len(sample_users) if sample_users.size > 0 else 0
            },
            'hybrid_algorithm': {
                'avg_recommendations': np.mean([m['num_recommendations'] for m in hybrid_metrics.values()]),
                'new_user_ratio': np.mean([m['is_new_user'] for m in hybrid_metrics.values()]),
                'coverage': len(hybrid_metrics) / len(sample_users) if sample_users.size > 0 else 0
            }
        }
        
        return {"comparison_results": result, "status": "success"}
    
    except Exception as e:
        logger.error(f"알고리즘 비교 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))