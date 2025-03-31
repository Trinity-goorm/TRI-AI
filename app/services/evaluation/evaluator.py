# app/services/evaluation/evaluator.py

import pandas as pd
import numpy as np
import logging
import json
from .metrics import calculate_rating_metrics, calculate_ranking_metrics
from app.services.model_trainer.recommendation import generate_recommendations

logger = logging.getLogger(__name__)

def evaluate_recommendation_model(model_dict, restaurant_df, user_features_df, test_data=None):
    """
    추천 모델 평가

    Args:
        model_dict: 모델 및 관련 객체 딕셔너리 (globals_dict)
        restaurant_df: 식당 데이터프레임
        user_features_df: 사용자 특성 데이터프레임
        test_data: 테스트 데이터 (없으면 찜, 예약 데이터에서 구성)

    Returns:
        dict: 평가 지표
    """
    if test_data is None:
        # 테스트 데이터가 없으면 찜과 예약 데이터에서 구성
        logger.info("테스트 데이터를 찜/예약 데이터에서 구성합니다")
        if "likes_df" in model_dict and "reservations_df" in model_dict:
            likes_df = model_dict["likes_df"]
            reservations_df = model_dict["reservations_df"]
            
            # 찜, 예약 데이터 결합
            test_data = pd.concat([
                likes_df[['user_id', 'restaurant_id']].assign(interaction_type='like'),
                reservations_df[['user_id', 'restaurant_id']].assign(interaction_type='reservation')
            ])
        else:
            logger.error("테스트 데이터를 구성할 수 없습니다. 찜/예약 데이터가 없습니다.")
            return {"error": "테스트 데이터를 구성할 수 없습니다."}

    # 랭킹 지표 계산을 위한 사용자별 추천 결과 생성
    recommendations_dict = {}
    # 대표적인 사용자 100명만 샘플링 (계산 시간 단축을 위해)
    sample_users = user_features_df['user_id'].sample(min(100, len(user_features_df))).tolist()
    
    for user_id in sample_users:
        try:
            # 사용자의 선호 카테고리 확인 (예: 모든 카테고리 선택)
            user_categories = []
            for i in range(1, 13):  # 카테고리 1~12
                cat_col = f"category_{i}"
                if cat_col in user_features_df.columns and user_features_df[user_features_df['user_id'] == user_id][cat_col].values[0] == 1:
                    user_categories.append(i)
            
            if not user_categories:
                # 선호 카테고리가 없으면 모든 카테고리 선택
                user_categories = list(range(1, 13))
            
            # 선호 카테고리의 식당만 필터링
            filtered_df = restaurant_df[restaurant_df['category_id'].isin(user_categories)].copy()
            
            # 추천 생성
            result_json = generate_recommendations(
                filtered_df, 
                model_dict["stacking_reg"], 
                model_dict["model_features"], 
                user_id,
                model_dict["scaler"],
                user_features=user_features_df
            )
            
            # JSON 문자열을 파이썬 객체로 변환
            result_data = json.loads(result_json)
            
            # 추천 식당 ID 리스트 추출
            recommended_items = [item['restaurant_id'] for item in result_data.get('recommendations', [])]
            recommendations_dict[user_id] = recommended_items
            
        except Exception as e:
            logger.error(f"사용자 {user_id}에 대한 추천 생성 중 오류 발생: {e}")
    
    # 랭킹 지표 계산
    ranking_metrics = calculate_ranking_metrics(recommendations_dict, test_data, k_values=[5, 10, 15])
    
    # 평점 예측 지표 계산 (평점 데이터가 있는 경우)
    rating_metrics = {"MAE": None, "RMSE": None}
    
    # 모든 지표 결합
    all_metrics = {**rating_metrics, **ranking_metrics}
    logger.info(f"평가 지표 계산 완료: {all_metrics}")
    return all_metrics