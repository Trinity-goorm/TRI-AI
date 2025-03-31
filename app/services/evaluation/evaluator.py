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
        # 파일에서 직접 테스트 데이터 로드
        logger.info("파일에서 직접 테스트 데이터 로드")
        try:
            from app.config import USER_DIR
            from pathlib import Path
            import json
            
            user_dir = Path(str(USER_DIR))
            
            # 가장 최근 파일 찾기 함수
            def get_latest_file(pattern):
                files = list(user_dir.glob(pattern))
                if not files:
                    return None
                return max(files, key=lambda x: x.stat().st_mtime)
            
            # 모든 관련 데이터 파일 찾기
            likes_file = get_latest_file('likes_*.json')
            reservations_file = get_latest_file('reservations_*.json')
            preferences_file = get_latest_file('user_preferences_*.json')
            recsys_file = get_latest_file('recsys_data_*.json')
            
            if not any([likes_file, reservations_file, preferences_file, recsys_file]):
                logger.error("상호작용 데이터 파일이 없습니다")
                return {"error": "테스트 데이터를 구성할 수 없습니다. 파일이 없습니다."}
            
            test_interactions = []
            
            # 찜 데이터 처리
            if likes_file:
                with open(likes_file, 'r') as f:
                    likes_data = json.load(f)
                logger.info(f"찜 데이터 로드 완료: {len(likes_data)}개 항목")
                
                for item in likes_data:
                    if 'user_id' in item and 'restaurant_id' in item:
                        test_interactions.append({
                            'user_id': item['user_id'],
                            'restaurant_id': item['restaurant_id'],
                            'interaction_type': 'like',
                            'weight': 1.0  # 찜은 명시적 관심 표현이므로 가중치 1.0
                        })
            
            # 예약 데이터 처리
            if reservations_file:
                with open(reservations_file, 'r') as f:
                    reservations_data = json.load(f)
                logger.info(f"예약 데이터 로드 완료: {len(reservations_data)}개 항목")
                
                for item in reservations_data:
                    if 'user_id' in item and 'restaurant_id' in item:
                        # 완료된 예약은 더 높은 가중치 부여
                        weight = 1.5 if item.get('status') == 'COMPLETED' else 1.0
                        test_interactions.append({
                            'user_id': item['user_id'],
                            'restaurant_id': item['restaurant_id'],
                            'interaction_type': 'reservation',
                            'weight': weight
                        })
            
            # 사용자 선호도 데이터 처리
            if preferences_file:
                with open(preferences_file, 'r') as f:
                    preferences_data = json.load(f)
                logger.info(f"선호도 데이터 로드 완료: {len(preferences_data)}개 항목")
                
                # 선호도 데이터는 직접적인 식당 상호작용이 아니므로 별도 처리가 필요할 수 있음
                # 예: 선호 카테고리 기반 평가 등
            
            # recsys 데이터 처리 (통합 데이터)
            if recsys_file:
                with open(recsys_file, 'r') as f:
                    recsys_data = json.load(f)
                logger.info(f"추천 시스템 데이터 로드 완료: {len(recsys_data)}개 항목")
                
                # recsys 데이터에서 추가 상호작용 추출
                for user_item in recsys_data:
                    if 'likes' in user_item and isinstance(user_item['likes'], list):
                        user_id = user_item.get('user_info', {}).get('user_id')
                        if not user_id:
                            continue
                        
                        for like in user_item['likes']:
                            if 'restaurant_id' in like:
                                test_interactions.append({
                                    'user_id': user_id,
                                    'restaurant_id': like['restaurant_id'],
                                    'interaction_type': 'like',
                                    'weight': 1.0
                                })
                    
                    if 'reservations' in user_item and isinstance(user_item['reservations'], list):
                        user_id = user_item.get('user_info', {}).get('user_id')
                        if not user_id:
                            continue
                        
                        for reservation in user_item['reservations']:
                            if 'restaurant_id' in reservation:
                                weight = 1.5 if reservation.get('status') == 'COMPLETED' else 1.0
                                test_interactions.append({
                                    'user_id': user_id,
                                    'restaurant_id': reservation['restaurant_id'],
                                    'interaction_type': 'reservation',
                                    'weight': weight
                                })
            
            if not test_interactions:
                logger.error("유효한 상호작용 데이터가 없습니다")
                return {"error": "유효한 테스트 데이터가 없습니다"}
            
            # 중복 제거 (같은 사용자-식당 쌍이 여러 개 있을 수 있음)
            unique_interactions = {}
            for interaction in test_interactions:
                key = (interaction['user_id'], interaction['restaurant_id'])
                if key not in unique_interactions or unique_interactions[key]['weight'] < interaction['weight']:
                    unique_interactions[key] = interaction
            
            # 데이터프레임 생성
            test_data = pd.DataFrame(list(unique_interactions.values()))
            logger.info(f"테스트 데이터 구성 완료: {len(test_data)}개 상호작용")
            
        except Exception as e:
            logger.error(f"테스트 데이터 로드 중 오류 발생: {e}", exc_info=True)
            return {"error": f"테스트 데이터 로드 중 오류 발생: {str(e)}"}
    
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