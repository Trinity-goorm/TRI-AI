import numpy as np
import pandas as pd
import json
import logging
from app.services.model_trainer.recommendation import generate_recommendations
from app.services.evaluation.metrics import calculate_ranking_metrics

logger = logging.getLogger(__name__)

def create_test_interactions(globals_dict):
    """
    더 현실적인 사용자-식당 상호작용 테스트 데이터 생성
    
    Args:
        globals_dict: 전역 변수 딕셔너리
    
    Returns:
        pd.DataFrame: 상호작용 테스트 데이터
    """
    try:
        df_model = globals_dict.get("df_model")
        user_features_df = globals_dict.get("user_features_df")
        
        if df_model is None or user_features_df is None:
            logger.warning("테스트 데이터 생성에 필요한 데이터가 부족합니다.")
            return pd.DataFrame(columns=['user_id', 'restaurant_id', 'score'])
        
        # 모든 사용자와 식당 데이터 가져오기
        all_users = user_features_df['user_id'].tolist()
        all_restaurants = df_model['restaurant_id'].unique().tolist()
        
        # 사용자별 프로필 특성 (있다면) 활용
        user_preferences = {}
        
        if 'preferred_category' in user_features_df.columns:
            for _, user_row in user_features_df.iterrows():
                user_id = user_row['user_id']
                preferred_category = user_row.get('preferred_category')
                user_preferences[user_id] = preferred_category
        
        # 테스트 상호작용 생성
        test_interactions = []
        
        for user_id in all_users[:50]:  # 테스트를 위해 50명 사용자만 선택
            # 사용자별 특성에 맞는 식당 선택
            user_preferred_category = user_preferences.get(user_id)
            
            if user_preferred_category:
                # 사용자 선호 카테고리가 있으면 해당 카테고리의 식당들 중에서 선택
                category_restaurants = df_model[
                    df_model['category_id'] == user_preferred_category
                ]['restaurant_id'].tolist()
                
                if category_restaurants:
                    # 선호 카테고리 식당 선택 (60%)
                    preferred_count = max(1, int(np.random.randint(3, 6) * 0.6))
                    preferred_restaurants = np.random.choice(
                        category_restaurants, 
                        size=min(preferred_count, len(category_restaurants)),
                        replace=False
                    )
                    
                    # 기타 랜덤 식당 선택 (40%)
                    other_restaurants = list(set(all_restaurants) - set(category_restaurants))
                    other_count = np.random.randint(1, 3)
                    other_selected = np.random.choice(
                        other_restaurants,
                        size=min(other_count, len(other_restaurants)),
                        replace=False
                    )
                    
                    selected_restaurants = np.concatenate([preferred_restaurants, other_selected])
                else:
                    # 선호 카테고리 식당이 없으면 랜덤 선택
                    selected_restaurants = np.random.choice(
                        all_restaurants, 
                        size=np.random.randint(3, 6),
                        replace=False
                    )
            else:
                # 사용자 선호도 정보가 없으면 랜덤 선택
                selected_restaurants = np.random.choice(
                    all_restaurants, 
                    size=np.random.randint(3, 6),
                    replace=False
                )
            
            # 선택된 식당에 대한 가상 평점 생성 (3.0 ~ 5.0)
            for restaurant_id in selected_restaurants:
                # 선호 카테고리 식당은 더 높은 평점 확률
                if user_preferred_category and restaurant_id in category_restaurants:
                    score = np.random.uniform(4.0, 5.0)  # 선호 카테고리 식당은 평점 높게
                else:
                    score = np.random.uniform(3.0, 5.0)  # 기타 식당은 일반적인 분포
                
                test_interactions.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant_id,
                    'score': round(score, 1)  # 소수점 첫째 자리까지
                })
        
        return pd.DataFrame(test_interactions)
    
    except Exception as e:
        logger.error(f"테스트 데이터 생성 중 오류: {e}", exc_info=True)
        return pd.DataFrame(columns=['user_id', 'restaurant_id', 'score'])

def evaluate_recommendation_model(globals_dict, df_model=None, user_features_df=None):
    """
    추천 모델 평가 함수
    
    Args:
        globals_dict: 전역 변수 딕셔너리
        df_model: 식당 데이터 (옵션)
        user_features_df: 사용자 특성 데이터 (옵션)
    
    Returns:
        dict: 추천 시스템 평가 지표
    """
    try:
        # 인자로 전달된 값 우선 사용, 없으면 globals_dict에서 가져오기
        df_model = df_model or globals_dict.get("df_model")
        user_features_df = user_features_df or globals_dict.get("user_features_df")
        
        # 필요한 객체 추출
        stacking_reg = globals_dict.get("stacking_reg")
        scaler = globals_dict.get("scaler")
        model_features = globals_dict.get("model_features")
        
        # DataFrame 객체 유효성 검사 수정
        missing_objects = []
        if stacking_reg is None:
            missing_objects.append("stacking_reg")
        if scaler is None:
            missing_objects.append("scaler")
        if model_features is None:
            missing_objects.append("model_features")
        if df_model is None or (isinstance(df_model, pd.DataFrame) and df_model.empty):
            missing_objects.append("df_model")
        if user_features_df is None or (isinstance(user_features_df, pd.DataFrame) and user_features_df.empty):
            missing_objects.append("user_features_df")
        
        if missing_objects:
            logger.error(f"모델 평가에 필요한 객체가 없습니다: {', '.join(missing_objects)}")
            return default_empty_metrics()
        
        # 테스트 상호작용 데이터 생성
        test_interactions = create_test_interactions({
            "df_model": df_model, 
            "user_features_df": user_features_df
        })
        
        if isinstance(test_interactions, pd.DataFrame) and test_interactions.empty:
            logger.warning("상호작용 테스트 데이터가 없어 평가를 수행할 수 없습니다.")
            return default_empty_metrics()
        
        # 데이터 검증
        logger.info(f"생성된 테스트 데이터: {len(test_interactions)}개 상호작용")
        logger.info(f"고유 사용자 수: {test_interactions['user_id'].nunique()}")
        logger.info(f"고유 식당 수: {test_interactions['restaurant_id'].nunique()}")
        
        # 추천 결과 생성
        recommendations_dict = {}
        sample_users = test_interactions['user_id'].unique()
        
        for user_id in sample_users:
            try:
                # 추천 결과 생성
                # 주의: user_id가 문자열이면 정수로 변환
                user_id_for_rec = int(user_id) if isinstance(user_id, str) else user_id
                
                result_json = generate_recommendations(
                    df_model.copy(), 
                    stacking_reg, 
                    model_features, 
                    user_id_for_rec, 
                    scaler, 
                    user_features=user_features_df
                )
                
                # JSON 파싱
                try:
                    result_data = json.loads(result_json)
                except json.JSONDecodeError:
                    logger.error(f"JSON 파싱 오류: {result_json[:100]}...")
                    continue
                
                # 추천 식당 ID 리스트 추출
                recommended_items = [item.get('restaurant_id') for item in result_data.get('recommendations', [])]
                
                # ID 타입 일관성 확인
                recommended_items = [int(item) if not isinstance(item, int) else item for item in recommended_items]
                
                # 유효한 추천 결과만 저장
                if recommended_items:
                    recommendations_dict[user_id] = recommended_items
                else:
                    logger.warning(f"사용자 {user_id}에 대한 추천 결과가 없습니다.")
            
            except Exception as e:
                logger.error(f"사용자 {user_id} 추천 생성 중 오류: {e}", exc_info=True)
        
        # 추천 결과 검증
        if not recommendations_dict:
            logger.error("모든 사용자에 대한 추천 생성 실패")
            return default_empty_metrics()
        
        logger.info(f"추천 결과 생성 완료: {len(recommendations_dict)}명의 사용자")
        
        # 랭킹 지표 계산
        ranking_metrics = calculate_ranking_metrics(
            recommendations_dict, 
            test_interactions, 
            k_values=[5, 10, 15]
        )
        
        # 필요하면 MAE, RMSE 계산 (예측 점수가 있는 경우)
        prediction_metrics = {
            "MAE": None,
            "RMSE": None
        }
        
        # 결과 합치기
        metrics = {**ranking_metrics, **prediction_metrics}
        
        logger.info(f"평가 지표: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {e}", exc_info=True)
        return default_empty_metrics()

def default_empty_metrics():
    """기본 빈 지표 반환"""
    return {
        "MAE": None,
        "RMSE": None,
        "Precision@5": 0,
        "Recall@5": 0,
        "NDCG@5": 0,
        "Hit_Rate@5": 0,
        "Precision@10": 0,
        "Recall@10": 0,
        "NDCG@10": 0,
        "Hit_Rate@10": 0,
        "Precision@15": 0,
        "Recall@15": 0,
        "NDCG@15": 0,
        "Hit_Rate@15": 0
    }