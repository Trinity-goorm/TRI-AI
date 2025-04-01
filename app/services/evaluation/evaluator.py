# app/services/evaluation/evaluator.py

import logging
import json
import pandas as pd
import numpy as np
from app.services.model_trainer.recommenation.basic import generate_recommendations
from app.services.evaluation.metrics import calculate_ranking_metrics
from app.services.evaluation.data_generation import (
    create_test_interactions, 
    create_stratified_train_test_split
)
from app.services.evaluation.utils import (
    validate_required_objects, 
    default_empty_metrics
)

logger = logging.getLogger(__name__)

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
        # 필요 객체 유효성 검증
        is_valid, missing_objects = validate_required_objects(globals_dict, df_model, user_features_df)
        
        if not is_valid:
            logger.error(f"모델 평가에 필요한 객체가 없습니다: {', '.join(missing_objects)}")
            return default_empty_metrics()
        
        # 인자로 전달된 값 우선 사용, 없으면 globals_dict에서 가져오기
        df_model = df_model or globals_dict.get("df_model")
        user_features_df = user_features_df or globals_dict.get("user_features_df")
        stacking_reg = globals_dict.get("stacking_reg")
        scaler = globals_dict.get("scaler")
        model_features = globals_dict.get("model_features")
        
        # 테스트 데이터 생성 방식 선택
        # 1. 층화 분할 방식 (실제 데이터 기반)
        if 'user_id' in df_model.columns:
            _, test_interactions = create_stratified_train_test_split(df_model)
        # 2. 랜덤 생성 방식 (완전 새 데이터)
        else:
            test_interactions = create_test_interactions(globals_dict)
        
        if isinstance(test_interactions, pd.DataFrame) and test_interactions.empty:
            logger.warning("상호작용 테스트 데이터가 없어 평가를 수행할 수 없습니다.")
            return default_empty_metrics()
        
        # 데이터 검증
        logger.info(f"생성된 테스트 데이터: {len(test_interactions)}개 상호작용")
        logger.info(f"고유 사용자 수: {test_interactions['user_id'].nunique()}")
        logger.info(f"고유 식당 수: {test_interactions['restaurant_id'].nunique()}")
        
        # 추천 결과 생성
        recommendations_dict = generate_recommendations_for_users(
            df_model, stacking_reg, model_features, 
            test_interactions['user_id'].unique(), scaler, user_features_df
        )
        
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

def generate_recommendations_for_users(df_model, stacking_reg, model_features, sample_users, scaler, user_features_df):
    """
    여러 사용자에 대한 추천 결과 생성
    
    Args:
        df_model: 식당 데이터
        stacking_reg: 학습된 스태킹 회귀 모델
        model_features: 모델 특성 리스트
        sample_users: 추천 대상 사용자 리스트
        scaler: 특성 스케일러
        user_features_df: 사용자 특성 데이터
        
    Returns:
        dict: 사용자별 추천 식당 ID 목록
    """
    recommendations_dict = {}
    
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
    
    return recommendations_dict

def evaluate_with_cross_validation(globals_dict, n_splits=5, k_values=[5, 10, 15]):
    """
    교차 검증을 통한 더 견고한 평가
    
    Args:
        globals_dict: 전역 변수 딕셔너리
        n_splits: 폴드 수
        k_values: 평가할 k 값 리스트
        
    Returns:
        dict: 평균 평가 지표
    """
    try:
        from sklearn.model_selection import KFold
        
        # 필요 객체 유효성 검증
        is_valid, missing_objects = validate_required_objects(globals_dict)
        
        if not is_valid:
            logger.error(f"평가에 필요한 객체가 없습니다: {', '.join(missing_objects)}")
            return default_empty_metrics()
        
        df_model = globals_dict.get("df_model")
        user_features_df = globals_dict.get("user_features_df")
        stacking_reg = globals_dict.get("stacking_reg")
        scaler = globals_dict.get("scaler")
        model_features = globals_dict.get("model_features")
        
        # 사용자별 데이터 그룹화
        if 'user_id' not in df_model.columns:
            logger.error("교차 검증을 위한 user_id 컬럼이 없습니다.")
            return default_empty_metrics()
            
        user_groups = df_model.groupby('user_id').apply(lambda x: x.index.tolist())
        users = list(user_groups.index)
        
        # 사용자를 기준으로 폴드 분할
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 평가 지표 저장
        all_metrics = []
        
        fold_idx = 1
        for train_users_idx, test_users_idx in kf.split(users):
            logger.info(f"폴드 {fold_idx}/{n_splits} 평가 중...")
            
            # 훈련/테스트 사용자 분할
            train_users = [users[i] for i in train_users_idx]
            test_users = [users[i] for i in test_users_idx]
            
            # 훈련/테스트 데이터 분할
            train_indices = [idx for user in train_users for idx in user_groups.get(user, [])]
            test_indices = [idx for user in test_users for idx in user_groups.get(user, [])]
            
            train_df = df_model.iloc[train_indices]
            test_df = df_model.iloc[test_indices]
            
            # 추천 생성 및 평가
            recommendations_dict = generate_recommendations_for_users(
                df_model, stacking_reg, model_features, test_users, scaler, user_features_df
            )
            
            # 평가 지표 계산
            if recommendations_dict:
                metrics = calculate_ranking_metrics(recommendations_dict, test_df, k_values=k_values)
                all_metrics.append(metrics)
                logger.info(f"폴드 {fold_idx} 평가 결과: {metrics}")
            else:
                logger.warning(f"폴드 {fold_idx}에서 추천 결과가 생성되지 않았습니다.")
            
            fold_idx += 1
        
        # 평균 지표 계산
        if not all_metrics:
            logger.error("모든 폴드에서 평가 지표 계산에 실패했습니다.")
            return default_empty_metrics()
            
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        avg_metrics.update({
            "MAE": None,
            "RMSE": None
        })
        
        logger.info(f"교차 검증 평균 평가 지표: {avg_metrics}")
        return avg_metrics
        
    except Exception as e:
        logger.error(f"교차 검증 평가 중 오류 발생: {e}", exc_info=True)
        return default_empty_metrics()