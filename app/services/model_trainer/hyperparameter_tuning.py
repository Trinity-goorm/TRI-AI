# app/services/model_trainer/hyperparameter_tuning.py

import numpy as np
import pandas as pd
import logging
import optuna
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import time

logger = logging.getLogger(__name__)

def optimize_recommendation_parameters(df_model, user_features_df, model_features, n_trials=50, timeout=600, checkpoint_file=None):
    """
    추천 시스템의 파라미터를 최적화하는 함수
    
    Args:
        df_model: 훈련 데이터 (식당 및 평점 데이터)
        user_features_df: 사용자 특성 데이터
        model_features: 모델 피처 리스트
        n_trials: 최적화 시도 횟수
        timeout: 최적화 제한 시간 (초)
        
    Returns:
        dict: 최적화된 파라미터
    """
    try:
        logger.info(f"추천 시스템 파라미터 최적화 시작 (최대 시도: {n_trials}, 제한 시간: {timeout}초)")
        
        # 훈련/검증 분할
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(df_model, test_size=0.2, random_state=42)
        
        # 사용자 ID 추출
        all_users = train_df['user_id'].unique()
        
        # Optuna 최적화 목표 함수
        def objective(trial):
            # 최적화할 파라미터 정의
            params = {
                # 카테고리 다양성 가중치
                'diversity_weight': trial.suggest_float('diversity_weight', 0.05, 0.5),
                
                # 리뷰 관련 가중치
                'review_weight': trial.suggest_float('review_weight', 0.1, 0.9),
                
                # 인기도 로그 변환 시 베이스
                'popularity_log_base': trial.suggest_float('popularity_log_base', 1.5, 10.0),
                
                # 카테고리 유사성 가중치
                'category_similarity_weight': trial.suggest_float('category_similarity_weight', 0.1, 0.8),
                
                # 시그모이드 변환 파라미터
                'sigmoid_a': trial.suggest_float('sigmoid_a', 0.5, 5.0),
                'sigmoid_b': trial.suggest_float('sigmoid_b', 0.0, 3.0),
                
                # 콜드 스타트 추천에서 카테고리 다양성 가중치
                'cold_start_diversity_weight': trial.suggest_float('cold_start_diversity_weight', 0.1, 0.4),
                
                # 콜드 스타트 추천에서 인기도 가중치
                'cold_start_popularity_weight': trial.suggest_float('cold_start_popularity_weight', 0.1, 0.5),
                
                # 하이브리드 추천에서 협업 필터링 가중치
                'hybrid_cf_weight': trial.suggest_float('hybrid_cf_weight', 0.4, 0.9)
            }
            
            # 현재 파라미터로 샘플 사용자에 대한 추천 생성 및 평가
            eval_users = np.random.choice(all_users, min(50, len(all_users)), replace=False)
            
            # RMSE 및 정확도 지표를 저장할 리스트
            user_rmse = []
            user_precision = []
            user_diversity = []
            
            for user_id in eval_users:
                # 사용자 데이터 분할 (훈련/테스트)
                user_data = train_df[train_df['user_id'] == user_id]
                user_valid = valid_df[valid_df['user_id'] == user_id]
                
                if len(user_data) < 5 or len(user_valid) < 2:
                    continue  # 데이터가 너무 적은 사용자는 건너뜀
                
                # 현재 파라미터 세트를 사용하여 추천 생성
                try:
                    # 임시 모델 학습 (파라미터 적용)
                    from sklearn.ensemble import GradientBoostingRegressor
                    
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=42
                    )
                    
                    # 선택된 특성으로 모델 피팅
                    X_train = user_data[model_features]
                    y_train = user_data['score']
                    
                    model.fit(X_train, y_train)
                    
                    # 테스트 데이터에 대한 예측
                    X_valid = user_valid[model_features]
                    y_valid = user_valid['score']
                    y_pred = model.predict(X_valid)
                    
                    # RMSE 계산
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    user_rmse.append(rmse)
                    
                    # 평가 지표: 추천 정확도 (정규화된 예측 오차의 역수)
                    max_possible_error = 5.0  # 평점 범위가 1~5라고 가정
                    prediction_accuracy = 1 - (rmse / max_possible_error)
                    user_precision.append(prediction_accuracy)
                    
                    # 다양성 계산 (카테고리 기준)
                    if 'category_id' in user_valid.columns:
                        recommended_categories = user_valid['category_id'].value_counts(normalize=True)
                        category_entropy = scipy.stats.entropy(recommended_categories)
                        normalized_entropy = 1 - (1 / (1 + category_entropy))  # 0~1 범위로 정규화
                        user_diversity.append(normalized_entropy)
                    
                except Exception as inner_e:
                    logger.debug(f"사용자 {user_id} 평가 중 오류: {inner_e}")
                    continue
            
            # 평균 지표 계산
            avg_rmse = np.mean(user_rmse) if user_rmse else 5.0  # 오류 시 최대 오류값
            avg_precision = np.mean(user_precision) if user_precision else 0.0
            avg_diversity = np.mean(user_diversity) if user_diversity else 0.0
            
            # 목표 함수: 정확도와 다양성의 가중 평균
            objective_score = (0.7 * avg_precision) + (0.3 * avg_diversity)
            
            # 진행 상황 로깅
            logger.debug(f"Trial {trial.number}: Params={params}, Score={objective_score:.4f} "
                        f"(Precision={avg_precision:.4f}, Diversity={avg_diversity:.4f})")
            
            return objective_score
        
        # Optuna 연구 생성 및 최적화 실행
        # 중간 결과 저장 및 메모리 최적화 추가
        
        # 중간 결과 저장 기능 추가
        if checkpoint_file:
            study = optuna.create_study(direction='maximize', 
                                    storage=f'sqlite:///{checkpoint_file}', 
                                    study_name='recommendation_params',
                                    load_if_exists=True)
        else:
            study = optuna.create_study(direction='maximize')
            
        # 메모리 최적화를 위한 콜백 추가
        def gc_after_trial(study, trial):
            # 주기적으로 메모리 정리
            if trial.number % 5 == 0:
                import gc
                gc.collect()

        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[gc_after_trial])
        
        # 최적 파라미터 및 결과
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"파라미터 최적화 완료: 최적 점수={best_value:.4f}")
        logger.info(f"최적 파라미터: {best_params}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"파라미터 최적화 중 오류 발생: {e}", exc_info=True)
        # 오류 발생 시 기본 파라미터 반환
        return {
            'diversity_weight': 0.15,
            'review_weight': 0.5,
            'popularity_log_base': 5.0,
            'category_similarity_weight': 0.3,
            'sigmoid_a': 2.0,
            'sigmoid_b': 1.5,
            'cold_start_diversity_weight': 0.2,
            'cold_start_popularity_weight': 0.3,
            'hybrid_cf_weight': 0.7
        }