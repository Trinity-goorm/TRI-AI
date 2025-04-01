# app/services/model_trainer/train_model.py

from .data_preparation import prepare_data, impute_and_clip, scale_and_split
from .model_training import train_ridge, train_rf, train_xgb, train_lgb, train_cat, train_mlp, train_stacking
from .model_evaluation import evaluate_model
import numpy as np
import logging
import warnings
import atexit
import shutil
import os
import tempfile
from joblib import parallel_backend

# joblib 경고 필터링
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", message="There appear to be")

logger = logging.getLogger(__name__)

def enhance_feature_engineering(df_prepared):
    """
    향상된 특성 엔지니어링 함수
    
    Args:
        df_prepared: 기본 전처리가 완료된 데이터프레임
    
    Returns:
        df_prepared: 향상된 특성이 추가된 데이터프레임
    """
    try:
        logger.debug("향상된 특성 엔지니어링 시작...")
        
        # 1. 카테고리별 희소성 계산
        category_counts = df_prepared['category_id'].value_counts()
        total_restaurants = len(df_prepared)
        category_sparsity = 1 - (category_counts / total_restaurants)
        
        # 2. 카테고리 다양성 특성 추가
        df_prepared['category_diversity_score'] = df_prepared['category_id'].map(
            category_sparsity.to_dict()
        ).fillna(0)
        
        # 3. 카테고리 인기도 측정
        category_avg_rating = df_prepared.groupby('category_id')['score'].mean()
        category_avg_reviews = df_prepared.groupby('category_id')['review'].mean()
        
        df_prepared['category_avg_rating'] = df_prepared['category_id'].map(
            category_avg_rating.to_dict()
        ).fillna(df_prepared['score'].mean())
        
        df_prepared['category_avg_reviews'] = df_prepared['category_id'].map(
            category_avg_reviews.to_dict()
        ).fillna(df_prepared['review'].mean())
        
        # 4. 식당 인기도 점수
        df_prepared['popularity_score'] = (
            df_prepared['score'] * 0.6 +
            np.log1p(df_prepared['review']) * 0.4
        )
        
        # 5. 리뷰 기반 상호작용 강도 - 가중치 최적화
        df_prepared['interaction_intensity'] = (
            df_prepared['review'] * 0.4 + 
            df_prepared['duration_hours'] * 0.25 + 
            np.log1p(df_prepared['review']) * 0.35
        )
        
        # 6. 식당 대비 카테고리 성능 (식당이 해당 카테고리 내에서 얼마나 좋은지)
        df_prepared['rating_vs_category'] = df_prepared['score'] - df_prepared['category_avg_rating']
        df_prepared['reviews_vs_category'] = df_prepared['review'] / (df_prepared['category_avg_reviews'] + 1)
        
        # 7. 복합 특성들
        # 복합 평점: 카테고리 다양성과 평점 결합
        df_prepared['composite_rating'] = (
            df_prepared['score'] * 0.7 + 
            df_prepared['category_diversity_score'] * 0.2 +
            df_prepared['rating_vs_category'] * 0.1
        )
        
        # 복합 인기도: 리뷰 수와 운영 시간 결합
        df_prepared['engagement_score'] = (
            np.log1p(df_prepared['review']) * 0.7 +
            (df_prepared['duration_hours'] / 24) * 0.3
        )
        
        # 8. 식당 특성과 카테고리 인기도의 상호작용
        df_prepared['category_quality_interaction'] = (
            df_prepared['score'] * df_prepared['category_avg_rating']
        )
        
        # 9. 리뷰 밀도 (시간당 리뷰 수)
        df_prepared['review_density'] = df_prepared['review'] / (df_prepared['duration_hours'] + 1)
        
        # 10. 편의 시설 복합 점수
        convenience_cols = [col for col in df_prepared.columns if col.startswith('conv_')]
        if convenience_cols:
            df_prepared['convenience_score'] = df_prepared[convenience_cols].sum(axis=1)
        
        # 11. 리뷰 영향력 비율
        global_avg_rating = df_prepared['score'].mean()
        df_prepared['bayesian_rating'] = (
            (df_prepared['review'] * df_prepared['score'] + 10 * global_avg_rating) /
            (df_prepared['review'] + 10)
        )
        
        logger.debug("향상된 특성 엔지니어링 완료")
        return df_prepared
        
    except Exception as e:
        logger.error(f"향상된 특성 엔지니어링 중 오류 발생: {e}", exc_info=True)
        # 오류가 발생해도 원본 데이터프레임 반환
        return df_prepared
    
def train_model(df_final):
    """
    전처리된 DataFrame을 입력받아 모델 학습과 평가, 앙상블 모델 학습까지 수행합니다.
    최종적으로 학습에 사용된 스케일러, 앙상블 모델, 모델 피처 목록, 사용 데이터 등을 딕셔너리 형태로 반환합니다.
    """
    try:
        # 1. 데이터 준비: 필수 컬럼 확인 및 결측치 제거
        required_cols = ['duration_hours', 'conv_WIFI', 'conv_주차', 'caution_예약가능', 'category_id', 'review', 'score']
        df_prepared = prepare_data(df_final, required_cols)
        logger.debug("학습 데이터 준비가 완료되었습니다.")
    except Exception as e:
        logger.error(f"train_model - 데이터 준비 오류: {e}", exc_info=True)
        raise e
        
    # 2. 결측치 보완: IterativeImputer를 이용
    try:
        impute_cols = ['score', 'review']
        df_prepared = impute_and_clip(df_prepared, impute_cols)
        logger.debug("평점 미제공 식당에 대한 가상 평점 계산이 완료되었습니다.")
    except Exception as e:
        logger.error(f"train_model - imputation 오류: {e}", exc_info=True)
        raise e

    try:
        # duration_hours가 문자열인 경우 숫자로 변환
        if df_prepared['duration_hours'].dtype == 'object':
            # "12:00 ~ 24:00" 형식에서 시간 차이 계산
            def extract_hours_diff(time_str):
                try:
                    if isinstance(time_str, str) and '~' in time_str:
                        start, end = time_str.split('~')
                        start_hour = float(start.strip().split(':')[0])
                        end_hour = float(end.strip().split(':')[0])
                        if end_hour < start_hour:  # 예: 22:00 ~ 02:00
                            return (24 - start_hour) + end_hour
                        else:
                            return end_hour - start_hour
                    else:
                        return 8.0  # 기본값
                except:
                    return 8.0  # 변환 실패 시 기본값
            
            df_prepared['duration_hours'] = df_prepared['duration_hours'].apply(extract_hours_diff)
        
        # 기본 피처 생성
        df_prepared['log_review'] = np.log(df_prepared['review'] + 1)
        df_prepared['review_duration'] = df_prepared['review'] * df_prepared['duration_hours']
        
        # 향상된 특성 엔지니어링 적용
        df_prepared = enhance_feature_engineering(df_prepared)
        
        logger.debug("특성 엔지니어링이 완료되었습니다.")
    except Exception as e:
        logger.error(f"train_model - 피처 생성 오류: {e}", exc_info=True)
        raise e

    # 이미 전처리 과정에서 로그 변환, 상호작용 등 피처 엔지니어링이 이루어졌다고 가정
    # 3. 모델 학습에 사용할 피처와 타깃을 설정합니다.
    try:
        # 추가된 특성들을 모델 피처에 포함
        model_features = [
            'review', 'duration_hours', 'conv_WIFI', 'conv_주차', 
            'caution_예약가능', 'log_review', 'review_duration',
            'category_diversity_score', 'interaction_intensity', 
            'composite_rating', 'popularity_score', 'rating_vs_category',
            'reviews_vs_category', 'engagement_score', 'category_quality_interaction',
            'review_density', 'bayesian_rating'
        ]
        
        # 존재하는 피처만 선택 (일부 특성이 생성되지 않을 수 있음)
        model_features = [f for f in model_features if f in df_prepared.columns]
        
        target = 'score'
        X = df_prepared[model_features]
        y = df_prepared[target]
        logger.debug(f"Feature 및 타깃 설정이 완료되었습니다: {model_features}")
    except Exception as e:
        logger.error(f"train_model - Feature 및 타깃 설정 오류: {e}", exc_info=True)
        raise e
        
    # 4. 특성 스케일링 및 데이터 분할
    try:
        scaler, X_train, X_test, y_train, y_test = scale_and_split(X, y)
        logger.debug(f"Type of scaler in train_model: {type(scaler)}")  # 이 로그가 <class 'sklearn.preprocessing._data.StandardScaler'>로 나와야 함.
    except Exception as e:
        logger.error(f"train_model - 스케일링/데이터 분할 오류: {e}", exc_info=True)
        raise e

    try:
        # 5. 개별 모델 학습 (하이퍼파라미터 튜닝 포함)
        best_ridge = train_ridge(X_train, y_train)
        best_rf = train_rf(X_train, y_train)
        best_xgb = train_xgb(X_train, y_train)
        best_lgb = train_lgb(X_train, y_train)
        best_cat = train_cat(X_train, y_train)
        best_mlp = train_mlp(X_train, y_train)
        logger.info("모델 학습을 시작합니다.")
    except Exception as e:
        logger.error(f"train_model - 개별 모델 학습 오류: {e}", exc_info=True)
        raise e
    
    try:
        # 6. 앙상블 모델 학습: 여러 모델을 Stacking하여 앙상블 모델 생성
        estimators = [
            ('ridge', best_ridge),
            ('rf', best_rf),
            ('xgb', best_xgb),
            ('lgb', best_lgb),
            ('cat', best_cat),
            ('mlp', best_mlp)
        ]
        stacking_reg, cv_stacking = train_stacking(estimators, X_train, y_train)
        logger.info(f"Stacking 앙상블 CV R²: {cv_stacking}")
    except Exception as e:
        logger.error(f"train_model - 앙상블 모델 학습 오류: {e}", exc_info=True)
        raise
    
    # (옵션) 각 모델의 평가 지표도 출력할 수 있습니다.
    # 예: evaluate_model(best_ridge, X_test, y_test)
    
    # 7. 반환할 객체들을 딕셔너리로 구성합니다.
    return {
        "scaler": scaler,
        "stacking_reg": stacking_reg,
        "model_features": model_features,
        "df_model": df_prepared
        }