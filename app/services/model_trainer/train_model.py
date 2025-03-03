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
        # 모델 학습에 필요한 새로운 피처 생성
        df_prepared['log_review'] = np.log(df_prepared['review'] + 1)
        df_prepared['review_duration'] = df_prepared['review'] * df_prepared['duration_hours']
        logger.debug("Feature 계산이 완료되었습니다.")
    except Exception as e:
        logger.error(f"train_model - 피처 생성 오류: {e}", exc_info=True)
        raise e

    # 3. (이미 전처리 과정에서 로그 변환, 상호작용 등 피처 엔지니어링이 이루어졌다고 가정)
    #    모델 학습에 사용할 피처와 타깃을 설정합니다.
    try:
        model_features = ['review', 'duration_hours', 'conv_WIFI', 'conv_주차', 'caution_예약가능', 'log_review', 'review_duration']
        target = 'score'
        X = df_prepared[model_features]
        y = df_prepared[target]
        logger.debug(f"Featuer 및 타깃 설정이 완료되었습니다: {model_features}")
    except Exception as e:
        logger.error(f"train_model - Featuer 및 타깃 설정 오류: {e}", exc_info=True)
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


