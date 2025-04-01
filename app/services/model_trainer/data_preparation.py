# app/services/model_trainer/data_preparation.py
# 데이터 전처리(결측치 보완, 스케일링, 데이터 분할 등)

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # 필요
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
import logging

logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    try:
        # 기존 코드 + 추가 데이터 품질 검증
        df_clean = df.dropna(subset=required_cols).copy()
        
        # 이상치 처리 강화
        df_clean['review'] = df_clean['review'].clip(lower=0)
        df_clean['score'] = df_clean['score'].clip(lower=0, upper=5)
        
        # 데이터 정규성 검증
        df_clean['log_review'] = np.log1p(df_clean['review'])
        df_clean['log_score'] = np.log1p(df_clean['score'])
        
        # 상호작용 특성 추가
        df_clean['review_duration'] = df_clean['review'] * df_clean['duration_hours']
        
        return df_clean
    except Exception as e:
        logger.error(f"prepare_data 오류: {e}", exc_info=True)
        raise e

def impute_and_clip(df: pd.DataFrame, impute_cols: list, min_values=None, max_values=None) -> pd.DataFrame:
    """
    IterativeImputer를 사용하여 결측치 보완 후, 지정된 범위로 클리핑.
    
    Args:
        df: 입력 데이터프레임
        impute_cols: 결측치를 보완할 컬럼 리스트
        min_values: 컬럼별 최소값 딕셔너리 (예: {'score': 0})
        max_values: 컬럼별 최대값 딕셔너리 (예: {'score': 5})
    """
    try:
        # 기본값 설정
        min_values = min_values or {'score': 0}
        max_values = max_values or {'score': 5.0}
        
        imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10, initial_strategy='median')
        df[impute_cols] = imputer.fit_transform(df[impute_cols])
        
        # 각 컬럼별로 클리핑 적용
        for col, min_val in min_values.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=min_val)
                
        for col, max_val in max_values.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=max_val)
                
        return df
    except Exception as e:
        logger.error(f"impute_and_clip 오류: {e}", exc_info=True)
        raise e

def scale_and_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """
    StandardScaler를 적용하고, 데이터를 학습/테스트 세트로 분할합니다.
    스케일링 후, DataFrame 형태로 변환하여 피처 이름을 유지합니다.
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # DataFrame으로 변환 (원본 피처 이름과 인덱스를 그대로 사용)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        return scaler, X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"scale_and_split 오류: {e}", exc_info=True)
        raise e