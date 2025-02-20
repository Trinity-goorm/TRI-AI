# app/services/model_trainer/data_preparation.py
# # 데이터 전처리(결측치 보완, 스케일링, 데이터 분할 등)

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
    """
    필수 컬럼에 결측치가 있는 행 제거 및 기본 전처리.
    """
    try:
        df_clean = df.dropna(subset=required_cols).copy()
        # 평점 미제공 처리: score와 review가 모두 0이면 score를 NaN으로 설정
        df_clean.loc[(df_clean['score'] == 0) & (df_clean['review'] == 0), 'score'] = np.nan
        df_clean['score_provided'] = df_clean['score'].notna().astype(int)
        return df_clean
    except Exception as e:
        logger.error(f"prepare_data 오류: {e}", exc_info=True)
        raise e

def impute_and_clip(df: pd.DataFrame, impute_cols: list) -> pd.DataFrame:
    """
    IterativeImputer를 사용하여 결측치 보완 후, score의 최댓값을 5.0으로 클리핑.
    """
    try:
        imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10, initial_strategy='median')
        df[impute_cols] = imputer.fit_transform(df[impute_cols])
        df.loc[df['score'] > 5, 'score'] = 5.0
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