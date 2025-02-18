# app/services/model_trainer.py

import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(df_final):
    """
    전처리된 DataFrame으로부터 모델 학습 및 스케일러, 모델 객체들을 반환합니다.
    """
    # 예시: 간단한 피처 엔지니어링
    df_model = df_final.dropna(subset=['duration_hours']).copy()
    df_model['log_review'] = np.log(df_model['review'] + 1)
    df_model['review_duration'] = df_model['review'] * df_model['duration_hours']
    
    model_features = ['review', 'duration_hours', 'log_review', 'review_duration']
    target = 'score'
    
    X_all = df_model[model_features]
    y_all = df_model[target]
    
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    X_train, _, y_train, _ = train_test_split(X_all_scaled, y_all, test_size=0.2, random_state=42)
    
    # 예시 모델: StackingRegressor 사용
    estimators = [
        ('ridge', Ridge()),
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42)),
        ('lgb', lgb.LGBMRegressor(random_state=42)),
        ('cat', CatBoostRegressor(random_state=42, verbose=0)),
        ('mlp', MLPRegressor(random_state=42, max_iter=1500))
    ]
    
    stacking_reg = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1)
    stacking_reg.fit(X_train, y_train)
    
    return {
        "scaler": scaler,
        "stacking_reg": stacking_reg,
        "model_features": model_features,
        "df_model": df_model
    }
