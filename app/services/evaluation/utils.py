# app/services/evaluation/utils.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)

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

def validate_required_objects(globals_dict, df_model=None, user_features_df=None):
    """
    모델 평가에 필요한 객체들의 유효성 검증
    
    Args:
        globals_dict: 전역 변수 딕셔너리
        df_model: 식당 데이터 (옵션)
        user_features_df: 사용자 특성 데이터 (옵션)
        
    Returns:
        tuple: (유효성 여부, 누락 객체 리스트)
    """
    # 인자로 전달된 값 우선 사용, 없으면 globals_dict에서 가져오기
    df_model = df_model or globals_dict.get("df_model")
    user_features_df = user_features_df or globals_dict.get("user_features_df")
    
    # 필요한 객체 추출
    stacking_reg = globals_dict.get("stacking_reg")
    scaler = globals_dict.get("scaler")
    model_features = globals_dict.get("model_features")
    
    # DataFrame 객체 유효성 검사
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
    
    return len(missing_objects) == 0, missing_objects