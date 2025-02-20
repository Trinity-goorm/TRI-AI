# app/services/model_trainer/model_evaluation.py
# 모델 평가 함수

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X, y):
    """
    주어진 모델에 대해 R², RMSE, MAE를 계산하여 반환합니다.
    """
    try:
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        return r2, rmse, mae
    except Exception as e:
        logger.error(f"evaluate_model 오류: {e}", exc_info=True)
        raise e
