# app/services/model_trainer/model_training.py
# # 개별 모델 학습 및 하이퍼파라미터 튜닝

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge as FinalRidge
import logging

logger = logging.getLogger(__name__)

def train_ridge(X, y):
    try:
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        ridge = Ridge()
        grid = GridSearchCV(ridge, param_grid, cv=3, scoring='r2', n_jobs=1)
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_ridge 오류: {e}", exc_info=True)
        raise e


def train_rf(X, y):
    try:
        param_grid = {'n_estimators': [50, 100],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]}
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_rf 오류: {e}", exc_info=True)
        raise e

def train_xgb(X, y):
    try:
        param_grid = {'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]}
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
        grid = GridSearchCV(xgb, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_xgb 오류: {e}", exc_info=True)
        raise e

def train_lgb(X, y):
    try:
        param_grid = {'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1]}
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, min_split_gain=0)
        grid = GridSearchCV(lgb_model, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_lgb 오류: {e}", exc_info=True)
        raise e

def train_cat(X, y):
    try:
        param_grid = {'iterations': [50, 100],
                    'depth': [3, 5],
                    'learning_rate': [0.01, 0.1]}
        cat = CatBoostRegressor(random_state=42, verbose=0)
        grid = GridSearchCV(cat, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_cat 오류: {e}", exc_info=True)
        raise e
        

def train_mlp(X, y):
    try:
        param_grid = {'hidden_layer_sizes': [(50,), (100,)],
                    'alpha': [0.0001, 0.001]}
        mlp = MLPRegressor(random_state=42, max_iter=1500, early_stopping=True, tol=1e-3)
        grid = GridSearchCV(mlp, param_grid, cv=3, scoring='r2')
        grid.fit(X, y)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"train_mlp 오류: {e}", exc_info=True)
        raise e

def train_stacking(estimators, X, y):
    try:
        final_estimator = FinalRidge()
        stacking = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=3, n_jobs=-1)
        stacking.fit(X, y)
        cv_score = cross_val_score(stacking, X, y, cv=3, scoring='r2', n_jobs=1).mean()
        return stacking, cv_score
    except Exception as e:
        logger.error(f"train_stacking 오류: {e}", exc_info=True)
        raise e