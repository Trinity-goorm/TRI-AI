# app/services/recommender.py

import numpy as np
import json

def sigmoid_transform(x: float, a: float, b: float) -> float:
    """
    Sigmoid 함수 변환 (a와 b는 조정 가능한 파라미터)
    """
    return 5 * (1 / (1 + np.exp(-a * (x - b))))

def get_recommendations(user_id: str, preferred_categories: list, globals_dict: dict) -> dict:
    """
    사용자 ID와 선호 카테고리 리스트를 받아 추천 결과를 dict로 반환합니다.
    """
    # 예시 카테고리 매핑 (전처리 시 사용한 매핑과 일치해야 함)
    category_mapping = {
        "중식": 0, "일식집": 1, "브런치카페": 2, "파스타": 3,
        "이탈리안": 4, "이자카야": 5, "한식집": 6, "치킨": 7,
        "스테이크": 8, "고깃집": 9, "다이닝바": 10, "오마카세": 11
    }
    preferred_ids = [category_mapping[cat] for cat in preferred_categories if cat in category_mapping]
    if not preferred_ids:
        return {"error": "유효한 카테고리 입력이 없습니다."}
    
    df_model = globals_dict["df_model"]
    data_filtered = df_model[df_model['category_id'].isin(preferred_ids)].copy()
    if data_filtered.empty:
        return {"error": "해당 선호 카테고리에 해당하는 식당 데이터가 없습니다."}
    
    # 간단한 피처 엔지니어링 (필요에 따라 추가)
    data_filtered['log_review'] = np.log(data_filtered['review'] + 1)
    data_filtered['review_duration'] = data_filtered['review'] * data_filtered['duration_hours']
    
    model_features = globals_dict["model_features"]
    scaler = globals_dict["scaler"]
    stacking_reg = globals_dict["stacking_reg"]
    
    X_all = data_filtered[model_features]
    X_all_scaled = scaler.transform(X_all)
    
    data_filtered['predicted_score'] = stacking_reg.predict(X_all_scaled)
    data_filtered['final_score'] = data_filtered['score']
    
    # 보정 로직 (예시)
    review_weight = 0.4
    def compute_composite(row):
        base = row['final_score']
        review_adjust = review_weight * (np.log(float(row['review']) + 50) / np.log(1000))
        return base + review_adjust
    
    data_filtered['composite_score'] = data_filtered.apply(lambda row: compute_composite(row), axis=1)
    a, b = 1.25, 2.5
    data_filtered['composite_score'] = data_filtered['composite_score'].apply(lambda x: sigmoid_transform(x, a, b))
    
    recommendations = data_filtered.sort_values(by='composite_score', ascending=False)\
                                     .head(15)[['id', 'category_id', 'score', 'predicted_score', 'composite_score']]
    recommendations['predicted_score'] = recommendations['predicted_score'].round(3)
    recommendations['composite_score'] = recommendations['composite_score'].round(3)
    
    return {
        "user": user_id,
        "recommendations": json.loads(recommendations.to_json(orient='records', force_ascii=False))
    }
