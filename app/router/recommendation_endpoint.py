# app/router/recommendation_endpoint.py

from flask import Blueprint, request, jsonify
from app.services.data_loader import load_data
from app.services.preprocessor import preprocess_data
from app.services.model_trainer import train_model
from app.services.recommender import get_recommendations

bp = Blueprint('recommendation', __name__, url_prefix='/recommend')

# 초기화 (실제 운영 시엔 별도 초기화 로직을 고려)
json_path = 'data/crawling_2nd_data/json/중식_restaurants_table.json'  # 실제 사용하는 데이터 경로
df_raw = load_data(json_path)
df_final = preprocess_data(df_raw)
globals_dict = train_model(df_final)

@bp.route('', methods=['POST'])
def recommendation():
    req_data = request.get_json()
    user_id = req_data.get("user_id")
    preferred_categories = req_data.get("preferred_categories")
    
    if not user_id or not preferred_categories:
        return jsonify({"error": "user_id와 preferred_categories를 입력해주세요."}), 400
    
    result = get_recommendations(user_id, preferred_categories, globals_dict)
    return jsonify(result)
