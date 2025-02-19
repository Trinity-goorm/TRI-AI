from flask import Blueprint, request, jsonify, Response
import json

from app.services.preprocess.data_loader import load_and_merge_json_files
from app.services.preprocess.preprocessor import preprocess_data
from app.services.model_trainer import train_model
from app.services.model_trainer.recommendation import generate_recommendations

bp = Blueprint('recommendation', __name__, url_prefix='/recommend')

# 초기화: 디렉토리 내의 여러 JSON 파일 병합 및 전처리
json_directory = 'data/crawling_2nd_data/json'
df_raw = load_and_merge_json_files(json_directory)
df_final = preprocess_data(df_raw)  # 병합된 DataFrame을 전달
globals_dict = train_model(df_final)

@bp.route('', methods=['POST'])
def recommendation():
    req_data = request.get_json()
    user_id = req_data.get("user_id")
    preferred_categories = req_data.get("preferred_categories")
    
    if not user_id or not preferred_categories:
        return jsonify({"error": "user_id와 preferred_categories를 입력해주세요."}), 400
    
    # 사용자 선호 카테고리를 기반으로 df_model 필터링
    category_mapping = {
        "중식": 1,
        "일식집": 2,
        "브런치카페": 3,
        "파스타": 4,
        "이탈리안": 5,
        "이자카야": 6,
        "한식집": 7,
        "치킨": 8,
        "스테이크": 9,
        "고깃집": 10,
        "다이닝바": 11,
        "오마카세": 12
    }
    # 선호 카테고리 이름을 해당 번호로 변환
    preferred_ids = [category_mapping.get(cat) for cat in preferred_categories if cat in category_mapping]
    
    df_model = globals_dict["df_model"]
    filtered_df = df_model[df_model["category_id"].isin(preferred_ids)].copy()
    
    if filtered_df.empty:
        return jsonify({"error": "해당 선호 카테고리에 해당하는 식당 데이터가 없습니다."}), 400
    
    result_json = generate_recommendations(filtered_df, globals_dict["stacking_reg"], globals_dict["model_features"], user_id, globals_dict["scaler"])
    return Response(result_json, mimetype='application/json')
