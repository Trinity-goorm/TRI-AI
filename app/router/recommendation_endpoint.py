from flask import Blueprint, request, jsonify, Response
import json
from app.config import UPLOAD_DIR
from app.services.preprocess.data_loader import load_and_merge_json_files
from app.services.preprocess.preprocessor import preprocess_data
from app.services.model_trainer import train_model
from app.services.model_trainer.recommendation import generate_recommendations
import logging

# 추천 결과를 FEEDBACK_DIR에 파일로 저장하는 코드 추가
import os, time
from app.config import FEEDBACK_DIR

# 라우터 설정
bp = Blueprint('recommendation', __name__, url_prefix='/recommend')
logger = logging.getLogger(__name__)

# 초기화: 디렉토리 내의 여러 JSON 파일 병합 및 전처리
# 실시간 데이터 갱신하는 방식일시, 추후 수정 필요.

json_directory = str(UPLOAD_DIR)

try:
    df_raw = load_and_merge_json_files(json_directory)
    df_final = preprocess_data(df_raw)  # 병합된 DataFrame 전달
    globals_dict = train_model(df_final)
except Exception as e:
    logger.error(f"Error during initialization: {e}", exc_info=True)
    # 초기화 실패 시 적절히 처리 (예: 프로그램 종료 등)
    globals_dict = {}

@bp.route('', methods=['POST'])
def recommendation():
    try:

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
        if df_model is None:
            return jsonify({"error": "모델 데이터가 초기화되지 않았습니다."}), 500

        filtered_df = df_model[df_model["category_id"].isin(preferred_ids)].copy()
        
        if filtered_df.empty:
            return jsonify({"error": "해당 선호 카테고리에 해당하는 식당 데이터가 없습니다."}), 400
        
        result_json = generate_recommendations(
            filtered_df, 
            globals_dict["stacking_reg"], 
            globals_dict["model_features"], 
            user_id,
            globals_dict["scaler"]
        )

        timestamp = int(time.time())
        feedback_filename = f"recommendation_{user_id}_{timestamp}.json"
        feedback_filepath = os.path.join(str(FEEDBACK_DIR), feedback_filename)
        
        with open(feedback_filepath, "w", encoding="utf-8") as f:
            f.write(result_json)

        return Response(result_json, mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
