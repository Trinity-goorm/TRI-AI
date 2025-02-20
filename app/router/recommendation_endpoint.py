
from flask_restx import Namespace, Resource, fields, abort
from flask import request, jsonify, Response
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
logger = logging.getLogger(__name__)

api = Namespace('recommendation', description="상위 15개 식당 추천 API")

# 요청 모델 정의 (Swagger 문서용)
user_data_model = api.model('UserData', {
    'user_id': fields.String(required=True, description="사용자 ID"),
    'preferred_categories': fields.List(fields.String, required=True, min_items=1, max_items=3, description="선호 카테고리 목록")
})

# 응답 모델 정의 (참고용)
recommendation_item = api.model('RecommendationItem', {
    'restaurant_id': fields.Integer(description="식당 고유 식별자"),
    'category_id': fields.Integer(description="카테고리 ID"),
    'score': fields.Float(description="원래 평점"),
    'predicted_score': fields.Float(description="예측 평점"),
    'composite_score': fields.Float(description="최종 추천 점수")
})
response_model = api.model('RecommendationResponse', {
    'user_id': fields.String(description="사용자 ID"),
    'recommendations': fields.List(fields.Nested(recommendation_item))
})

json_directory = str(UPLOAD_DIR)

try:
    df_raw = load_and_merge_json_files(json_directory)
    df_final = preprocess_data(df_raw)  # 병합된 DataFrame 전달
    globals_dict = train_model(df_final)
except Exception as e:
    logger.error(f"Error during initialization: {e}", exc_info=True)
    # 초기화 실패 시 적절히 처리 (예: 프로그램 종료 등)
    globals_dict = {}

@api.route('', methods=['POST'])
class RecommendationResource(Resource):
    @api.expect(user_data_model)
    @api.response(200, 'Success', response_model)
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """사용자 데이터를 받아 추천 결과를 생성하고, 결과를 파일로 저장합니다."""
        try:

            req_data = request.get_json()
            user_id = req_data.get("user_id")
            preferred_categories = req_data.get("preferred_categories")
        
            if not user_id or not preferred_categories:
                abort(400, "user_id와 preferred_categories를 입력해주세요.")
        
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
            
            df_model = globals_dict.get("df_model")
            if df_model is None:
                abort(500, "모델 데이터가 초기화되지 않았습니다.")
            
            filtered_df = df_model[df_model["category_id"].isin(preferred_ids)].copy()
            if filtered_df.empty:
                abort(400, "해당 선호 카테고리에 해당하는 식당 데이터가 없습니다.")
            
            # 추천 결과 생성 (generate_recommendations는 JSON 문자열을 반환)
            result_json = generate_recommendations(
                filtered_df, 
                globals_dict["stacking_reg"], 
                globals_dict["model_features"], 
                user_id,
                globals_dict["scaler"]
            )

            # 추천 결과를 FEEDBACK_DIR에 파일로 저장
            timestamp = int(time.time())
            feedback_filename = f"recommendation_{user_id}_{timestamp}.json"
            feedback_filepath = os.path.join(str(FEEDBACK_DIR), feedback_filename)
            
            try:
                with open(feedback_filepath, "w", encoding="utf-8") as f:
                    f.write(result_json)
                logger.info(f"추천 결과가 {feedback_filepath}에 저장되었습니다.")
            except Exception as file_err:
                logger.error(f"추천 결과 저장 실패: {file_err}", exc_info=True)

            return Response(result_json, mimetype='application/json')
    
        except Exception as e:
            logger.error(f"Error in recommendation endpoint: {e}", exc_info=True)
            abort(500, str(e))