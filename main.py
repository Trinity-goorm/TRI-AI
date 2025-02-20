# main.py

import os
import json
import logging.config
from flask import Flask, request
from flask_restx import Api
from app.router.recommendation_endpoint import api as recommendation_ns  # flask_restx Namespace

def create_app():
    app = Flask(__name__)
    
    # logging_config.json 파일의 경로 (프로젝트 루트에 위치)
    logging_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_config.json")
    with open(logging_config_path, "r", encoding="utf-8") as f:
        logging_config = json.load(f)
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured.")
    
    # Flask-RESTX API 객체 생성 및 Swagger UI 설정 (Swagger UI는 /swagger에서 제공)
    api = Api(
        app,
        version="1.0",
        title="Recommendation API",
        description="식당 추천 API",
        doc="/swagger"
    )
    
    # 추천 API 네임스페이스 등록
    api.add_namespace(recommendation_ns, path="/recommend")
    
    # 요청 로깅 미들웨어 (모든 요청 정보를 로깅)
    @app.before_request
    def log_request_info():
        logger.info(f"Request: {request.method} {request.url}")
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
