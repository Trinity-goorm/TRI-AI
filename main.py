# main.py

import os
import json
import logging.config
from flask import Flask, request
from app.router.recommendation_endpoint import bp as recommendation_bp  # 절대 경로 사용

def create_app():
    
    app = Flask(__name__)
    
    # logging_config.json 파일의 경로 (프로젝트 루트에 위치한다고 가정)
    logging_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_config.json")
    
    # 로깅 설정 적용
    with open(logging_config_path, "r", encoding="utf-8") as f:
        logging_config = json.load(f)
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured.")
    
    # Blueprint 등록
    app.register_blueprint(recommendation_bp)
    
    # 요청 로깅 미들웨어 (선택 사항)
    @app.before_request
    def log_request_info():
        logger.info(f"Request: {request.method} {request.url}")
    
    return app

app = create_app()

if __name__ == '__main__':
    # Flask 개발 서버 실행 (운영 시에는 WSGI 서버 사용)
    app.run(debug=True, host="127.0.0.1", port=5000)

# source venv/bin/activate - macOS/Linux
# 실행 명령어: python -m app.main