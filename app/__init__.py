# app/__init__.py
from flask import Flask
from flask_restx import Api
from app.router.recommendation_endpoint import api as recommendation_ns

def create_app():
    app = Flask(__name__)
    
    # 앱 설정, 예: config.py에서 설정값 로드 (필요시)
    # app.config.from_object('app.config')

    # Swagger UI는 기본적으로 /swagger에서 제공 (doc 인자를 조정 가능)
    api = Api(
        app,
        version="1.0",
        title="Recommendation API",
        description="API for restaurant recommendation",
        doc="/swagger"  # Swagger UI 경로
    )
    
    # 추천 API 네임스페이스 등록
    api.add_namespace(recommendation_ns, path="/recommend")

    # 전역 예외 핸들러 (선택 사항)
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled Exception: {e}", exc_info=True)
        return {"error": "Internal Server Error", "message": str(e)}, 500
    
    return app
