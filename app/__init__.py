# app/__init__.py
from flask import Flask, jsonify
from app.router.recommendation_endpoint import bp as recommendation_bp

def create_app():
    app = Flask(__name__)
    
    # 앱 설정, 예: config.py에서 설정값 로드 (필요시)
    # app.config.from_object('app.config')
    
    # Blueprint 등록
    app.register_blueprint(recommendation_bp)
    
    return app
