# app/main.py

from flask import Flask
from app.router.recommendation_endpoint import bp as recommendation_bp  # 절대 경로 사용

def create_app():
    app = Flask(__name__)
    app.register_blueprint(recommendation_bp)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

# 실행 명령어: python -m app.main