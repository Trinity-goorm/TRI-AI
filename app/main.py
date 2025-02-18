# app/main.py

from flask import Flask
from app.router.recommendation_endpoint import bp as recommendation_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(recommendation_bp)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
