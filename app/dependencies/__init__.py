# app/dependencies/__init__.py

from datetime import datetime

# 글로벌 변수 초기화
globals_dict = {}
model_initializing = False  # 모델 초기화 상태를 추적하는 전역 변수
last_initialization_attempt = None  # 마지막 초기화 시도 시간

def get_globals_dict():
    return globals_dict