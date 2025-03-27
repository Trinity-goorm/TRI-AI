# test_data_sync.py
import os
import logging
from dotenv import load_dotenv
from app.services.mongo_data_sync import fetch_data_from_mongodb

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# 환경 변수 확인 (비밀번호는 출력하지 않음)
print(f"MongoDB 연결 정보:")
print(f"MONGO_HOST: {os.environ.get('MONGO_HOST')}")
print(f"MONGO_PORT: {os.environ.get('MONGO_PORT')}")
print(f"MONGO_USER: {os.environ.get('MONGO_USER')}")
print(f"MONGO_DATABASE: {os.environ.get('MONGO_DATABASE')}")
print(f"MONGO_PASSWORD: {'*' * len(os.environ.get('MONGO_PASSWORD', ''))} (길이: {len(os.environ.get('MONGO_PASSWORD', ''))})")

# SSH 터널링 정보
use_ssh_tunnel = os.environ.get('USE_SSH_TUNNEL', 'false').lower() == 'true'
if use_ssh_tunnel:
    print("\nSSH 터널링 설정:")
    print(f"SSH_HOST: {os.environ.get('SSH_HOST')}")
    print(f"SSH_PORT: {os.environ.get('SSH_PORT')}")
    print(f"SSH_USER: {os.environ.get('SSH_USER')}")
    
    if os.environ.get('SSH_PASSWORD'):
        print(f"SSH_PASSWORD: {'*' * len(os.environ.get('SSH_PASSWORD', ''))} (길이: {len(os.environ.get('SSH_PASSWORD', ''))})")
    
    if os.environ.get('SSH_KEY_PATH'):
        print(f"SSH_KEY_PATH: {os.environ.get('SSH_KEY_PATH')}")

# MongoDB 데이터 동기화 함수 실행
print("\n데이터 동기화 시작...")
result = fetch_data_from_mongodb()
print(f"데이터 동기화 결과: {'성공' if result else '실패'}")