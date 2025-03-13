# test_data_sync.py
import os
import logging
from dotenv import load_dotenv
from app.services.data_sync import fetch_data_from_rds

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# 환경 변수 확인 (비밀번호는 출력하지 않음)
print(f"RDS_HOST: {os.environ.get('RDS_HOST')}")
print(f"RDS_USER: {os.environ.get('RDS_USER')}")
print(f"RDS_DATABASE: {os.environ.get('RDS_DATABASE')}")
print(f"RDS_PASSWORD: {'*' * len(os.environ.get('RDS_PASSWORD', ''))} (길이: {len(os.environ.get('RDS_PASSWORD', ''))})")

# RDS 데이터 동기화 함수 실행
result = fetch_data_from_rds()
print(f"데이터 동기화 결과: {'성공' if result else '실패'}")