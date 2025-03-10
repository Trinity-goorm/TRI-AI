# test_data_sync.py
import logging
from app.services.data_sync import fetch_data_from_rds

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# RDS 데이터 동기화 함수 실행
result = fetch_data_from_rds()
print(f"데이터 동기화 결과: {'성공' if result else '실패'}")