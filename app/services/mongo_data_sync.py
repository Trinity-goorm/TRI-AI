# app/servies/mongo_data_sync.py

import logging
from datetime import datetime
from app.services.mongodb.connection import get_mongodb_connection
from app.services.mongodb.data_collector import process_restaurant_data, process_user_data

logger = logging.getLogger("data_sync")

def fetch_data_from_mongodb():
    """MongoDB에서 식당 데이터와 사용자 관련 데이터를 가져와 JSON 파일로 저장"""
    try:
        logger.info("MongoDB에서 데이터 가져오기 시작")
        
        # MongoDB 연결 (SSH 터널링 자동 설정)
        result = get_mongodb_connection()
        
        # SSH 터널링을 사용하는 경우
        if len(result) == 3:
            client, db, tunnel = result
            cleanup_tunnel = True
        else:
            client, db = result
            cleanup_tunnel = False
        
        try:
            # 타임스탬프 생성 (모든 파일에 동일한 타임스탬프 사용)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 레스토랑 데이터 처리
            success_restaurant = process_restaurant_data(db, timestamp)
            
            # 2. 사용자 관련 데이터 처리
            try:
                success_user = process_user_data(db, timestamp)
                if success_user:
                    logger.info("사용자 데이터 동기화 완료")
                else:
                    logger.warning("일부 사용자 데이터 동기화 실패")
            except Exception as e:
                logger.warning(f"사용자 데이터 동기화 중 오류 발생: {e}")
                success_user = False
            
            return success_restaurant
            
        finally:
            # MongoDB 연결 종료
            client.close()
            logger.info("MongoDB 연결 종료")
            
            # SSH 터널이 있는 경우 터널도 종료
            if cleanup_tunnel:
                tunnel.stop()
                logger.info("SSH 터널 종료")
    
    except Exception as e:
        logger.error(f"MongoDB 데이터 가져오기 오류: {str(e)}", exc_info=True)
        return False