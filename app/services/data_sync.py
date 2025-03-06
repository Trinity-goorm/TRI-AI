# app/services/data_sync.py

import pymysql
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from app.config import UPLOAD_DIR

logger = logging.getLogger("data_sync")

def fetch_data_from_rds():
    """RDS에서 식당 데이터와 사용자 데이터를 가져와 JSON 파일로 저장"""
    try:
        logger.info("RDS에서 데이터 가져오기 시작")
        
        # RDS 연결 설정
        connection = pymysql.connect(
            host=os.environ.get('RDS_HOST', 'localhost'),
            user=os.environ.get('RDS_USER', 'username'),
            password=os.environ.get('RDS_PASSWORD', 'password'),
            database=os.environ.get('RDS_DATABASE', 'database')
        )
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:         
                # 식당 데이터 쿼리 - 필요한 필드만 선택
                cursor.execute("""
                    SELECT 
                        r.id as restaurant_id, 
                        r.name, 
                        r.address, 
                        r.average_price, 
                        r.caution, 
                        r.convenience, 
                        r.expanded_days, 
                        r.is_deleted, 
                        r.operating_hour, 
                        r.phone_number, 
                        r.rating, 
                        r.review_count, 
                        r.time_range,
                        rc.category_id as db_category_id
                    FROM restaurant r
                    JOIN restaurant_category rc ON r.id = rc.Restaurant_id
                """)
                restaurant_data = cursor.fetchall()
                            
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 식당 데이터 저장
            rest_filepath = Path(UPLOAD_DIR) / f"restaurant_data_{timestamp}.json"
            with open(rest_filepath, 'w', encoding='utf-8') as f:
                json.dump(restaurant_data, f, ensure_ascii=False, indent=2)
            
            # 사용자 데이터 저장
            user_filepath = Path(UPLOAD_DIR) / f"user_data_{timestamp}.json"
            with open(user_filepath, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"데이터 저장 완료: 식당 {len(restaurant_data)}개, 사용자 관련 데이터 {len(user_data)}개")
            
            # 오래된 파일 정리 (최신 3개만 유지)
            cleanup_old_files(UPLOAD_DIR, "restaurant_data_", 3)
            cleanup_old_files(UPLOAD_DIR, "user_data_", 3)
            
            return True
            
        finally:
            connection.close()
    
    except Exception as e:
        logger.error(f"데이터 가져오기 오류: {str(e)}", exc_info=True)
        return False

def cleanup_old_files(directory, prefix, keep_count):
    """특정 접두사를 가진 파일 중 최신 파일 n개만 유지"""
    try:
        dir_path = Path(directory)
        files = [f for f in dir_path.glob(f"{prefix}*.json")]
        
        # 파일의 생성 시간에 따라 정렬
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 유지할 파일 수보다 많으면 오래된 파일 삭제
        if len(files) > keep_count:
            for old_file in files[keep_count:]:
                old_file.unlink()
                logger.info(f"오래된 파일 삭제: {old_file}")

    except Exception as e:
        logger.error(f"파일 정리 중 오류: {str(e)}", exc_info=True)