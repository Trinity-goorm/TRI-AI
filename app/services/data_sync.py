import pymysql
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from app.config import UPLOAD_DIR
from app.config.queries import RESTAURANT_QUERY, USER_PREFERENCES_QUERY, USER_PREFERENCE_CATEGORIES_QUERY, LIKES_QUERY, RESERVATIONS_QUERY

logger = logging.getLogger("data_sync")

def fetch_data_from_rds():
    """RDS에서 식당 데이터와 사용자 관련 데이터를 가져와 JSON 파일로 저장"""
    try:
        logger.info("RDS에서 데이터 가져오기 시작")
        
        # RDS 연결 설정
        connection = pymysql.connect(
            host=os.environ.get('RDS_HOST', 'localhost'),
            user=os.environ.get('RDS_USER', 'username'),
            password=os.environ.get('RDS_PASSWORD', 'password'),
            database=os.environ.get('RDS_DATABASE', 'database'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        try:
            # 데이터 디렉토리 확인 및 생성
            upload_dir = Path(UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # 타임스탬프 생성 (모든 파일에 동일한 타임스탬프 사용)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with connection.cursor() as cursor:
                # 1. 식당 데이터 쿼리 및 저장
                cursor.execute(RESTAURANT_QUERY)
                restaurant_data = cursor.fetchall()
                
                rest_filepath = upload_dir / f"restaurant_data_{timestamp}.json"
                with open(rest_filepath, 'w', encoding='utf-8') as f:
                    json.dump(restaurant_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"식당 데이터 저장 완료: {len(restaurant_data)}개 항목")
                
                # 2. 사용자 가격 범위 선호도 데이터 쿼리 및 저장
                cursor.execute(USER_PREFERENCES_QUERY)
                user_preferences_data = cursor.fetchall()
                
                pref_filepath = upload_dir / f"user_preferences_{timestamp}.json"
                with open(pref_filepath, 'w', encoding='utf-8') as f:
                    json.dump(user_preferences_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"사용자 가격 범위 선호도 데이터 저장 완료: {len(user_preferences_data)}개 항목")
                
                # 3. 사용자 카테고리 선호도 데이터 쿼리 및 저장
                cursor.execute(USER_PREFERENCE_CATEGORIES_QUERY)
                user_preference_categories_data = cursor.fetchall()
                
                cat_pref_filepath = upload_dir / f"user_preference_categories_{timestamp}.json"
                with open(cat_pref_filepath, 'w', encoding='utf-8') as f:
                    json.dump(user_preference_categories_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"사용자 카테고리 선호도 데이터 저장 완료: {len(user_preference_categories_data)}개 항목")
                
                # 4. 찜 데이터 쿼리 및 저장
                cursor.execute(LIKES_QUERY)
                likes_data = cursor.fetchall()
                
                likes_filepath = upload_dir / f"likes_{timestamp}.json"
                with open(likes_filepath, 'w', encoding='utf-8') as f:
                    json.dump(likes_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"찜 데이터 저장 완료: {len(likes_data)}개 항목")
                
                # 5. 예약 데이터 쿼리 및 저장
                cursor.execute(RESERVATIONS_QUERY)
                reservations_data = cursor.fetchall()
                
                reservations_filepath = upload_dir / f"reservations_{timestamp}.json"
                with open(reservations_filepath, 'w', encoding='utf-8') as f:
                    json.dump(reservations_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"예약 데이터 저장 완료: {len(reservations_data)}개 항목")
            
            # 오래된 파일 정리 (각 유형별로 최신 3개만 유지)
            cleanup_old_files(str(upload_dir), "restaurant_data_", 3)
            cleanup_old_files(str(upload_dir), "user_preferences_", 3)
            cleanup_old_files(str(upload_dir), "user_preference_categories_", 3)
            cleanup_old_files(str(upload_dir), "likes_", 3)
            cleanup_old_files(str(upload_dir), "reservations_", 3)
            
            logger.info("데이터 동기화 완료")
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