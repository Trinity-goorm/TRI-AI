# app/servies/mongodb/data_collector.py

import logging
from pathlib import Path
from datetime import datetime
from app.config import RESTAURANTS_DIR, USER_DIR
from app.services.mongodb.data_converter import process_and_save_data, cleanup_old_files

logger = logging.getLogger(__name__)

def process_restaurant_data(db, timestamp):
    """MongoDB에서 레스토랑 관련 데이터 처리 및 저장"""
    try:
        # 디렉토리 경로
        restaurant_dir = Path(RESTAURANTS_DIR)
        restaurant_dir.mkdir(parents=True, exist_ok=True)
        
        # 레스토랑 컬렉션에서 데이터 가져오기
        restaurant_collection = db['restaurants']
        restaurant_data = list(restaurant_collection.find({}, {'_id': 0}))
        
        if not restaurant_data:
            logger.warning("MongoDB에서 식당 데이터를 찾을 수 없습니다.")
            return False
            
        logger.info(f"MongoDB에서 {len(restaurant_data)}개의 레스토랑 레코드 가져옴")
        
        # 데이터 처리 및 저장
        rest_filepath = restaurant_dir / f"restaurant_data_{timestamp}.json"
        success = process_and_save_data(
            restaurant_data, 
            rest_filepath, 
            "식당 데이터"
        )
        
        if success:
            # 오래된 파일 정리 (최신 3개만 유지)
            cleanup_old_files(str(restaurant_dir), "restaurant_data_", 3)
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"레스토랑 데이터 처리 오류: {str(e)}", exc_info=True)
        return False

def process_user_data(db, timestamp):
    """MongoDB에서 사용자 관련 데이터 처리 및 저장"""
    # 디렉토리 경로
    user_dir = Path(USER_DIR)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # 적어도 하나의 쿼리가 성공했는지 추적
    success_count = 0
    total_queries = 5  # 총 실행할 쿼리 수
    
    # 1. 사용자 기본 정보 가져오기 및 저장
    success_count += process_collection(
        db, 'users', 
        user_dir / f"user_data_{timestamp}.json", 
        "사용자 기본 정보",
        user_dir, "user_data_", 3
    )
    
    # 2. 사용자 선호도 데이터 가져오기 및 저장
    success_count += process_collection(
        db, 'user_preferences', 
        user_dir / f"user_preferences_{timestamp}.json", 
        "사용자 선호도 데이터",
        user_dir, "user_preferences_", 3
    )
    
    # 3. 찜 데이터 가져오기 및 저장
    success_count += process_collection(
        db, 'likes', 
        user_dir / f"likes_{timestamp}.json", 
        "찜 데이터",
        user_dir, "likes_", 3
    )
    
    # 4. 예약 데이터 가져오기 및 저장
    success_count += process_collection(
        db, 'reservations', 
        user_dir / f"reservations_{timestamp}.json", 
        "예약 데이터",
        user_dir, "reservations_", 3
    )
    
    # 5. 추천 시스템 통합 데이터 가져오기 및 저장 (선택사항)
    success_count += process_collection(
        db, 'recsys_data', 
        user_dir / f"recsys_data_{timestamp}.json", 
        "추천 시스템 데이터",
        user_dir, "recsys_data_", 3
    )
    
    # 일부 쿼리라도 성공했으면 일부 성공으로 간주
    if success_count > 0:
        logger.info(f"사용자 데이터 동기화: {success_count}/{total_queries} 쿼리 성공")
        return True
    else:
        logger.warning("모든 사용자 데이터 쿼리 실패")
        return False

def process_collection(db, collection_name, filepath, prefix, dir_path, file_prefix, keep_count):
    """특정 컬렉션에서 데이터를 가져와 저장하는 헬퍼 함수"""
    try:
        collection = db[collection_name]
        data = list(collection.find({}, {'_id': 0}))
        
        if data:
            logger.info(f"MongoDB에서 {len(data)}개의 {prefix} 레코드 가져옴")
            
            if process_and_save_data(data, filepath, prefix):
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(dir_path), file_prefix, keep_count)
                return 1
        else:
            logger.warning(f"MongoDB에서 {collection_name} 데이터를 찾을 수 없습니다.")
        
        return 0
    except Exception as e:
        logger.warning(f"{prefix} 처리 실패: {e}")
        return 0