# app/servies/mongodb/direct_mongodb.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient

# SSH 터널링이 필요한 경우에만 import
try:
    import sshtunnel
    has_sshtunnel = True
except ImportError:
    has_sshtunnel = False

from app.config.mongo_config import (
    MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD, MONGO_DATABASE, MONGO_COLLECTION,
    USE_SSH_TUNNEL, SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD, SSH_KEY_PATH
)
from app.services.mongodb.connection import get_mongodb_connection
from app.services.mongodb.data_converter import convert_numpy_types

logger = logging.getLogger("direct_mongodb")

def get_restaurants_from_mongodb():
    """MongoDB에서 레스토랑 데이터를 직접 가져와 DataFrame으로 반환"""
    try:
        # MongoDB 연결
        result = get_mongodb_connection()
        
        # SSH 터널링을 사용하는 경우
        if len(result) == 3:
            client, db, tunnel = result
            cleanup_tunnel = True
        else:
            client, db = result
            cleanup_tunnel = False
        
        try:
            # 레스토랑 컬렉션에서 데이터 가져오기
            restaurant_collection = db['restaurants']
            restaurant_data = list(restaurant_collection.find({}, {'_id': 0}))
            
            if not restaurant_data:
                logger.warning("MongoDB에서 식당 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()
                
            logger.info(f"MongoDB에서 {len(restaurant_data)}개의 레스토랑 레코드 가져옴")
            
            # DataFrame으로 변환
            df_restaurant = pd.DataFrame(restaurant_data)
            return df_restaurant
            
        finally:
            # MongoDB 연결 종료
            client.close()
            logger.info("MongoDB 연결 종료")
            
            # SSH 터널이 있는 경우 터널도 종료
            if cleanup_tunnel:
                tunnel.stop()
                logger.info("SSH 터널 종료")
    
    except Exception as e:
        logger.error(f"MongoDB 레스토랑 데이터 가져오기 오류: {str(e)}", exc_info=True)
        return pd.DataFrame()

def get_user_data_from_mongodb():
    """MongoDB에서 사용자 관련 데이터를 직접 가져와 DataFrame 사전으로 반환"""
    try:
        # MongoDB 연결
        result = get_mongodb_connection()
        
        # SSH 터널링을 사용하는 경우
        if len(result) == 3:
            client, db, tunnel = result
            cleanup_tunnel = True
        else:
            client, db = result
            cleanup_tunnel = False
        
        user_data_frames = {}
        
        try:
            # 1. 사용자 기본 정보
            users_collection = db['users']
            user_data = list(users_collection.find({}, {'_id': 0}))
            
            if user_data:
                logger.info(f"MongoDB에서 {len(user_data)}개의 사용자 레코드 가져옴")
                user_data_frames['users'] = pd.DataFrame(user_data)
            else:
                logger.warning("MongoDB에서 사용자 데이터를 찾을 수 없습니다.")
            
            # 2. 사용자 선호도 정보
            user_preferences_collection = db['user_preferences']
            user_preferences_data = list(user_preferences_collection.find({}, {'_id': 0}))
            
            if user_preferences_data:
                logger.info(f"MongoDB에서 {len(user_preferences_data)}개의 사용자 선호도 레코드 가져옴")
                user_data_frames['user_preferences'] = pd.DataFrame(user_preferences_data)
            else:
                logger.warning("MongoDB에서 사용자 선호도 데이터를 찾을 수 없습니다.")
            
            # 3. 찜 데이터
            likes_collection = db['likes']
            likes_data = list(likes_collection.find({}, {'_id': 0}))
            
            if likes_data:
                logger.info(f"MongoDB에서 {len(likes_data)}개의 찜 레코드 가져옴")
                user_data_frames['likes'] = pd.DataFrame(likes_data)
            else:
                logger.warning("MongoDB에서 찜 데이터를 찾을 수 없습니다.")
            
            # 4. 예약 데이터
            reservations_collection = db['reservations']
            reservations_data = list(reservations_collection.find({}, {'_id': 0}))
            
            if reservations_data:
                logger.info(f"MongoDB에서 {len(reservations_data)}개의 예약 레코드 가져옴")
                user_data_frames['reservations'] = pd.DataFrame(reservations_data)
            else:
                logger.warning("MongoDB에서 예약 데이터를 찾을 수 없습니다.")
            
            # 5. 추천 시스템 통합 데이터 (선택사항)
            recsys_collection = db[MONGO_COLLECTION]
            recsys_data = list(recsys_collection.find({}, {'_id': 0}))
            
            if recsys_data:
                logger.info(f"MongoDB에서 {len(recsys_data)}개의 추천 시스템 레코드 가져옴")
                # 데이터가 복잡한 중첩 구조를 가질 수 있으므로 변환 필요
                recsys_data = convert_numpy_types(recsys_data)
                user_data_frames['recsys_data'] = pd.DataFrame(recsys_data)
            else:
                logger.warning("MongoDB에서 추천 시스템 데이터를 찾을 수 없습니다.")
            
            return user_data_frames
            
        finally:
            # MongoDB 연결 종료
            client.close()
            logger.info("MongoDB 연결 종료")
            
            # SSH 터널이 있는 경우 터널도 종료
            if cleanup_tunnel:
                tunnel.stop()
                logger.info("SSH 터널 종료")
    
    except Exception as e:
        logger.error(f"MongoDB 사용자 데이터 가져오기 오류: {str(e)}", exc_info=True)
        return {}