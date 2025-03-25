# app/servies/background_tasks.py

import asyncio
import logging
from datetime import datetime
import time

from app.services.mongo_data_sync import fetch_data_from_mongodb
from app.services.model_initialization import initialize_model

logger = logging.getLogger(__name__)

async def periodic_data_sync(hours_interval=24):
    """주기적인 데이터 동기화 및 모델 재학습 수행"""
    try:
        # 초기 지연 (5초)
        await asyncio.sleep(5)
        
        while True:
            try:
                logger.info(f"주기적 데이터 동기화 시작 (간격: {hours_interval}시간)")
                # MongoDB에서 데이터 가져오기
                sync_result = await asyncio.get_event_loop().run_in_executor(None, fetch_data_from_mongodb)
                
                if sync_result:
                    logger.info("데이터 동기화 완료, 모델 재학습 시작")
                    # 직접 MongoDB 사용 옵션을 False로 하여 JSON 파일로부터 모델 초기화
                    await initialize_model(force_reload=True, use_direct_mongodb=False)
                    logger.info(f"모델 재학습 완료 ({datetime.now().isoformat()})")
                else:
                    logger.error("데이터 동기화 실패, 모델 재학습 스킵")
            
            except Exception as e:
                logger.error(f"주기적 동기화 중 오류: {str(e)}", exc_info=True)
            
            # 다음 실행까지 대기 (시간 -> 초)
            next_sync_seconds = hours_interval * 3600
            logger.info(f"다음 동기화까지 {next_sync_seconds}초 대기")
            await asyncio.sleep(next_sync_seconds)
    
    except asyncio.CancelledError:
        logger.info("주기적 동기화 태스크 취소됨")

async def run_initial_sync():
    """애플리케이션 시작 시 데이터 동기화 및 모델 초기화 수행"""
    try:
        logger.info("초기 데이터 동기화 시작")
        # MongoDB에서 데이터 가져오기
        sync_result = await asyncio.get_event_loop().run_in_executor(None, fetch_data_from_mongodb)
        
        if sync_result:
            logger.info("초기 데이터 동기화 완료, 모델 초기화 시작")
            # 직접 MongoDB 사용 옵션을 True로 설정하여 MongoDB에서 직접 데이터를 가져와 모델 초기화
            await initialize_model(force_reload=True, use_direct_mongodb=True)
            logger.info("초기화 완료")
        else:
            logger.error("초기 데이터 동기화 실패, 모델 초기화 스킵")
    
    except Exception as e:
        logger.error(f"초기 동기화 중 오류: {str(e)}", exc_info=True)