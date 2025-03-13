import asyncio
import logging
from datetime import datetime
import time

from app.services.data_sync import fetch_data_from_rds
from app.services.model_initialization import initialize_model

logger = logging.getLogger(__name__)

async def periodic_data_sync(interval_hours=24):
    """주기적으로 RDS에서 데이터를 가져와 모델 재학습"""
    interval_seconds = interval_hours * 3600
    
    while True:
        try:
            logger.info(f"주기적 데이터 동기화 시작 (간격: {interval_hours}시간)")
            start_time = time.time()
            
            # RDS에서 데이터 가져오기
            success = fetch_data_from_rds()
            
            if success:
                # 모델 재학습 (force_reload=True로 강제 재로딩)
                await initialize_model(force_reload=True)
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"데이터 동기화 및 모델 재학습 완료 (소요시간: {duration:.2f}초)")
            else:
                logger.warning("데이터 가져오기 실패, 모델 재학습 건너뜀")
                
        except Exception as e:
            logger.error(f"주기적 태스크 실행 중 오류: {str(e)}", exc_info=True)
        
        # 다음 실행 전 대기
        next_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"다음 데이터 동기화 예정 시간: {next_sync_time} (현재로부터 {interval_hours}시간 후)")
        await asyncio.sleep(interval_seconds)

async def run_initial_sync():
    """애플리케이션 시작 시 초기 데이터 동기화 수행"""
    try:
        logger.info("초기 데이터 동기화 시작")
        success = fetch_data_from_rds()
        if success:
            await initialize_model(force_reload=True)
            logger.info("초기 데이터 동기화 및 모델 재학습 완료")
        else:
            logger.warning("초기 데이터 동기화 실패")
    except Exception as e:
        logger.error(f"초기 데이터 동기화 중 오류: {str(e)}", exc_info=True)