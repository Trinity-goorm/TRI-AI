# app/services/background_tasks.py
import asyncio
import logging
from datetime import datetime
import time

from app.services.mongo_data_sync import fetch_data_from_mongodb
from app.services.model_initialization import initialize_model

logger = logging.getLogger(__name__)

# 초기 동기화 상태를 추적하는 변수
_initial_sync_completed = False

# 첫 번째 주기적 동기화 지연 시간(시간)
FIRST_PERIODIC_SYNC_DELAY = 1.0  # 첫 주기적 동기화는 1시간 후에 시작

async def periodic_data_sync(hours_interval=24):
    """주기적인 데이터 동기화 및 모델 재학습 수행"""
    global _initial_sync_completed
    
    try:
        # 초기 동기화가 완료될 때까지 대기
        while not _initial_sync_completed:
            logger.info("초기 동기화가 완료되지 않았습니다. 주기적 동기화는 초기 동기화 완료 후 시작됩니다.")
            await asyncio.sleep(10)  # 10초마다 확인
        
        # 첫 주기적 동기화를 위한 추가 지연
        logger.info(f"첫 주기적 동기화는 {FIRST_PERIODIC_SYNC_DELAY}시간 후에 시작됩니다.")
        await asyncio.sleep(FIRST_PERIODIC_SYNC_DELAY * 3600)  # 시간을 초로 변환
        
        while True:
            start_time = datetime.now()
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
            
            # 다음 실행 시간 계산 (실행 소요 시간 고려)
            end_time = datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            
            # 간격에서 이미 소요된 시간을 뺀 만큼만 대기
            wait_seconds = max(1, hours_interval * 3600 - elapsed_seconds)  # 최소 1초 대기
            logger.info(f"다음 동기화까지 {wait_seconds/3600:.2f}시간 대기")
            await asyncio.sleep(wait_seconds)
    
    except asyncio.CancelledError:
        logger.info("주기적 동기화 태스크 취소됨")
    except Exception as e:
        logger.error(f"주기적 동기화 태스크 오류: {e}", exc_info=True)
        # 태스크가 중단되지 않도록 다시 시작 (5초 후)
        await asyncio.sleep(5)
        asyncio.create_task(periodic_data_sync(hours_interval))

async def run_initial_sync():
    """애플리케이션 시작 시 데이터 동기화 및 모델 초기화 수행"""
    global _initial_sync_completed
    
    try:
        logger.info("초기 데이터 동기화 시작")
        
        # 이미 초기화가 완료되었는지 확인
        if _initial_sync_completed:
            logger.info("초기 동기화가 이미 완료되었습니다. 건너뜁니다.")
            return True
        
        # MongoDB에서 데이터 가져오기
        sync_result = await asyncio.get_event_loop().run_in_executor(None, fetch_data_from_mongodb)
        
        if sync_result:
            logger.info("초기 데이터 동기화 완료, 모델 초기화 시작")
            # 직접 MongoDB 사용 옵션을 True로 설정하여 MongoDB에서 직접 데이터를 가져와 모델 초기화
            init_result = await initialize_model(force_reload=True, use_direct_mongodb=True)
            
            if init_result:
                logger.info("초기화 완료")
                _initial_sync_completed = True  # 초기화 완료 상태 설정
                return True
            else:
                logger.error("모델 초기화 실패")
                return False
        else:
            logger.error("초기 데이터 동기화 실패, 모델 초기화 스킵")
            return False
    
    except Exception as e:
        logger.error(f"초기 동기화 중 오류: {str(e)}", exc_info=True)
        return False