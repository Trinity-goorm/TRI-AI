from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

from app.config import FEEDBACK_DIR
from app.router.recommendation_api import router as recommendation_api

from pathlib import Path
import json
import os
import uvicorn
import logging
import logging.config
import asyncio

# 로깅 설정
# JSON 기반 로깅 설정 적용
logging_config_path = Path(__file__).resolve().parent / "logging_config.json"  # 프로젝트 루트에 위치한 파일 경로
with open(logging_config_path, "r", encoding="utf-8") as f:
    logging_config = json.load(f)

logging.config.dictConfig(logging_config)
logger = logging.getLogger("recommendation_api")

app = FastAPI(
    title="Restaurant Recommendation API",
    description="식당 추천 API",
    version="1.0",
    docs_url="/docs"  # Swagger UI 경로
)

# 모든 출처를 허용하는 CORS 설정 (자격 증명 포함 불가)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # credentials를 반드시 False로 설정
)

# 라우터 포함
app.include_router(recommendation_api, prefix="/recommend", tags=["Restaurant Recommendation"])

# 피드백 파일을 저장하는 디렉토리를 정적 파일로 제공 (필요한 경우)
# 디렉토리가 존재하는지 확인
feedback_dir = Path(str(FEEDBACK_DIR))
if feedback_dir.exists():
    app.mount("/feedback", StaticFiles(directory=str(feedback_dir)), name="feedback")

# 루트 엔드포인트 (선택 사항)
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Restaurant Recommendation API", "docs_url": "/docs"}

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise e

# 예외 처리기
class RecommendationProcessingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@app.exception_handler(RecommendationProcessingError)
async def recommendation_processing_exception_handler(request: Request, exc: RecommendationProcessingError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."},
    )

# 첫 번째 초기화 여부를 추적하는 전역 변수
initial_sync_completed = False

# 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    global initial_sync_completed
    
    logger.info("애플리케이션 시작")
    
    # FEEDBACK_DIR이 존재하는지 확인하고 없으면 생성
    os.makedirs(str(FEEDBACK_DIR), exist_ok=True)
    
    # 최초 한 번만 실행되도록 함
    if not initial_sync_completed:
        # 데이터 동기화 실행
        logger.info("MongoDB 데이터 동기화 시작...")
        from app.services.background_tasks import run_initial_sync
        sync_result = await run_initial_sync()  # 여기서 await를 사용하여 동기화 완료 기다림
        
        if sync_result:
            logger.info("MongoDB 데이터 동기화 완료, 모델 초기화 시작...")
            # 모델 초기화 함수 호출
            from app.router.recommendation_api import initialize_model
            init_result = initialize_model()
            
            if init_result:
                logger.info("모델 초기화 성공")
            else:
                logger.error("모델 초기화 실패")
        else:
            logger.error("MongoDB 데이터 동기화 실패")
        
        # 초기화 완료 표시
        initial_sync_completed = True
        
        # 주기적 데이터 동기화 작업 시작 (최초 동기화와 별도로 실행)
        sync_interval = float(os.environ.get('SYNC_INTERVAL_HOURS', '24'))
        if sync_interval > 0:
            from app.services.background_tasks import periodic_data_sync
            asyncio.create_task(periodic_data_sync(sync_interval))
            logger.info(f"주기적 데이터 동기화 태스크 시작 (간격: {sync_interval}시간)")
    
    logger.info("애플리케이션 초기화 완료")

# 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("서버를 종료합니다.")
    # 필요한 정리 작업 수행

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_config=str(logging_config_path)  # 로깅 설정 파일 지정
    )

    import multiprocessing
    if hasattr(multiprocessing, 'freeze_support'):
        multiprocessing.freeze_support()

# 실행 명령어 (터미널에서 실행 시):
# uvicorn main:app --host 0.0.0.0 --port 5000 --reload --log-config logging_config.json