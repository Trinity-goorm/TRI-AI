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

# 로깅 설정
# JSON 기반 로깅 설정 적용
logging_config_path = Path(__file__).resolve().parent / "logging_config.json"  # 프로젝트 루트에 위치한 파일 경로
with open(logging_config_path, "r", encoding="utf-8") as f:
    logging_config = json.load(f)

logging.config.dictConfig(logging_config)
logger = logging.getLogger("recommendation_api")

app = FastAPI()

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
    return {"message": "Hello, Toby!"}

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
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."},
    )

# 서버 실행
if __name__ == "__main__":
    # FEEDBACK_DIR이 존재하는지 확인하고 없으면 생성
    os.makedirs(str(FEEDBACK_DIR), exist_ok=True)

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