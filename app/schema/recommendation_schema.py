# app/models/recommendation_schema.py

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Annotated

class RecommendationItem(BaseModel):
    category_id: int
    restaurant_id: int
    score: float
    predicted_score: float
    composite_score: float

# 카테고리 매핑 참고용 (선택 사항)
CATEGORY_MAPPING = {
    "중식": 1,
    "일식집": 2,
    "브런치카페": 3,
    "파스타": 4,
    "이탈리안": 5,
    "이자카야": 6,
    "한식집": 7,
    "치킨": 8,
    "스테이크": 9,
    "고깃집": 10,
    "다이닝바": 11,
    "오마카세": 12
}

class UserData(BaseModel):
    user_id: Annotated[int, Field(alias="userId")]
    # 선호 카테고리는 최소 1개, 최대 3개 (Pydantic v2 방식)
    preferred_categories: Annotated[List[str], Field(alias="preferredCategories", min_length=1, max_length=3)]

class RecommendationEvaluationResponse(BaseModel):
    metrics: Dict[str, Any]
    status: str

class HyperparameterOptimizationRequest(BaseModel):
    n_trials: Optional[int] = 30
    timeout: Optional[int] = 300

class HyperparameterOptimizationResponse(BaseModel):
    best_parameters: Dict[str, Any]
    status: str