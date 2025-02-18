# app/models/recommendation_schema.py

from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    id: str
    category_id: int
    score: float
    predicted_score: float
    composite_score: float

class RecommendationResponse(BaseModel):
    user: str
    recommendations: List[RecommendationItem]
