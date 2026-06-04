from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to classify")
    return_attention: bool = Field(False, description="Return token-level attention weights")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64, description="List of texts (max 64)")
    return_attention: bool = False

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        return [t.strip() for t in v if t.strip()]


class SentimentResult(BaseModel):
    text: str
    label: str
    emoji: str
    confidence: float
    probabilities: Dict[str, float]
    tokens: Optional[List[str]] = None
    attention: Optional[List] = None


class PredictResponse(BaseModel):
    result: SentimentResult
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[SentimentResult]
    latency_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    total_predictions: int


class MetricsResponse(BaseModel):
    latency: Dict
    total_predictions: int
    uptime_seconds: float
