"""Pydantic javob sxemalari (OpenAPI hujjatlari uchun)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LivenessInfo(BaseModel):
    is_real: bool
    real_probability: float = Field(..., description="Ansambl bo'yicha 'tirik' ehtimoli [0..1]")
    spoof_probability: float
    threshold: float
    per_model: Dict[str, float] = Field(default_factory=dict)


class EmbedResponse(BaseModel):
    success: bool = True
    embedding: List[float]
    embedding_dim: int
    det_score: float
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")


class VerifyFrameResponse(EmbedResponse):
    liveness: LivenessInfo


class LivenessResponse(BaseModel):
    success: bool = True
    bbox: List[int]
    liveness: LivenessInfo


class ErrorResponse(BaseModel):
    success: bool = False
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    device: str
    providers: List[str]
    insightface_model: str
    insightface_loaded: bool
    embedding_dim: int
    liveness_enabled: bool
    liveness_loaded: bool
    liveness_models: List[str]
    liveness_threshold: float
    liveness_require_unanimous: bool
    liveness_error: Optional[str] = None
    ready: bool
