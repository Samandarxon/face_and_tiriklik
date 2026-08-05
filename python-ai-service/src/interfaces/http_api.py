"""HTTP interfeys adapteri (FastAPI).

Bu qatlam gRPC bilan bir xil use-case'larni chaqiradi — biznes mantiq
takrorlanmaydi. HTTP asosan hujjatlar (`/docs`), qo'lda tekshirish va
monitoring uchun kerak; Golang backend gRPC orqali ulanadi.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from ..container import container
from ..domain.errors import AIServiceError, InvalidImageError
from ..infrastructure.config import settings
from ..runner import run_inference, run_io
from .http_schemas import (
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    LivenessResponse,
    VerifyFrameResponse,
)

logger = logging.getLogger("ai-service.http")
router = APIRouter()


# ------------------------------------------------------------------ yordamchi

async def _read_image(file: UploadFile):
    if file is None:
        raise InvalidImageError("`image` maydoni majburiy.")
    if file.content_type and file.content_type not in settings.allowed_mime_types:
        raise InvalidImageError(
            f"Qo'llab-quvvatlanmaydigan format: {file.content_type}.",
            {"allowed": settings.allowed_mime_types},
        )
    raw = await file.read()
    return await run_io(container.decoder.decode, raw)


def _face_body(face) -> dict:
    body = {
        "det_score": round(face.det_score, 4),
        "bbox": list(face.bbox.as_tuple()),
    }
    # O'lchangan sifat — chaqiruvchi o'z darvozasini qo'yishi uchun.
    # Bu yerdagi chegara eng past pol, undan qattig'i yuqorida qo'llanadi.
    if getattr(face, "quality", None) is not None:
        body["quality"] = face.quality.as_dict()
    if getattr(face, "embedding_norm", 0.0):
        body["embedding_norm"] = round(face.embedding_norm, 2)
    return body


# ----------------------------------------------------------------- endpointlar

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> JSONResponse:
    status = container.status().as_dict()
    payload = {"status": "ok" if status["ready"] else "degraded", **status}
    return JSONResponse(status_code=200 if status["ready"] else 503, content=payload)


@router.post(
    "/v1/face/embed",
    response_model=EmbedResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["face"],
)
async def face_embed(image: UploadFile = File(...)):
    """REGISTER oqimi: kadrda aynan 1 ta yuz borligini tekshirib, vektorini qaytaradi."""
    frame = await _read_image(image)
    face = await run_inference(container.embed_face.execute, frame)
    return {
        "success": True,
        "embedding": face.embedding.tolist(),
        "embedding_dim": face.embedding_dim,
        **_face_body(face),
    }


@router.post(
    "/v1/face/verify-frame",
    response_model=VerifyFrameResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
    tags=["face"],
)
async def face_verify_frame(image: UploadFile = File(...)):
    """VERIFY oqimi.

    STEP 1 — Anti-Spoofing. Soxta bo'lsa 400 `spoof_detected` va jarayon TO'XTAYDI.
    STEP 2 — Faqat tirik inson tasdiqlangandan keyin vektor qaytariladi.
    """
    frame = await _read_image(image)
    analysis = await run_inference(container.verify_frame.execute, frame)
    return {
        "success": True,
        "embedding": analysis.face.embedding.tolist(),
        "embedding_dim": analysis.face.embedding_dim,
        "liveness": analysis.liveness.as_dict(),
        **_face_body(analysis.face),
    }


@router.post(
    "/v1/liveness/check",
    response_model=LivenessResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["liveness"],
)
async def liveness_check(image: UploadFile = File(...)):
    """Faqat anti-spoofing (vektor hisoblanmaydi)."""
    frame = await _read_image(image)
    analysis = await run_inference(container.probe_liveness.execute, frame)
    return {
        "success": True,
        "bbox": list(analysis.face.bbox.as_tuple()),
        "liveness": analysis.liveness.as_dict(),
    }


# --------------------------------------------------------------------- ilova

def create_app(lifespan) -> FastAPI:
    app = FastAPI(
        title="Face AI Service",
        description=(
            "InsightFace (buffalo_l) + MiniFASNet Anti-Spoofing.\n\n"
            "Golang backend gRPC (:50051) orqali ulanadi; bu HTTP interfeysi "
            "hujjatlar va qo'lda tekshirish uchun."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(AIServiceError)
    async def handle_domain_error(request: Request, exc: AIServiceError) -> JSONResponse:
        logger.info("[%s] %s -> %s: %s",
                    getattr(request.state, "request_id", "-"), request.url.path, exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("[%s] Kutilmagan xato: %s", getattr(request.state, "request_id", "-"), exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(code="internal_error",
                                  message="AI servisda ichki xatolik yuz berdi.").model_dump(),
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("[%s] %s %s -> %d (%.1f ms)", request.state.request_id, request.method,
                    request.url.path, response.status_code, elapsed_ms)
        return response

    return app
