"""Markazlashgan konfiguratsiya (environment variable orqali boshqariladi)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Server ---
    # HTTP (FastAPI) — hujjatlar, debug va qo'lda tekshirish uchun.
    host: str = field(default_factory=lambda: _env_str("AI_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("AI_PORT", 8000))
    # gRPC — Golang backend aynan shu orqali ulanadi.
    grpc_host: str = field(default_factory=lambda: _env_str("GRPC_HOST", "0.0.0.0"))
    grpc_port: int = field(default_factory=lambda: _env_int("GRPC_PORT", 50051))
    grpc_enabled: bool = field(default_factory=lambda: _env_bool("GRPC_ENABLED", True))
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())

    # --- Qurilma (GPU/CPU avto-moslashuv) ---
    # "auto" -> CUDA bo'lsa GPU, aks holda CPU.
    device: str = field(default_factory=lambda: _env_str("DEVICE", "auto").lower())
    gpu_id: int = field(default_factory=lambda: _env_int("GPU_ID", 0))

    # --- InsightFace ---
    insightface_model: str = field(default_factory=lambda: _env_str("INSIGHTFACE_MODEL", "buffalo_l"))
    insightface_root: str = field(default_factory=lambda: _env_str("INSIGHTFACE_ROOT", "/root/.insightface"))
    det_size: int = field(default_factory=lambda: _env_int("DET_SIZE", 640))

    # ONNX Runtime oqimlari soni (bitta inference uchun).
    #
    # 0 = avtomatik: yadrolar / parallel inference soni.
    #
    # Standart ORT qiymati — BARCHA yadro. Bir vaqtda bir nechta inference
    # ketayotganda bu ortiqcha bandlik (oversubscription) beradi: 4 ta
    # inference × 32 oqim = 128 oqim 32 ta apparat oqimida. Ular bir-birini
    # siqib chiqaradi, kesh buziladi, va o'tkazuvchanlik TUSHADI.
    #
    # Shu mashinada o'lchandi (w600k_r50, 4 parallel, i9-14900KF):
    #   intra=0 (standart)  p50=229 ms  p99=322 ms   16.6/s
    #   intra=8 (sozlangan) p50= 91 ms  p99=112 ms   39.8/s   <- 2.4x
    #   intra=16            p50=225 ms  p99=499 ms   16.3/s
    ort_intra_op_threads: int = field(
        default_factory=lambda: _env_int("ORT_INTRA_OP_THREADS", 0))
    det_score_threshold: float = field(default_factory=lambda: _env_float("DET_SCORE_THRESHOLD", 0.60))

    # --- Sifat filtri (liveness aniqligini oshiradi) ---
    min_face_size: int = field(default_factory=lambda: _env_int("MIN_FACE_SIZE", 80))
    min_blur_score: float = field(default_factory=lambda: _env_float("MIN_BLUR_SCORE", 45.0))
    max_image_side: int = field(default_factory=lambda: _env_int("MAX_IMAGE_SIDE", 1920))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

    # --- Liveness / Anti-Spoofing ---
    liveness_enabled: bool = field(default_factory=lambda: _env_bool("LIVENESS_ENABLED", True))
    liveness_model_dir: str = field(
        default_factory=lambda: _env_str("LIVENESS_MODEL_DIR", "./resources/anti_spoof_models")
    )
    # Ansambl "real" ehtimoli shu qiymatdan katta bo'lsa -> tirik inson.
    liveness_threshold: float = field(default_factory=lambda: _env_float("LIVENESS_THRESHOLD", 0.75))
    # True bo'lsa ansambldagi HAR BIR model "real" deyishi shart (FAR pasayadi, FRR biroz oshadi).
    liveness_require_unanimous: bool = field(
        default_factory=lambda: _env_bool("LIVENESS_REQUIRE_UNANIMOUS", True)
    )
    # Model yuklanmasa: True -> so'rovni rad etamiz (fail-closed, xavfsiz standart).
    liveness_fail_closed: bool = field(default_factory=lambda: _env_bool("LIVENESS_FAIL_CLOSED", True))

    # --- Embedding ---
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 512))

    @property
    def allowed_mime_types(self) -> List[str]:
        return ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"]


settings = Settings()
