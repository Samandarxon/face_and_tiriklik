"""Domain xatolari va ular uchun barqaror (stable) error kodlari.

Golang backend aynan shu `code` maydoniga tayanadi, matn (message) o'zgarishi mumkin.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AIServiceError(Exception):
    """Barcha biznes xatolari uchun asosiy sinf."""

    status_code: int = 400
    code: str = "ai_error"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class InvalidImageError(AIServiceError):
    status_code = 400
    code = "invalid_image"


class NoFaceDetectedError(AIServiceError):
    status_code = 422
    code = "no_face_detected"


class MultipleFacesError(AIServiceError):
    status_code = 422
    code = "multiple_faces"


class LowQualityFaceError(AIServiceError):
    status_code = 422
    code = "low_quality_face"


class SpoofDetectedError(AIServiceError):
    """Anti-Spoofing: kadrda tirik inson emas (foto / ekran / video)."""

    status_code = 400
    code = "spoof_detected"


class LivenessUnavailableError(AIServiceError):
    """Liveness modeli yuklanmagan va fail-closed rejim yoqilgan."""

    status_code = 503
    code = "liveness_unavailable"


class ModelNotReadyError(AIServiceError):
    status_code = 503
    code = "model_not_ready"
