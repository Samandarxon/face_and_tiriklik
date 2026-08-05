"""Domain ob'ektlari — freymvorksiz, faqat standart kutubxona + numpy.

Clean Architecture: bu qatlam FastAPI, gRPC, torch yoki insightface haqida
hech narsa bilmaydi. Shu tufayli biznes qoidalarini modellarsiz test qilish
mumkin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # faqat tip tekshiruvi uchun — ishlash vaqtida import yo'q
    from ..infrastructure.imaging import FaceQuality


@dataclass(frozen=True)
class BoundingBox:
    """Yuz koordinatalari (piksel, chap-yuqori boshlang'ich)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    def as_xywh(self) -> Tuple[int, int, int, int]:
        """Anti-spoofing modellari [x, y, w, h] formatini kutadi."""
        return self.x1, self.y1, self.width, self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class HeadPose:
    """Boshning burilishi (gradus). Aktiv liveness uchun.

    `available=False` bo'lsa qolgan maydonlar ma'nosiz — detektor kalit
    nuqtalarni bermagan yoki yuz juda kichik. Bunday kadr aktiv tekshiruvda
    ishlatilmaydi: noto'g'ri yaw tekshiruvni aldab o'tkazishi mumkin.
    """

    yaw: float
    pitch: float
    roll: float
    available: bool


@dataclass(frozen=True)
class DetectedFace:
    """Kadrdan ajratib olingan yuz."""

    bbox: BoundingBox
    det_score: float
    embedding: np.ndarray  # 512 o'lchamli, L2-normalizatsiyalangan
    # Aktiv liveness uchun. Eski oqimlar buni ishlatmaydi, shuning uchun
    # standart qiymat bilan — mavjud chaqiruvchilar o'zgarmaydi.
    pose: Optional[HeadPose] = None
    # Sifat o'lchovlari — tekshiruv o'tkazilganda to'ldiriladi. Bu QAROR
    # emas, FAKT: past pol shu yerda tekshirilgan, qattiqroq talabni esa
    # chaqiruvchi qatlam shu qiymatlar asosida qo'yadi.
    quality: Optional["FaceQuality"] = None
    # Normallashtirilmagan vektor uzunligi — MagFace uslubidagi sifat
    # signali. 0 = o'lchanmagan.
    embedding_norm: float = 0.0

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[0])


@dataclass(frozen=True)
class LivenessResult:
    """Anti-spoofing ansamblining qarori."""

    is_real: bool
    real_probability: float
    spoof_probability: float
    threshold: float
    per_model: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "is_real": self.is_real,
            "real_probability": round(self.real_probability, 4),
            "spoof_probability": round(self.spoof_probability, 4),
            "threshold": self.threshold,
            "per_model": {k: round(v, 4) for k, v in self.per_model.items()},
        }


@dataclass(frozen=True)
class FaceAnalysis:
    """To'liq tahlil natijasi (verify oqimi)."""

    face: DetectedFace
    liveness: LivenessResult | None = None


@dataclass(frozen=True)
class EngineStatus:
    """Servis holati (health uchun)."""

    ready: bool
    device: str
    providers: Sequence[str]
    insightface_model: str
    insightface_loaded: bool
    embedding_dim: int
    liveness_enabled: bool
    liveness_loaded: bool
    liveness_models: Sequence[str]
    liveness_threshold: float
    liveness_require_unanimous: bool
    liveness_error: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "ready": self.ready,
            "device": self.device,
            "providers": list(self.providers),
            "insightface_model": self.insightface_model,
            "insightface_loaded": self.insightface_loaded,
            "embedding_dim": self.embedding_dim,
            "liveness_enabled": self.liveness_enabled,
            "liveness_loaded": self.liveness_loaded,
            "liveness_models": list(self.liveness_models),
            "liveness_threshold": self.liveness_threshold,
            "liveness_require_unanimous": self.liveness_require_unanimous,
            "liveness_error": self.liveness_error,
        }
