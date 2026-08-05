"""Domain portlari — tashqi dunyoga abstraksiyalar.

Use-case'lar AYNAN shu protokollarga bog'lanadi, konkret modellarga emas.
Shu tufayli InsightFace'ni boshqa detektorga yoki MiniFASNet'ni boshqa
anti-spoofing modeliga almashtirish uchun `application/` qatlamiga tegilmaydi.

`typing.Protocol` ishlatilgan (ABC emas) — implementatsiyalar hech narsadan
meros olishi shart emas, faqat imzoga mos kelsa yetarli (structural typing).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .entities import BoundingBox, DetectedFace, LivenessResult


@runtime_checkable
class FaceDetector(Protocol):
    """Kadrdan yuz topib, uning vektorini beruvchi model."""

    def detect(self, image: np.ndarray) -> Sequence[DetectedFace]:
        """Kadrdagi barcha yuzlarni qaytaradi (ishonch chegarasidan o'tganlari).

        Yuz topilmasa bo'sh ketma-ketlik qaytaradi — bu xato emas, qaror
        use-case qatlamida qabul qilinadi.
        """
        ...

    @property
    def embedding_dim(self) -> int:
        ...


@runtime_checkable
class LivenessChecker(Protocol):
    """Kadrdagi yuz tirik insonnikimi yoki soxta (foto/ekran) ekanini aniqlaydi."""

    def check(self, image: np.ndarray, bbox: BoundingBox) -> LivenessResult:
        ...

    @property
    def model_names(self) -> Sequence[str]:
        ...


@runtime_checkable
class ImageDecoder(Protocol):
    """Bayt oqimini rasm massiviga aylantiradi."""

    def decode(self, raw: bytes) -> np.ndarray:
        ...


@runtime_checkable
class QualityInspector(Protocol):
    """Kadr anti-spoofing uchun yetarli sifatdaligini baholaydi.

    Xira yoki juda kichik yuzda anti-spoofing modeli ishonchsiz bo'ladi,
    shuning uchun bunday kadrlar "noaniq" deb o'tkazilmaydi, balki rad etiladi.
    """

    def inspect(self, image: np.ndarray, bbox: BoundingBox) -> None:
        """Sifat yetarli bo'lmasa `LowQualityFaceError` ko'taradi."""
        ...
