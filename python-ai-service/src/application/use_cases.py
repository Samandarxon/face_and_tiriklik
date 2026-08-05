"""Ilova stsenariylari (application layer).

Bu qatlam domain portlarini birlashtiradi va biznes qoidalarini qo'llaydi.
FastAPI, gRPC, torch, insightface — hech biri bu yerda import qilinmaydi.
"""

from __future__ import annotations

from dataclasses import replace

import logging
from typing import Optional

import numpy as np

from ..domain.entities import DetectedFace, FaceAnalysis, LivenessResult
from ..domain.errors import (
    LivenessUnavailableError,
    MultipleFacesError,
    NoFaceDetectedError,
    SpoofDetectedError,
)
from ..domain.ports import FaceDetector, LivenessChecker, QualityInspector

logger = logging.getLogger(__name__)


class ExtractFace:
    """Kadrdan AYNAN BITTA sifatli yuzni ajratib oladi.

    Bir nechta yuz — rad etiladi: autentifikatsiyada kadrda kim borligi
    aniq bo'lishi shart, aks holda begona odamning yuzi ishlatilishi mumkin.
    """

    def __init__(self, detector: FaceDetector, quality: QualityInspector) -> None:
        self._detector = detector
        self._quality = quality

    def execute(self, image: np.ndarray, *, check_quality: bool = True) -> DetectedFace:
        faces = self._detector.detect(image)

        if not faces:
            raise NoFaceDetectedError("Rasmda yuz aniqlanmadi. Yuzingiz kameraga to'liq ko'rinsin.")
        if len(faces) > 1:
            raise MultipleFacesError(
                f"Rasmda {len(faces)} ta yuz aniqlandi. Kadrda faqat bitta inson bo'lishi kerak.",
                {"faces_found": len(faces)},
            )

        face = faces[0]
        if check_quality:
            # O'lchangan qiymatlar yuzga biriktiriladi: yuqoridagi qatlam
            # o'z, qattiqroq darvozasini qo'yishi uchun ularni ko'rishi kerak.
            face = replace(face, quality=self._quality.inspect(image, face.bbox))
        return face


class CheckLiveness:
    """STEP 1 — Anti-Spoofing.

    Soxta bo'lsa `SpoofDetectedError` ko'tariladi va jarayon SHU YERDA to'xtaydi.
    """

    def __init__(
        self,
        checker: Optional[LivenessChecker],
        *,
        enabled: bool,
        fail_closed: bool,
        threshold: float,
        unavailable_reason: Optional[str] = None,
    ) -> None:
        self._checker = checker
        self._enabled = enabled
        self._fail_closed = fail_closed
        self._threshold = threshold
        self._reason = unavailable_reason

    @property
    def available(self) -> bool:
        return self._checker is not None

    def execute(self, image: np.ndarray, face: DetectedFace) -> LivenessResult:
        if self._checker is None:
            if self._fail_closed:
                # Fail-closed: modelsiz ishlashdan ko'ra so'rovni rad etamiz.
                raise LivenessUnavailableError(
                    "Liveness moduli ishlamayapti, xavfsizlik sababli so'rov rad etildi.",
                    {"reason": self._reason or "model yuklanmagan"},
                )
            logger.warning("Liveness o'chirilgan holatda so'rov o'tkazildi (fail-open).")
            return LivenessResult(
                is_real=True, real_probability=1.0, spoof_probability=0.0,
                threshold=self._threshold, per_model={},
            )

        result = self._checker.check(image, face.bbox)
        if not result.is_real:
            raise SpoofDetectedError(
                "Soxta yuz/rasm aniqlandi. Tirik inson kameraga qarashi shart!",
                {"liveness": result.as_dict()},
            )
        return result


class EmbedFace:
    """REGISTER oqimi: liveness'siz, faqat vektor.

    Liveness bu yerda ataylab tekshirilmaydi — odamlar odatda passport skani
    yoki arxiv fotosidan ro'yxatga olinadi, ular esa ta'rifga ko'ra "spoof".
    """

    def __init__(self, extract: ExtractFace) -> None:
        self._extract = extract

    def execute(self, image: np.ndarray) -> DetectedFace:
        return self._extract.execute(image)


class VerifyFrame:
    """VERIFY oqimi: STEP 1 liveness -> STEP 2 embedding.

    Vektor detektsiya bosqichida allaqachon hisoblangan bo'ladi, lekin
    liveness yiqilsa u CHAQIRUVCHIGA UMUMAN BERILMAYDI — shu tufayli Golang
    vektor bazaga so'rov yubora olmaydi.
    """

    def __init__(self, extract: ExtractFace, liveness: CheckLiveness) -> None:
        self._extract = extract
        self._liveness = liveness

    def execute(self, image: np.ndarray) -> FaceAnalysis:
        face = self._extract.execute(image)
        result = self._liveness.execute(image, face)  # STEP 1
        return FaceAnalysis(face=face, liveness=result)  # STEP 2


class ProbeLiveness:
    """Faqat anti-spoofing (vektor qaytarilmaydi)."""

    def __init__(self, extract: ExtractFace, liveness: CheckLiveness) -> None:
        self._extract = extract
        self._liveness = liveness

    def execute(self, image: np.ndarray) -> FaceAnalysis:
        face = self._extract.execute(image)
        result = self._liveness.execute(image, face)
        return FaceAnalysis(face=face, liveness=result)
