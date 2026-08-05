"""Rasm dekodlash va sifat baholash — `ImageDecoder`, `QualityInspector` portlari."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..domain.entities import BoundingBox
from ..domain.errors import InvalidImageError, LowQualityFaceError


class OpenCVImageDecoder:
    """Bayt oqimini BGR `np.ndarray` ga aylantiradi."""

    def __init__(self, max_side: int, max_bytes: int) -> None:
        self._max_side = max_side
        self._max_bytes = max_bytes

    def decode(self, raw: bytes) -> np.ndarray:
        if not raw:
            raise InvalidImageError("Rasm fayli bo'sh.")
        if len(raw) > self._max_bytes:
            raise InvalidImageError(
                f"Rasm hajmi juda katta (maksimum {self._max_bytes // (1024 * 1024)} MB)."
            )

        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Rasmni o'qib bo'lmadi. JPEG/PNG/WEBP formatini yuboring.")

        if image.ndim == 2:  # grayscale -> BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return self._limit_size(image)

    def _limit_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self._max_side:
            return image
        scale = self._max_side / float(longest)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class FaceQuality:
    """O'lchangan sifat ko'rsatkichlari — qaror emas, fakt."""

    face_size: int
    blur_score: float

    def as_dict(self) -> dict:
        return {"face_size": self.face_size, "blur_score": self.blur_score}


class LaplacianQualityInspector:
    """Yuz o'lchami va aniqligini (blur) tekshiradi.

    Anti-spoofing modeli xira yoki juda kichik yuzda ishonchsiz bo'ladi.
    Bunday kadrni "noaniq" deb o'tkazib yuborish xavfsizlik teshigi bo'lardi,
    shuning uchun uni ochiq rad etamiz.
    """

    def __init__(self, min_face_size: int, min_blur_score: float) -> None:
        self._min_face_size = min_face_size
        self._min_blur_score = min_blur_score

    def inspect(self, image: np.ndarray, bbox: BoundingBox) -> "FaceQuality":
        """Tekshiradi VA o'lchangan qiymatlarni qaytaradi.

        Qaytarish muhim: bu yerdagi chegara — eng past, majburiy pol.
        Undan qattiqroq talab (masalan bazaga yozishda) chaqiruvchi qatlamda
        qo'llanadi, chunki qaror siyosati bu yerda emas. Qiymatlarni
        qaytarmasak, yuqoridagi qatlam ularni ko'rmasdi va o'z darvozasini
        qo'ya olmasdi.
        """
        if bbox.min_side < self._min_face_size:
            raise LowQualityFaceError(
                "Yuz juda kichik. Kameraga yaqinroq turing.",
                {"face_size": bbox.min_side, "min_required": self._min_face_size},
            )

        sharpness = self._blur_score(image, bbox)
        if sharpness < self._min_blur_score:
            raise LowQualityFaceError(
                "Rasm xira chiqdi. Kamerani barqaror ushlang va yorug'likni yaxshilang.",
                {"blur_score": round(sharpness, 2), "min_required": self._min_blur_score},
            )
        return FaceQuality(face_size=bbox.min_side, blur_score=round(sharpness, 2))

    @staticmethod
    def _blur_score(image: np.ndarray, bbox: BoundingBox) -> float:
        """Laplasian dispersiyasi: qiymat qanchalik past bo'lsa, rasm shunchalik xira."""
        h, w = image.shape[:2]
        x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
        x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
