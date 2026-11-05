"""Oqimda kameraga eng yaqin yuzni tanlash testlari. Model ham, rasm ham kerak emas.

Ishga tushirish:
    python -m pytest tests/test_dominant_face.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.application.use_cases import ExtractFace
from src.domain.entities import BoundingBox, DetectedFace
from src.domain.errors import MultipleFacesError, NoFaceDetectedError


def face(side: int, x: int = 0) -> DetectedFace:
    return DetectedFace(
        bbox=BoundingBox(x, 0, x + side, side), det_score=0.9, embedding=np.zeros(512)
    )


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


class FakeQuality:
    def __init__(self):
        self.seen = []

    def inspect(self, image, bbox):
        self.seen.append(bbox)
        return None


def extract(faces, ratio=2.0):
    quality = FakeQuality()
    return ExtractFace(FakeDetector(faces), quality, dominant_ratio=ratio), quality


IMAGE = np.zeros((10, 10, 3))


def test_aniq_yaqin_yuz_tanlanadi_tartibdan_qatiy_nazar():
    near, far = face(200), face(100, x=300)
    for faces in ([far, near], [near, far]):
        extractor, quality = extract(faces)
        assert extractor.execute(IMAGE).bbox == near.bbox
        assert quality.seen == [near.bbox]


def test_uch_yuzda_ikkinchi_eng_katta_bilan_solishtiriladi():
    extractor, _ = extract([face(150), face(200, x=300), face(60, x=600)])
    with pytest.raises(MultipleFacesError):
        extractor.execute(IMAGE)


def test_deyarli_teng_yuzlar_rad_etiladi():
    extractor, _ = extract([face(140), face(100, x=300)])  # maydon nisbati 1.96
    with pytest.raises(MultipleFacesError):
        extractor.execute(IMAGE)


def test_nisbat_aynan_chegarada_tanlanadi():
    near = face(200)
    extractor, _ = extract([near, DetectedFace(
        bbox=BoundingBox(300, 0, 400, 200), det_score=0.9, embedding=np.zeros(512)
    )])  # 40000 / 20000
    assert extractor.execute(IMAGE).bbox == near.bbox


def test_standart_holatda_bir_nechta_yuz_doim_rad_etiladi():
    quality = FakeQuality()
    extractor = ExtractFace(FakeDetector([face(500), face(50, x=600)]), quality)
    with pytest.raises(MultipleFacesError):
        extractor.execute(IMAGE)


def test_bitta_va_nol_yuz_ozgarmaydi():
    only = face(100)
    assert extract([only])[0].execute(IMAGE).bbox == only.bbox
    with pytest.raises(NoFaceDetectedError):
        extract([])[0].execute(IMAGE)
