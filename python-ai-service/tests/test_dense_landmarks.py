"""Zich nuqtalar faqat passiv-birinchi liveness yoqilganda chiqishi kerak.

Model ham, rasm ham kerak emas: InsightFace soxta ilova bilan almashtiriladi.

    python -m pytest tests/test_dense_landmarks.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.infrastructure.device import DeviceSelection
from src.infrastructure.insightface_adapter import InsightFaceDetector


def _raw_face() -> SimpleNamespace:
    vector = np.ones(512, dtype=np.float32)
    return SimpleNamespace(
        det_score=0.9,
        bbox=np.array([10.0, 10.0, 200.0, 200.0]),
        normed_embedding=vector / np.linalg.norm(vector),
        embedding=vector,
        kps=np.array([[60, 80], [140, 80], [100, 120], [70, 160], [130, 160]], dtype=float),
        landmark_2d_106=np.arange(212, dtype=float).reshape(106, 2),
    )


def _detector(dense: bool) -> InsightFaceDetector:
    detector = InsightFaceDetector(
        model_name="buffalo_l", model_root="/tmp", det_size=640,
        det_score_threshold=0.6,
        device=DeviceSelection.__new__(DeviceSelection),
        expected_dim=512, dense_landmarks=dense,
    )
    detector._app = SimpleNamespace(get=lambda image: [_raw_face()])
    return detector


def test_landmarks_are_exposed_when_enabled() -> None:
    faces = _detector(True).detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert faces[0].landmarks is not None
    assert faces[0].landmarks.shape == (106, 2)
    # Proto tartibi: x0, y0, x1, y1, ...
    assert faces[0].landmarks.reshape(-1).tolist()[:3] == [0.0, 1.0, 2.0]


def test_landmarks_stay_hidden_when_disabled() -> None:
    faces = _detector(False).detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert faces[0].landmarks is None


def test_old_model_pack_without_landmarks_is_not_an_error() -> None:
    detector = _detector(True)
    raw = _raw_face()
    del raw.landmark_2d_106
    detector._app = SimpleNamespace(get=lambda image: [raw])
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))[0].landmarks is None
