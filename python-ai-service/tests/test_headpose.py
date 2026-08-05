"""Bosh burilishini hisoblash testlari.

Sintetik kalit nuqtalar bilan ishlaydi — model ham, rasm ham kerak emas.
Shu sababli bu testlar CI'da har doim ishlaydi va aktiv liveness qarori
qanday chiqarilishini aniq qulflaydi.

Ishga tushirish:
    python -m pytest tests/test_headpose.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.infrastructure.headpose import MIN_EYE_DISTANCE_PX, estimate


def landmarks(nose_x: float, nose_y: float = 100.0, eye_y: float = 100.0) -> np.ndarray:
    """Ko'zlar (60, eye_y) va (140, eye_y); burun ko'rsatilgan joyda."""
    return np.array([
        [60.0, eye_y],    # chap ko'z
        [140.0, eye_y],   # o'ng ko'z
        [nose_x, nose_y], # burun
        [80.0, 160.0],    # chap lab
        [120.0, 160.0],   # o'ng lab
    ])


def rotate(points: np.ndarray, degrees: float, center: np.ndarray) -> np.ndarray:
    t = math.radians(degrees)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return np.array([center + r @ (p - center) for p in points])


# --------------------------------------------------------------- yo'nalish

def test_centred_nose_gives_zero_yaw():
    assert abs(estimate(landmarks(100.0)).yaw) < 2.0


def test_yaw_sign_follows_nose_offset():
    """Chapga burilganda manfiy, o'ngga burilganda musbat.

    Ishorani chalkashtirish "chapga qarang" buyrug'ini o'ngga burilish bilan
    bajarib bo'ladigan qilib qo'yadi — ya'ni tekshiruvni butunlay ma'nosiz.
    """
    left = estimate(landmarks(80.0)).yaw
    right = estimate(landmarks(120.0)).yaw

    assert left < -5.0, f"chapga burilishda yaw manfiy bo'lishi kerak: {left}"
    assert right > 5.0, f"o'ngga burilishda yaw musbat bo'lishi kerak: {right}"
    assert abs(left + right) < 1.0, "chap va o'ng simmetrik bo'lishi kerak"


def test_yaw_is_scale_invariant():
    """Kameradan uzoq turgan odam ham bir xil yaw berishi kerak.

    Aks holda chegara faqat ma'lum masofada to'g'ri ishlardi.
    """
    near = estimate(landmarks(120.0)).yaw
    far = estimate(np.array([
        [20.0, 100.0], [180.0, 100.0], [140.0, 100.0],
        [60.0, 220.0], [140.0, 220.0],
    ])).yaw

    assert abs(near - far) < 2.0, f"masshtab yaw'ga ta'sir qildi: {near} vs {far}"


def test_head_tilt_does_not_leak_into_yaw():
    """Boshni YONBOSHLATISH yaw hosil qilmasligi kerak.

    Bu aldash yo'li: agar siljish oddiy X o'qi bo'ylab o'lchansa, boshni egish
    burunni yon tomonga surib, "chapga qarang" buyrug'ini burilmasdan
    bajarilgandek ko'rsatardi. Siljish ko'zlar CHIZIG'I bo'ylab o'lchanadi.
    """
    centre = np.array([100.0, 100.0])
    tilted = rotate(landmarks(100.0), 30.0, centre)
    tilted[2] = centre  # burun ko'zlar markazida qoladi

    pose = estimate(tilted)
    assert abs(pose.yaw) < 5.0, f"egilish yaw'ga sizib chiqdi: {pose.yaw}"
    assert abs(abs(pose.roll) - 30.0) < 5.0, f"roll noto'g'ri: {pose.roll}"


# ------------------------------------------------------- xavfsiz rad etish

@pytest.mark.parametrize("bad", [
    None,
    np.zeros((3, 2)),               # nuqtalar yetarli emas
    np.zeros((5, 3)),               # noto'g'ri o'lcham
    np.full((5, 2), np.nan),        # NaN
    np.full((5, 2), np.inf),        # cheksizlik
])
def test_unusable_input_is_reported_not_guessed(bad):
    """Noto'g'ri kirishda taxmin qilinmaydi.

    Noto'g'ri yaw aktiv tekshiruvni aldab o'tkazishi mumkin, shuning uchun
    "bilmayman" javobi taxmindan xavfsizroq.
    """
    assert estimate(bad).available is False


def test_tiny_face_is_rejected():
    """Yuz juda kichik bo'lsa piksel shovqini burun siljishidan katta bo'ladi."""
    d = MIN_EYE_DISTANCE_PX / 2
    tiny = np.array([
        [100.0 - d / 2, 100.0], [100.0 + d / 2, 100.0],
        [100.0, 100.0], [99.0, 110.0], [101.0, 110.0],
    ])
    assert estimate(tiny).available is False


def test_valid_input_is_available():
    assert estimate(landmarks(100.0)).available is True
