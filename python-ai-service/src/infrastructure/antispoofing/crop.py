"""Anti-spoofing uchun yuz atrofidagi kontekstni saqlagan holda kesish.

MiniFASNet modellari yuzning o'zi emas, yuz ATROFIDAGI fon bilan birga bo'lgan
patch ustida o'qitilgan — qog'oz cheti, telefon ramkasi, moire naqshi kabi
spoof belgilarini aynan shu kontekst ochib beradi. Shuning uchun har bir model
o'zining `scale` koeffitsiyenti bilan kengaytirilgan bbox'dan foydalanadi.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def _expand_box(src_w: int, src_h: int, bbox: Sequence[int], scale: float) -> Tuple[int, int, int, int]:
    x, y, box_w, box_h = bbox

    # Kengaytirish rasm chegarasidan chiqib ketmasligi kerak.
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    # Chegaradan chiqsa — kesib tashlamay, oynani ichkariga suramiz (o'lcham saqlanadi).
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop_face(
    image: np.ndarray,
    bbox: Sequence[int],
    scale: Optional[float],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """`bbox` = [x, y, w, h]. `scale` None bo'lsa butun kadr resize qilinadi."""
    if scale is None:
        return cv2.resize(image, (out_w, out_h))

    src_h, src_w = image.shape[:2]
    x1, y1, x2, y2 = _expand_box(src_w, src_h, bbox, scale)
    patch = image[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        return cv2.resize(image, (out_w, out_h))
    return cv2.resize(patch, (out_w, out_h))
