#!/usr/bin/env python3
"""Anti-Spoofing modelining to'g'ri ishlayotganini tekshiradi (regression test).

Silent-Face-Anti-Spoofing repozitoriysining namuna rasmlarini yuklab olib,
javob ma'lum bo'lgan 3 ta holatda modelni sinaydi:

    image_T1.jpg -> Real Face
    image_F1.jpg -> Fake Face
    image_F2.jpg -> Fake Face

Nima uchun kerak: MiniFASNet vaznlari 0..255 oralig'idagi kirishga o'qitilgan.
Agar kimdir preprocessing'ga `/255` qo'shib qo'ysa, model KIRISHDAN QAT'I NAZAR
doim "soxta" deydi — tizim tashqaridan "ishlayotgandek" ko'rinadi, lekin hech
kimni o'tkazmaydi. Shu skript aynan shuni ushlaydi.

Ishlatish:
    python scripts/verify_liveness.py

Docker ichida:
    docker compose exec ai-service python scripts/verify_liveness.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

import cv2
import numpy as np
import torch

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)
# Sozlamalardagi yo'llar nisbiy (`./resources/...`), shuning uchun skript
# qayerdan chaqirilganidan qat'i nazar servis ildizidan ishlaymiz.
os.chdir(SERVICE_ROOT)

from src.domain.entities import BoundingBox  # noqa: E402
from src.infrastructure.antispoofing.minifasnet_adapter import MiniFASNetChecker  # noqa: E402
from src.infrastructure.config import settings  # noqa: E402

SAMPLE_URL = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/images/sample"
SAMPLES = {"image_T1.jpg": True, "image_F1.jpg": False, "image_F2.jpg": False}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".samples")


def fetch_samples() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in SAMPLES:
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            continue
        print(f"[get ] {name}")
        urllib.request.urlretrieve(f"{SAMPLE_URL}/{name}", path)


def detect_bbox(image: np.ndarray) -> list[int] | None:
    """Yuz bbox'i [x1, y1, x2, y2]. Bu yerda InsightFace o'rniga Haar yetarli —
    tekshirilayotgan narsa detektor emas, anti-spoofing modeli."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return [int(x), int(y), int(x + w), int(y + h)]


def main() -> int:
    fetch_samples()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checker = MiniFASNetChecker(
        model_dir=settings.liveness_model_dir,
        device=device,
        threshold=settings.liveness_threshold,
        require_unanimous=settings.liveness_require_unanimous,
    )
    print(f"\nQurilma: {device} | modellar: {checker.model_names}")
    print(f"Chegara: {settings.liveness_threshold} | unanimous: {settings.liveness_require_unanimous}\n")

    failures = 0
    for name, want_real in SAMPLES.items():
        image = cv2.imread(os.path.join(CACHE_DIR, name))
        if image is None:
            print(f"❌ {name}: rasmni o'qib bo'lmadi")
            failures += 1
            continue

        bbox = detect_bbox(image)
        if bbox is None:
            print(f"❌ {name}: yuz topilmadi")
            failures += 1
            continue

        result = checker.check(image, BoundingBox(*bbox))
        want = "REAL" if want_real else "FAKE"
        got = "REAL" if result.is_real else "FAKE"
        ok = result.is_real == want_real
        failures += 0 if ok else 1

        print(f"{'✅' if ok else '❌'} {name:15s} kutilgan={want:4s} olindi={got:4s} "
              f"real_prob={result.real_probability:.4f}")
        for model_name, score in result.per_model.items():
            print(f"     {model_name:34s} {score:.4f}")

    print()
    if failures:
        print(f"XATO: {failures} ta holat mos kelmadi.")
        print("Eng ehtimoliy sabab: `src/infrastructure/antispoofing/minifasnet_adapter.py`")
        print("-> `_to_tensor()` da kirish 255 ga bo'linmoqda.")
        print("MiniFASNet 0..255 oralig'ini kutadi.")
        return 1

    print("OK — anti-spoofing modeli to'g'ri ishlayapti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
