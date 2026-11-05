"""Tekis yuzaning (surat) real 2d106det nuqtalarini yig'adi — passiv liveness uchun.

InsightFace bilan keladigan namunaviy rasm (`t1.jpg`) 3D fazoda aylantiriladigan
tekislik sifatida kadrlarga proyeksiya qilinadi va model har kadrni baholaydi.
Natija Go testiga fixture bo'ladi: u gomografiya emas, modelning HAQIQIY
xatolari bilan suratni ushlab turganda nima chiqishini ko'rsatadi.

    docker exec -i face-ai python - < scripts/passive_flat_fixture.py 2>/dev/null | grep '^{' > flat.jsonl

Faqat suratni o'lchaydi. Tirik odam uchun chegara bundan chiqmaydi — u real
kamera logi bilan sozlanadi (YUZ_IMKONIYATLARI_VA_MODELLAR.md).
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore")

FRAMES = 12
SIZES = (130, 200)
SAMPLE = os.path.join(os.path.dirname(insightface.__file__), "data/images/t1.jpg")
KAMERA = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])

rng = np.random.default_rng(7)
app = FaceAnalysis(
    name="buffalo_l", root=os.getenv("INSIGHTFACE_ROOT", "/models/insightface"),
    providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    a, b, c = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


def plane(size: int, yaw=0.0, pitch=0.0, roll=0.0, tx=0.0, ty=0.0, zoom=1.0) -> np.ndarray:
    """Yuz kengligi 15 sm, kamera f=600 bo'lgan tekislikning gomografiyasi."""
    k = 15 / size
    r = rotation(yaw, pitch, roll)
    t = np.array([tx, ty, 600 * k / zoom])
    back = np.array([[k, 0, -320 * k], [0, k, -240 * k], [0, 0, 1]])
    return KAMERA @ np.c_[r[:, 0], r[:, 1], t] @ back


def shot(picture: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Webcam'ga o'xshash: shovqin va JPEG (mijoz 0.82 sifat bilan yuboradi)."""
    out = cv2.warpPerspective(picture, matrix, (640, 480), borderValue=(128, 128, 128))
    out = np.clip(out + rng.normal(0, 2.0, out.shape), 0, 255).astype(np.uint8)
    _, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def centred(image: np.ndarray, face, size: int) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    margin = int((x2 - x1) * 0.8)
    x0, y0 = max(0, x1 - margin), max(0, y1 - margin)
    crop = image[y0:y2 + margin, x0:x2 + margin]
    s = size / (x2 - x1)
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    shift = np.array([[1, 0, 320 - ((x1 + x2) / 2 - x0) * s], [0, 1, 240 - ((y1 + y2) / 2 - y0) * s], [0, 0, 1.0]])
    return cv2.warpPerspective(crop, shift, (640, 480), borderValue=(128, 128, 128))


def bent(picture: np.ndarray, size: int, depth_cm: float, yaw: float) -> np.ndarray:
    """Vertikal o'q bo'ylab egilgan qog'oz (taxminiy proyeksiya)."""
    k = 15 / size
    xs, ys = np.meshgrid(np.arange(640), np.arange(480))
    x, y = (xs - 320) * k, (ys - 240) * k
    a = np.radians(yaw)
    z = -depth_cm * (x / 7.5) ** 2
    u = (x * np.cos(a) + z * np.sin(a)) / (1 + (z * np.cos(a) - x * np.sin(a)) / (600 * k))
    return cv2.remap(picture, (2 * x - u).astype(np.float32) / k + 320, ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=(128, 128, 128))


def motions(picture: np.ndarray, size: int):
    g = lambda **kw: shot(picture, plane(size, **kw))
    wave = lambda i, a, ph: a * np.sin(i * 0.8 + ph)
    yield "still", lambda i: g()
    yield "tremor", lambda i: g(yaw=rng.normal(0, 0.6), pitch=rng.normal(0, 0.6), roll=rng.normal(0, 0.6), tx=rng.normal(0, 0.2), ty=rng.normal(0, 0.2))
    for amp in (5, 10, 15, 25):
        ph = rng.uniform(0, 6)
        yield f"yaw{amp}", lambda i, a=amp, p=ph: g(yaw=wave(i, a, p))
        yield f"pitch{amp}", lambda i, a=amp, p=ph: g(pitch=wave(i, a, p))
    for amp in (8, 15):
        yield f"wobble{amp}", lambda i, a=amp: g(yaw=rng.normal(0, a), pitch=rng.normal(0, a), roll=rng.normal(0, 5), tx=rng.normal(0, 1), ty=rng.normal(0, 1), zoom=1 + rng.normal(0, 0.05))
    yield "blur", lambda i: cv2.GaussianBlur(g(yaw=rng.normal(0, 8), pitch=rng.normal(0, 8)), (5, 5), 0)
    yield "bent", lambda i: shot(bent(picture, size, 3.0, 8 * np.sin(i * 0.8)), np.eye(3))


def largest(image: np.ndarray):
    found = app.get(image)
    return max(found, key=lambda f: f.bbox[2] - f.bbox[0]) if found else None


image = cv2.imread(SAMPLE)
for index, face in enumerate(sorted(app.get(image), key=lambda f: f.bbox[0])):
    for size in SIZES:
        picture = centred(image, face, size)
        for name, frame in motions(picture, size):
            seq = []
            for i in range(FRAMES):
                got = largest(frame(i))
                if got is not None:
                    seq.append([round(float(v), 2) for v in got.landmark_2d_106.reshape(-1)])
            json.dump({"motion": name, "face": index, "size": size, "frames": seq}, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
