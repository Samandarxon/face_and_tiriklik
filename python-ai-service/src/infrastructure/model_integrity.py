"""Model fayllarining yaxlitligi (SHA-256).

# Nima uchun

Model fayllari tashqi manbadan (GitHub) yuklanadi. `.pth` ni `torch.load`
o'qiydi — torch 2.4 da `weights_only=True` ham to'liq himoya emas
(CVE-2025-32434), ya'ni almashtirilgan `.pth` kod bajarishi mumkin. ONNX
almashtirilsa esa tanish yoki anti-spoofing jimgina boshqa natija beradi.

Shuning uchun har bir fayl yuklanishdan OLDIN ma'lum xesh bilan
solishtiriladi va mos kelmasa servis ishga tushmaydi (fail-closed).

Xeshlar rasmiy manbalardan olingan va 2026-10-27 da tekshirilgan:
  - buffalo_l: github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  - MiniFASNet: github.com/minivision-ai/Silent-Face-Anti-Spoofing (resources/anti_spoof_models)

Boshqa model to'plami (masalan, o'zingiz kvantlagan) ishlatilsa, uning
xeshlarini MODEL_SHA256_FILE ("<sha256>  <fayl nomi>" qatorlari) orqali
bering.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

KNOWN_SHA256: dict[str, str] = {
    # buffalo_l
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    # MiniFASNet
    "2.7_80x80_MiniFASNetV2.pth": "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    "4_0_0_80x80_MiniFASNetV1SE.pth": "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
}


class ModelIntegrityError(RuntimeError):
    """Model fayli kutilgan xeshga mos kelmadi yoki noma'lum."""


def _extra_hashes() -> dict[str, str]:
    path = os.environ.get("MODEL_SHA256_FILE", "").strip()
    if not path:
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                out[os.path.basename(parts[1])] = parts[0].lower()
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: str) -> None:
    """Faylni tekshiradi; mos kelmasa yoki noma'lum bo'lsa ModelIntegrityError."""
    if os.environ.get("MODEL_INTEGRITY_CHECK", "true").lower() == "false":
        logger.warning("MODEL_INTEGRITY_CHECK=false — %s tekshirilmadi", path)
        return
    name = os.path.basename(path)
    expected = {**KNOWN_SHA256, **_extra_hashes()}.get(name)
    if expected is None:
        raise ModelIntegrityError(
            f"{name}: noma'lum model fayli. Xeshini MODEL_SHA256_FILE orqali bering."
        )
    actual = sha256_of(path)
    if actual != expected:
        raise ModelIntegrityError(f"{name}: SHA-256 mos kelmadi — fayl almashtirilgan yoki buzilgan.")
    logger.info("Model yaxlitligi tasdiqlandi: %s", name)


def verify_dir(directory: str, suffix: str) -> None:
    """Katalogdagi `suffix` bilan tugaydigan HAR bir faylni tekshiradi."""
    for name in sorted(os.listdir(directory)):
        if name.endswith(suffix):
            verify_file(os.path.join(directory, name))
