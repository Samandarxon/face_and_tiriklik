"""`LivenessChecker` portining implementatsiyasi — MiniFASNet ansambli.

Ishlash mantiqi:
  1. Har bir model nomidan uning kirish o'lchami va `scale` koeffitsiyenti o'qiladi
     (masalan `2.7_80x80_MiniFASNetV2.pth` -> 80x80, scale=2.7).
  2. Yuz shu scale bilan kesiladi va modelga beriladi.
  3. Model 3 sinf ustidan softmax qaytaradi: [0]=spoof(print), [1]=real, [2]=spoof(replay).
  4. Ansambl natijalari o'rtachalanadi va yagona `real_probability` hosil bo'ladi.

`require_unanimous=True` bo'lganda har bir model alohida "real" deb topishi shart —
bu False Accept Rate'ni (soxta yuzni o'tkazib yuborish) sezilarli pasaytiradi.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...domain.entities import BoundingBox, LivenessResult
from .crop import crop_face
from .networks import MODEL_MAPPING
from ..model_integrity import verify_file

logger = logging.getLogger(__name__)

# "2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth",
# "org_1_80x80_MiniFASNetV1.pth" (org = kesishsiz, butun kadr).
_MODEL_NAME_RE = re.compile(
    r"^(?P<scale>org_\d+|[\d._]+?)_(?P<h>\d+)x(?P<w>\d+)_(?P<arch>MiniFASNet\w+)\.pth$"
)


@dataclass(frozen=True)
class ModelSpec:
    path: str
    name: str
    arch: str
    height: int
    width: int
    scale: Optional[float]
    kernel: Tuple[int, int]


def parse_model_name(filename: str) -> ModelSpec:
    match = _MODEL_NAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Model fayl nomi kutilgan formatga mos emas: {filename!r} "
            "(masalan: 2.7_80x80_MiniFASNetV2.pth)"
        )

    raw_scale = match.group("scale")
    if raw_scale.startswith("org"):
        scale: Optional[float] = None
    else:
        # "4_0_0" -> 4.0 ; "2.7" -> 2.7
        parts = raw_scale.split("_")
        scale = float(parts[0]) if len(parts) == 1 else float(f"{parts[0]}.{parts[1]}")

    height, width = int(match.group("h")), int(match.group("w"))
    kernel = ((height + 15) // 16, (width + 15) // 16)
    return ModelSpec(
        path="",
        name=filename,
        arch=match.group("arch"),
        height=height,
        width=width,
        scale=scale,
        kernel=kernel,
    )


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """HWC uint8 BGR -> CHW float32, qiymatlar **0..255 oralig'ida qoladi**.

    DIQQAT — bu yerda oson yo'l qo'yiladigan xato bor. `torchvision`ning
    `ToTensor()` si qiymatlarni 255 ga bo'lib [0,1] ga keltiradi, lekin
    Silent-Face-Anti-Spoofing repozitoriysi o'zining `src/data_io/transform.py`
    dagi ToTensor'idan foydalanadi va u **bo'lmaydi**. MiniFASNet vaznlari
    aynan shunday, 0..255 oralig'idagi kirish bilan o'qitilgan:
    `conv1.bn.running_mean ≈ -49`, `running_var ≈ 958` — bu qiymatlar
    [0,1] kirish uchun juda katta.

    255 ga bo'linsa BatchNorm to'yinadi va tarmoq KIRISHDAN QAT'I NAZAR
    doim bir xil natija beradi (haqiqiy yuz ham "soxta" deb topiladi).
    Regressiyani `scripts/verify_liveness.py` ushlaydi.
    """
    array = np.ascontiguousarray(image_bgr.transpose(2, 0, 1)).astype(np.float32)
    return torch.from_numpy(array)


class MiniFASNetChecker:
    """`LivenessChecker` portining MiniFASNet ansambli bilan implementatsiyasi."""

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        threshold: float = 0.75,
        require_unanimous: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold
        self.require_unanimous = require_unanimous
        self._models: List[Tuple[ModelSpec, torch.nn.Module]] = []
        self._load_all()

    # ------------------------------------------------------------------ loading

    def _load_all(self) -> None:
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Liveness model papkasi topilmadi: {self.model_dir}. "
                "`scripts/download_models.sh` ni ishga tushiring."
            )

        files = sorted(f for f in os.listdir(self.model_dir) if f.endswith(".pth"))
        if not files:
            raise FileNotFoundError(
                f"{self.model_dir} ichida `.pth` model topilmadi. "
                "`scripts/download_models.sh` ni ishga tushiring."
            )

        for filename in files:
            spec = parse_model_name(filename)
            spec = ModelSpec(**{**spec.__dict__, "path": os.path.join(self.model_dir, filename)})
            model = self._build_and_load(spec)
            self._models.append((spec, model))
            logger.info("Liveness modeli yuklandi: %s (scale=%s, %dx%d)",
                        spec.name, spec.scale, spec.height, spec.width)

    def _build_and_load(self, spec: ModelSpec) -> torch.nn.Module:
        factory = MODEL_MAPPING.get(spec.arch)
        if factory is None:
            raise ValueError(f"Noma'lum MiniFASNet arxitekturasi: {spec.arch}")

        model = factory(conv6_kernel=spec.kernel).to(self.device)
        # Xesh torch.load DAN OLDIN: almashtirilgan .pth yuklanishning o'zida
        # kod bajarishi mumkin (CVE-2025-32434).
        verify_file(spec.path)
        state_dict = torch.load(spec.path, map_location=self.device, weights_only=True)

        # DataParallel bilan saqlangan checkpoint'larda `module.` prefiksi bo'ladi.
        if next(iter(state_dict)).startswith("module."):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ----------------------------------------------------------------- inference

    @property
    def model_names(self) -> List[str]:
        return [spec.name for spec, _ in self._models]

    @torch.inference_mode()
    def check(self, image_bgr: np.ndarray, bbox: BoundingBox) -> LivenessResult:
        """InsightFace topgan yuz atrofidagi kontekst bo'yicha qaror chiqaradi.

        Har bir model uchun kadr o'z scale'i bilan kesiladi, softmax olinadi va
        natijalar o'rtachalanadi.
        """
        if not self._models:
            raise RuntimeError("Liveness modellari yuklanmagan.")

        per_model: Dict[str, float] = {}
        probabilities: List[np.ndarray] = []

        for spec, model in self._models:
            patch = crop_face(image_bgr, bbox.as_xywh(), spec.scale, spec.width, spec.height)
            tensor = _to_tensor(patch).unsqueeze(0).to(self.device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            probabilities.append(probs)
            per_model[spec.name] = float(probs[1])

        mean_probs = np.mean(probabilities, axis=0)
        real_probability = float(mean_probs[1])
        spoof_probability = float(mean_probs[0] + mean_probs[2])

        # Ansambl qarori: o'rtacha ehtimol + (ixtiyoriy) har bir modelning roziligi.
        is_real = real_probability >= self.threshold and int(np.argmax(mean_probs)) == 1
        if is_real and self.require_unanimous:
            is_real = all(score >= self.threshold for score in per_model.values())

        return LivenessResult(
            is_real=is_real,
            real_probability=real_probability,
            spoof_probability=spoof_probability,
            threshold=self.threshold,
            per_model=per_model,
        )
