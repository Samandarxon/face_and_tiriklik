"""GPU/CPU avto-moslashuv.

Bitta joyda hal qilinadi va ikkala model (InsightFace/ONNX va MiniFASNet/torch)
bir xil qurilmani ishlatadi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSelection:
    name: str            # "cpu" yoki "cuda:0"
    ctx_id: int          # InsightFace: >=0 GPU, -1 CPU
    providers: List[str]  # onnxruntime execution providers

    @property
    def is_gpu(self) -> bool:
        return self.ctx_id >= 0


def resolve_device(preference: str, gpu_id: int) -> DeviceSelection:
    """`preference`: "auto" | "cpu" | "cuda"/"gpu"."""
    preference = (preference or "auto").lower()
    want_gpu = preference in {"auto", "cuda", "gpu"}
    cuda_available = False

    if want_gpu:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            # `is_available()` YETARLI EMAS.
            #
            # U qurilma bor-yo'qligini aytadi, lekin o'rnatilgan torch shu
            # qurilma arxitekturasi uchun kompilyatsiya qilinganmi — yo'q.
            # Yangi kartada (masalan RTX 50xx, sm_120) eski torch bilan u
            # True qaytaradi, model yuklanadi, va faqat BIRINCHI inference'da
            # yiqiladi: "no kernel image is available for execution".
            #
            # Bu eng yomon nosozlik turi — ishga tushish muvaffaqiyatli
            # ko'rinadi, xato esa birinchi haqiqiy foydalanuvchida chiqadi.
            if cuda_available and not _arch_supported(torch, gpu_id):
                cuda_available = False
        except Exception as exc:  # pragma: no cover - torch bo'lmagan holat
            logger.warning("torch.cuda tekshiruvi muvaffaqiyatsiz: %s", exc)

        if cuda_available:
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    logger.warning(
                        "CUDA GPU bor, lekin onnxruntime-gpu o'rnatilmagan. "
                        "InsightFace CPU'da ishlaydi."
                    )
            except Exception:  # pragma: no cover
                pass

    if preference in {"cuda", "gpu"} and not cuda_available:
        logger.warning("DEVICE=%s so'ralgan, lekin CUDA topilmadi — CPU'ga o'tildi.", preference)

    if cuda_available:
        return DeviceSelection(
            name=f"cuda:{gpu_id}",
            ctx_id=gpu_id,
            providers=_gpu_providers(),
        )
    return DeviceSelection(name="cpu", ctx_id=-1, providers=["CPUExecutionProvider"])


def _arch_supported(torch, gpu_id: int) -> bool:
    """O'rnatilgan torch shu GPU arxitekturasini qo'llab-quvvatlaydimi.

    Ikki yo'l bilan qo'llab-quvvatlanadi:
      1. Aniq moslik — `sm_120` build ro'yxatida bor;
      2. PTX forward-compat — build'da `compute_XX` bor va XX <= qurilma
         darajasi; u holda drayver ishga tushishda JIT qiladi (sekinroq,
         lekin ishlaydi).
    """
    try:
        major, minor = torch.cuda.get_device_capability(gpu_id)
    except Exception as exc:
        logger.warning("GPU imkoniyatini o'qib bo'lmadi: %s", exc)
        return False

    device_cap = major * 10 + minor
    arches = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    if not arches:
        # Ro'yxat bo'sh — CPU build. `is_available()` bunda False bo'lishi
        # kerak edi, lekin ishonmaymiz.
        logger.warning("torch CPU uchun qurilgan (arch ro'yxati bo'sh) — CPU'da ishlaymiz.")
        return False

    if f"sm_{device_cap}" in arches:
        return True

    ptx = [int(a.split("_")[1]) for a in arches if a.startswith("compute_") and a.split("_")[1].isdigit()]
    if ptx and max(ptx) <= device_cap:
        logger.info(
            "GPU sm_%d uchun aniq kernel yo'q, PTX (compute_%d) orqali JIT qilinadi — "
            "birinchi ishga tushish sekinroq bo'ladi.",
            device_cap, max(ptx),
        )
        return True

    logger.warning(
        "GPU sm_%d, lekin o'rnatilgan torch faqat %s uchun qurilgan — CPU'ga o'tildi. "
        "GPU'ni yoqish uchun requirements-gpu.txt dagi arxitektura jadvaliga qarang.",
        device_cap, ",".join(arches),
    )
    return False


def _gpu_providers() -> List[str]:
    """ONNX Runtime provayderlari, tezligi bo'yicha tartibda.

    TensorRT eng tez (≈1.5-2x), lekin har GPU modeli uchun alohida engine
    kompilyatsiya qiladi — birinchi ishga tushish bir necha daqiqa oladi.
    Mavjud bo'lmasa ORT o'zi CUDA'ga tushadi, shuning uchun ro'yxatga
    shartsiz qo'shish xavfsiz.
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    chain = []
    if "TensorrtExecutionProvider" in available:
        chain.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    else:
        logger.warning(
            "CUDA GPU topildi, lekin onnxruntime-gpu o'rnatilmagan — "
            "InsightFace CPU'da ishlaydi (MiniFASNet GPU'da qoladi)."
        )
    chain.append("CPUExecutionProvider")
    return chain
