"""Composition root — barcha qatlamlar shu yerda bir-biriga ulanadi.

Bu YAGONA joy bo'lib, unda konkret implementatsiyalar (InsightFace, MiniFASNet,
OpenCV) tanlanadi. Use-case va domain qatlamlari ular haqida bilmaydi.
"""

from __future__ import annotations

import logging
from typing import Optional

from .application.use_cases import CheckLiveness, EmbedFace, ExtractFace, ProbeLiveness, VerifyFrame
from .domain.entities import EngineStatus
from .infrastructure.antispoofing.minifasnet_adapter import MiniFASNetChecker
from .infrastructure.config import Settings, settings as default_settings
from .infrastructure.device import resolve_device
from .infrastructure.imaging import LaplacianQualityInspector, OpenCVImageDecoder
from .infrastructure.insightface_adapter import InsightFaceDetector

logger = logging.getLogger(__name__)


class Container:
    """Yuklangan modellar va tayyor use-case'lar."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device, settings.gpu_id)

        self.decoder = OpenCVImageDecoder(
            max_side=settings.max_image_side,
            max_bytes=settings.max_upload_bytes,
        )
        self._quality = LaplacianQualityInspector(
            min_face_size=settings.min_face_size,
            min_blur_score=settings.min_blur_score,
        )
        self._detector = InsightFaceDetector(
            model_name=settings.insightface_model,
            model_root=settings.insightface_root,
            det_size=settings.det_size,
            det_score_threshold=settings.det_score_threshold,
            device=self.device,
            expected_dim=settings.embedding_dim,
            intra_op_threads=settings.ort_intra_op_threads,
        )

        self._checker: Optional[MiniFASNetChecker] = None
        self._liveness_error: Optional[str] = None

        # Use-case'lar `load()` dan keyin to'ldiriladi.
        self.embed_face: EmbedFace
        self.verify_frame: VerifyFrame
        self.probe_liveness: ProbeLiveness

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Modellarni yuklaydi va use-case'larni yig'adi. Bloklovchi chaqiruv."""
        self._detector.load()

        if self.settings.liveness_enabled:
            self._load_liveness()
        else:
            self._liveness_error = "LIVENESS_ENABLED=false"
            logger.warning("DIQQAT: Liveness tekshiruvi o'chirilgan (faqat test uchun!).")

        extract = ExtractFace(detector=self._detector, quality=self._quality)
        liveness = CheckLiveness(
            checker=self._checker,
            enabled=self.settings.liveness_enabled,
            fail_closed=self.settings.liveness_fail_closed,
            threshold=self.settings.liveness_threshold,
            unavailable_reason=self._liveness_error,
        )

        self.embed_face = EmbedFace(extract)
        self.verify_frame = VerifyFrame(extract, liveness)
        self.probe_liveness = ProbeLiveness(extract, liveness)

    def _load_liveness(self) -> None:
        try:
            import torch

            device = torch.device(self.device.name if self.device.is_gpu else "cpu")
            self._checker = MiniFASNetChecker(
                model_dir=self.settings.liveness_model_dir,
                device=device,
                threshold=self.settings.liveness_threshold,
                require_unanimous=self.settings.liveness_require_unanimous,
            )
            logger.info("Liveness ansambli tayyor: %s", self._checker.model_names)
        except Exception as exc:
            self._checker = None
            self._liveness_error = str(exc)
            level = logging.ERROR if self.settings.liveness_fail_closed else logging.WARNING
            logger.log(level, "Liveness modelini yuklab bo'lmadi: %s", exc)

    # ---------------------------------------------------------------- status

    @property
    def ready(self) -> bool:
        if not self._detector.loaded:
            return False
        if (self.settings.liveness_enabled
                and self.settings.liveness_fail_closed
                and self._checker is None):
            return False
        return True

    def status(self) -> EngineStatus:
        return EngineStatus(
            ready=self.ready,
            device=self.device.name,
            providers=self.device.providers,
            insightface_model=self.settings.insightface_model,
            insightface_loaded=self._detector.loaded,
            embedding_dim=self.settings.embedding_dim,
            liveness_enabled=self.settings.liveness_enabled,
            liveness_loaded=self._checker is not None,
            liveness_models=self._checker.model_names if self._checker else [],
            liveness_threshold=self.settings.liveness_threshold,
            liveness_require_unanimous=self.settings.liveness_require_unanimous,
            liveness_error=self._liveness_error,
        )


# Yagona nusxa — `main.py` uni `load()` qiladi, adapterlar esa import qiladi.
container = Container()
