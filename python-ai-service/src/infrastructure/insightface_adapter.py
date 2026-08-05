"""`FaceDetector` portining InsightFace (buffalo_l) implementatsiyasi."""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

from ..domain.entities import BoundingBox, DetectedFace
from ..domain.entities import HeadPose as DomainHeadPose
from ..domain.errors import ModelNotReadyError
from . import headpose
from .device import DeviceSelection

logger = logging.getLogger(__name__)


def _cpu_thread_budget(requested: int) -> int:
    """Bitta inference uchun nechta oqim.

    0 so'ralsa: apparat oqimlari / parallel inference soni. Ya'ni har bir
    inference o'z ulushini oladi va qo'shnisining oqimlari bilan
    urishmaydi.
    """
    if requested > 0:
        return requested
    cores = os.cpu_count() or 4
    concurrent = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", "4")))
    return max(1, cores // concurrent)


def _tune_cpu_threads(app, requested: int) -> None:
    """Sessiyalarni oqim chegarasi bilan qayta quradi.

    # Nima uchun qayta qurish kerak

    ONNX Runtime oqim sonini faqat `SessionOptions` orqali qabul qiladi,
    insightface 0.7.3 esa uni o'tkazishga yo'l bermaydi (`get_model` faqat
    `providers` va `provider_options` ni uzatadi). Shuning uchun sessiya
    yuklangandan keyin almashtiriladi — model fayli o'sha, faqat oqim
    byudjeti boshqacha.

    # Nima uchun umuman kerak

    ORT standarti — barcha yadro. Bir vaqtda 4 ta inference ketayotganda
    bu 4x ortiqcha bandlik beradi va o'tkazuvchanlik 2.4x tushadi
    (o'lchov config.py da). GPU'da bu muammo yo'q, shuning uchun faqat
    CPU yo'lida qo'llanadi.
    """
    import onnxruntime as ort

    threads = _cpu_thread_budget(requested)
    tuned = []
    for task, model in app.models.items():
        path = getattr(model, "model_file", None)
        if not path:
            continue
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            # Tugunlarni parallel bajarish CNN'da deyarli foyda bermaydi,
            # lekin yana bir oqim havzasini yaratadi — o'chirib qo'yamiz.
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model.session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"])
            tuned.append(task)
        except Exception as exc:  # sozlash muvaffaqiyatsiz bo'lsa ham servis ishlaydi
            logger.warning("'%s' sessiyasini sozlab bo'lmadi: %s", task, exc)

    if tuned:
        logger.info(
            "CPU oqimlari sozlandi: intra_op=%d, inter_op=1 (%s). "
            "Standart qiymat barcha yadroni oladi va parallel so'rovlarda "
            "o'tkazuvchanlikni tushiradi.",
            threads, ", ".join(tuned),
        )


class InsightFaceDetector:
    """buffalo_l: SCRFD detektsiya + ArcFace (w600k_r50) 512-dim embedding."""

    def __init__(
        self,
        model_name: str,
        model_root: str,
        det_size: int,
        det_score_threshold: float,
        device: DeviceSelection,
        expected_dim: int,
        intra_op_threads: int = 0,
    ) -> None:
        self._model_name = model_name
        self._model_root = model_root
        self._det_size = det_size
        self._det_score_threshold = det_score_threshold
        self._device = device
        self._expected_dim = expected_dim
        self._intra_op_threads = intra_op_threads
        self._app = None

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        logger.info("InsightFace '%s' yuklanmoqda (device=%s, providers=%s)...",
                    self._model_name, self._device.name, self._device.providers)

        app = FaceAnalysis(
            name=self._model_name,
            root=self._model_root,
            providers=self._device.providers,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=self._device.ctx_id, det_size=(self._det_size, self._det_size))
        if not self._device.is_gpu:
            _tune_cpu_threads(app, self._intra_op_threads)
        self._app = app
        logger.info("InsightFace tayyor.")

    @property
    def loaded(self) -> bool:
        return self._app is not None

    @property
    def embedding_dim(self) -> int:
        return self._expected_dim

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        if self._app is None:
            raise ModelNotReadyError("InsightFace modeli hali yuklanmagan.")

        height, width = image.shape[:2]
        faces: List[DetectedFace] = []

        for raw in self._app.get(image):
            if float(raw.det_score) < self._det_score_threshold:
                continue

            x1, y1, x2, y2 = (int(v) for v in raw.bbox)
            bbox = BoundingBox(
                x1=max(0, x1), y1=max(0, y1),
                x2=min(width, x2), y2=min(height, y2),
            )

            # `normed_embedding` L2-normalizatsiyalangan — Qdrant'dagi Cosine
            # masofa aynan shuni kutadi.
            embedding = np.asarray(raw.normed_embedding, dtype=np.float32)

            # Normallashtirilmagan vektorning UZUNLIGI — sifat signali.
            #
            # MagFace (CVPR 2021) ning asosiy kuzatuvi: margin-loss bilan
            # o'qitilgan modelda ||f|| tasvir sifati va shaxsiyat ishonchi
            # bilan korrelyatsiya qiladi. Sifatli, frontal, aniq yuzda vektor
            # uzun; xira, burchakli, yopilgan yuzda qisqa.
            #
            # ArcFace bu xossani ataylab o'rgatmaydi, lekin u qisman baribir
            # paydo bo'ladi — shuning uchun bu MagFace o'rnini bosmaydi,
            # ammo ALOHIDA MODEL YUKLAMASDAN qo'shimcha signal beradi.
            #
            # Bu QAROR emas, o'lchov: chegara yuqoridagi qatlamda qo'llanadi.
            raw_vec = getattr(raw, "embedding", None)
            norm = float(np.linalg.norm(raw_vec)) if raw_vec is not None else 0.0
            if embedding.shape[0] != self._expected_dim:
                raise ModelNotReadyError(
                    f"Kutilgan embedding o'lchami {self._expected_dim}, "
                    f"olindi {embedding.shape[0]}."
                )

            # Kalit nuqtalar aktiv liveness uchun. Detektor ularni ba'zan
            # bermaydi (juda kichik yuz, kuchli burchak) — estimate() bunday
            # holatda available=False qaytaradi va aktiv tekshiruv kadrni
            # e'tiborsiz qoldiradi.
            pose = headpose.estimate(getattr(raw, "kps", None))

            faces.append(DetectedFace(
                bbox=bbox,
                det_score=float(raw.det_score),
                embedding=embedding,
                embedding_norm=norm,
                pose=DomainHeadPose(
                    yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll,
                    available=pose.available,
                ),
            ))

        return faces
