#!/usr/bin/env python3
"""O'LCHASH HARNESS — CPU-only serverda AI oqimining unumdorligini o'lchaydi.

NIMA UCHUN BU FAYL ENG MUHIM
============================
Optimizatsiya haqidagi har qanday qaror (DET_SIZE ni tushirish, INT8 kvantlash,
liveness'ni ONNX'ga o'tkazish, thread sonini o'zgartirish) faqat O'LCHOV
asosida qabul qilinishi kerak. O'lchovsiz "optimizatsiya" — bu taxmin, va
xavfsizlik tizimida taxmin qimmatga tushadi.

Bu skript hech qanday sozlamani O'ZGARTIRMAYDI. U faqat o'lchaydi va
`.env` ga qo'lda kiritishingiz mumkin bo'lgan raqamlarni ko'rsatadi.

NIMA O'LCHANADI
===============
1. LATENSIYA — bosqichlar ALOHIDA:
      detection  (SCRFD)      — odatda oqimdagi eng katta tor joy
      embedding  (ArcFace)    — yuz soniga chiziqli bog'liq
      liveness   (MiniFASNet) — ansambl, ya'ni modellar soniga ko'paytiriladi
   Har biri uchun p50 / p95 / p99. O'rtacha (mean) yolg'on tinchlik beradi:
   foydalanuvchi p95 ni sezadi, p50 ni emas.

2. THROUGHPUT — 1 / 2 / 4 / 8 parallel so'rovda. Latensiya yaxshi bo'lib,
   throughput yomon bo'lishi mumkin (va aksincha).

3. DET_SIZE 640 va 320 taqqoslash. ★ Skript nafaqat tezlikni, balki HAR BIR
   o'lchamda TOPILGAN YUZLAR SONINI ham ko'rsatadi. 320 doim tezroq — savol
   shundaki, u qancha yuzni YO'QOTADI. Tezlikka qarab qaror qabul qilib,
   aniqlikni ko'rmaslik — eng ko'p uchraydigan xato.

4. XOTIRA — RSS. CPU-only serverda bitta mashinaga nechta worker sig'ishini
   aynan shu raqam belgilaydi.

5. ★ THREAD SOZLAMALARI (intra_op / inter_op). Bu ko'pincha e'tibordan
   chetda qoladi, holbuki noto'g'ri sozlash throughput'ni PASAYTIRADI.
   Klassik xato: 16 yadroli serverda `intra_op=16` qo'yib, ustiga 4 ta worker
   ishga tushirish. Natijada 64 ta thread 16 ta yadro uchun kurashadi,
   kontekst almashinuvi vaqtning yarmini yeydi.

   To'g'ri qoida:   intra_op × worker_soni ≈ yadrolar soni

Ishlatish:
    python scripts/benchmark.py --images ./bench_frames
    python scripts/benchmark.py --images ./bench_frames --det-sizes 640,320 \
        --thread-configs 1:1,2:1,4:1,8:1 --concurrency 1,2,4,8
    python scripts/benchmark.py --images ./bench_frames --json natija.json

    # Liveness'ni PyTorch o'rniga ONNX'da o'lchash (export_onnx.py dan keyin):
    python scripts/benchmark.py --images ./bench_frames --liveness-onnx-dir resources/anti_spoof_models
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ORIGINAL_CWD = os.getcwd()
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)
os.chdir(SERVICE_ROOT)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ONNX Runtime sessiyalariga uzatiladigan sozlamalar. Patch orqali
# insightface ichidagi sessiya yaratishga "ilashtiriladi" (pastga qarang).
_SESSION_OPTIONS: Dict[str, object] = {"opts": None}


def _die(message: str, code: int = 2) -> None:
    print(f"\nXATO: {message}", file=sys.stderr)
    raise SystemExit(code)


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError:
        _die(f"`{module}` o'rnatilmagan. {hint}")


def _user_path(path: Optional[str]) -> Optional[str]:
    """CLI'dan kelgan nisbiy yo'lni ORIGINAL papkaga nisbatan hal qiladi.

    Skript servis ildiziga `chdir` qiladi (sozlamalardagi yo'llar nisbiy),
    shuning uchun foydalanuvchi bergan yo'llarni oldindan absolyutlashtiramiz.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ORIGINAL_CWD, path))


# ------------------------------------------------------------------ o'lchovlar


@dataclass
class LatencyStats:
    """Bitta bosqichning latensiya statistikasi (millisekundlarda)."""

    stage: str
    count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @classmethod
    def build(cls, stage: str, samples: Sequence[float]) -> "LatencyStats":
        if not samples:
            return cls(stage=stage)
        ordered = sorted(samples)

        def percentile(fraction: float) -> float:
            # Nearest-rank: rank = ceil(p * N), indeks = rank - 1.
            # Interpolyatsiyali usuldan ko'ra halolroq — chunki u haqiqatan
            # o'lchangan qiymatni qaytaradi, o'ylab topilganini emas.
            index = math.ceil(fraction * len(ordered)) - 1
            return ordered[max(0, min(len(ordered) - 1, index))]

        return cls(
            stage=stage,
            count=len(ordered),
            p50=percentile(0.50),
            p95=percentile(0.95),
            p99=percentile(0.99),
            mean=statistics.fmean(ordered),
            min=ordered[0],
            max=ordered[-1],
        )


@dataclass
class ConfigResult:
    """Bitta (det_size, intra_op, inter_op) konfiguratsiyasi natijasi."""

    det_size: int
    intra_op: int
    inter_op: int
    liveness_backend: str
    faces_found: int = 0
    frames_with_no_face: int = 0
    rss_after_load_mb: float = 0.0
    peak_rss_mb: float = 0.0
    latency: Dict[str, dict] = field(default_factory=dict)
    throughput: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------- thread patch


def _install_thread_patch() -> Tuple[bool, str]:
    """insightface ichidagi ONNX sessiyalarga o'z `SessionOptions`imizni ilashtiradi.

    Nima uchun patch kerak: `FaceAnalysis.prepare()` thread sozlamalarini
    tashqaridan qabul qilmaydi, sessiyalarni o'zi yaratadi. ONNX Runtime esa
    intra_op/inter_op ni environment variable orqali o'qimaydi. Ya'ni patch'siz
    bu benchmark'ning thread ustuni SOXTA bo'lardi — raqamlar o'zgarardi-yu,
    aslida hech narsa qo'llanilmagan bo'lardi.

    Patch ishlamasa — buni JIM o'tkazib yubormaymiz, natijaga izoh qo'shamiz.
    """
    try:
        from insightface.model_zoo import model_zoo as mz
    except Exception as exc:  # noqa: BLE001
        return False, f"insightface.model_zoo import qilinmadi ({exc})"

    base = getattr(mz, "PickableInferenceSession", None)
    if base is None:
        return False, "PickableInferenceSession topilmadi (insightface versiyasi boshqa?)"

    if getattr(base, "_bench_patched", False):
        return True, ""

    original_init = base.__init__

    def _patched_init(self, model_path, **kwargs):  # noqa: ANN001
        opts = _SESSION_OPTIONS.get("opts")
        if opts is not None and kwargs.get("sess_options") is None:
            kwargs["sess_options"] = opts
        original_init(self, model_path, **kwargs)

    base.__init__ = _patched_init
    base._bench_patched = True
    return True, ""


def _make_session_options(intra_op: int, inter_op: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op
    opts.inter_op_num_threads = inter_op
    # inter_op faqat PARALLEL rejimda ma'noga ega. SEQUENTIAL'da ORT uni
    # e'tiborsiz qoldiradi — shuni ochiq boshqaramiz.
    opts.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL if inter_op > 1 else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    return opts


def _set_torch_threads(intra_op: int, inter_op: int, notes: List[str]) -> None:
    """MiniFASNet PyTorch'da ishlaydi — uning thread pool'i ORT'nikidan alohida.

    Ikkalasini sozlamasak, ORT 4 ta thread, torch yana 4 ta thread ochadi va
    ular bir xil yadrolarni talashadi. Aynan shu holat "yadro qo'shdim, tezlik
    tushdi" degan hodisaning sababi.
    """
    try:
        import torch
    except ImportError:
        return

    try:
        torch.set_num_threads(max(1, intra_op))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"torch.set_num_threads bajarilmadi: {exc}")

    try:
        torch.set_num_interop_threads(max(1, inter_op))
    except RuntimeError:
        # PyTorch inter-op thread sonini faqat BIRINCHI parallel ishdan oldin
        # o'rnatishga ruxsat beradi. Bir process ichida bir nechta
        # konfiguratsiyani sinaganda bu kutilgan holat.
        notes.append("torch inter-op thread soni allaqachon o'rnatilgan (o'zgartirilmadi)")


# ------------------------------------------------------------------- xotira


def _rss_mb() -> float:
    """Joriy RSS (MB). psutil bo'lsa aniqroq, bo'lmasa /proc dan o'qiymiz."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def _peak_rss_mb() -> float:
    try:
        import resource

        # Linux'da ru_maxrss kilobaytda.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001
        return 0.0


# -------------------------------------------------------------- ONNX liveness


class _OnnxLivenessRunner:
    """MiniFASNet ansamblini ONNX Runtime'da ishlatadi (taqqoslash uchun).

    Preprocessing produksiyadagi bilan AYNAN bir xil: `crop_face` + CHW float32,
    0..255 (255 ga bo'linmaydi). Aks holda taqqoslash ma'nosiz bo'lardi.
    """

    def __init__(self, model_dir: str, session_options) -> None:
        import numpy as np
        import onnxruntime as ort

        from src.infrastructure.antispoofing.minifasnet_adapter import parse_model_name

        self._np = np
        paths = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        paths = [p for p in paths if "_int8" not in os.path.basename(p)] or paths
        if not paths:
            raise FileNotFoundError(
                f"{model_dir} ichida `.onnx` liveness modeli yo'q. "
                "Avval `python scripts/export_onnx.py` ni ishga tushiring."
            )

        self._entries = []
        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            spec = parse_model_name(base + ".pth")
            session = ort.InferenceSession(
                path, sess_options=session_options, providers=["CPUExecutionProvider"]
            )
            self._entries.append((spec, session, session.get_inputs()[0].name))

    @property
    def model_names(self) -> List[str]:
        return [spec.name for spec, _, _ in self._entries]

    def check(self, image, bbox) -> None:
        from src.infrastructure.antispoofing.crop import crop_face

        np = self._np
        for spec, session, input_name in self._entries:
            patch = crop_face(image, bbox.as_xywh(), spec.scale, spec.width, spec.height)
            blob = np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32)[np.newaxis, ...]
            session.run(None, {input_name: blob})


# ------------------------------------------------------------------- pipeline


class Pipeline:
    """Bitta konfiguratsiya uchun yuklangan modellar to'plami."""

    def __init__(
        self,
        det_size: int,
        intra_op: int,
        inter_op: int,
        liveness_mode: str,
        liveness_dir: str,
        notes: List[str],
    ) -> None:
        self.det_size = det_size
        self.intra_op = intra_op
        self.inter_op = inter_op
        self.liveness_mode = liveness_mode
        self.liveness_dir = liveness_dir
        self.notes = notes

        self.app = None
        self.det_model = None
        self.rec_model = None
        self.liveness = None

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        from src.infrastructure.config import settings

        _SESSION_OPTIONS["opts"] = _make_session_options(self.intra_op, self.inter_op)
        _set_torch_threads(self.intra_op, self.inter_op, self.notes)

        app = FaceAnalysis(
            name=settings.insightface_model,
            root=settings.insightface_root,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        # DIQQAT: bu yerda det_size benchmark parametridan olinadi, lekin bu
        # FAQAT o'lchov uchun. `.env` dagi DET_SIZE o'zgarmaydi.
        app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))

        self.app = app
        self.det_model = getattr(app, "det_model", None) or app.models.get("detection")
        self.rec_model = app.models.get("recognition")

        if self.det_model is None:
            _die("InsightFace detektsiya modeli topilmadi (insightface versiyasi mos emasmi?).")

        if self.liveness_mode == "off":
            return

        if self.liveness_mode == "onnx":
            self.liveness = _OnnxLivenessRunner(self.liveness_dir, _SESSION_OPTIONS["opts"])
            return

        from src.infrastructure.antispoofing.minifasnet_adapter import MiniFASNetChecker
        import torch

        self.liveness = MiniFASNetChecker(
            model_dir=self.liveness_dir,
            device=torch.device("cpu"),
            threshold=settings.liveness_threshold,
            require_unanimous=settings.liveness_require_unanimous,
        )

    # ------------------------------------------------------------- bosqichlar

    def detect(self, image):
        return self.det_model.detect(image, max_num=0, metric="default")

    def make_face(self, bbox_row, kps):
        from insightface.app.common import Face
        import numpy as np

        return Face(
            bbox=np.asarray(bbox_row[:4], dtype=np.float32),
            kps=None if kps is None else np.asarray(kps, dtype=np.float32),
            det_score=float(bbox_row[4]) if len(bbox_row) > 4 else 1.0,
        )

    def embed(self, image, face) -> None:
        self.rec_model.get(image, face)

    def check_liveness(self, image, bbox_xyxy) -> None:
        from src.domain.entities import BoundingBox

        self.liveness.check(image, BoundingBox(*bbox_xyxy))

    def full_request(self, image) -> bool:
        """Bitta so'rovning to'liq oqimi — throughput o'lchovi uchun."""
        bboxes, kpss = self.detect(image)
        if bboxes is None or len(bboxes) == 0:
            return False

        index = int(max(range(len(bboxes)), key=lambda i: (bboxes[i][2] - bboxes[i][0]) *
                                                          (bboxes[i][3] - bboxes[i][1])))
        kps = None if kpss is None else kpss[index]
        face = self.make_face(bboxes[index], kps)
        if self.rec_model is not None and kps is not None:
            self.embed(image, face)
        if self.liveness is not None:
            x1, y1, x2, y2 = (int(v) for v in bboxes[index][:4])
            self.check_liveness(image, (max(0, x1), max(0, y1), x2, y2))
        return True


# --------------------------------------------------------------------- o'lchov


def _time_it(func: Callable[[], object]) -> Tuple[float, object]:
    start = time.perf_counter()
    result = func()
    return (time.perf_counter() - start) * 1000.0, result


def measure_latency(
    pipeline: Pipeline, frames: Sequence, iterations: int, warmup: int
) -> Tuple[Dict[str, LatencyStats], int, int]:
    """Bosqichlarni ALOHIDA o'lchaydi.

    Har bir kadr uchun avval (o'lchovsiz) yuz topiladi, keyin har bir bosqich
    o'z vaqti bilan o'lchanadi. Shunday qilinmasa, embedding va liveness
    o'lchovlari detektsiya vaqtini o'z ichiga olib qolardi.
    """
    detection: List[float] = []
    embedding: List[float] = []
    liveness: List[float] = []
    faces_found = 0
    no_face = 0

    # Isitish: birinchi chaqiruvlarda ORT arena ajratadi, lazy init bo'ladi —
    # ularni o'lchovga qo'shsak p99 ma'nosiz katta chiqadi.
    for i in range(warmup):
        pipeline.full_request(frames[i % len(frames)])

    for i in range(iterations):
        image = frames[i % len(frames)]

        elapsed, (bboxes, kpss) = _time_it(lambda: pipeline.detect(image))
        detection.append(elapsed)

        if bboxes is None or len(bboxes) == 0:
            no_face += 1
            continue

        faces_found += len(bboxes)
        index = int(max(range(len(bboxes)), key=lambda k: (bboxes[k][2] - bboxes[k][0]) *
                                                          (bboxes[k][3] - bboxes[k][1])))
        kps = None if kpss is None else kpss[index]
        face = pipeline.make_face(bboxes[index], kps)

        if pipeline.rec_model is not None and kps is not None:
            elapsed, _ = _time_it(lambda: pipeline.embed(image, face))
            embedding.append(elapsed)

        if pipeline.liveness is not None:
            x1, y1, x2, y2 = (int(v) for v in bboxes[index][:4])
            box = (max(0, x1), max(0, y1), x2, y2)
            elapsed, _ = _time_it(lambda: pipeline.check_liveness(image, box))
            liveness.append(elapsed)

    stats = {
        "detection": LatencyStats.build("detection", detection),
        "embedding": LatencyStats.build("embedding", embedding),
        "liveness": LatencyStats.build("liveness", liveness),
    }
    total = [
        d + (embedding[i] if i < len(embedding) else 0.0) + (liveness[i] if i < len(liveness) else 0.0)
        for i, d in enumerate(detection)
    ]
    stats["total"] = LatencyStats.build("total", total)
    return stats, faces_found, no_face


def measure_throughput(
    pipeline: Pipeline, frames: Sequence, concurrency_levels: Sequence[int], requests: int
) -> Dict[str, float]:
    """Turli parallellik darajalarida so'rov/sekund.

    ThreadPoolExecutor ataylab tanlangan: ONNX Runtime ham, PyTorch ham
    inferens paytida GIL'ni bo'shatadi, ya'ni thread'lar haqiqatan ham
    parallel ishlaydi — bu produksiyadagi (uvicorn/grpc worker) holatga
    eng yaqin model.
    """
    results: Dict[str, float] = {}

    for level in concurrency_levels:
        payload = [frames[i % len(frames)] for i in range(requests)]

        start = time.perf_counter()
        if level == 1:
            for image in payload:
                pipeline.full_request(image)
        else:
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(pipeline.full_request, payload))
        elapsed = time.perf_counter() - start

        results[f"concurrency_{level}"] = round(requests / elapsed, 2) if elapsed > 0 else 0.0

    return results


# ---------------------------------------------------------------------- chop


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print("  (ma'lumot yo'q)")
        return
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print("  " + line)
    print("  " + "-" * len(line))
    for row in cells:
        print("  " + "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


# ---------------------------------------------------------------------- args


def _parse_int_list(raw: str, name: str) -> List[int]:
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        _die(f"`{name}` noto'g'ri formatda: {raw!r}. Masalan: 1,2,4,8")
    return []


def _parse_thread_configs(raw: str) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            intra, inter = item.split(":", 1)
        else:
            intra, inter = item, "1"
        try:
            configs.append((int(intra), int(inter)))
        except ValueError:
            _die(f"`--thread-configs` noto'g'ri: {item!r}. Format: intra:inter (masalan 4:1)")
    return configs


def _default_thread_configs() -> str:
    cores = os.cpu_count() or 4
    candidates = sorted({1, 2, max(1, cores // 2), cores})
    return ",".join(f"{c}:1" for c in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description=(
            "AI oqimining latensiya / throughput / xotira ko'rsatkichlarini o'lchaydi. "
            "Hech qanday sozlamani o'zgartirmaydi — faqat raqam beradi."
        ),
        epilog=(
            "THREAD QOIDASI:\n"
            "  intra_op × worker_soni ≈ yadrolar soni\n"
            f"  (bu mashinada yadrolar: {os.cpu_count()})\n\n"
            "DET_SIZE haqida:\n"
            "  320 doim tezroq. Jadvaldagi `faces` ustuniga qarang — 640 ga\n"
            "  nisbatan qancha yuz YO'QOLGANINI ko'rsatadi. Tezlik uchun aniqlikni\n"
            "  qurbon qilishdan oldin shu raqamni ko'ring.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images", required=True,
        help="MAJBURIY. O'lchov uchun haqiqiy kadrlar papkasi (yuzli rasmlar).",
    )
    parser.add_argument(
        "--det-sizes", default="640,320",
        help="Taqqoslanadigan DET_SIZE qiymatlari (standart: 640,320).",
    )
    parser.add_argument(
        "--concurrency", default="1,2,4,8",
        help="Throughput uchun parallellik darajalari (standart: 1,2,4,8).",
    )
    parser.add_argument(
        "--thread-configs", default=None,
        help=(
            "ONNX Runtime thread sozlamalari `intra:inter` ro'yxati. "
            f"Standart shu mashina uchun: {_default_thread_configs()}"
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Latensiya uchun o'lchov soni (standart: 50).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="O'lchovga kirmaydigan isitish chaqiruvlari (standart: 5).",
    )
    parser.add_argument(
        "--throughput-requests", type=int, default=40,
        help="Har bir parallellik darajasi uchun so'rovlar soni (standart: 40).",
    )
    parser.add_argument(
        "--liveness", choices=["torch", "onnx", "off"], default="torch",
        help="Liveness backend: torch (produksiyadagi), onnx (taqqoslash), off.",
    )
    parser.add_argument(
        "--liveness-dir", default=None,
        help="Liveness modellari papkasi (standart: LIVENESS_MODEL_DIR sozlamasi).",
    )
    parser.add_argument(
        "--json", default=None,
        help="Natijani JSON faylga yozish (ixtiyoriy).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()

    _require("numpy", "`pip install -r requirements.txt` ni bajaring.")
    cv2 = _require("cv2", "`pip install -r requirements.txt` ni bajaring.")
    _require("onnxruntime", "`pip install -r requirements.txt` ni bajaring.")
    _require("insightface", "`pip install -r requirements.txt` ni bajaring.")

    args.images = _user_path(args.images)
    args.liveness_dir = _user_path(args.liveness_dir)
    args.json = _user_path(args.json)

    if not os.path.isdir(args.images):
        _die(
            f"Kadrlar papkasi topilmadi: {args.images}\n"
            "Benchmark HAQIQIY kadrlarni talab qiladi — tasodifiy shovqinda yuz "
            "topilmaydi va embedding/liveness bosqichlari umuman ishlamaydi, "
            "natijada raqamlar aldamchi chiqadi."
        )

    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(args.images, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(args.images, f"*{ext.upper()}")))
    paths = sorted(set(paths))

    if not paths:
        _die(f"{args.images} ichida rasm topilmadi ({', '.join(IMAGE_EXTENSIONS)}).")

    frames = []
    for path in paths:
        image = cv2.imread(path)
        if image is not None:
            frames.append(image)

    if not frames:
        _die("Hech qanday rasm o'qilmadi (fayllar buzuq bo'lishi mumkin).")

    from src.infrastructure.config import settings

    liveness_dir = args.liveness_dir or settings.liveness_model_dir
    det_sizes = _parse_int_list(args.det_sizes, "--det-sizes")
    concurrency = _parse_int_list(args.concurrency, "--concurrency")
    thread_configs = _parse_thread_configs(args.thread_configs or _default_thread_configs())

    patched, patch_error = _install_thread_patch()

    print("=" * 78)
    print("AI SERVIS BENCHMARK (CPU-only)")
    print("=" * 78)
    print(f"Yadrolar          : {os.cpu_count()}")
    print(f"Kadrlar           : {len(frames)} ta ({args.images})")
    print(f"DET_SIZE          : {det_sizes}   (produksiyadagi qiymat: {settings.det_size} — o'zgarmaydi)")
    print(f"Thread konfig.    : {['%d:%d' % c for c in thread_configs]}")
    print(f"Parallellik       : {concurrency}")
    print(f"Liveness backend  : {args.liveness}")
    print(f"Iteratsiyalar     : {args.iterations} (warmup {args.warmup})")
    if not patched:
        print(f"\n⚠️  THREAD PATCH ISHLAMADI: {patch_error}")
        print("    Thread ustunlaridagi farqlar ISHONCHSIZ — sozlama qo'llanmagan bo'lishi mumkin.")
    print()

    results: List[ConfigResult] = []

    for det_size in det_sizes:
        for intra_op, inter_op in thread_configs:
            notes: List[str] = []
            if not patched:
                notes.append(f"thread patch ishlamadi: {patch_error}")

            label = f"det_size={det_size} intra_op={intra_op} inter_op={inter_op}"
            print(f"--- {label} yuklanmoqda ...")

            rss_before = _rss_mb()
            pipeline = Pipeline(
                det_size=det_size,
                intra_op=intra_op,
                inter_op=inter_op,
                liveness_mode=args.liveness,
                liveness_dir=liveness_dir,
                notes=notes,
            )
            try:
                pipeline.load()
            except FileNotFoundError as exc:
                _die(str(exc))
            except Exception as exc:  # noqa: BLE001
                _die(
                    f"Modellarni yuklab bo'lmadi: {exc}\n"
                    "InsightFace modellari yo'q bo'lsa ular birinchi ishga tushishda "
                    "yuklab olinadi (internet kerak). Liveness modellari uchun: "
                    "bash scripts/download_models.sh"
                )

            rss_after = _rss_mb()

            stats, faces, no_face = measure_latency(
                pipeline, frames, args.iterations, args.warmup
            )
            throughput = measure_throughput(
                pipeline, frames, concurrency, args.throughput_requests
            )

            result = ConfigResult(
                det_size=det_size,
                intra_op=intra_op,
                inter_op=inter_op,
                liveness_backend=args.liveness,
                faces_found=faces,
                frames_with_no_face=no_face,
                rss_after_load_mb=round(rss_after - rss_before, 1),
                peak_rss_mb=round(_peak_rss_mb(), 1),
                latency={k: asdict(v) for k, v in stats.items()},
                throughput=throughput,
                notes=notes,
            )
            results.append(result)

            # Keyingi konfiguratsiya toza o'lchansin.
            del pipeline

    # ---------------------------------------------------------------- jadval
    print()
    print("=" * 78)
    print("LATENSIYA (ms) — bosqichlar alohida")
    print("=" * 78)
    headers = ["det_size", "intra:inter", "bosqich", "p50", "p95", "p99", "mean", "n"]
    rows = []
    for res in results:
        for stage in ("detection", "embedding", "liveness", "total"):
            data = res.latency.get(stage, {})
            if not data.get("count"):
                continue
            rows.append([
                res.det_size,
                f"{res.intra_op}:{res.inter_op}",
                stage,
                f"{data['p50']:.1f}",
                f"{data['p95']:.1f}",
                f"{data['p99']:.1f}",
                f"{data['mean']:.1f}",
                data["count"],
            ])
    _print_table(headers, rows)

    print()
    print("=" * 78)
    print("THROUGHPUT (so'rov/sekund)")
    print("=" * 78)
    conc_headers = ["det_size", "intra:inter"] + [f"c={c}" for c in concurrency]
    conc_rows = []
    for res in results:
        conc_rows.append(
            [res.det_size, f"{res.intra_op}:{res.inter_op}"]
            + [f"{res.throughput.get(f'concurrency_{c}', 0):.2f}" for c in concurrency]
        )
    _print_table(conc_headers, conc_rows)

    print()
    print("=" * 78)
    print("XOTIRA va TOPILGAN YUZLAR")
    print("=" * 78)
    mem_headers = ["det_size", "intra:inter", "RSS o'sishi (MB)", "peak RSS (MB)",
                   "topilgan yuz", "yuzsiz kadr"]
    mem_rows = [
        [res.det_size, f"{res.intra_op}:{res.inter_op}", res.rss_after_load_mb,
         res.peak_rss_mb, res.faces_found, res.frames_with_no_face]
        for res in results
    ]
    _print_table(mem_headers, mem_rows)

    # ------------------------------------------------------------ ★ ogohlantirish
    print()
    print("=" * 78)
    print("O'QISH BO'YICHA ESLATMALAR")
    print("=" * 78)

    by_det: Dict[int, int] = {}
    for res in results:
        by_det.setdefault(res.det_size, 0)
        by_det[res.det_size] = max(by_det[res.det_size], res.faces_found)

    if len(by_det) > 1:
        best = max(by_det.items(), key=lambda kv: kv[1])
        for det_size, faces in sorted(by_det.items(), reverse=True):
            if det_size == best[0] or best[1] == 0:
                continue
            lost = best[1] - faces
            if lost > 0:
                percent = lost / best[1] * 100
                print(f"⚠️  DET_SIZE={det_size}: DET_SIZE={best[0]} ga nisbatan "
                      f"{lost} ta yuz kamroq topildi ({percent:.1f}%).")
                print("    Bu to'g'ridan-to'g'ri FRR (o'zinikini rad etish) o'sishi degani.")

    best_throughput = None
    for res in results:
        for key, value in res.throughput.items():
            if best_throughput is None or value > best_throughput[1]:
                best_throughput = (f"det_size={res.det_size} "
                                   f"intra:inter={res.intra_op}:{res.inter_op} {key}", value)
    if best_throughput:
        print(f"ℹ️  Eng yuqori throughput: {best_throughput[0]} -> {best_throughput[1]:.2f} req/s")

    print()
    print("⚠️  DET_SIZE ni yoki thread sonini o'zgartirishdan oldin:")
    print("    - `faces` ustunini ko'ring (aniqlik yo'qotilishi),")
    print("    - `scripts/accuracy_check.py` bilan FAR/FRR ni o'lchang.")
    print("    Bu skript hech qanday sozlamani o'zgartirmadi.")

    # ------------------------------------------------------------------- JSON
    if args.json:
        payload = {
            "cpu_count": os.cpu_count(),
            "frames": len(frames),
            "images_dir": os.path.abspath(args.images),
            "liveness_backend": args.liveness,
            "production_det_size": settings.det_size,
            "thread_patch_applied": patched,
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nJSON yozildi: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
