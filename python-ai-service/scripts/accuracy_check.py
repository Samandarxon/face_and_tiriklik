#!/usr/bin/env python3
"""FAR / FRR / ROC / EER o'lchash — kvantlash yoki model almashtirish uchun DARVOZA.

NIMA UCHUN BU ENG MUHIM VOSITA
==============================
Tezlik optimizatsiyasining xavfi shundaki, u JIMGINA aniqlikni buzadi.
INT8 kvantlash yoki boshqa modelga o'tish natijasida:

    FRR oshsa  -> xodim taniilmaydi, u shikoyat qiladi, muammo darhol ko'rinadi.
    FAR oshsa  -> BEGONA ODAM "o'ziniki" deb qabul qilinadi.
                  Hech kim shikoyat qilmaydi. Log'da xato yo'q. Metrikalar toza.
                  Tizim "yaxshi ishlayapti" — begonalarni ham kiritgani uchun.

Ikkinchisi kirish-chiqishni nazorat qiladigan tizim uchun qabul qilib
bo'lmaydigan xavf. Uni faqat O'LCHASH orqali ko'rish mumkin — shuning uchun bu
skript kvantlashni yoqishdan oldingi MAJBURIY darvoza.

KIRISH FORMATI
==============
CSV, uchta ustun (sarlavha ixtiyoriy):

    img_a,img_b,same_person
    /data/faces/ali_1.jpg,/data/faces/ali_2.jpg,1
    /data/faces/ali_1.jpg,/data/faces/vali_1.jpg,0

`same_person`: 1 = bir xil odam (positive), 0 = turli odamlar (negative).

Juftliklar SIZNING ma'lumotingizdan bo'lishi shart — o'sha kameralar, o'sha
yorug'lik. Ochiq datasetdagi (LFW va h.k.) natija sizning reseptioningizdagi
FAR'ni bashorat qilmaydi.

Negative juftliklar soni muhim: FAR = 0.1% ni ishonchli o'lchash uchun kamida
~1000 ta negative juftlik kerak. 20 ta juftlik bilan chiqqan "FAR = 0" hech
narsani anglatmaydi.

REJIMLAR
========
1) Bitta model:
     python scripts/accuracy_check.py --pairs pairs.csv

2) Ikki modelni taqqoslash (FP32 vs INT8) — kvantlashdan keyin MAJBURIY:
     python scripts/accuracy_check.py --pairs pairs.csv \
         --model  /root/.insightface/models/buffalo_l/w600k_r50.onnx \
         --compare-model /root/.insightface/models/buffalo_l/w600k_r50_int8.onnx

Skript hech qanday sozlamani O'ZGARTIRMAYDI. MATCH_THRESHOLD (hozir 0.65)
faqat hisobot uchun "ishchi nuqta" sifatida ishlatiladi.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ORIGINAL_CWD = os.getcwd()
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)
os.chdir(SERVICE_ROOT)

# Golang backend'dagi MATCH_THRESHOLD standart qiymati. Bu yerda faqat
# HISOBOT uchun ishlatiladi — hech narsa o'zgartirilmaydi.
DEFAULT_OPERATING_THRESHOLD = 0.65


def _die(message: str, code: int = 2) -> None:
    print(f"\nXATO: {message}", file=sys.stderr)
    raise SystemExit(code)


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError:
        _die(f"`{module}` o'rnatilmagan. {hint}")


def _user_path(path: Optional[str]) -> Optional[str]:
    """Foydalanuvchi bergan nisbiy yo'lni ORIGINAL papkaga nisbatan hal qiladi.

    Skript servis ildiziga `chdir` qiladi (sozlamalardagi yo'llar nisbiy),
    shuning uchun CLI'dan kelgan nisbiy yo'llarni oldindan absolyutlashtiramiz.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ORIGINAL_CWD, path))


# ------------------------------------------------------------------ metrikalar


@dataclass
class ThresholdPoint:
    threshold: float
    far: float           # False Accept Rate — begona qabul qilindi
    frr: float           # False Reject Rate  — o'ziniki rad etildi
    tar: float           # True Accept Rate = 1 - FRR
    false_accepts: int
    false_rejects: int


@dataclass
class ModelMetrics:
    name: str
    model_path: str
    positives: int
    negatives: int
    eer: float
    eer_threshold: float
    points: List[ThresholdPoint] = field(default_factory=list)
    similarities_pos: List[float] = field(default_factory=list)
    similarities_neg: List[float] = field(default_factory=list)

    def at(self, threshold: float) -> ThresholdPoint:
        """Berilgan chegaraga eng yaqin nuqta."""
        return min(self.points, key=lambda p: abs(p.threshold - threshold))


def compute_metrics(
    name: str,
    model_path: str,
    pos_scores: Sequence[float],
    neg_scores: Sequence[float],
    thresholds: Sequence[float],
) -> ModelMetrics:
    points: List[ThresholdPoint] = []
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    for threshold in thresholds:
        # "Qabul qilindi" = o'xshashlik chegaradan katta yoki teng.
        false_accepts = sum(1 for s in neg_scores if s >= threshold)
        false_rejects = sum(1 for s in pos_scores if s < threshold)
        far = false_accepts / n_neg if n_neg else 0.0
        frr = false_rejects / n_pos if n_pos else 0.0
        points.append(
            ThresholdPoint(
                threshold=threshold,
                far=far,
                frr=frr,
                tar=1.0 - frr,
                false_accepts=false_accepts,
                false_rejects=false_rejects,
            )
        )

    # EER — FAR va FRR teng bo'ladigan nuqta.
    eer_point = min(points, key=lambda p: abs(p.far - p.frr))
    return ModelMetrics(
        name=name,
        model_path=model_path,
        positives=n_pos,
        negatives=n_neg,
        eer=(eer_point.far + eer_point.frr) / 2.0,
        eer_threshold=eer_point.threshold,
        points=points,
        similarities_pos=list(pos_scores),
        similarities_neg=list(neg_scores),
    )


# ------------------------------------------------------------------- modellar


def _default_recognition_model() -> str:
    from src.infrastructure.config import settings

    return os.path.join(
        settings.insightface_root, "models", settings.insightface_model, "w600k_r50.onnx"
    )


def _load_recognition(model_path: str):
    """ArcFace ONNX modelini yuklaydi (FP32 ham, INT8 ham bir xil yo'l bilan)."""
    try:
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    except ImportError as exc:
        _die(f"insightface import qilinmadi ({exc}). `pip install -r requirements.txt`.")

    if not os.path.isfile(model_path):
        _die(
            f"Tanish (recognition) modeli topilmadi: {model_path}\n"
            "InsightFace modellari birinchi ishga tushishda yuklab olinadi. "
            "Yo'lni `--model` bilan ochiq bering."
        )

    try:
        model = ArcFaceONNX(model_file=model_path)
        model.prepare(ctx_id=-1)
    except Exception as exc:  # noqa: BLE001
        _die(f"Modelni yuklab bo'lmadi ({model_path}): {exc}")
    return model


def _load_detector():
    """Detektor BITTA bo'ladi — ikkala tanish modeli uchun ham AYNAN o'sha yuz.

    Bu muhim: agar har bir model o'z detektsiyasini ishlatsa, farq
    kvantlashdanmi yoki boshqacha kesilgan yuzdanmi — ajratib bo'lmasdi.
    """
    from insightface.app import FaceAnalysis

    from src.infrastructure.config import settings

    app = FaceAnalysis(
        name=settings.insightface_model,
        root=settings.insightface_root,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    # DET_SIZE produksiyadagi qiymatda qoladi — o'zgartirilmaydi.
    app.prepare(ctx_id=-1, det_size=(settings.det_size, settings.det_size))
    return app


# ------------------------------------------------------------------------ CSV


def read_pairs(csv_path: str, image_root: Optional[str]) -> List[Tuple[str, str, int]]:
    if not os.path.isfile(csv_path):
        _die(
            f"Juftliklar fayli topilmadi: {csv_path}\n\n"
            "Format (CSV, sarlavha ixtiyoriy):\n"
            "    img_a,img_b,same_person\n"
            "    /data/ali_1.jpg,/data/ali_2.jpg,1\n"
            "    /data/ali_1.jpg,/data/vali_1.jpg,0"
        )

    pairs: List[Tuple[str, str, int]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle), start=1):
            row = [cell.strip() for cell in row if cell.strip() != ""]
            if len(row) < 3:
                continue
            label_raw = row[2].lower()
            if label_raw not in {"0", "1", "true", "false", "yes", "no"}:
                # Sarlavha qatori — o'tkazib yuboramiz.
                if line_no == 1:
                    continue
                print(f"[ogoh] {line_no}-qator o'tkazib yuborildi (same_person = {row[2]!r})")
                continue

            label = 1 if label_raw in {"1", "true", "yes"} else 0
            img_a, img_b = row[0], row[1]
            if image_root:
                if not os.path.isabs(img_a):
                    img_a = os.path.join(image_root, img_a)
                if not os.path.isabs(img_b):
                    img_b = os.path.join(image_root, img_b)
            pairs.append((img_a, img_b, label))

    if not pairs:
        _die(f"{csv_path} ichida yaroqli juftlik topilmadi.")
    return pairs


# --------------------------------------------------------------------- chiqish


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


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_model_report(metrics: ModelMetrics, operating: float, report_thresholds: Sequence[float]) -> None:
    print()
    print("-" * 78)
    print(f"MODEL: {metrics.name}")
    print(f"  fayl      : {metrics.model_path}")
    print(f"  juftliklar: {metrics.positives} positive, {metrics.negatives} negative")
    print(f"  EER       : {_percent(metrics.eer)} (chegara {metrics.eer_threshold:.3f})")
    print()

    rows = []
    for threshold in report_thresholds:
        point = metrics.at(threshold)
        marker = "  <-- ishchi nuqta" if abs(threshold - operating) < 1e-9 else ""
        rows.append([
            f"{point.threshold:.2f}",
            _percent(point.far),
            _percent(point.frr),
            _percent(point.tar),
            point.false_accepts,
            point.false_rejects,
            marker,
        ])
    _print_table(
        ["chegara", "FAR", "FRR", "TAR", "noto'g'ri qabul", "noto'g'ri rad", ""],
        rows,
    )


def _print_roc(metrics: ModelMetrics, count: int) -> None:
    print()
    print(f"ROC nuqtalari ({metrics.name}) — FAR ga qarab TAR:")
    step = max(1, len(metrics.points) // count)
    rows = [
        [f"{p.threshold:.2f}", _percent(p.far), _percent(p.tar)]
        for p in metrics.points[::step]
    ]
    _print_table(["chegara", "FAR (FPR)", "TAR (TPR)"], rows)


# ---------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="accuracy_check.py",
        description=(
            "Yuz tanish aniqligini o'lchaydi: FAR, FRR, ROC, EER. "
            "Ikki modelni (masalan FP32 va INT8) taqqoslash rejimi bor. "
            "Hech qanday sozlamani o'zgartirmaydi."
        ),
        epilog=(
            "⚠️  KVANTLANGAN (INT8) MODELNI SHU TEKSHIRUVSIZ PRODUCTION'GA CHIQARMANG.\n\n"
            "FAR o'sishi hech qanday log'da ko'rinmaydi — begona odam ichkariga\n"
            "kiradi va tizim buni \"muvaffaqiyatli tanish\" deb yozadi.\n\n"
            "Statistik eslatma: FAR ni ishonchli o'lchash uchun negative juftliklar\n"
            "ko'p bo'lishi kerak. 1000 ta negative juftlikda o'lchanadigan eng kichik\n"
            "ma'noli FAR ~ 0.1%.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs", required=True,
        help="MAJBURIY. CSV: img_a,img_b,same_person(0/1).",
    )
    parser.add_argument(
        "--image-root", default=None,
        help="CSV ichidagi nisbiy yo'llar uchun ildiz papka.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Asosiy (baseline) ArcFace ONNX modeli. Standart: buffalo_l/w600k_r50.onnx.",
    )
    parser.add_argument(
        "--compare-model", default=None,
        help="Taqqoslanadigan model (masalan `*_int8.onnx`). Berilsa FP32 vs INT8 hisoboti chiqadi.",
    )
    parser.add_argument(
        "--operating-threshold", type=float, default=DEFAULT_OPERATING_THRESHOLD,
        help=(
            f"Hisobotdagi ishchi nuqta (standart: {DEFAULT_OPERATING_THRESHOLD} — "
            "backend'dagi MATCH_THRESHOLD). Faqat hisobot uchun, hech narsa o'zgarmaydi."
        ),
    )
    parser.add_argument(
        "--far-tolerance", type=float, default=0.002,
        help=(
            "Ruxsat etilgan FAR o'sishi (absolyut, standart: 0.002 = 0.2 foiz nuqta). "
            "Ko'proq bo'lsa ogohlantirish chiqadi va exit-code 1 bo'ladi."
        ),
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.01,
        help="Chegara qadami (standart: 0.01).",
    )
    parser.add_argument(
        "--roc-points", type=int, default=15,
        help="Chop etiladigan ROC nuqtalari soni (standart: 15).",
    )
    parser.add_argument(
        "--json", default=None,
        help="Natijani JSON faylga yozish (ixtiyoriy).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()

    np = _require("numpy", "`pip install -r requirements.txt` ni bajaring.")
    cv2 = _require("cv2", "`pip install -r requirements.txt` ni bajaring.")
    _require("insightface", "`pip install -r requirements.txt` ni bajaring.")
    _require("onnx", "`pip install -r requirements-dev.txt` ni bajaring "
                     "(ArcFaceONNX model grafigini o'qish uchun kerak).")

    pairs_path = _user_path(args.pairs)
    image_root = _user_path(args.image_root)
    model_path = _user_path(args.model) or _default_recognition_model()
    compare_path = _user_path(args.compare_model)
    json_path = _user_path(args.json)

    pairs = read_pairs(pairs_path, image_root)
    n_pos = sum(1 for *_, label in pairs if label == 1)
    n_neg = len(pairs) - n_pos

    print("=" * 78)
    print("ANIQLIK TEKSHIRUVI (FAR / FRR / ROC / EER)")
    print("=" * 78)
    print(f"Juftliklar fayli : {pairs_path}")
    print(f"Juftliklar       : {len(pairs)} ta ({n_pos} positive, {n_neg} negative)")
    print(f"Baseline model   : {model_path}")
    if compare_path:
        print(f"Taqqoslanadigan  : {compare_path}")
    print(f"Ishchi chegara   : {args.operating_threshold} (MATCH_THRESHOLD — o'zgartirilmaydi)")
    print()

    if n_pos == 0 or n_neg == 0:
        _die(
            "FAR va FRR ni birga hisoblash uchun HAM positive (same_person=1), "
            "HAM negative (same_person=0) juftliklar kerak.\n"
            f"Hozir: positive={n_pos}, negative={n_neg}."
        )

    if n_neg < 100:
        print(f"⚠️  OGOHLANTIRISH: negative juftliklar soni juda kam ({n_neg}).")
        print(f"    Bu to'plamda o'lchash mumkin bo'lgan eng kichik FAR ~ {1 / n_neg * 100:.2f}%.")
        print("    Ya'ni \"FAR = 0\" natijasi xavfsizlik kafolati EMAS.\n")

    print("Modellar yuklanmoqda ...")
    detector = _load_detector()
    baseline = _load_recognition(model_path)
    compare = _load_recognition(compare_path) if compare_path else None

    # ------------------------------------------------------- embedding olish
    from insightface.app.common import Face

    cache: Dict[str, Optional[Tuple[object, object]]] = {}

    def embeddings_for(path: str) -> Optional[Tuple[object, object]]:
        """Rasm uchun (baseline_emb, compare_emb) — ikkalasi AYNAN bir yuzdan."""
        if path in cache:
            return cache[path]

        image = cv2.imread(path)
        if image is None:
            cache[path] = None
            return None

        faces = detector.get(image)
        if not faces:
            cache[path] = None
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if getattr(face, "kps", None) is None:
            cache[path] = None
            return None

        def embed(model) -> object:
            # Har bir model uchun toza `Face` — ArcFaceONNX.get() obyektni o'zgartiradi.
            fresh = Face(bbox=face.bbox, kps=face.kps, det_score=face.det_score)
            vector = np.asarray(model.get(image, fresh), dtype=np.float32).flatten()
            norm = float(np.linalg.norm(vector))
            # Qdrant Cosine masofasi normalizatsiyalangan vektorni kutadi.
            return vector / norm if norm > 0 else vector

        result = (embed(baseline), embed(compare) if compare is not None else None)
        cache[path] = result
        return result

    pos_base: List[float] = []
    neg_base: List[float] = []
    pos_cmp: List[float] = []
    neg_cmp: List[float] = []
    drift: List[float] = []
    skipped: List[str] = []

    for index, (img_a, img_b, label) in enumerate(pairs, start=1):
        if index % 50 == 0:
            print(f"  ... {index}/{len(pairs)}")

        emb_a = embeddings_for(img_a)
        emb_b = embeddings_for(img_b)
        if emb_a is None or emb_b is None:
            skipped.append(img_a if emb_a is None else img_b)
            continue

        base_sim = float(np.dot(emb_a[0], emb_b[0]))
        (pos_base if label == 1 else neg_base).append(base_sim)

        if compare is not None:
            cmp_sim = float(np.dot(emb_a[1], emb_b[1]))
            (pos_cmp if label == 1 else neg_cmp).append(cmp_sim)
            # Bir xil rasmda ikki modelning embedding'lari qanchalik farq qiladi.
            drift.append(float(np.dot(emb_a[0], emb_a[1])))

    if skipped:
        print(f"\n⚠️  {len(skipped)} ta rasm o'tkazib yuborildi (yuz topilmadi yoki o'qilmadi).")
        for path in skipped[:5]:
            print(f"    {path}")
        if len(skipped) > 5:
            print(f"    ... va yana {len(skipped) - 5} ta")

    if not pos_base or not neg_base:
        _die(
            "Yaroqli juftliklar qolmadi (ko'p rasmda yuz topilmadi). "
            "Rasmlar sifatini va yo'llarni tekshiring."
        )

    # ------------------------------------------------------------ metrikalar
    step = max(0.001, args.threshold_step)
    thresholds = [round(-0.2 + i * step, 4) for i in range(int(1.2 / step) + 1)]

    baseline_metrics = compute_metrics(
        "baseline (FP32)", model_path, pos_base, neg_base, thresholds
    )
    report_thresholds = [t for t in (0.20, 0.30, 0.40, 0.50, 0.55, 0.60,
                                     args.operating_threshold, 0.70, 0.75, 0.80)]
    report_thresholds = sorted(set(round(t, 4) for t in report_thresholds))

    _print_model_report(baseline_metrics, args.operating_threshold, report_thresholds)
    _print_roc(baseline_metrics, args.roc_points)

    compare_metrics: Optional[ModelMetrics] = None
    exit_code = 0

    if compare is not None:
        compare_metrics = compute_metrics(
            "taqqoslanuvchi (INT8?)", compare_path, pos_cmp, neg_cmp, thresholds
        )
        _print_model_report(compare_metrics, args.operating_threshold, report_thresholds)

        # ------------------------------------------------------- taqqoslash
        print()
        print("=" * 78)
        print("TAQQOSLASH — aniqlik qancha o'zgardi")
        print("=" * 78)

        base_point = baseline_metrics.at(args.operating_threshold)
        cmp_point = compare_metrics.at(args.operating_threshold)
        far_delta = cmp_point.far - base_point.far
        frr_delta = cmp_point.frr - base_point.frr
        eer_delta = compare_metrics.eer - baseline_metrics.eer

        rows = [
            ["FAR (chegara %.2f)" % args.operating_threshold,
             _percent(base_point.far), _percent(cmp_point.far),
             f"{far_delta * 100:+.2f} p.p."],
            ["FRR (chegara %.2f)" % args.operating_threshold,
             _percent(base_point.frr), _percent(cmp_point.frr),
             f"{frr_delta * 100:+.2f} p.p."],
            ["EER", _percent(baseline_metrics.eer), _percent(compare_metrics.eer),
             f"{eer_delta * 100:+.2f} p.p."],
        ]
        _print_table(["ko'rsatkich", "baseline", "taqqoslanuvchi", "farq"], rows)

        if drift:
            mean_drift = float(np.mean(drift))
            min_drift = float(np.min(drift))
            print()
            print("Embedding siljishi (bir xil rasmda ikki model kosinus o'xshashligi):")
            print(f"  o'rtacha {mean_drift:.4f} | eng yomon {min_drift:.4f}  (1.0000 = farq yo'q)")
            if mean_drift < 0.98:
                print("  ⚠️  0.98 dan past — modellar sezilarli farq qilmoqda.")

        # ------------------------------------------------------ ★ ogohlantirish
        print()
        relative = (far_delta / base_point.far) if base_point.far > 0 else float("inf")

        if far_delta > args.far_tolerance:
            exit_code = 1
            print("=" * 78)
            print("⛔  OGOHLANTIRISH: FAR SEZILARLI OSHDI")
            print("=" * 78)
            print(f"  {args.operating_threshold:.2f} chegarada FAR "
                  f"{_percent(base_point.far)} -> {_percent(cmp_point.far)} "
                  f"({far_delta * 100:+.2f} foiz nuqta)")
            print(f"  Ruxsat etilgan chegara: {args.far_tolerance * 100:.2f} foiz nuqta")
            print()
            print("  Bu shuni anglatadi: taqqoslanuvchi model bilan har "
                  f"{int(1 / cmp_point.far) if cmp_point.far > 0 else 0} ta begona urinishdan")
            print("  bittasi ICHKARIGA O'TADI. Bu xato hech qanday log'da ko'rinmaydi.")
            print()
            print("  TAVSIYA: bu modelni production'ga CHIQARMANG.")
            print("  Variantlar:")
            print("    - kalibrlash to'plamini kattalashtiring / produksiyaga yaqinlashtiring,")
            print("    - `--no-per-channel` ni olib tashlang (per-channel aniqroq),")
            print("    - embedding modelini umuman kvantlamang (faqat liveness'ni kvantlang),")
            print("    - yoki MATCH_THRESHOLD ni ko'tarish orqali FAR ni qaytaring —")
            print("      lekin bu FRR ni oshiradi, ya'ni o'z xodimlaringiz rad etiladi.")
        elif far_delta > 0:
            print(f"ℹ️  FAR biroz oshdi ({far_delta * 100:+.2f} p.p.), ruxsat etilgan "
                  f"chegara ichida ({args.far_tolerance * 100:.2f} p.p.).")
            if relative != float("inf") and relative > 0.2:
                print(f"    Lekin nisbiy o'sish katta ({relative * 100:.0f}%) — "
                      "juftliklar sonini oshirib qayta o'lchang.")
        else:
            print("✅ FAR oshmadi.")

        if frr_delta > 0.01:
            print(f"\n⚠️  FRR {frr_delta * 100:+.2f} p.p. oshdi — o'z xodimlaringiz "
                  "ko'proq rad etiladi (shikoyatlar ko'payadi).")

    # ------------------------------------------------------------------- JSON
    if json_path:
        def dump(metrics: Optional[ModelMetrics]) -> Optional[dict]:
            if metrics is None:
                return None
            return {
                "name": metrics.name,
                "model_path": metrics.model_path,
                "positives": metrics.positives,
                "negatives": metrics.negatives,
                "eer": metrics.eer,
                "eer_threshold": metrics.eer_threshold,
                "points": [
                    {"threshold": p.threshold, "far": p.far, "frr": p.frr, "tar": p.tar,
                     "false_accepts": p.false_accepts, "false_rejects": p.false_rejects}
                    for p in metrics.points
                ],
            }

        payload = {
            "pairs_file": pairs_path,
            "pairs_total": len(pairs),
            "pairs_skipped": len(skipped),
            "operating_threshold": args.operating_threshold,
            "far_tolerance": args.far_tolerance,
            "baseline": dump(baseline_metrics),
            "compare": dump(compare_metrics),
            "far_regression": exit_code == 1,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nJSON yozildi: {json_path}")

    print()
    print("Eslatma: bu skript hech qanday sozlamani o'zgartirmadi "
          "(MATCH_THRESHOLD, DET_SIZE, LIVENESS_THRESHOLD o'z joyida).")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
