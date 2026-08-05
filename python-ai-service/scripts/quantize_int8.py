#!/usr/bin/env python3
"""ONNX modellarni INT8 ga kvantlash — STATIC quantization.

NIMA UCHUN STATIC, DYNAMIC EMAS
===============================
Dynamic quantization faqat vaznlarni INT8 qiladi, aktivatsiyalar runtime'da
har safar qayta o'lchanadi. Bu Transformer/LSTM (GEMM ustun) modellarda yaxshi
ishlaydi, lekin MiniFASNet va ArcFace kabi KONVOLYUTSION tarmoqlarda deyarli
tezlik bermaydi — konvolyutsiya INT8 kernel'iga tushmaydi.

Static quantization esa aktivatsiya diapazonlarini OLDINDAN, haqiqiy kadrlar
ustida o'lchaydi (kalibrlash) va konvolyutsiyalarni to'liq INT8 ga o'tkazadi.
CPU'da real foyda aynan shundan keladi.

Shuning uchun bu skript kalibrlash rasmlari bo'lmasa DYNAMIC'ga TUSHIB
QOLMAYDI — u shunchaki to'xtaydi. "Ishladi shekilli" degan yolg'on natijadan
ko'ra ochiq xato yaxshiroq.

⚠️ ENG MUHIM OGOHLANTIRISH
==========================
INT8 kvantlash aniqlikni PASAYTIRADI. Yuzni tanishda bu pasayish FAR (begona
odamni "o'ziniki" deb qabul qilish) ko'rinishida chiqadi va uni oddiy ko'z
bilan sezib bo'lmaydi: tizim baribir ishlaydi, xodimlarni taniydi, hech qanday
xato log'ga tushmaydi. Faqat begonalar ham o'ta boshlaydi.

Shuning uchun bu skript hech narsani YOQMAYDI. U `*_int8.onnx` faylini alohida
nom bilan yozadi, asl fayl joyida qoladi. Kvantlangan modelni ishlatishdan
oldin MAJBURIY:

    python scripts/accuracy_check.py --pairs pairs.csv \
        --model <fp32>.onnx --compare-model <int8>.onnx

va FAR o'sishini o'z ko'zingiz bilan ko'ring.

KALIBRLASH MA'LUMOTI QANDAY BO'LISHI KERAK
==========================================
Kalibrlash rasmlari PRODUKSIYA taqsimotini aks ettirishi shart — ya'ni
o'sha kameralardan, o'sha yorug'likda olingan haqiqiy kadrlar. Internetdan
olingan chiroyli portretlar bilan kalibrlangan model reseptiondagi qorong'i
kadrda buziladi, chunki aktivatsiya diapazonlari boshqacha bo'ladi.

Ishlatish:
    # 1) avval ONNX ga eksport qiling
    python scripts/export_onnx.py

    # 2) kalibrlash (kamida 100 ta haqiqiy kadr)
    python scripts/quantize_int8.py --calibration-dir ./calib_frames

    # ArcFace (embedding) modelini kvantlash:
    python scripts/quantize_int8.py \
        --models /root/.insightface/models/buffalo_l/w600k_r50.onnx \
        --calibration-dir ./calib_frames
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Iterator, List, Optional, Sequence, Tuple

ORIGINAL_CWD = os.getcwd()
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)
os.chdir(SERVICE_ROOT)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Static quantization uchun minimal kadr soni. 100 — amaliy minimum:
# undan kam bo'lsa aktivatsiya diapazonlari tasodifiy chiqadi va kvantlash
# natijasi har safar boshqacha bo'ladi.
DEFAULT_MIN_IMAGES = 100


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


# --------------------------------------------------------------------- detektor


class _FaceLocator:
    """Kalibrlash kadrlaridan yuz koordinatalarini oladi.

    Nima uchun kerak: MiniFASNet butun kadrni emas, yuz atrofidagi `scale`
    bilan kengaytirilgan patch'ni ko'radi (`antispoofing/crop.py`). Agar
    kalibrlashda butun kadrni bersak, aktivatsiya diapazonlari produksiyadagidan
    butunlay boshqacha chiqadi va kvantlash yomon bo'ladi.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._app = None
        self._cascade = None
        self.active = "?"
        self._init(mode)

    def _init(self, mode: str) -> None:
        if mode in {"auto", "insightface"}:
            try:
                from insightface.app import FaceAnalysis

                from src.infrastructure.config import settings

                app = FaceAnalysis(
                    name=settings.insightface_model,
                    root=settings.insightface_root,
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection"],
                )
                # DET_SIZE sozlamasi O'ZGARTIRILMAYDI — produksiyadagi qiymat.
                app.prepare(ctx_id=-1, det_size=(settings.det_size, settings.det_size))
                self._app = app
                self.active = "insightface"
                return
            except Exception as exc:  # noqa: BLE001
                if mode == "insightface":
                    _die(f"InsightFace detektorini yuklab bo'lmadi: {exc}")
                print(f"[ogoh] InsightFace mavjud emas ({exc}); Haar cascade'ga o'tildi.")

        import cv2

        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.active = "haar"

    def locate(self, image) -> Optional[Tuple[Tuple[int, int, int, int], Optional[object]]]:
        """Eng katta yuz uchun ((x1,y1,x2,y2), kps) qaytaradi. kps faqat InsightFace'da."""
        import cv2
        import numpy as np

        if self._app is not None:
            faces = self._app.get(image)
            if not faces:
                return None
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            return (x1, y1, x2, y2), getattr(face, "kps", None)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(found) == 0:
            return None
        x, y, w, h = max(found, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(x + w), int(y + h)), None


# ------------------------------------------------------------- preprocessing


def _preprocess_minifasnet(image, bbox, kps, spec) -> "object":
    """Produksiyadagi bilan BIR XIL: crop_face + CHW float32, **0..255**.

    255 ga BO'LINMAYDI. Sabab `minifasnet_adapter._to_tensor()` docstring'ida:
    MiniFASNet vaznlari 0..255 kirishga o'qitilgan. Agar kalibrlashda boshqa
    diapazon bersak, o'lchangan aktivatsiya chegaralari noto'g'ri chiqadi va
    INT8 model butunlay buziladi.
    """
    import numpy as np

    from src.infrastructure.antispoofing.crop import crop_face

    x1, y1, x2, y2 = bbox
    xywh = (x1, y1, x2 - x1, y2 - y1)
    patch = crop_face(image, xywh, spec.scale, spec.width, spec.height)
    array = np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32)
    return array[np.newaxis, ...]


def _preprocess_arcface(image, bbox, kps, spec) -> "object":
    """ArcFace: 5 nuqta bo'yicha tekislangan 112x112 RGB, (x-127.5)/127.5."""
    import numpy as np

    if kps is None:
        _die(
            "ArcFace kalibratsiyasi uchun 5 ta yuz nuqtasi (landmarks) kerak, "
            "Haar cascade esa ularni bermaydi.\n"
            "Yechim: InsightFace modellarini yuklang va `--detector insightface` bering."
        )

    from insightface.utils import face_align

    aligned = face_align.norm_crop(image, landmark=np.asarray(kps), image_size=112)
    blob = aligned[:, :, ::-1].astype(np.float32)          # BGR -> RGB
    blob = (blob - 127.5) / 127.5
    return np.ascontiguousarray(blob.transpose(2, 0, 1))[np.newaxis, ...]


PREPROCESSORS = {
    "minifasnet": _preprocess_minifasnet,
    "arcface": _preprocess_arcface,
}


def _resolve_preprocess(model_path: str, explicit: str, session_input_shape) -> str:
    """`auto` rejimida model kirish shaklidan preprocessing turini aniqlaydi."""
    if explicit != "auto":
        return explicit

    name = os.path.basename(model_path).lower()
    if "minifasnet" in name:
        return "minifasnet"

    # Kirish shakli: [batch, 3, H, W]
    try:
        height = int(session_input_shape[2])
        width = int(session_input_shape[3])
    except (TypeError, ValueError, IndexError):
        height = width = -1

    if (height, width) == (112, 112):
        return "arcface"
    if (height, width) == (80, 80):
        return "minifasnet"

    _die(
        f"{os.path.basename(model_path)} uchun preprocessing turini avtomatik "
        f"aniqlab bo'lmadi (kirish shakli: {session_input_shape}).\n"
        "`--preprocess minifasnet` yoki `--preprocess arcface` ni ochiq bering.\n"
        "Noto'g'ri preprocessing bilan kalibrlangan INT8 model jimgina buziladi — "
        "shuning uchun taxmin qilmaymiz."
    )
    return ""


# ------------------------------------------------------------------- reader


def _build_calibration_reader(samples: Sequence["object"], input_name: str):
    """`onnxruntime.quantization.CalibrationDataReader` implementatsiyasi."""
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter: Iterator = iter(samples)

        def get_next(self):
            item = next(self._iter, None)
            return None if item is None else {input_name: item}

        def rewind(self) -> None:
            self._iter = iter(samples)

    return _Reader()


# --------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quantize_int8.py",
        description=(
            "ONNX modellarni STATIC INT8 ga kvantlaydi. Asl fayl saqlanadi, "
            "natija `*_int8.onnx` nomi bilan yoziladi. Servis sozlamalari "
            "O'ZGARTIRILMAYDI."
        ),
        epilog=(
            "⚠️  KVANTLANGAN MODELNI FAR/FRR O'LCHAMASDAN PRODUCTION'GA CHIQARMANG:\n"
            "      python scripts/accuracy_check.py --pairs pairs.csv \\\n"
            "          --model model.onnx --compare-model model_int8.onnx\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=(
            "Kvantlanadigan `.onnx` fayllar yoki papka "
            "(standart: LIVENESS_MODEL_DIR ichidagi barcha `.onnx`)."
        ),
    )
    parser.add_argument(
        "--calibration-dir", default=None,
        help="MAJBURIY. Haqiqiy produksiya kadrlari joylashgan papka.",
    )
    parser.add_argument(
        "--min-images", type=int, default=DEFAULT_MIN_IMAGES,
        help=f"Minimal kalibrlash kadrlari soni (standart: {DEFAULT_MIN_IMAGES}).",
    )
    parser.add_argument(
        "--max-images", type=int, default=500,
        help="Ishlatiladigan maksimal kadrlar soni (standart: 500).",
    )
    parser.add_argument(
        "--detector", choices=["auto", "insightface", "haar"], default="auto",
        help="Kalibrlash kadrlarida yuzni topish usuli (standart: auto).",
    )
    parser.add_argument(
        "--preprocess", choices=["auto", "minifasnet", "arcface"], default="auto",
        help="Kirishni tayyorlash usuli (standart: auto — model shaklidan aniqlanadi).",
    )
    parser.add_argument(
        "--suffix", default="_int8",
        help="Chiqish fayl nomiga qo'shiladigan qo'shimcha (standart: _int8).",
    )
    parser.add_argument(
        "--per-channel", dest="per_channel", action="store_true", default=True,
        help="Vaznlarni kanal bo'yicha kvantlash (standart: yoqilgan, aniqroq).",
    )
    parser.add_argument(
        "--no-per-channel", dest="per_channel", action="store_false",
        help="Kanal bo'yicha kvantlashni o'chirish (eski runtime'lar uchun).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Mavjud `*_int8.onnx` ustiga qayta yozish.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------- main


def _collect_models(raw: Optional[List[str]]) -> List[str]:
    from src.infrastructure.config import settings

    if not raw:
        raw = [settings.liveness_model_dir]

    paths: List[str] = []
    for item in raw:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.onnx"))))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            matched = sorted(glob.glob(item))
            if not matched:
                _die(f"Model topilmadi: {item}")
            paths.extend(matched)

    # Allaqachon kvantlangan fayllarni qayta kvantlamaymiz.
    return [p for p in paths if "_int8" not in os.path.basename(p)]


def main() -> int:
    args = parse_args()

    args.calibration_dir = _user_path(args.calibration_dir)
    if args.models:
        args.models = [_user_path(m) for m in args.models]

    # ---------------- kalibrlash papkasi: MAJBURIY tekshiruv ----------------
    # Bu tekshiruv ataylab og'ir import'lardan OLDIN turadi — foydalanuvchi
    # eng muhim xabarni darhol ko'rsin.
    if not args.calibration_dir:
        _die(
            "`--calibration-dir` berilmadi.\n\n"
            f"Static quantization uchun kamida {args.min_images} ta HAQIQIY kadr kerak — "
            "aktivatsiya diapazonlari aynan shu kadrlar ustida o'lchanadi.\n\n"
            "Bu skript dynamic quantization'ga TUSHIB QOLMAYDI: konvolyutsion "
            "tarmoqlarda dynamic quantization deyarli tezlik bermaydi, lekin "
            "\"kvantladik\" degan yolg'on tuyg'u beradi.\n\n"
            "Kadrlarni produksiya kameralaringizdan yig'ing (o'sha yorug'lik, "
            "o'sha burchak) va qayta urinib ko'ring:\n"
            "    python scripts/quantize_int8.py --calibration-dir ./calib_frames"
        )

    if not os.path.isdir(args.calibration_dir):
        _die(f"Kalibrlash papkasi topilmadi: {args.calibration_dir}")

    files: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(args.calibration_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(args.calibration_dir, f"*{ext.upper()}")))
    files = sorted(set(files))[: args.max_images]

    if len(files) < args.min_images:
        _die(
            f"Kalibrlash uchun kadrlar yetarli emas: {len(files)} ta topildi, "
            f"kamida {args.min_images} ta kerak.\n\n"
            f"Papka: {args.calibration_dir}\n"
            f"Qidirilgan formatlar: {', '.join(IMAGE_EXTENSIONS)}\n\n"
            "Static quantization aktivatsiya diapazonlarini shu kadrlardan o'lchaydi. "
            "Kadr kam bo'lsa diapazonlar tasodifiy chiqadi va INT8 model FAR'ni "
            "beixtiyor oshirib yuborishi mumkin.\n\n"
            "Agar ataylab kamroq kadr bilan sinamoqchi bo'lsangiz: --min-images N "
            "(lekin natijani production'ga chiqarmang)."
        )

    # Kalibrlash ma'lumoti joyida — endi og'ir bog'liqliklarni tekshiramiz.
    _require("numpy", "`pip install -r requirements.txt` ni bajaring.")
    cv2 = _require("cv2", "`pip install -r requirements.txt` ni bajaring.")
    ort = _require("onnxruntime", "`pip install -r requirements.txt` ni bajaring.")

    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        _die(
            f"onnxruntime.quantization mavjud emas ({exc}).\n"
            "`pip install -r requirements-dev.txt` ni bajaring (onnx + onnxruntime kerak)."
        )

    models = _collect_models(args.models)
    if not models:
        _die(
            "Kvantlanadigan `.onnx` model topilmadi.\n"
            "Avval eksport qiling: python scripts/export_onnx.py"
        )

    locator = _FaceLocator(args.detector)

    print("=" * 72)
    print("STATIC INT8 KVANTLASH")
    print("=" * 72)
    print(f"Kalibrlash papkasi : {args.calibration_dir}")
    print(f"Kadrlar            : {len(files)} ta")
    print(f"Detektor           : {locator.active}")
    print(f"Modellar           : {len(models)} ta")
    print(f"per_channel        : {args.per_channel}")
    print()

    # Rasmlarni bir marta o'qib olamiz — har bir model uchun qayta o'qish shart emas.
    frames = []
    unreadable = 0
    for path in files:
        image = cv2.imread(path)
        if image is None:
            unreadable += 1
            continue
        frames.append((path, image))

    if unreadable:
        print(f"[ogoh] {unreadable} ta fayl o'qilmadi (buzuq yoki rasm emas).")

    failures = 0
    produced: List[Tuple[str, float, float]] = []

    for model_path in models:
        base = os.path.basename(model_path)
        out_path = os.path.join(
            os.path.dirname(model_path),
            os.path.splitext(base)[0] + args.suffix + ".onnx",
        )

        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {os.path.basename(out_path)} mavjud (--force bilan qayta yozing)")
            continue

        print(f"\n[model] {base}")

        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ❌ modelni ochib bo'lmadi: {exc}")
            continue

        inp = session.get_inputs()[0]
        preprocess_kind = _resolve_preprocess(model_path, args.preprocess, inp.shape)
        preprocess = PREPROCESSORS[preprocess_kind]
        print(f"  preprocessing: {preprocess_kind} (kirish `{inp.name}` {inp.shape})")

        # MiniFASNet uchun scale/o'lcham model NOMIDAN o'qiladi — bu mantiq
        # `minifasnet_adapter.parse_model_name()` da, uni takrorlamaymiz.
        spec = None
        if preprocess_kind == "minifasnet":
            from src.infrastructure.antispoofing.minifasnet_adapter import parse_model_name

            try:
                # parse_model_name `.pth` kutadi; ONNX nomi bir xil, faqat kengaytmasi boshqa.
                spec = parse_model_name(os.path.splitext(base)[0] + ".pth")
            except ValueError as exc:
                failures += 1
                print(f"  ❌ {exc}")
                continue

        # ---------------- kalibrlash namunalarini yig'ish ----------------
        samples = []
        no_face = 0
        for _, image in frames:
            located = locator.locate(image)
            if located is None:
                no_face += 1
                continue
            bbox, kps = located
            try:
                samples.append(preprocess(image, bbox, kps, spec))
            except Exception as exc:  # noqa: BLE001
                no_face += 1
                if no_face <= 3:
                    print(f"  [ogoh] kadr tayyorlanmadi: {exc}")

        print(f"  kalibrlash namunalari: {len(samples)} ta "
              f"(yuz topilmagan/o'tkazib yuborilgan: {no_face})")

        if len(samples) < args.min_images:
            failures += 1
            print(
                f"  ❌ Yaroqli namuna yetarli emas: {len(samples)} < {args.min_images}.\n"
                f"     Kadrlarning ko'pida yuz topilmadi. Kalibrlash to'plamini "
                f"yuzi aniq ko'rinadigan kadrlar bilan to'ldiring."
            )
            continue

        # ---------------- kvantlash ----------------
        # `quant_pre_process` shakl inferensiyasini bajaradi. Usiz ba'zi tugunlar
        # kvantlanmay qolib ketadi va tezlik kutilganidan past chiqadi.
        prepared = model_path
        tmp_prepared = os.path.splitext(model_path)[0] + ".preproc.onnx"
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            quant_pre_process(model_path, tmp_prepared, skip_symbolic_shape=False)
            prepared = tmp_prepared
        except Exception as exc:  # noqa: BLE001
            print(f"  [ogoh] quant_pre_process bajarilmadi ({exc}); asl model ishlatiladi.")

        reader = _build_calibration_reader(samples, inp.name)

        try:
            quantize_static(
                model_input=prepared,
                model_output=out_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=args.per_channel,
                # x86 CPU'da (AVX2/VNNI) aktivatsiya uchun QUInt8, vazn uchun QInt8 —
                # ONNX Runtime rasman shu kombinatsiyani tavsiya qiladi. QInt8
                # aktivatsiya VNNI bo'lmagan protsessorda sekinroq ishlaydi.
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ❌ kvantlash muvaffaqiyatsiz: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            continue
        finally:
            if os.path.exists(tmp_prepared):
                os.remove(tmp_prepared)

        # Kvantlangan model umuman yuklanadimi va shakl saqlanganmi?
        try:
            q_session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            q_input = q_session.get_inputs()[0].name
            q_session.run(None, {q_input: samples[0]})
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ❌ kvantlangan model ishlamadi: {exc}")
            os.remove(out_path)
            continue

        fp32_mb = os.path.getsize(model_path) / (1024 * 1024)
        int8_mb = os.path.getsize(out_path) / (1024 * 1024)
        produced.append((os.path.basename(out_path), fp32_mb, int8_mb))
        print(f"  [ok] {os.path.basename(out_path)} — "
              f"{fp32_mb:.2f} MB -> {int8_mb:.2f} MB ({int8_mb / fp32_mb * 100:.0f}%)")

    # ------------------------------------------------------------------ xulosa
    print()
    print("=" * 72)
    if produced:
        print(f"Kvantlandi: {len(produced)} ta model")
        for name, fp32_mb, int8_mb in produced:
            print(f"  {name:40s} {fp32_mb:6.2f} MB -> {int8_mb:6.2f} MB")
    else:
        print("Yangi INT8 fayl yozilmadi.")

    if failures:
        print(f"\n{failures} ta model kvantlanmadi (yuqoridagi xabarlarga qarang).")

    if produced:
        print()
        print("⚠️  MAJBURIY KEYINGI QADAM — usiz bu modellarni ISHLATMANG:")
        print()
        print("    python scripts/accuracy_check.py --pairs pairs.csv \\")
        print("        --model <fp32>.onnx --compare-model <int8>.onnx")
        print()
        print("INT8 aniqlik pasayishi FAR (begonani qabul qilish) ko'rinishida chiqadi")
        print("va uni faqat o'lchash orqali ko'rish mumkin. Asl fayllar joyida qoldi —")
        print("servis hozircha o'sha FP32 modellar bilan ishlayapti.")

    return 1 if failures and not produced else 0


if __name__ == "__main__":
    sys.exit(main())
