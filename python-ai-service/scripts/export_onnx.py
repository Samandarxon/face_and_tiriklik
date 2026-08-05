#!/usr/bin/env python3
"""MiniFASNet (.pth) vaznlarini ONNX ga eksport qiladi + ekvivalentlikni tekshiradi.

NIMA UCHUN KERAK
================
Hozir oqimning uch bosqichidan ikkitasi (detektsiya + embedding) ONNX Runtime'da,
uchinchisi (liveness / anti-spoofing) esa PyTorch'da ishlaydi. Bu juda qimmat:

    torch (CPU wheel)          ~900 MB disk, ~250 MB RSS
    MiniFASNet vaznlari        ~1.8 MB har biri (ikkitasi ~3.6 MB)

Ya'ni 3.6 MB modelni ishlatish uchun 900 MB kutubxona olib yuriladi. Agar
liveness ham ONNX'ga o'tsa:

  * Docker image bir necha yuz MB ga kichrayadi (CPU-only serverda muhim),
  * konteyner RSS'i kamayadi — bitta mashinaga ko'proq worker sig'adi,
  * yagona runtime qoladi, ya'ni thread sozlash bitta joydan boshqariladi
    (hozir ORT va torch thread pool'lari bir-biri bilan yadro talashadi).

BU SKRIPT HECH NARSANI YOQMAYDI
===============================
Skript faqat `.onnx` fayl YOZADI. `minifasnet_adapter.py` o'zgarmaydi va servis
baribir `.pth` bilan ishlashda davom etadi. ONNX'ga haqiqiy o'tish alohida
qaror bo'lib, u `scripts/benchmark.py` (tezlik) va `scripts/accuracy_check.py`
(FAR/FRR) natijalaridan KEYIN qabul qilinadi.

TEKSHIRUV — SKRIPTNING ENG MUHIM QISMI
======================================
Eksportdan keyin bir xil tasodifiy kirishda PyTorch va ONNX chiqishlari
`numpy.allclose(rtol=1e-3)` bilan solishtiriladi. Mos kelmasa skript XATO
bilan to'xtaydi va yarim-yaroqsiz faylni O'CHIRADI.

Sabab: jimgina noto'g'ri model yozib qo'yish — eng yomon natija. Liveness
tashqaridan "ishlayotgandek" ko'rinadi (200 OK qaytaradi), lekin qarorlari
ma'nosiz bo'ladi va buni hech kim sezmaydi.

Kirish diapazoni haqida: dummy tensor ataylab 0..255 oralig'ida yasaladi.
MiniFASNet vaznlari aynan shunday kirishga o'qitilgan — sababi
`minifasnet_adapter._to_tensor()` docstring'ida batafsil yozilgan.

Ishlatish:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --out-dir resources/anti_spoof_onnx
    python scripts/export_onnx.py --batch-check 4 --rtol 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys

ORIGINAL_CWD = os.getcwd()
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)
# Sozlamalardagi yo'llar nisbiy (`./resources/...`), shuning uchun skript
# qayerdan chaqirilganidan qat'i nazar servis ildizidan ishlaymiz.
os.chdir(SERVICE_ROOT)


def _die(message: str, code: int = 2) -> "None":
    """Stack trace o'rniga tushunarli xato. Skript modellarsiz ham yiqilmasligi kerak."""
    print(f"\nXATO: {message}", file=sys.stderr)
    raise SystemExit(code)


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError:
        _die(f"`{module}` o'rnatilmagan. {hint}")


def _user_path(path):
    """CLI'dan kelgan nisbiy yo'lni ORIGINAL papkaga nisbatan hal qiladi.

    Skript servis ildiziga `chdir` qiladi (sozlamalardagi yo'llar nisbiy),
    shuning uchun foydalanuvchi bergan yo'llarni oldindan absolyutlashtiramiz.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ORIGINAL_CWD, path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description=(
            "MiniFASNet PyTorch vaznlarini ONNX ga eksport qiladi va PyTorch bilan "
            "bir xil natija berishini tekshiradi. Servis sozlamalarini O'ZGARTIRMAYDI."
        ),
        epilog=(
            "Keyingi qadam:\n"
            "  1) python scripts/benchmark.py       -> tezlik farqini o'lchash\n"
            "  2) python scripts/accuracy_check.py  -> aniqlik farqini o'lchash\n"
            "Faqat shundan keyin ONNX'ga o'tish haqida qaror qabul qiling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="`.pth` fayllar papkasi (standart: LIVENESS_MODEL_DIR sozlamasi).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="`.onnx` fayllar yoziladigan papka (standart: --model-dir bilan bir xil).",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset versiyasi (standart: 17).",
    )
    parser.add_argument(
        "--batch-check", type=int, default=4,
        help=(
            "Dynamic batch o'qini tekshirish uchun ikkinchi sinov batch o'lchami "
            "(standart: 4). 0 -> faqat batch=1 tekshiriladi."
        ),
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3,
        help="numpy.allclose uchun nisbiy tolerantlik (standart: 1e-3).",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="numpy.allclose uchun absolyut tolerantlik (standart: 1e-4).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Mavjud `.onnx` fayl ustiga qayta yozish.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    torch = _require("torch", "`pip install -r requirements.txt` ni bajaring.")
    np = _require("numpy", "`pip install -r requirements.txt` ni bajaring.")
    _require("onnx", "`pip install -r requirements-dev.txt` ni bajaring.")
    ort = _require("onnxruntime", "`pip install -r requirements.txt` ni bajaring.")

    from src.infrastructure.config import settings
    from src.infrastructure.antispoofing.minifasnet_adapter import MiniFASNetChecker

    model_dir = _user_path(args.model_dir) or settings.liveness_model_dir
    out_dir = _user_path(args.out_dir) or model_dir
    os.makedirs(out_dir, exist_ok=True)

    # Modellarni PRODUKSIYA yo'li bilan yuklaymiz: bir xil arxitektura tanlash,
    # bir xil `module.` prefiks tozalash, bir xil `eval()`. Shu kodni bu yerda
    # takrorlasak — eksport qilingan model servisdagidan farq qilib qolishi
    # mumkin va buni hech qanday test ushlamaydi.
    try:
        checker = MiniFASNetChecker(
            model_dir=model_dir,
            device=torch.device("cpu"),   # eksport har doim CPU'da
            threshold=settings.liveness_threshold,
            require_unanimous=settings.liveness_require_unanimous,
        )
    except FileNotFoundError as exc:
        _die(str(exc))
    except Exception as exc:  # noqa: BLE001 - foydalanuvchiga tushunarli xabar muhimroq
        _die(f"Liveness modellarini yuklab bo'lmadi: {exc}")

    # `_models` — adapterning ichki ro'yxati [(ModelSpec, nn.Module), ...].
    # Ataylab shundan foydalanamiz: bu servis ishlatadigan AYNAN o'sha obyektlar.
    pairs = list(checker._models)  # noqa: SLF001
    if not pairs:
        _die(f"{model_dir} ichida eksport qilinadigan model topilmadi.")

    print(f"Manba papka : {model_dir}")
    print(f"Chiqish     : {out_dir}")
    print(f"Opset       : {args.opset} | rtol={args.rtol} atol={args.atol}")
    print(f"Modellar    : {len(pairs)} ta\n")

    exported: list[tuple[str, str, float]] = []
    failures = 0

    for spec, model in pairs:
        out_name = os.path.splitext(spec.name)[0] + ".onnx"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {out_name} allaqachon mavjud (--force bilan qayta yozing)")
            continue

        print(f"[export] {spec.name} -> {out_name} "
              f"({spec.height}x{spec.width}, arch={spec.arch}, scale={spec.scale})")

        # DIQQAT: 0..255 oralig'i. MiniFASNet vaznlari [0,1] ga normalizatsiya
        # QILINMAGAN kirish bilan o'qitilgan (batafsil: `_to_tensor()` docstring).
        # Tracing paytida ham xuddi shu diapazonni ishlatamiz.
        rng = np.random.default_rng(seed=1234)
        dummy_np = rng.uniform(0.0, 255.0, size=(1, 3, spec.height, spec.width)).astype(np.float32)
        dummy = torch.from_numpy(dummy_np)

        try:
            model.eval()
            torch.onnx.export(
                model,
                dummy,
                out_path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                # Dynamic batch: kelajakda bir nechta yuzni bitta chaqiruvda
                # o'tkazish imkonini qoldiradi (ansambl uchun ham foydali).
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ❌ eksport muvaffaqiyatsiz: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            continue

        # ---------------- ekvivalentlik tekshiruvi ----------------
        try:
            session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ❌ ONNX faylni ochib bo'lmadi: {exc}")
            os.remove(out_path)
            continue

        batches = [1] + ([args.batch_check] if args.batch_check and args.batch_check > 1 else [])
        max_diff_overall = 0.0
        ok = True

        for batch in batches:
            sample = rng.uniform(0.0, 255.0, size=(batch, 3, spec.height, spec.width)).astype(np.float32)

            with torch.inference_mode():
                torch_out = model(torch.from_numpy(sample)).cpu().numpy()
            onnx_out = session.run(None, {input_name: sample})[0]

            if torch_out.shape != onnx_out.shape:
                ok = False
                print(f"  ❌ batch={batch}: shakl mos emas "
                      f"(torch={torch_out.shape}, onnx={onnx_out.shape})")
                break

            max_diff = float(np.max(np.abs(torch_out - onnx_out)))
            max_diff_overall = max(max_diff_overall, max_diff)

            if not np.allclose(torch_out, onnx_out, rtol=args.rtol, atol=args.atol):
                ok = False
                print(f"  ❌ batch={batch}: chiqishlar mos kelmadi (max_diff={max_diff:.3e})")
                break

            print(f"  ✅ batch={batch}: mos (max_diff={max_diff:.3e})")

        if not ok:
            failures += 1
            # Yaroqsiz faylni QOLDIRMAYMIZ — kimdir uni ishlatib yuborishi mumkin.
            os.remove(out_path)
            print(f"  -> {out_name} o'chirildi (noto'g'ri modelni saqlab qolishdan ko'ra yo'q bo'lgani xavfsizroq)")
            continue

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        exported.append((spec.name, out_name, size_mb))
        print(f"  [ok] {out_name} — {size_mb:.2f} MB\n")

    # ------------------------------------------------------------------ xulosa
    print("=" * 72)
    if exported:
        print(f"Eksport qilindi: {len(exported)} ta model")
        for src, dst, size_mb in exported:
            print(f"  {src:36s} -> {dst:36s} {size_mb:6.2f} MB")
    else:
        print("Yangi fayl yozilmadi.")

    if failures:
        print(f"\nXATO: {failures} ta model tekshiruvdan o'tmadi.")
        print("ONNX chiqishi PyTorch bilan mos kelmasa uni ISHLATMANG.")
        print("Ehtimoliy sabablar: opset mos emas, torch/onnx versiyalari zid,")
        print("yoki model ichida ONNX qo'llab-quvvatlamaydigan operator bor.")
        return 1

    print("\nKeyingi qadamlar:")
    print("  1) python scripts/benchmark.py                 # tezlik farqi")
    print("  2) python scripts/quantize_int8.py --help      # (ixtiyoriy) INT8")
    print("  3) python scripts/accuracy_check.py --help     # FAR/FRR — MAJBURIY")
    print("\nEslatma: bu skript servis sozlamalarini o'zgartirmadi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
