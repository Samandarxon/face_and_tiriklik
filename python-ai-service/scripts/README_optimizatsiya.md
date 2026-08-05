# CPU-only serverda AI servisni tezlashtirish — qadamma-qadam qo'llanma

Bu hujjat `python-ai-service` ni GPU'siz serverda tezlashtirish uchun yozilgan
**vositalar** to'plamini tushuntiradi.

> ## ⛔ ENG MUHIM QOIDA
>
> **Kvantlashni (INT8) FAR/FRR o'lchamasdan production'ga chiqarmang.**
>
> Bu shunchaki tavsiya emas. Yuz tanish tizimida aniqlik pasayishi ikki xil
> ko'rinishda chiqadi:
>
> | Nima buziladi | Qanday bilinadi |
> |---|---|
> | **FRR** oshsa (o'zinikini rad etish) | Xodim shikoyat qiladi. Muammo **darhol** ko'rinadi. |
> | **FAR** oshsa (begonani qabul qilish) | **Hech kim shikoyat qilmaydi.** Log toza. Metrikalar yaxshi. Begona odam binoga kirib ketadi va tizim buni "muvaffaqiyatli tanish" deb yozadi. |
>
> Kirish-chiqishni nazorat qiladigan tizimda ikkinchisi qabul qilib bo'lmaydigan
> xavf. Uni **faqat o'lchash** orqali ko'rish mumkin:
>
> ```bash
> python scripts/accuracy_check.py --pairs pairs.csv \
>     --model  w600k_r50.onnx \
>     --compare-model w600k_r50_int8.onnx
> ```

---

## 0. Bu skriptlar hech narsani o'zgartirmaydi

Barcha to'rt skript **faqat o'lchaydi va fayl yozadi**. Ular
`DET_SIZE`, `MATCH_THRESHOLD`, `LIVENESS_THRESHOLD` yoki boshqa biror
sozlamani **o'zgartirmaydi**, `.env` ga tegmaydi, servis kodini tahrirlamaydi.

Sabab: kvantlash yoki `DET_SIZE` ni o'lchovsiz o'zgartirish FAR'ni **jimgina**
oshirishi mumkin. Shuning uchun qaror **siz o'z ma'lumotingizda o'lchagandan
keyin** qabul qilinadi va `.env` ga **qo'lda** kiritiladi.

### Vositalar

| Skript | Vazifasi |
|---|---|
| `export_onnx.py` | MiniFASNet `.pth` → `.onnx` + PyTorch bilan mosligini tekshirish |
| `quantize_int8.py` | ONNX modellarni **static** INT8 ga kvantlash |
| `benchmark.py` | Latensiya (p50/p95/p99), throughput, xotira, thread sozlamalari |
| `accuracy_check.py` | **FAR / FRR / ROC / EER** — kvantlash uchun majburiy darvoza |

### O'rnatish

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt    # onnx, psutil — faqat offline ishlar uchun
```

---

## 1. Hozirgi holatni o'lchang (BIRINCHI QADAM)

**Optimizatsiyani o'lchovdan boshlang.** Qaysi bosqich sekinligini bilmasdan
tezlashtirishga urinish — vaqtni behuda sarflash.

```bash
python scripts/benchmark.py --images ./bench_frames --json baseline.json
```

`bench_frames` — sizning kameralaringizdan olingan **haqiqiy kadrlar**
(20–50 ta yetarli). Tasodifiy shovqinda yuz topilmaydi va embedding/liveness
bosqichlari umuman ishlamaydi — natija aldamchi chiqadi.

Chiqishda uchta jadval bo'ladi: latensiya (bosqichlar alohida), throughput
(parallellik bo'yicha) va xotira.

### Odatda tor joy qayerda

| Bosqich | Model | Odatiy ulush |
|---|---|---|
| **detection** | SCRFD (`det_10g`), `DET_SIZE=640` | **eng katta** — ko'pincha 50–70% |
| liveness | MiniFASNet ×2 (ansambl) | o'rtacha, PyTorch runtime'da |
| embedding | ArcFace `w600k_r50` | yuz soniga chiziqli |

Baseline'ni `baseline.json` ga saqlang — keyingi har bir o'zgarishni shunga
nisbatan taqqoslaysiz.

---

## 2. Thread sozlash — eng arzon va eng ko'p unutiladigan yutuq

Bu bosqich **hech qanday aniqlik xavfi yo'q** (model o'zgarmaydi), lekin
throughput'ni sezilarli o'zgartiradi.

### ★ Qoida

```
intra_op × worker_soni  ≈  yadrolar soni
```

* `intra_op` — bitta operator ichidagi parallellik (bitta inferens ichida).
* `inter_op` — mustaqil tugunlarni parallel bajarish. CNN'larda deyarli
  foydasiz — **`inter_op=1` qoldiring**.
* `worker_soni` — uvicorn/gRPC worker (process) soni.

### Klassik xato

16 yadroli serverda `intra_op=16` qo'yib, ustiga **4 ta worker** ishga tushirish.
Natijada `4 × 16 = 64` ta thread 16 ta yadro uchun kurashadi, kontekst
almashinuvi vaqtning yarmini yeydi va **throughput pasayadi**.

| Yadro | Worker | To'g'ri `intra_op` | Izoh |
|---|---|---|---|
| 4 | 1 | 4 | past yuklama, latensiya muhim |
| 8 | 2 | 4 | muvozanatli |
| 16 | 4 | 4 | ko'p so'rov, throughput muhim |
| 16 | 1 | 16 | faqat bitta og'ir so'rov bo'lsa |

### O'lchang

```bash
python scripts/benchmark.py --images ./bench_frames \
    --thread-configs 1:1,2:1,4:1,8:1,16:1 \
    --concurrency 1,2,4,8 --json threads.json
```

Latensiya uchun eng yaxshi konfiguratsiya bilan throughput uchun eng yaxshisi
**har xil bo'ladi**. Qaysi biri muhimligini o'zingiz hal qilasiz.

> **Muhim nuqta:** MiniFASNet PyTorch'da, qolgan ikkalasi ONNX Runtime'da
> ishlaydi. Ular **alohida thread pool** ochadi va bir xil yadrolarni talashadi.
> `benchmark.py` ikkalasini birga sozlaydi (`torch.set_num_threads` +
> `SessionOptions.intra_op_num_threads`). Production'da ham ikkalasini
> sozlashni unutmang.

**Kutilayotgan foyda:** throughput bo'yicha 1.2–2x (noto'g'ri sozlangan bo'lsa).
**Xavf:** yo'q — aniqlik o'zgarmaydi.

---

## 3. Liveness'ni ONNX'ga o'tkazish (torch'dan qutulish)

### Nima uchun

| | Hajm |
|---|---|
| `torch` (CPU wheel) | **~900 MB** disk, ~250 MB RSS |
| MiniFASNet vaznlari | **~1.8 MB** har biri (ikkitasi ~3.6 MB) |

3.6 MB model uchun 900 MB kutubxona olib yurilmoqda. Detektsiya va embedding
allaqachon ONNX Runtime'da — agar liveness ham o'tsa, `torch` **butunlay**
olib tashlanishi mumkin.

```bash
python scripts/export_onnx.py
```

Skript har bir `.pth` uchun `.onnx` yasaydi va **darhol tekshiradi**: bir xil
tasodifiy kirishda PyTorch va ONNX chiqishlari `numpy.allclose(rtol=1e-3)` bilan
mos kelishi shart. Mos kelmasa — fayl **o'chiriladi** va skript xato bilan
to'xtaydi. Jimgina noto'g'ri model yozib qo'yish eng yomon natija.

Keyin tezlikni taqqoslang:

```bash
python scripts/benchmark.py --images ./bench_frames --liveness torch --json liv_torch.json
python scripts/benchmark.py --images ./bench_frames --liveness onnx  --json liv_onnx.json
```

**Kutilayotgan foyda:** image hajmi ~600–800 MB kichrayadi; liveness latensiyasi
odatda 1.1–1.5x tezlashadi (ORT'ning CNN grafik optimizatsiyasi hisobiga).

**Xavf:** past, lekin nolga teng emas. Preprocessing **aynan bir xil** bo'lishi
shart: `crop_face` + CHW `float32`, qiymatlar **0..255** (255 ga **bo'linmaydi**).
Sabab `src/infrastructure/antispoofing/minifasnet_adapter.py` dagi `_to_tensor()`
docstring'ida yozilgan — vaznlar 0..255 kirishga o'qitilgan, `/255` qo'yilsa
BatchNorm to'yinadi va model **kirishdan qat'i nazar** doim bir xil javob beradi.

O'tkazgandan keyin regressiya testini ishga tushiring:

```bash
python scripts/verify_liveness.py
```

---

## 4. INT8 kvantlash — eng katta foyda, eng katta xavf

### Nima uchun static, dynamic emas

| | Dynamic | **Static** |
|---|---|---|
| Vaznlar | INT8 | INT8 |
| Aktivatsiyalar | runtime'da qayta o'lchanadi | **oldindan kalibrlanadi** |
| Kalibrlash kerakmi | yo'q | **ha** |
| CNN'da tezlik | deyarli yo'q | **1.5–3x** |
| Transformer/LSTM'da | yaxshi | yaxshi |

MiniFASNet va ArcFace — **konvolyutsion** tarmoqlar. Dynamic quantization'da
ularning konvolyutsiyalari INT8 kernel'iga tushmaydi, ya'ni tezlik bermaydi,
lekin "kvantladik" degan yolg'on tuyg'u beradi.

Shuning uchun `quantize_int8.py` kalibrlash rasmlari bo'lmasa **dynamic'ga
tushib qolmaydi** — u aniq xato beradi va to'xtaydi.

### 4.1. Kalibrlash to'plamini yig'ing

```bash
mkdir calib_frames
# Reseption kameralaringizdan kamida 100 ta (yaxshisi 300–500) kadr ko'chiring
```

**Kalibrlash kadrlari produksiya taqsimotini aks ettirishi shart** — o'sha
kameralar, o'sha yorug'lik, o'sha burchak. Internetdan olingan chiroyli
portretlar bilan kalibrlangan model reseptiondagi qorong'i kadrda buziladi,
chunki aktivatsiya diapazonlari boshqacha bo'ladi.

### 4.2. Kvantlang

```bash
# Liveness modellari (avval export_onnx.py bajarilgan bo'lishi kerak)
python scripts/quantize_int8.py --calibration-dir ./calib_frames

# Embedding modeli
python scripts/quantize_int8.py \
    --models /root/.insightface/models/buffalo_l/w600k_r50.onnx \
    --calibration-dir ./calib_frames
```

Natija `*_int8.onnx` nomi bilan yoziladi. **Asl fayl joyida qoladi** va servis
hozircha o'sha FP32 model bilan ishlashda davom etadi.

### 4.3. ⛔ ANIQLIKNI O'LCHANG — bu qadamni O'TKAZIB YUBORMANG

Juftliklar CSV faylini tayyorlang:

```csv
img_a,img_b,same_person
/data/faces/ali_1.jpg,/data/faces/ali_2.jpg,1
/data/faces/ali_1.jpg,/data/faces/vali_1.jpg,0
```

```bash
python scripts/accuracy_check.py --pairs pairs.csv \
    --model  /root/.insightface/models/buffalo_l/w600k_r50.onnx \
    --compare-model /root/.insightface/models/buffalo_l/w600k_r50_int8.onnx \
    --json accuracy.json
```

Skript chiqaradi:

* FAR / FRR turli chegaralarda (`MATCH_THRESHOLD=0.65` ishchi nuqta sifatida belgilanadi),
* ROC nuqtalari va **EER**,
* ikki model o'rtasidagi **farq** (p.p. — foiz nuqta),
* embedding siljishi (bir xil rasmda ikki modelning kosinus o'xshashligi),
* FAR ruxsat etilgan chegaradan ko'p oshsa — **ochiq ogohlantirish** va
  `exit code 1` (CI'da darvoza sifatida ishlatish mumkin).

#### Nechta juftlik kerak

| Negative juftlik | O'lchash mumkin bo'lgan eng kichik FAR |
|---|---|
| 100 | 1% |
| 1 000 | 0.1% |
| 10 000 | 0.01% |

20 ta juftlikda chiqqan "FAR = 0" **hech narsani anglatmaydi** — bu statistik
shovqin, xavfsizlik kafolati emas. Skript negative juftliklar 100 tadan kam
bo'lsa o'zi ogohlantiradi.

### 4.4. Qaror

| Natija | Qaror |
|---|---|
| FAR o'zgarmadi yoki kamaydi, tezlik oshdi | ✅ INT8 ga o'tish mumkin |
| FAR biroz oshdi (tolerantlik ichida), FRR barqaror | ⚠️ Juftliklarni ko'paytirib qayta o'lchang |
| **FAR sezilarli oshdi** | ⛔ **Chiqarmang.** Pastdagi variantlarga qarang |
| FRR oshdi, FAR barqaror | ⚠️ Xodimlar shikoyat qiladi — foyda arziydimi? |

FAR oshgan bo'lsa:

1. Kalibrlash to'plamini kattalashtiring / produksiyaga yaqinlashtiring.
2. `--per-channel` yoqilganiga ishonch hosil qiling (standart — yoqilgan, aniqroq).
3. **Faqat liveness'ni kvantlang, embedding modelini FP32 qoldiring.** Ko'pincha
   eng to'g'ri yechim: embedding aniqligi xavfsizlikka to'g'ridan-to'g'ri
   ta'sir qiladi, liveness esa ansambl bo'lgani uchun bardoshliroq.
4. `MATCH_THRESHOLD` ni ko'tarish orqali FAR ni qaytarish mumkin — **lekin** bu
   FRR ni oshiradi, ya'ni o'z xodimlaringiz rad etila boshlaydi.

**Kutilayotgan foyda:** 1.5–3x tezlik, model hajmi ~4x kichrayadi.
**Xavf: YUQORI** — o'lchamasdan ishlatmang.

---

## 5. `DET_SIZE` — tezlik va aniqlik o'rtasidagi savdo

`DET_SIZE=640` → `320` o'tish detektsiyani taxminan **4x** tezlashtiradi
(piksellar soni 4 barobar kamayadi). Bu oqimdagi eng katta bitta yutuq.

**Lekin:** kichik yoki uzoqdagi yuzlar **topilmay qoladi**. Bu to'g'ridan-to'g'ri
FRR o'sishi — odam kameraga qaraydi, lekin tizim uni umuman ko'rmaydi.

```bash
python scripts/benchmark.py --images ./bench_frames --det-sizes 640,320
```

Jadvaldagi **`topilgan yuz`** ustuniga qarang — `benchmark.py` 320 da 640 ga
nisbatan qancha yuz yo'qolganini foizda ham ko'rsatadi. Tezlikka qarab qaror
qabul qilib, bu ustunni ko'rmaslik — eng ko'p uchraydigan xato.

Kadrlaringizda odamlar kameraga yaqin tursa (reseption — aynan shunday), 320
yo'qotishsiz ishlashi mumkin. Buni **faqat o'z kadrlaringizda** tekshirasiz.

**Kutilayotgan foyda:** detektsiyada ~4x.
**Xavf: o'rta-yuqori** — FRR o'sishi. `DET_SIZE` hozirgi qiymatida (`640`)
qoldirilgan; o'zgartirish uchun `.env` ga qo'lda kiriting.

---

## 6. ONNX Runtime vs OpenVINO

OpenVINO — Intel'ning CPU inferens runtime'i. ONNX Runtime uni
**Execution Provider** sifatida ishlata oladi (`OpenVINOExecutionProvider`),
ya'ni kodni qayta yozish shart emas.

| | ONNX Runtime (CPU EP) | OpenVINO |
|---|---|---|
| **Intel Xeon / Core** (AVX-512, VNNI) | baseline | **1.5–2x tezroq** |
| **Intel** (eski, AVX2) | baseline | 1.2–1.4x |
| **AMD EPYC / Ryzen** | baseline | **~1.0–1.1x — foyda deyarli yo'q** |
| **ARM** (Graviton, Ampere) | baseline | qo'llab-quvvatlanmaydi |
| O'rnatish | `requirements.txt` da bor | qo'shimcha paket (~400 MB) |
| Image hajmi | baseline | **+300–400 MB** |
| INT8 qo'llab-quvvatlash | ha (QDQ) | ha (o'zining kvantlash oqimi) |
| Murakkablik | past | o'rta (model konvertatsiyasi, EP sozlash) |

### Tavsiya

1. **Avval protsessorni aniqlang:**
   ```bash
   lscpu | grep -E "Model name|Flags" | head -2
   ```
   `avx512_vnni` yoki `avx_vnni` bayrog'i bo'lsa — OpenVINO va INT8 dan katta
   foyda kutish mumkin.

2. **AMD bo'lsa OpenVINO'ga vaqt sarflamang.** Foyda ~10% atrofida, image esa
   400 MB kattalashadi. O'rniga thread sozlash (2-bo'lim) va INT8 (4-bo'lim)
   bilan shug'ullaning.

3. **Intel bo'lsa** OpenVINO'ni sinash arziydi — lekin **avval** 2 va 3-bo'limlarni
   bajaring, chunki ular arzonroq va xavfsizroq.

> OpenVINO'ga o'tsangiz ham `accuracy_check.py` ni qayta ishga tushiring:
> boshqa runtime = boshqa sonli natijalar. Farq odatda juda kichik, lekin
> **tekshirilgan** kichik farq bilan **taxmin qilingan** kichik farq — bir xil
> narsa emas.

---

## 7. Tavsiya etilgan tartib (xavf bo'yicha o'sish tartibida)

```
1. benchmark.py            -> baseline o'lchash            [xavf: yo'q]
2. thread sozlash          -> intra_op × worker ≈ yadro    [xavf: yo'q]
3. export_onnx.py          -> torch'dan qutulish           [xavf: past]
4. DET_SIZE 640 vs 320     -> faqat o'lchash, qaror sizniki [xavf: o'rta]
5. quantize_int8.py        -> INT8                          [xavf: YUQORI]
6. accuracy_check.py       -> MAJBURIY darvoza              [xavf: —]
```

**1–3 qadamlar deyarli xavfsiz va ko'pincha yetarli foyda beradi.**
4–5 qadamlarga faqat 1–3 dan keyin ham tezlik yetmasa o'ting.

### To'liq oqim

```bash
# 1) baseline
python scripts/benchmark.py --images ./bench_frames --json 1_baseline.json

# 2) thread sozlamalari
python scripts/benchmark.py --images ./bench_frames \
    --thread-configs 1:1,2:1,4:1,8:1 --json 2_threads.json

# 3) liveness -> ONNX
python scripts/export_onnx.py
python scripts/benchmark.py --images ./bench_frames --liveness onnx --json 3_onnx.json
python scripts/verify_liveness.py

# 4) DET_SIZE taqqoslash (faqat o'lchov)
python scripts/benchmark.py --images ./bench_frames --det-sizes 640,320 --json 4_detsize.json

# 5) INT8
python scripts/quantize_int8.py --calibration-dir ./calib_frames

# 6) ⛔ MAJBURIY darvoza
python scripts/accuracy_check.py --pairs pairs.csv \
    --model w600k_r50.onnx --compare-model w600k_r50_int8.onnx --json 6_accuracy.json
```

---

## 8. Xulosa jadvali

| Qadam | Kutilayotgan foyda | Xavf | Majburiy tekshiruv |
|---|---|---|---|
| Thread sozlash | throughput 1.2–2x | yo'q | `benchmark.py` |
| Liveness → ONNX | image −600..800 MB, latensiya 1.1–1.5x | past | `verify_liveness.py` |
| `DET_SIZE` 640→320 | detektsiya ~4x | **o'rta-yuqori** (FRR) | `benchmark.py` `topilgan yuz` ustuni |
| INT8 (liveness) | 1.5–2x, hajm −4x | o'rta | `accuracy_check.py` + `verify_liveness.py` |
| INT8 (embedding) | 1.5–3x, hajm −4x | **YUQORI** (FAR) | `accuracy_check.py` — **shartsiz** |
| OpenVINO (Intel) | 1.5–2x | past-o'rta | `accuracy_check.py` |
| OpenVINO (AMD) | ~1.0–1.1x | past | arzimaydi |

---

## 9. Tez-tez uchraydigan xatolar

**"Kvantladim, hech narsa tezlashmadi."**
Dynamic quantization ishlatgansiz yoki `quant_pre_process` bajarilmagan.
`quantize_int8.py` static ishlatadi va shakl inferensiyasini o'zi bajaradi.

**"Yadro qo'shdim, tezlik tushdi."**
Klassik thread oshib ketishi. `intra_op × worker ≈ yadro` qoidasiga qayting.
Esda tuting: torch va ONNX Runtime **alohida** thread pool ochadi.

**"INT8 da hamma narsa yaxshi ko'rinadi."**
Negative juftliklaringiz kammi? 100 ta juftlikda o'lchanadigan eng kichik FAR —
1%. Bu darajada "FAR = 0" hech narsani isbotlamaydi.

**"Liveness ONNX'ga o'tkazdim, hamma yuz 'soxta' bo'lib qoldi."**
Preprocessing'da kirish 255 ga bo'lingan. MiniFASNet **0..255** kutadi.
`verify_liveness.py` aynan shuni ushlaydi.

**"320 da benchmark tezroq, o'tsam bo'ladimi?"**
`topilgan yuz` ustuniga qarang. Tezlik har doim oshadi — savol qancha yuz
yo'qolishida.
