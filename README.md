# Face Authentication Microservice

Golang (Gin) + Python (FastAPI) + Qdrant asosidagi, **Anti-Spoofing** himoyasiga ega
yuz orqali autentifikatsiya tizimi.

Asosiy qoida: **kamera oldida tirik inson bo'lmasa, bazaga qidiruv so'rovi umuman
yuborilmaydi.** Telefon ekranidagi surat, qog'ozga chop etilgan foto yoki ekranda
o'ynatilayotgan video `400 spoof_detected` bilan rad etiladi.

---

## 1. Arxitektura

```
  Brauzer / Mobil / Kiosk
        │            │
        │ REST       │ WebSocket (JPEG kadrlar)
        ▼            ▼
  ┌──────────────────────────────────────────┐
  │        Golang Backend (Gin)  :8080       │
  │                                          │
  │  delivery → usecase → domain ← adapter   │
  └────────┬───────────────────────┬─────────┘
           │ gRPC                  │ REST (JSON)
           │ (unary + bidi stream) │
           ▼                       ▼
  ┌──────────────────────┐   ┌────────────────────┐
  │  Python AI Service   │   │      Qdrant        │
  │  gRPC :50051         │   │      :6333         │
  │  HTTP :8000 (docs)   │   │                    │
  │                      │   │  collection: faces │
  │  ① MiniFASNet        │   │  size:  512        │
  │     (Anti-Spoofing)  │   │  distance: Cosine  │
  │  ② InsightFace       │   │                    │
  │     buffalo_l 512-d  │   │                    │
  └──────────────────────┘   └────────────────────┘
```

**Transport tanlovi va sabablari**

| Yo'nalish | Protokol | Nega |
|---|---|---|
| Brauzer → Golang (bir martalik) | HTTP REST | `FormData` bilan rasm yuborish tabiiy, hech qanday kutubxona kerak emas |
| Brauzer ↔ Golang (real-time) | **WebSocket** | Brauzer HTTP/2 trailer'larga kira olmaydi, ya'ni sof gRPC brauzerda ishlamaydi (gRPC-Web + Envoy proxy kerak bo'lardi) |
| Golang ↔ Python | **gRPC** | Qat'iy shartnoma (`.proto`), binar serializatsiya, **ikki tomonlama oqim** — ulanish bir marta o'rnatiladi, har kadrda handshake yo'q |
| Golang ↔ Qdrant | HTTP REST | Qdrant gRPC ham beradi, lekin REST bog'liqlik qo'shmaydi va bu yerda tor joy emas |

**Nima uchun AI logikasi Python'da, biznes mantiq Golang'da?**
Model inference Python ekotizimida (PyTorch/ONNX) barqarorroq, Gin esa yuqori
yuklamali gateway sifatida tezroq va kam resurs talab qiladi. Ikkalasi mustaqil
masshtablanadi: og'ir bo'lgan AI servisni GPU-node'larda ko'paytirish mumkin.

### gRPC shartnomasi

`proto/face/v1/face.proto`:

```protobuf
service FaceService {
  rpc Embed(ImageRequest) returns (EmbedResponse);             // register
  rpc VerifyFrame(ImageRequest) returns (VerifyFrameResponse); // liveness -> vektor
  rpc CheckLiveness(ImageRequest) returns (LivenessResponse);  // probe
  rpc StreamFrames(stream FrameRequest) returns (stream FrameResponse);  // ★ real-time
  rpc Health(HealthRequest) returns (HealthResponse);
}
```

Kodni qayta generatsiya qilish (protoni o'zgartirgandan keyin):

```bash
make proto     # yoki: bash scripts/gen-proto.sh
```

Generatsiya qilingan kod repozitoriyga **kommit qilinadi** — shunda Docker
build'da `protoc` kerak bo'lmaydi.

> ⚠️ **`grpcio-tools` versiyasi muhim.** U generatsiya qilgan kod ichiga
> o'zining protobuf *gencode* versiyasini yozib qo'yadi, runtime esa uni
> tekshiradi. Gencode runtime'dan yangi bo'lsa AI servis **ishga tushmaydi**:
>
> ```
> ValidateProtobufRuntimeVersion -> Detected incompatible Protobuf
> Gencode/Runtime versions
> ```
>
> Shuning uchun versiya `python-ai-service/requirements-dev.txt` da pinlangan
> va `gen-proto.sh` uni tekshiradi — mos kelmasa generatsiya to'xtaydi:
>
> ```bash
> pip install -r python-ai-service/requirements-dev.txt
> make proto
> ```
>
> `requirements.txt` dagi `grpcio`/`protobuf` pinlarini o'zgartirsangiz,
> `requirements-dev.txt` ni ham birga yangilang va kodni qayta generatsiya qiling.

**Xatolarni uzatish — ikki xil strategiya, ataylab:**

- **Unary** chaqiruvlarda biznes xatolari gRPC status + `FaceError` detali
  (`google.rpc.Status`) bilan keladi. `spoof_detected` bilan birga liveness
  ballari ham o'tadi — mijozga sababni ko'rsatish uchun.
- **Streamingda** xatolar *in-band* (`FrameResponse.error`). Bitta soxta kadr
  butun oqimni uzib qo'ymasligi kerak: foydalanuvchi telefonini olib qo'ygach
  oqim ishlashda davom etadi.

### Katalog tuzilmasi — Clean Architecture

Ikkala servis ham bir xil qatlam mantiqiga amal qiladi. **Bog'liqlik yo'nalishi
har doim ichkariga**: `delivery → usecase → domain ← adapter`. Domain qatlami
hech qanday freymvorkni import qilmaydi.

```
face_detecting_microservice/
├── proto/face/v1/face.proto         # ★ ikki servis o'rtasidagi shartnoma
├── scripts/gen-proto.sh
├── docker-compose.yml · .env.example · Makefile
│
├── golang-backend/
│   ├── cmd/server/main.go           # composition root — bog'liqliklar shu yerda ulanadi
│   ├── genproto/face/v1/            # generatsiya qilingan (qo'lda tahrirlanmaydi)
│   ├── internal/
│   │   ├── domain/                  # ★ eng ichki qatlam, freymvorksiz
│   │   │   ├── models.go            #   Person, Face, Liveness, Embedding
│   │   │   ├── errors.go            #   barqaror xato kodlari
│   │   │   └── ports.go             #   FaceAnalyzer, PersonRepository, FrameStream
│   │   ├── usecase/                 # biznes stsenariylari (portlarga bog'lanadi)
│   │   │   ├── register.go
│   │   │   ├── verify.go            #   ★ STEP 1/2/3 qoidasi shu yerda
│   │   │   └── directory.go
│   │   ├── adapter/                 # portlarning implementatsiyalari
│   │   │   ├── faceai/client.go     #   FaceAnalyzer  <- gRPC
│   │   │   └── vectordb/qdrant.go   #   PersonRepository <- Qdrant REST
│   │   ├── delivery/httpapi/        # HTTP + WebSocket adapterlari
│   │   │   ├── router.go · response.go · person_handler.go
│   │   │   └── stream_handler.go    #   ★ WebSocket ⟷ gRPC bidi ko'prigi
│   │   └── config/config.go
│   └── web/                         # demo interfeys (index.html + app.js)
│
└── python-ai-service/
    ├── main.py                      # composition root: HTTP + gRPC bitta loop'da
    ├── genproto/face/v1/
    ├── scripts/                     # download_models.sh · verify_liveness.py
    └── src/
        ├── domain/                  # ★ entities.py · errors.py · ports.py
        ├── application/use_cases.py # ExtractFace, CheckLiveness, VerifyFrame...
        ├── infrastructure/          # adapterlar
        │   ├── config.py · device.py · imaging.py
        │   ├── insightface_adapter.py
        │   └── antispoofing/
        │       ├── minifasnet_adapter.py
        │       ├── crop.py
        │       └── networks.py      #   MiniFASNet V1/V2/V1SE/V2SE
        ├── interfaces/
        │   ├── grpc_api.py          #   ★ FaceService implementatsiyasi
        │   └── http_api.py          #   Swagger/debug
        └── container.py             # bog'liqliklarni ulash

## 2. Anti-Spoofing qanday ishlaydi

Ishlatilgan model — **MiniFASNet** (MiniVision AI, *Silent-Face-Anti-Spoofing*, MIT).
Bu **passiv** usul: foydalanuvchidan ko'z qisish yoki bosh burish talab qilinmaydi,
qaror bitta kadr asosida chiqariladi.

Ikkita model ansambl qilib ishlatiladi:

| Model | Kirish | `scale` | Vazifasi |
|---|---|---|---|
| `2.7_80x80_MiniFASNetV2.pth` | 80×80 | 2.7 | Yuzga yaqin kontekst — teri tekstura, moire naqsh |
| `4_0_0_80x80_MiniFASNetV1SE.pth` | 80×80 | 4.0 | Kengroq kontekst — qog'oz cheti, telefon ramkasi, qo'l |

Har bir model 3 sinf ustidan `softmax` beradi:
`[0] = spoof (chop etilgan foto)`, `[1] = real`, `[2] = spoof (ekran/replay)`.

Qaror qabul qilish (`app/liveness/predictor.py`):

1. Ikki modelning ehtimollari **o'rtachalanadi** → `real_probability`.
2. `real_probability >= LIVENESS_THRESHOLD` **va** argmax == "real" bo'lishi kerak.
3. `LIVENESS_REQUIRE_UNANIMOUS=true` bo'lsa — **har bir model alohida** ham
   chegaradan o'tishi shart. Bu False Accept Rate'ni (soxta yuzni o'tkazib
   yuborish) sezilarli kamaytiradi.

Bundan tashqari, liveness'dan **oldin** sifat filtri ishlaydi
(`MIN_FACE_SIZE`, `MIN_BLUR_SCORE`): juda kichik yoki xira yuzlarda
anti-spoofing modeli ishonchsiz bo'lgani uchun, bunday kadrlar
"noaniq" deb o'tkazilmaydi, balki `422 low_quality_face` bilan rad etiladi.

> ### Aniqlik haqida — halol baho
>
> Siz 98%+ aniqlik so'radingiz. MiniFASNet o'zining test to'plamida shunga yaqin
> natija beradi, lekin **haqiqiy sharoitda bu raqamni hech kim kafolatlay olmaydi** —
> u kamera sifatiga, yorug'likka va hujum turiga bog'liq. Amaliy kutish:
>
> | Hujum turi | Ushlanish darajasi |
> |---|---|
> | Telefon/planshet ekranidagi foto yoki video | juda yuqori (~97–99%) |
> | Oddiy printerda chop etilgan qog'oz foto | yuqori (~95–98%) |
> | Yuqori sifatli katta bosma / kesilgan niqob | o'rtacha (~80–90%) |
> | 3D silikon niqob, deepfake video oqimi | **past — bu model bunga mo'ljallanmagan** |
>
> Yuqori xavfli stsenariylar (bank, chegara nazorati) uchun tavsiya:
> `LIVENESS_THRESHOLD=0.85`, `LIVENESS_REQUIRE_UNANIMOUS=true` va qo'shimcha
> **aktiv challenge** (tasodifiy "boshingizni chapga buring" / "ko'zingizni qisib
> oching") yoki IR/depth kamera. Bu tizim passiv qatlamni to'liq beradi;
> aktiv qatlam kerak bo'lsa, `/v1/liveness/check` endpointi ustiga
> ko'p kadrli oqim qurish mumkin.

---

## 3. Ishga tushirish (Step-by-Step)

> **Docker'siz, to'g'ridan-to'g'ri serverda ishga tushirmoqchimisiz?**
> Alohida qo'llanma: [`DOCKERSIZ_ISHGA_TUSHIRISH.md`](DOCKERSIZ_ISHGA_TUSHIRISH.md)
> (systemd + `deploy/bare-metal/install.sh`). Ikkala rejim ham qo'llab-quvvatlanadi;
> quyidagi bo'lim Docker rejimi uchun.

### Talablar
- Docker 24+ va Docker Compose v2
- ~4 GB bo'sh joy (buffalo_l ≈ 300 MB, torch ≈ 900 MB)
- GPU ixtiyoriy (NVIDIA drayver + `nvidia-container-toolkit`)

### 1-qadam — Sozlamalar va sirlar

```bash
cd face_detecting_microservice
cp .env.example .env
make secrets          # FACE_ID_PEPPER va ADMIN_BOOTSTRAP_KEY ni yasaydi
```

`make secrets` faqat **bo'sh** qatorlarni to'ldiradi — mavjud `FACE_ID_PEPPER`
hech qachon qayta yozilmaydi. (`make setup` buni o'zi chaqiradi.)

| O'zgaruvchi | Nima uchun |
|---|---|
| `FACE_ID_PEPPER` | `person_id` ni hisoblaydigan maxfiy kalit. **Yo'qotmang** — o'zgarsa butun baza qayta to'ldiriladi |
| `ADMIN_BOOTSTRAP_KEY` | Birinchi API kalitni olish uchun; keyin o'chiriladi |
| `LIVENESS_THRESHOLD` | Anti-spoofing qat'iyligi |
| `MATCH_THRESHOLD` | Tanish chegarasi |

> Servis bu sirlarsiz **ishga tushmaydi** va aniq xato bilan to'xtaydi. Tekshiruv
> ataylab `docker-compose.yml` da emas, servisning o'zida: compose'dagi majburiy
> o'zgaruvchi (`${VAR:?...}`) `docker compose down` va `ps` ni ham bloklab
> qo'yadi, ya'ni sozlama bo'sh bo'lsa konteynerlarni to'xtatib ham bo'lmaydi.

### 2-qadam — Anti-spoofing vaznlarini yuklab olish

```bash
bash python-ai-service/scripts/download_models.sh
```

Natija:
```
python-ai-service/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth       (1.8M)
python-ai-service/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth   (1.8M)
```

> Bu qadamni o'tkazib yuborsangiz ham bo'ladi — Docker build vaqtida
> avtomatik yuklab olinadi. Lekin oldindan yuklab qo'ysangiz build tezroq bo'ladi.

### 3-qadam — Ishga tushirish

```bash
docker compose up -d --build
```

Yoki qisqacha: `make setup && make up`

Birinchi build 5–15 daqiqa oladi (torch + insightface + buffalo_l yuklanadi).

### 4-qadam — Holatni tekshirish

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

Kutilgan javob:

```json
{
  "service": "golang-backend",
  "collection": "faces",
  "ai_service": {
    "ok": true,
    "details": {
      "device": "cpu",
      "insightface_loaded": true,
      "liveness_loaded": true,
      "liveness_models": ["2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth"],
      "liveness_threshold": 0.75,
      "embedding_dim": 512,
      "ready": true
    }
  },
  "qdrant": { "ok": true, "registered_persons": 0 },
  "status": "ok"
}
```

`"liveness_loaded": false` bo'lsa — vaznlar topilmagan. `docker compose logs ai-service`
ga qarang va 2-qadamni takrorlang.

### 5-qadam — Demo web-interfeys

Brauzerda oching: **http://localhost:8080**

1. "Kamerani yoqish" → ruxsat bering
2. Ismni kiriting → "Kadrni olib ro'yxatga qo'shish"
   (`person_id` so'ralmaydi — server UUID beradi va javobda ko'rsatadi)
3. Boshqa hech narsa bosish shart emas. Kamera yoqilishi bilan **WebSocket
   oqimi** ochiladi va tiriklik tasdiqlangach server o'zi to'liq tekshiruvni
   boshlaydi.

   Kamerada yuz atrofida ramka chiziladi, rangi holatni bildiradi:
   🟡 tekshirilmoqda · 🟢 tirik inson (foizi bilan) · 🔴 soxta ·
   🔵 **tanildi — ism va score ramka ustida chiqadi**.

4. **Anti-spoofing testi:** telefoningizda o'zingizning suratingizni oching va
   kameraga tuting → ramka qizil bo'lib `SOXTA` yozuvi chiqadi va avtomatik
   tekshiruv ishga **tushmaydi**.

> Kamera faqat `localhost` yoki `https://` da ishlaydi (brauzer cheklovi).
> Boshqa mashinadan kirmoqchi bo'lsangiz, oldiga TLS bilan reverse proxy qo'ying —
> WebSocket uchun `Upgrade`/`Connection` sarlavhalarini o'tkazishni unutmang.

### 6-qadam — gRPC ishlayotganini tekshirish (ixtiyoriy)

```bash
grpcurl -plaintext localhost:50051 list face.v1.FaceService
grpcurl -plaintext localhost:50051 face.v1.FaceService/Health
```

`grpcurl` bo'lmasa: `go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest`

---

## 3.5. Kirish nazorati — servis YOPIQ

`/health` va `/health/ready` dan tashqari **har bir endpoint API kalit talab
qiladi**. Kalit `X-API-Key` sarlavhasida yuboriladi (yoki
`Authorization: Bearer <kalit>`).

### Nima uchun shunday qilingan

Bu servis yuz vektorlari bazasini boshqaradi. Kalitsiz ochiq qoldirilgan
endpoint bitta `curl` bilan butun bazani ag'darish, yangi yuz qo'shish yoki
mavjudini o'chirish imkonini beradi. Shuning uchun sukut bo'yicha holat —
yopiq, va uni ochib qo'yadigan sozlama umuman yo'q.

### Model

```
  Mijoz (platforma)  ──►  api_clients        "kirdi_chiqdi", "kadrlar_tizimi", ...
        │
        └── kalitlar  ──►  api_keys          har biri scope'lar to'plami bilan
                            │
                            └── audit_log     kim, qachon, nima so'radi
```

Kalitning ochiq matni **hech qayerda saqlanmaydi** — bazada faqat SHA-256
hash va qidiruv uchun prefiks yotadi. Kalitni yo'qotsangiz uni tiklab
bo'lmaydi, faqat rotatsiya qilish mumkin.

### Scope'lar

| Scope | Nima qila oladi |
|---|---|
| `face:verify` | Rasm bo'yicha tanish, liveness, real-time oqim |
| `face:register` | Bazaga yuz qo'shish / yangilash |
| `face:read` | Yozuvlarni o'qish, `subject_id` → `person_id` hisoblash |
| `face:delete` | Yozuvni o'chirish |
| `admin:keys` | Mijoz va kalitlarni boshqarish, jurnalni ko'rish, `/metrics` |

Kalitga faqat kerakli scope'ni bering. Masalan kirish-chiqish terminaliga
`face:verify` yetarli — u bazani o'zgartira olmaydi.

### Birinchi kalitni olish (bootstrap)

`ADMIN_BOOTSTRAP_KEY` faqat bazada **birorta ham faol kalit bo'lmaganda**
ishlaydi. Birinchi kalit berilishi bilan bu yo'l avtomatik yopiladi.

```bash
# 1. Mijozni ro'yxatga olish
curl -X POST http://localhost:8080/api/v1/admin/clients \
  -H "X-API-Key: $ADMIN_BOOTSTRAP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"kirdi_chiqdi","contact":"admin@example.uz"}'
# -> {"client":{"id":"<CLIENT_ID>", ...}}

# 2. Unga kalit berish
curl -X POST http://localhost:8080/api/v1/admin/keys \
  -H "X-API-Key: $ADMIN_BOOTSTRAP_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "client_id": "<CLIENT_ID>",
        "label": "kirdi_chiqdi prod",
        "scopes": ["face:verify","face:register","face:read","face:delete"],
        "rate_limit_per_min": 600
      }'
# -> {"secret":"fdsk_a1b2c3d4e5f6_<sir>", ...}
#    ★ Bu qiymat BOSHQA KO'RSATILMAYDI. Darhol .env ga ko'chiring.

# 3. .env: FACE_API_KEY=fdsk_... , so'ng ADMIN_BOOTSTRAP_KEY ni o'chiring
docker compose up -d --force-recreate
```

### Boshqa platformani ulash

Aynan shu uch qadam, faqat `name` va `scopes` boshqacha:

```bash
curl -X POST .../admin/clients -d '{"name":"kadrlar_tizimi"}'
curl -X POST .../admin/keys    -d '{"client_id":"...","scopes":["face:verify"]}'
```

Barcha platformalar **bitta umumiy yuz bazasidan** foydalanadi va bitta odam
uchun bitta `person_id` oladi — shuning uchun bir xil odam ikki xil platformada
ikki marta ro'yxatga olinmaydi.

### Boshqaruv endpointlari

| Metod | Yo'l | Vazifa |
|---|---|---|
| `POST` | `/admin/clients` | Yangi platforma qo'shish |
| `GET` | `/admin/clients` | Ro'yxat |
| `PUT` | `/admin/clients/{id}` | Yangilash / faolsizlantirish |
| `DELETE` | `/admin/clients/{id}` | O'chirish (kalitlari bilan) |
| `POST` | `/admin/keys` | Kalit berish |
| `GET` | `/admin/keys?client_id=` | Kalitlar (ochiq matnsiz) |
| `POST` | `/admin/keys/{id}/rotate` | Almashtirish (eskisi darhol o'ladi) |
| `DELETE` | `/admin/keys/{id}` | Bekor qilish |
| `GET` | `/admin/audit?limit=100` | So'rovlar jurnali |
| `GET` | `/admin/scopes` | Mavjud scope'lar |

Bekor qilish va rotatsiya **darhol** kuchga kiradi: kalit keshdan ham
chiqariladi.

### Chastota cheklovi

Har bir kalit uchun `rate_limit_per_min`. Oshib ketsa `429 rate_limited` va
`Retry-After` sarlavhasi qaytadi. Ko'rsatilmasa `DEFAULT_RATE_LIMIT_PER_MIN`
qo'llanadi.

---

## 4. API hujjati

Barcha javoblar JSON. Xato javoblarida barqaror `code` maydoni bor —
**mijoz kod bo'yicha ishlashi kerak, matn bo'yicha emas.**

Interaktiv hujjat: `http://localhost:8080/swagger/index.html`

### Identifikatsiya modeli — `subject_id` va `person_id`

Servis **shaxsiy ma'lumot saqlamaydi**. Siz `subject_id` yuborasiz (masalan
PNFL), servis uni maxfiy kalit bilan HMAC-SHA256 qiladi va faqat natijani —
`person_id` ni — saqlaydi:

```
person_id = HMAC-SHA256(FACE_ID_PEPPER, normalize(subject_id))   → 64 belgi hex
```

| Xossa | Natija |
|---|---|
| Deterministik | Bitta odam = butun tizimda bitta `person_id` |
| Qaytarilmas | Bazani o'g'irlagan odam PNFL larni tiklay olmaydi |
| Kalitli | `pepper` siz mos qiymatni hisoblab bo'lmaydi (oddiy SHA-256 dan farqi) |

> **Nega oddiy SHA-256 emas?** PNFL — atigi 14 raqam va tug'ilgan sana bilan
> strukturalangan. Uning butun fazosini oddiy noutbukda bir necha soniyada
> hisoblab chiqish mumkin, ya'ni tuzsiz hash hech narsani yashirmaydi.
>
> **Nega bcrypt/argon2 emas?** Ular har safar tasodifiy tuz qo'shadi — bir xil
> PNFL har safar boshqa natija beradi va uni qidiruv kaliti sifatida ishlatib
> bo'lmaydi. HMAC esa deterministik va tez.
>
> **`FACE_ID_PEPPER` ni o'zgartirmang.** O'zgartirilsa barcha `person_id` lar
> boshqa bo'ladi va mavjud yozuvlar "yo'qolgandek" ko'rinadi.

### `POST /api/v1/persons/register`

Scope: `face:register`

| Maydon | Majburiy | Izoh |
|---|---|---|
| `image` | ✅ | Rasm fayli (`photo` / `file` nomi ham qabul qilinadi) |
| `subject_id` | ❌ | **Tavsiya etiladi.** PNFL yoki tabel raqami — servis uni hashlaydi |
| `person_id` | ❌ | Tayyor `person_id` (`subject_id` bilan birga yuborilmaydi) |
| `name` | ❌ | Ixtiyoriy yorliq. Shaxsiy ma'lumot yubormaslik afzal |
| `replace` | ❌ | `true` — mavjud yozuvning yuzini almashtirish. Standart `false`: yozuv bor bo'lsa `409 already_exists` |

Ikkalasi ham berilmasa servis tasodifiy ID beradi (anonim yozuv).

```bash
curl -X POST http://localhost:8080/api/v1/persons/register \
  -F "name=Ali Valiyev" \
  -F "image=@ali.jpg"
```

**201 Created**
```json
{
  "success": true,
  "message": "Inson muvaffaqiyatli ro'yxatdan o'tkazildi.",
  "person_id": "3f2a8c14-7b91-4d6e-a052-1c8f9e4b7d30",
  "generated_id": true,
  "name": "Ali Valiyev",
  "point_id": "0a4c1f7e-...-9b2d",
  "updated": false,
  "det_score": 0.8912,
  "bbox": [220, 118, 452, 430],
  "vector_dim": 512
}
```

`generated_id: true` — ID'ni server bergan, **mijoz uni saqlab qo'yishi kerak**
(keyinchalik `GET`/`DELETE` va yuzni yangilash uchun kerak bo'ladi).

Ketma-ketlik: Golang → AI servis (`/v1/face/embed`) → 512-dim vektor →
Qdrant `upsert` (`person_id` + `name` payload bilan).

**Mavjud insonning yuzini yangilash** — o'sha `person_id` ni `replace=true`
bilan yuboring. `replace` siz so'rov `409 already_exists` qaytaradi: aks holda
PNFL'ni bilgan har qanday mijoz boshqa odamning yuzini o'zinikiga almashtirib,
uning nomidan yuz orqali kira olardi. `replace=true` ni faqat ishonchli,
audit qilinadigan administrator amali uchun yuboring.

```bash
curl -X POST http://localhost:8080/api/v1/persons/register \
  -F "person_id=3f2a8c14-7b91-4d6e-a052-1c8f9e4b7d30" \
  -F "name=Ali Valiyev" \
  -F "replace=true" \
  -F "image=@ali-yangi.jpg"
# -> 200 OK, "updated": true, "generated_id": false — dublikat yaratilmaydi
```

O'z tizimingizdagi ID'ni ishlatmoqchi bo'lsangiz (`emp-001`, tabel raqami va h.k.)
uni ham shu maydonda yuborsangiz bo'ladi — backend uni o'zgartirmasdan saqlaydi.

> **Dizayn qarori:** register'da liveness **majburiy emas**. Sabab — odamlar
> odatda passport skani yoki HR bazasidagi arxiv fotosidan ro'yxatga olinadi,
> ular esa ta'rifga ko'ra "spoof". Agar sizda ro'yxatga olish ham jonli kamera
> orqali bo'lsa, `handlers.go` dagi `h.ai.Embed(...)` ni `h.ai.VerifyFrame(...)`
> ga almashtiring — boshqa hech narsa o'zgartirish shart emas.

---

### `POST /api/v1/persons/resolve`

Scope: `face:read`. **Rasmsiz** identifikatsiya — yuz tanish ishlamagan
holatlar uchun zaxira yo'l (yorug'lik yomon, odam niqobda, kamera buzuq).

```bash
curl -X POST http://localhost:8080/api/v1/persons/resolve \
  -H "X-API-Key: $FACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"12345678901234"}'
```

```json
{ "success": true, "person_id": "9f2a...c7b1", "registered": true }
```

Operator PNFL kiritadi → mijoz shu yerdan `person_id` ni oladi → o'z bazasidan
yozuvni topadi. Ya'ni yuz — qulay yo'l, `subject_id` — kafolatli zaxira yo'l.

### `POST /api/v1/persons/verify` ⭐

Scope: `face:verify`

```bash
curl -X POST http://localhost:8080/api/v1/persons/verify \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "image=@frame.jpg"
```

Ketma-ketlik:

| Qadam | Bajaruvchi | Natija |
|---|---|---|
| **STEP 1** | Python — MiniFASNet ansambli | Soxta bo'lsa → `400` va **JARAYON TO'XTAYDI** |
| **STEP 2** | Python — InsightFace | 512-dim vektor (faqat tirik inson tasdiqlansa) |
| **STEP 3** | Golang — Qdrant | Cosine qidiruv, `score >= MATCH_THRESHOLD` |

**200 OK — topildi**
```json
{
  "success": true,
  "message": "Inson muvaffaqiyatli tanildi.",
  "person_id": "emp-001",
  "name": "Ali Valiyev",
  "score": 0.8134,
  "threshold": 0.65,
  "registered_at": "2026-07-25T09:14:02Z",
  "liveness": {
    "is_real": true,
    "checked": true,
    "real_probability": 0.9673,
    "spoof_probability": 0.0327,
    "threshold": 0.75,
    "per_model": {
      "2.7_80x80_MiniFASNetV2.pth": 0.9721,
      "4_0_0_80x80_MiniFASNetV1SE.pth": 0.9625
    }
  },
  "det_score": 0.9134,
  "bbox": [231, 102, 470, 441]
}
```

**400 Bad Request — SOXTA YUZ** (Qdrant'ga so'rov **yuborilmagan**)
```json
{
  "success": false,
  "code": "spoof_detected",
  "message": "Soxta yuz/rasm aniqlandi. Tirik inson kameraga qarashi shart!",
  "details": {
    "liveness": {
      "is_real": false,
      "real_probability": 0.0412,
      "spoof_probability": 0.9588,
      "threshold": 0.75,
      "per_model": {
        "2.7_80x80_MiniFASNetV2.pth": 0.0388,
        "4_0_0_80x80_MiniFASNetV1SE.pth": 0.0436
      }
    }
  }
}
```

**404 Not Found — tirik, lekin bazada yo'q**
```json
{
  "success": false,
  "code": "person_not_found",
  "message": "Bazadan inson topilmadi.",
  "threshold": 0.65,
  "best_score": 0.4127,
  "liveness": { "is_real": true, "checked": true, "real_probability": 0.9512, "...": "..." }
}
```

---

### `GET /api/v1/stream?ticket=...` — real-time oqim (WebSocket) ⭐

Brauzer kadrlarni uzluksiz yuboradi, server hodisalarni qaytaradi. Butun
avtomatik mantiq **serverda**: mijoz faqat kadr yuboradi.

#### Avval ticket oling

Brauzerdagi `new WebSocket(...)` maxsus sarlavha yubora olmaydi — ya'ni
`X-API-Key` ni WS ulanishida berish imkonsiz. Kalitni URL query'ga qo'yish esa
uni brauzer tarixiga, proksi loglariga va `Referer` sarlavhasiga chiqarib
yuboradi.

Shuning uchun **server tarafingiz** (kalit bilan) bir martalik ticket oladi va
faqat shuni brauzerga beradi:

```bash
curl -X POST http://face-backend:8080/api/v1/stream/tickets \
  -H "X-API-Key: $FACE_API_KEY"
```

```json
{ "ticket": "kZ9x...", "expires_in": 60, "stream_path": "/api/v1/stream" }
```

```js
new WebSocket(`wss://.../api/v1/stream?ticket=${ticket}`);
```

Ticket **60 soniya** yashaydi va **bir marta** ishlatiladi — ushlab qolishdan
foyda yo'q. `kirdi_chiqdi` da buning uchun tayyor proksi bor:
`POST /api/v1/face/stream-ticket`.

```
brauzer --WebSocket(binary JPEG)--> Golang --gRPC bidi--> Python
        <---------JSON hodisa------        <-------------
```

**Mijoz → server:** binar JPEG kadrlar (matnli xabarlar e'tiborga olinmaydi).
Server sekundiga 4 tadan ortiq kadrni qabul qilmaydi (ortiqchasi tashlanadi).

**Server → mijoz:** JSON hodisalar.

| `type` | Qachon | Muhim maydonlar |
|---|---|---|
| `ready` | oqim ochilganda | — |
| `liveness` | tirik yuz aniqlandi | `liveness.real_probability`, `bbox` |
| `spoof` | soxta yuz | `liveness`, `bbox` |
| `recognized` | avtomatik tanish muvaffaqiyatli | `person_id`, `name`, `score` |
| `not_found` | tirik, lekin bazada yo'q | `details.best_score`, `details.threshold` |
| `error` | kadr xatosi (yuz yo'q, xira) | `code`, `message` |

Avtomatik mantiq (`internal/delivery/httpapi/stream_handler.go`):

```
har kadr  ──► gRPC STREAM_MODE_LIVENESS_ONLY   (tez: faqat anti-spoofing)
                     │
   ketma-ket 2 ta "tirik" ──┐
                            ▼
keyingi kadr ──► gRPC STREAM_MODE_FULL          (liveness + 512-dim vektor)
                            │
                            ▼
                    Qdrant qidiruv ──► recognized / not_found
                            │
                    3 soniya tanaffus
```

Muvaffaqiyatsiz urinishdan keyin 3 soniya tanaffus qo'yiladi — notanish odam
oldida cheksiz qidiruv yubormaslik uchun.

Python bilan sinash namunasi:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8080/api/v1/stream") as ws:
        await ws.send(open("frame.jpg", "rb").read())
        async for msg in ws:
            print(json.loads(msg))

asyncio.run(main())
```

---

### Ramka harakati tezligi

Ramka sekin/kechikib harakatlansa — sababi va yechimlari:

| Sabab | Yechim | Qayerda |
|---|---|---|
| Kadr yuborish qat'iy intervalda edi | **Pipelining**: mijoz keyingi kadrni javob kelishi bilan yuboradi, ya'ni tezlik AI servisning haqiqiy imkoniyatiga moslashadi | `web/app.js` → `scheduleNextFrame()` |
| Server sun'iy cheklab turardi | `minFrameInterval` 250 ms → **50 ms** (endi u faqat DoS himoyasi) | `stream_handler.go` |
| Ramka faqat javob kelganda sakrardi | **Eksponensial silliqlash** 60 Hz da (`SMOOTH_TAU_MS`) | `web/app.js` → `drawLoop()` |
| Ramka aylanma vaqt qadar orqada qolardi | **Tezlik bo'yicha bashorat**: oxirgi ikki o'lchovdan tezlik hisoblanib, ramka o'lchangan kechikish qadar oldinga suriladi | `web/app.js` → `aimBox()` |
| Har kadrda ortiqcha piksel | Oqim kadri 640 → **480 px**, sifat 0.8 → 0.7 | `web/app.js` |

**O'lchangan natijalar.** Ikki xil o'lchov, chunki ular ikki xil narsani ko'rsatadi.

*1. Sof transport tezligi* (soxta AI bilan — inference deyarli 0 ms):

```
ESKI  (400 ms interval + 250 ms server cheklovi):   2.0 hodisa/s,  401 ms oraliq
YANGI (pipelining + 50 ms cheklov):                16.3 hodisa/s,   61 ms oraliq
```

*2. Haqiqiy stack* (Docker, CPU, InsightFace buffalo_l + MiniFASNet, `DET_SIZE=640`):

```
haqiqiy yuz:  4.6 fps, 220 ms oraliq, AI inference mediana 156 ms
soxta (ekran): 4.3 fps, 230 ms oraliq, AI inference mediana 169 ms
```

Ya'ni **haqiqiy tizimda tor joy — AI inference (~156 ms)**, transport emas.
Pipelining eski 2.0 fps chegarasini olib tashladi va 4.6 fps ga chiqardi
(~2.3x), qolgan ravonlikni esa silliqlash va bashorat beradi.

Silliqlash + bashorat matematikasi (yuz 300 px/s tezlikda harakatlanganda,
ramkaning haqiqiy yuzdan o'rtacha chetlanishi):

| Holat | O'rtacha xato | Tebranish |
|---|---|---|
| Eski (silliqlashsiz, 400 ms) | 132 px | 0 |
| Faqat silliqlash | 94 px | 0 |
| **Silliqlash + bashorat** | **28 px** | 0 |
| Sekin AI (300 ms), bashoratsiz | 161 px | 0 |
| Sekin AI (300 ms), bashorat bilan | 95 px | 0 |

> Silliqlash va bashorat 60 Hz da ishlaydi va serverga umuman yuk qo'shmaydi —
> AI qanchalik sekin bo'lsa, ularning foydasi shunchalik katta.
>
> Inference'ning o'zini tezlashtirish uchun eng samarali ikki chora:
> `DET_SIZE=320` (detektsiya sezilarli tezlashadi) va GPU rejimi (6-bo'limga qarang).

Status satrida jonli ko'rsatkich chiqadi: `6.4 fps · 156 ms` — bu haqiqiy
o'lchov, sozlash paytida foydali.

---

### Qo'shimcha endpointlar

| Metod | Yo'l | Tavsif |
|---|---|---|
| `POST` | `/api/v1/liveness/check` | **Probe**: faqat tiriklik. Soxta kadrda ham `200` qaytadi, `liveness.is_real=false` bilan — real-time ko'rsatkich uchun |
| `GET` | `/health` | Backend + AI servis + Qdrant holati |
| `GET` | `/api/v1/persons?limit=50&offset=` | Ro'yxatdan o'tganlar (sahifalab) |
| `GET` | `/api/v1/persons/:person_id` | Bitta insonning metadatasi |
| `DELETE` | `/api/v1/persons/:person_id` | Bazadan o'chirish |

### Python AI servisi — gRPC (`:50051`)

Golang backend AYNAN shu interfeys orqali ulanadi. `grpcurl` bilan qo'lda sinash
(reflection yoqilgan, `.proto` fayli kerak emas):

```bash
grpcurl -plaintext localhost:50051 list face.v1.FaceService
grpcurl -plaintext localhost:50051 face.v1.FaceService/Health
```

### Python AI servisi — HTTP (`:8000`)

gRPC bilan bir xil use-case'larni chaqiradi; hujjatlar va qo'lda tekshirish uchun.

| Metod | Yo'l | Tavsif |
|---|---|---|
| `GET` | `/health` | Model holati |
| `GET` | `/docs` | Interaktiv Swagger UI |
| `POST` | `/v1/face/embed` | Faqat embedding (register uchun) |
| `POST` | `/v1/face/verify-frame` | Liveness → embedding |
| `POST` | `/v1/liveness/check` | Faqat anti-spoofing (debug/tuning) |

Chegarani sozlash uchun foydali:

```bash
# Haqiqiy yuz bilan
curl -s -X POST http://localhost:8000/v1/liveness/check -F "image=@real.jpg" | python3 -m json.tool
# Telefon ekranidagi surat bilan
curl -s -X POST http://localhost:8000/v1/liveness/check -F "image=@screen.jpg" | python3 -m json.tool
```

### Barcha xato kodlari

| `code` | HTTP | Sabab |
|---|---|---|
| `spoof_detected` | 400 | Soxta yuz — **qidiruv bajarilmadi** |
| `invalid_request` | 400 | Maydon yetishmayapti yoki fayl rasm emas |
| `no_face_detected` | 422 | Kadrda yuz yo'q |
| `multiple_faces` | 422 | Kadrda bir nechta yuz |
| `low_quality_face` | 422 | Yuz juda kichik yoki rasm xira |
| `person_not_found` | 404 | Tirik inson, lekin bazada mos keladigani yo'q |
| `ai_service_unavailable` | 503 | AI servis javob bermayapti / model yuklanmagan |
| `vector_db_error` | 503 | Qdrant xatosi |
| `internal_error` | 500 | Kutilmagan xato |

---

## 5. Qdrant sozlamalari

Kollektsiya backend ishga tushganda **avtomatik** yaratiladi
(`qdrant/client.go` → `EnsureCollection`):

```json
{
  "vectors": { "size": 512, "distance": "Cosine" },
  "hnsw_config": { "m": 16, "ef_construct": 128 },
  "optimizers_config": { "default_segment_number": 2 },
  "on_disk_payload": true
}
```

Muhim jihatlar:

- **`size: 512`** — InsightFace `buffalo_l` recognition modeli (ArcFace w600k_r50)
  aynan 512 o'lchamli vektor beradi. Boshqa modelga o'tsangiz `VECTOR_SIZE` ni
  ham o'zgartiring; mos kelmasa xizmat ishga tushmaydi va aniq xato beradi.
- **`distance: Cosine`** — `normed_embedding` ishlatilgani uchun (L2 = 1) cosine
  similarity to'g'ridan-to'g'ri o'xshashlik darajasini beradi: `1.0` = bir xil.
- **Point ID = UUIDv5(person_id)** — deterministik. Shu tufayli bitta odamni
  qayta ro'yxatga olish dublikat yaratmaydi, mavjud vektorni almashtiradi.
  Kod: `qdrant.PointIDFor()`, testi: `qdrant/client_test.go`.
- **`person_id` payload indeksi** — keyword tipida, filtrli qidiruvlar uchun.

Qdrant Web UI: **http://localhost:6333/dashboard**

### `MATCH_THRESHOLD` ni qanday tanlash

`buffalo_l` uchun amaliy oraliqlar:

| Qiymat | Xulq-atvor |
|---|---|
| `0.50` | Juda yumshoq — begonalar ham "tanilishi" mumkin |
| `0.65` | **Muvozanatli (standart)** — ko'pchilik holatlar uchun |
| `0.75` | Qat'iy — yorug'lik/burchak o'zgarsa o'zini ham tanimasligi mumkin |
| `0.80+` | Faqat deyarli bir xil sharoitdagi kadrlar uchun |

Aniq qiymatni **o'z ma'lumotlaringizda** o'lchang: 20–30 kishini ro'yxatga oling,
har biridan turli sharoitda 3–5 ta test kadr oling, `verify` javobidagi
`score` va `best_score` qiymatlarini yig'ing.

---

## 6. GPU rejimi

Kod GPU/CPU'ga **avtomatik moslashadi** (`app/engine.py` → `resolve_device`):
`torch.cuda.is_available()` tekshiriladi, mavjud bo'lsa PyTorch `cuda:N` da,
InsightFace esa `CUDAExecutionProvider` bilan ishlaydi. Aks holda ikkalasi CPU'da.
`DEVICE=cpu` qo'ysangiz, GPU bor bo'lsa ham CPU ishlatiladi.

Yoqish uchun:

1. `docker-compose.yml` da `ai-service.build.dockerfile` ni `Dockerfile.gpu` ga o'zgartiring
2. O'sha servisdagi `deploy.resources.reservations.devices` blokini oching
3. `docker compose up -d --build`

Tekshirish: `curl -s localhost:8000/health | grep device` → `"device": "cuda:0"`

Taxminiy tezlik (bitta kadr, to'liq verify oqimi):

| Rejim | Vaqt |
|---|---|
| CPU (4 yadro) | ~250–450 ms |
| GPU (RTX 3060) | ~35–70 ms |

---

## 7. Production'ga chiqishdan oldin

Bu kod to'liq ishlaydi, lekin ochiq internetga chiqarishdan oldin quyidagilar zarur —
ular ataylab kiritilmagan, chunki har bir tashkilotda boshqacha hal qilinadi:

- [ ] **Autentifikatsiya.** Hozir API ochiq. Kamida `/register` va `/persons` ga
      API-key yoki JWT middleware qo'shing.
- [ ] **Rate limiting.** `/verify` — brute-force uchun jozibador nishon.
- [ ] **TLS.** Kamera oqimi va biometrik ma'lumot faqat HTTPS orqali.
- [ ] **`CORS_ALLOW_ORIGIN`** — `*` o'rniga `none` (server-server integratsiya)
      yoki aniq domen (brauzer to'g'ridan-to'g'ri ulansa).
- [ ] **`QDRANT_API_KEY`** — hozir bo'sh; production'da to'ldiring va
      6333/6334 portlarini tashqariga ochmang.
- [ ] **Audit log.** Kim, qachon, qaysi score bilan tanildi — alohida jurnalga.
- [ ] **Huquqiy jihat.** Yuz vektori — biometrik shaxsiy ma'lumot. Saqlash muddati,
      rozilik va o'chirish tartibini oldindan hal qiling (`DELETE /persons/:id` shu uchun bor).

---

## 8. Testlar

```bash
make test-go              # unit testlar — Docker ham, Qdrant ham KERAK EMAS (~5 ms)
make test-integration     # + haqiqiy Qdrant'ga qarshi integratsion testlar
```

Clean Architecture'ning amaliy foydasi shu yerda ko'rinadi: biznes qoidalari
portlar orqali ishlagani uchun ularni gRPC, Qdrant yoki Docker'siz test qilamiz.

**Use-case testlari** (`internal/usecase/`) — soxta (fake) adapterlar bilan:

| Test | Nimani isbotlaydi |
|---|---|
| `TestVerifyStopsAtSpoofAndNeverSearches` | **Soxta kadrda `FindNearest` 0 marta chaqiriladi.** Fake ombor chaqiruvlarni sanaydi |
| `TestMatchEmbeddingRejectsNonLiveFrame` | Streaming yo'lida ham ikkinchi himoya qatlami ishlaydi: tirik bo'lmagan kadr bilan chaqirilsa qidiruv bajarilmaydi |
| `TestVerifyReturnsPersonAboveThreshold` | Chegaradan o'tgan eng yaqin yozuv qaytariladi |
| `TestVerifyReturnsNotFoundBelowThreshold` | `best_score` va liveness 404 javobda ham beriladi |
| `TestThresholdIsInclusive` | `score == threshold` qabul qilinadi |
| `TestRegisterGeneratesIDWhenOmitted` | `person_id` berilmasa UUID generatsiya qilinadi |
| `TestRegisterHonoursClientIDAndUpdates` | O'z ID'si bilan qayta yozish dublikat yaratmaydi |
| `TestRegisterDoesNotRunLivenessCheck` | Register arxiv fotolari uchun liveness'siz ishlaydi |
| `TestLivenessProbeReturnsResultInsteadOfError` | Probe uchun soxta kadr xato emas |

**Adapter testlari:**

| Test | Nimani isbotlaydi |
|---|---|
| `TestTranslateErrorMapsFaceErrorDetails` | gRPC `FaceError` detallari domain kodlariga to'g'ri o'giriladi (7 ta holat) |
| `TestTranslateErrorCarriesLivenessScores` | Spoof xatosida liveness ballari Python'dan Go'ga buzilmasdan yetib boradi |
| `TestTranslateErrorWithoutDetailsIsInfrastructure` | Detalsiz gRPC xatosi transport muammosi sifatida qaraladi |
| `TestIntegrationFullLifecycle` | Haqiqiy Qdrant: 512/Cosine kollektsiya, save, search, dedup, list, delete, count |
| `TestIntegrationUnauthorizedIsDistinguishable` | 401 "hali tayyor emas" dan farqlanadi (qayta urinish befoyda) |
| `TestPointIDFor` / `TestUUIDGeneratorProducesValidV4` | UUIDv5 RFC 4122 ga mos (Python `uuid.uuid5` bilan solishtirilgan), UUIDv4 formati va takrorlanmasligi |

MiniFASNet arxitekturasi `.pth` checkpoint'lar bilan kalit-ma-kalit tekshirilgan:
V2 uchun 334/334, V1SE uchun 370/370 parametr nomi va shakli mos keladi.

### Anti-Spoofing modelini tekshirish

```bash
make verify-liveness
# yoki: docker compose exec ai-service python scripts/verify_liveness.py
```

Original repozitoriyning javobi ma'lum 3 ta namunasida modelni sinaydi:

```
✅ image_T1.jpg    kutilgan=REAL olindi=REAL real_prob=0.9977
✅ image_F1.jpg    kutilgan=FAKE olindi=FAKE real_prob=0.0281
✅ image_F2.jpg    kutilgan=FAKE olindi=FAKE real_prob=0.0192
```

Bu skript preprocessing buzilishini ushlaydi — quyidagi bo'limga qarang.

---

## 9. Muammolarni bartaraf etish

| Alomat | Sabab / Yechim |
|---|---|
| `liveness_loaded: false` | Vaznlar yo'q → `bash python-ai-service/scripts/download_models.sh`, so'ng `docker compose up -d --build ai-service` |
| Har bir so'rov `503 ai_service_unavailable` | Modellar hali yuklanmoqda (birinchi start 1–3 daqiqa). `docker compose logs -f ai-service` |
| **Haqiqiy odam DOIM `spoof_detected`, `real_probability` ~0.015 va o'zgarmaydi** | Preprocessing buzilgan. `make verify-liveness` ni ishga tushiring. Sabab: MiniFASNet **0..255** oralig'idagi kirishga o'qitilgan; agar `app/liveness/predictor.py` → `_to_tensor()` da `/255` bo'lsa, BatchNorm to'yinadi va model kirishdan qat'i nazar doim bir xil natija beradi |
| Haqiqiy odam ba'zan `spoof_detected` bo'lyapti | Yorug'lik yetarli emas yoki yuz kichik. `LIVENESS_THRESHOLD` ni `0.65` ga tushiring yoki `LIVENESS_REQUIRE_UNANIMOUS=false` |
| Telefon ekrani `spoof` deb topilmayapti | `LIVENESS_THRESHOLD` ni `0.85` ga ko'taring, `MIN_FACE_SIZE` ni `100` ga oshiring |
| `422 low_quality_face` tez-tez chiqadi | `MIN_BLUR_SCORE` ni pasaytiring (masalan `25`) yoki kamera fokusini tekshiring |
| `verify` doim `404` beradi | `MATCH_THRESHOLD` juda yuqori, yoki register va verify kadrlari juda farqli |
| Brauzerda kamera ochilmayapti | `localhost` yoki `https://` dan foydalaning |
| `qdrant xatosi (HTTP 401): Must provide an API key` va 8080 port ochilmaydi | `.env` da `QDRANT__SERVICE__API_KEY=` **bo'sh qiymat** bilan turibdi. Qdrant bunda auth'ni yoqadi, lekin hech qanday kalit mos kelmaydi. O'sha qatorni butunlay izohga oling yoki o'chiring, so'ng `docker compose up -d --force-recreate` |
| `ai_service.ok=false`, xato `connection refused` | Backend gRPC portiga ulanmoqda. `AI_SERVICE_TARGET` **sxemasiz** bo'lishi kerak (`ai-service:50051`, `http://` emas). Python logida `gRPC server tinglamoqda` qatorini qidiring |
| `/api/v1/stream` 400 yoki darhol yopiladi | Reverse proxy WebSocket'ni o'tkazmayapti. Nginx uchun: `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";` |
| Oqim ~1 daqiqadan keyin uziladi | Proxy idle timeout. Backend har 25 s da ping yuboradi; proxy timeout'ini undan katta qiling |
| AI servis start'da yiqiladi: `Detected incompatible Protobuf Gencode/Runtime versions` | Generatsiya qilingan kod noto'g'ri `grpcio-tools` versiyasi bilan yaratilgan. `pip install -r python-ai-service/requirements-dev.txt && make proto`, so'ng `docker compose build ai-service` |
| `protoc-gen-go: program not found` | `make proto` plaginlarni o'zi o'rnatadi, lekin `$(go env GOPATH)/bin` PATH'da bo'lishi kerak |
| Backend qayta-qayta restart bo'ladi, `curl localhost:8080` → connection refused | Backend Qdrant'ga ulanmaguncha portni **ochmaydi** (kollektsiya kafolatlanishi kerak). Sabab har doim yuqoridagi `qdrant kutilmoqda` loglarida. `docker compose logs backend \| tail -20` |
| `mavjud 'faces' kollektsiyasi N o'lchamli` | Eski kollektsiya boshqa o'lchamda. `make clean` (ma'lumot o'chadi!) yoki `QDRANT_COLLECTION` ga yangi nom bering |

Foydali buyruqlar:

```bash
make health          # to'liq holat
make logs            # barcha loglar
make persons         # ro'yxatdan o'tganlar
make verify IMG=test.jpg
docker compose logs -f ai-service   # model yuklanishini kuzatish
```

---

## 10. Litsenziya va manbalar

- **InsightFace** — [deepinsight/insightface](https://github.com/deepinsight/insightface),
  MIT. `buffalo_l` modeli **tadqiqot maqsadlari uchun** taqdim etilgan —
  tijorat loyihasida ishlatishdan oldin model kartasidagi shartlarni tekshiring.
- **Silent-Face-Anti-Spoofing (MiniFASNet)** —
  [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), MIT.
  `app/liveness/model_lib/mini_fasnet.py` shu repodagi arxitekturaning qayta
  yozilgan versiyasi; vaznlar o'zgartirilmagan.
- **Qdrant** — Apache 2.0.
