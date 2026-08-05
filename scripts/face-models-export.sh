#!/usr/bin/env bash
#
# InsightFace vaznlarini Docker volume'idan arxivga chiqaradi.
#
# NEGA KERAK
#
#   Butun tizimda internetdan yuklanadigan YAGONA narsa shu qoldi: InsightFace
#   `buffalo_l` to'plamini birinchi ishga tushganda GitHub'dan o'zi yuklab
#   oladi (~300 MB). Bir marta yuklab olgach volume'da qoladi va servis
#   internetsiz ishlayveradi — lekin YANGI mashinaga o'rnatishda o'sha birinchi
#   yuklash yana kerak bo'ladi.
#
#   Ichki tarmoqdagi server esa internetni umuman ko'rmasligi mumkin. Shuning
#   uchun modellar internetdan emas, shu arxivdan ko'chiriladi.
#
#   Modellar repoga qo'shilmadi ataylab: 300 MB lik binar fayl versiya
#   nazoratida — har bir klon uchun 300 MB, va uni keyin olib tashlab
#   bo'lmaydi. Arxiv esa oddiy fayl: flesh xotira, ichki fayl serveri yoki
#   relizga biriktirilgan asset sifatida tarqatiladi.
#
# ISHLATISH
#
#   ./scripts/face-models-export.sh                 # face-models.tar.gz yaratadi
#   ./scripts/face-models-export.sh /media/usb/     # boshqa joyga
#
# Yangi mashinada: face-models-import.sh

set -euo pipefail

OUT_DIR="${1:-$(pwd)}"
ARCHIVE="$OUT_DIR/face-models.tar.gz"

# Volume nomi compose loyihasiga bog'liq: `kirdi_chiqdi` ildizidagi
# docker-compose.yml uni `kirdi-chiqdi_insightface_models` deb ataydi,
# face servisning o'z compose fayli esa `face-auth_insightface_models` deb.
# Ikkalasini ham qidiramiz va to'lgan bo'lganini olamiz.
CANDIDATES=(
  "${FACE_MODELS_VOLUME:-}"
  "kirdi-chiqdi_insightface_models"
  "face-auth_insightface_models"
  "insightface_models"
)

VOLUME=""
for v in "${CANDIDATES[@]}"; do
  [[ -z "$v" ]] && continue
  if ! docker volume inspect "$v" >/dev/null 2>&1; then
    continue
  fi
  # Bo'sh volume ham mavjud bo'ladi — ichida model bor-yo'qligini tekshiramiz.
  if docker run --rm -v "$v:/m" alpine test -d /m/models/buffalo_l 2>/dev/null; then
    VOLUME="$v"
    break
  fi
done

if [[ -z "$VOLUME" ]]; then
  echo "XATO: buffalo_l modeli topilmadi." >&2
  echo "Avval face-ai servisini bir marta internetli mashinada ishga tushiring," >&2
  echo "u modelni yuklab oladi, keyin shu skriptni qayta bajaring." >&2
  exit 1
fi

echo "→ volume: $VOLUME"
mkdir -p "$OUT_DIR"

# Arxiv volume ILDIZIDAN olinadi, shunda import qilishda yo'llar bir xil
# joyga tushadi va qo'lda ko'chirish kerak bo'lmaydi.
docker run --rm -v "$VOLUME:/m:ro" -v "$(cd "$OUT_DIR" && pwd):/out" alpine \
  tar czf /out/face-models.tar.gz -C /m .

echo "✓ $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "Yangi mashinada:"
echo "  ./scripts/face-models-import.sh $ARCHIVE"
