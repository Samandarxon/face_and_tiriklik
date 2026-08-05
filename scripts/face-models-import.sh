#!/usr/bin/env bash
#
# InsightFace vaznlarini arxivdan Docker volume'iga tiklaydi.
#
# Bu — face-models-export.sh ning teskarisi, va u internetsiz o'rnatishning
# oxirgi bo'lagi: shu skript bajarilgach tizimda internetdan yuklanadigan
# hech narsa qolmaydi.
#
# ISHLATISH
#
#   ./scripts/face-models-import.sh face-models.tar.gz
#   FACE_MODELS_VOLUME=my_volume ./scripts/face-models-import.sh arxiv.tar.gz
#
# Skript `docker compose up` DAN OLDIN bajarilishi kerak: servis ishga
# tushganda modelni joyida topsa, uni yuklab olishga urinmaydi.

set -euo pipefail

ARCHIVE="${1:-face-models.tar.gz}"
# Ildizdagi docker-compose.yml yaratadigan nom. Boshqa compose loyihasi uchun
# FACE_MODELS_VOLUME orqali almashtiriladi.
VOLUME="${FACE_MODELS_VOLUME:-kirdi-chiqdi_insightface_models}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "XATO: arxiv topilmadi: $ARCHIVE" >&2
  exit 1
fi

# Volume avvaldan bo'lmasa yaratamiz — `docker compose up` ni kutish shart
# emas, aks holda servis modelsiz ko'tarilib, yuklashga urinib qolardi.
docker volume create "$VOLUME" >/dev/null

# Ustiga yozishdan oldin ogohlantiramiz: volume'da allaqachon boshqa (masalan
# yangiroq) model turgan bo'lishi mumkin.
if docker run --rm -v "$VOLUME:/m" alpine test -d /m/models/buffalo_l 2>/dev/null; then
  echo "DIQQAT: $VOLUME da model allaqachon bor, ustiga yoziladi."
  read -r -p "Davom etilsinmi? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 0
fi

ARCHIVE_DIR="$(cd "$(dirname "$ARCHIVE")" && pwd)"
ARCHIVE_NAME="$(basename "$ARCHIVE")"

docker run --rm -v "$VOLUME:/m" -v "$ARCHIVE_DIR:/in:ro" alpine \
  tar xzf "/in/$ARCHIVE_NAME" -C /m

echo "✓ modellar tiklandi: $VOLUME"
docker run --rm -v "$VOLUME:/m" alpine sh -c 'du -sh /m; ls /m/models'
