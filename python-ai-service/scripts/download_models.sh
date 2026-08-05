#!/usr/bin/env bash
# MiniFASNet anti-spoofing vaznlarini yuklab oladi (Silent-Face-Anti-Spoofing, MIT).
#
# Ishlatish:
#   bash python-ai-service/scripts/download_models.sh
#
# Natija:
#   python-ai-service/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
#   python-ai-service/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-$SCRIPT_DIR/../resources/anti_spoof_models}"
REPO_RAW="https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models"

MODELS=(
  "2.7_80x80_MiniFASNetV2.pth"
  "4_0_0_80x80_MiniFASNetV1SE.pth"
)

mkdir -p "$TARGET_DIR"

for model in "${MODELS[@]}"; do
  dest="$TARGET_DIR/$model"
  if [[ -s "$dest" ]]; then
    echo "[skip] $model allaqachon mavjud"
    continue
  fi
  echo "[get ] $model ..."
  curl -fL --retry 3 --retry-delay 2 -o "$dest.tmp" "$REPO_RAW/$model"
  # Yuklangan fayl HTML xato sahifasi emasligiga ishonch hosil qilamiz.
  if [[ $(stat -c%s "$dest.tmp" 2>/dev/null || stat -f%z "$dest.tmp") -lt 100000 ]]; then
    rm -f "$dest.tmp"
    echo "XATO: $model to'liq yuklanmadi. Tarmoqni tekshiring." >&2
    exit 1
  fi
  mv "$dest.tmp" "$dest"
  echo "[ok  ] $model"
done

echo
echo "Modellar joyi: $TARGET_DIR"
ls -lh "$TARGET_DIR"
