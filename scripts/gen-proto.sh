#!/usr/bin/env bash
# .proto dan Go va Python kodini generatsiya qiladi.
#
#   bash scripts/gen-proto.sh
#
# Generatsiya qilingan kod repozitoriyga KOMMIT QILINADI — shunda build uchun
# protoc kerak bo'lmaydi va Docker image'lar yengil qoladi. Protoni
# o'zgartirgandan keyin shu skriptni qayta ishga tushiring.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT/proto"
GO_OUT="$ROOT/golang-backend/genproto"
PY_OUT="$ROOT/python-ai-service/genproto"

command -v protoc >/dev/null || {
  echo "XATO: protoc topilmadi. O'rnatish:" >&2
  echo "  Ubuntu/Debian: sudo apt install -y protobuf-compiler" >&2
  echo "  macOS:         brew install protobuf" >&2
  exit 1
}

# --- Go plaginlari ---
export PATH="$PATH:$(go env GOPATH)/bin"
if ! command -v protoc-gen-go >/dev/null || ! command -v protoc-gen-go-grpc >/dev/null; then
  echo "[go  ] plaginlar o'rnatilmoqda..."
  go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

echo "[go  ] generatsiya -> $GO_OUT"
rm -rf "$GO_OUT"
mkdir -p "$GO_OUT"
protoc -I "$PROTO_DIR" \
  --go_out="$GO_OUT" --go_opt=paths=source_relative \
  --go-grpc_out="$GO_OUT" --go-grpc_opt=paths=source_relative \
  "$PROTO_DIR"/face/v1/*.proto

# --- Python ---
echo "[py  ] generatsiya -> $PY_OUT"
PY_BIN="${PYTHON:-python3}"
DEV_REQ="$ROOT/python-ai-service/requirements-dev.txt"
WANT_TOOLS="$(grep -oP '(?<=^grpcio-tools==)[\d.]+' "$DEV_REQ")"

if ! "$PY_BIN" -c "import grpc_tools" 2>/dev/null; then
  echo "XATO: grpcio-tools topilmadi. O'rnatish:" >&2
  echo "  $PY_BIN -m pip install -r $DEV_REQ" >&2
  exit 1
fi

# grpcio-tools generatsiya qilgan kod ichiga o'zining protobuf gencode
# versiyasini yozadi va runtime uni tekshiradi. Gencode runtime'dan YANGI
# bo'lsa AI servis ishga tushmaydi ("Detected incompatible Protobuf
# Gencode/Runtime versions"). Shuning uchun versiyani qat'iy tekshiramiz.
HAVE_TOOLS="$("$PY_BIN" -c 'import importlib.metadata as m; print(m.version("grpcio-tools"))' 2>/dev/null || echo "?")"
if [[ "$HAVE_TOOLS" != "$WANT_TOOLS" ]]; then
  cat >&2 <<MSG
XATO: grpcio-tools versiyasi mos emas.
  o'rnatilgan: $HAVE_TOOLS
  kerak:       $WANT_TOOLS   (requirements-dev.txt dan)

Boshqa versiya bilan generatsiya qilingan kod runtime protobuf bilan mos
kelmaydi va servis ishga tushmaydi. Tuzatish:

  $PY_BIN -m pip install -r $DEV_REQ
MSG
  exit 1
fi

rm -rf "$PY_OUT"
mkdir -p "$PY_OUT"
"$PY_BIN" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PY_OUT" \
  --pyi_out="$PY_OUT" \
  --grpc_python_out="$PY_OUT" \
  "$PROTO_DIR"/face/v1/*.proto

# grpc_tools paket `__init__.py` yaratmaydi — import ishlashi uchun qo'shamiz.
find "$PY_OUT" -type d -exec touch {}/__init__.py \;

echo
echo "Tayyor:"
echo "  Go:     $(find "$GO_OUT" -name '*.go' | wc -l) fayl"
echo "  Python: $(find "$PY_OUT" -name '*.py' | wc -l) fayl"
