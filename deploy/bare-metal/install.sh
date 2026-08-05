#!/usr/bin/env bash
# =============================================================================
#  Face servisni DOCKER'SIZ, to'g'ridan-to'g'ri serverga o'rnatadi (systemd).
#
#  Ishlatish (root yoki sudo bilan):
#      sudo bash deploy/bare-metal/install.sh              # hamma qadam
#      sudo bash deploy/bare-metal/install.sh python       # faqat bitta qadam
#      sudo bash deploy/bare-metal/install.sh --help
#
#  Skript IDEMPOTENT: qayta ishga tushirsangiz mavjud narsalarni buzmaydi.
#  Mavjud sozlama fayllari va parollar qayta yozilmaydi.
#
#  Qo'llab-quvvatlanadi: Ubuntu 22.04 / 24.04, Debian 12 (amd64, arm64).
# =============================================================================
set -euo pipefail

# --- Sozlanadigan yo'llar ----------------------------------------------------
FACE_USER="${FACE_USER:-face}"
APP_ROOT="${APP_ROOT:-/opt/face}"          # venv, bin, models
APP_DIR="$APP_ROOT/app"                     # manba kod nusxasi
QDRANT_DIR="${QDRANT_DIR:-/opt/qdrant}"
QDRANT_DATA="${QDRANT_DATA:-/var/lib/qdrant}"
ENV_DIR="${ENV_DIR:-/etc/face}"
LOG_DIR="${LOG_DIR:-/var/log/face}"

QDRANT_VERSION="${QDRANT_VERSION:-v1.12.4}"
GO_VERSION="${GO_VERSION:-1.25.7}"

# Manba kod — shu skriptdan ikki qavat yuqorida.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Chiroyli chiqish --------------------------------------------------------
c_ok=$'\033[32m'; c_info=$'\033[36m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_off=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$c_info" "$1" "$c_off"; }
ok()   { printf '%s  ✓ %s%s\n' "$c_ok"   "$1" "$c_off"; }
warn() { printf '%s  ! %s%s\n' "$c_warn" "$1" "$c_off"; }
die()  { printf '%s  ✗ %s%s\n' "$c_err"  "$1" "$c_off" >&2; exit 1; }

need_root() { [[ $EUID -eq 0 ]] || die "root kerak: sudo bash $0 $*"; }

# =============================================================================
#  0 — Tekshiruv
# =============================================================================
do_preflight() {
  step "Server tekshirilmoqda"

  [[ -f /etc/os-release ]] || die "/etc/os-release topilmadi — qo'llab-quvvatlanmaydigan OS"
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID:-}${ID_LIKE:-}" in
    *debian*|*ubuntu*) ok "OS: $PRETTY_NAME" ;;
    # Diqqat: ${VAR:-...} ichida apostrof yozib bo'lmaydi — bash uni qo'shtirnoq
    # deb oladi va butun skript sintaksisi buziladi.
    *) die "Bu skript Debian/Ubuntu uchun. Sizda: ${PRETTY_NAME:-aniqlanmadi}" ;;
  esac

  local arch; arch="$(uname -m)"
  case "$arch" in
    x86_64|aarch64) ok "Arxitektura: $arch" ;;
    *) die "Qo'llab-quvvatlanmaydigan arxitektura: $arch" ;;
  esac

  local ram_mb; ram_mb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 ))
  if (( ram_mb < 3500 )); then
    warn "RAM ${ram_mb} MB — kam. Tavsiya: kamida 4 GB (torch + buffalo_l og'ir)."
  else
    ok "RAM: ${ram_mb} MB"
  fi

  local free_gb; free_gb=$(df -BG --output=avail /opt 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if (( free_gb < 6 )); then
    warn "/opt da ${free_gb} GB bo'sh joy — kam. Tavsiya: kamida 6 GB."
  else
    ok "Bo'sh joy: ${free_gb} GB"
  fi

  ok "Manba kod: $SRC_DIR"
}

# =============================================================================
#  1 — Tizim paketlari
# =============================================================================
do_deps() {
  need_root
  step "Tizim paketlari o'rnatilmoqda"

  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq

  # opencv libgl/libglib talab qiladi; insightface esa Cython bilan kompilyatsiya
  # qilinadi — shuning uchun build-essential va cmake kerak.
  local pkgs=(
    ca-certificates curl wget gnupg tar
    # Minimal Debian o'rnatmasida sudo bo'lmasligi mumkin — postgres qadami
    # `sudo -u postgres` ga tayanadi.
    sudo openssl
    build-essential cmake pkg-config
    libgl1 libglib2.0-0
    postgresql postgresql-client
  )
  apt-get install -y -qq "${pkgs[@]}" || die "apt-get install muvaffaqiyatsiz"
  ok "Asosiy paketlar o'rnatildi"

  _ensure_python
  _ensure_go
}

# numpy 1.26.4 / torch 2.4.1 / onnxruntime 1.19.2 uchun Python 3.11 yoki 3.12
# kerak. 3.13 da wheel'lar YO'Q — pip manbadan qurishga urinadi va yiqiladi.
_ensure_python() {
  local found=""
  for v in 3.11 3.12; do
    if command -v "python$v" >/dev/null 2>&1; then found="python$v"; break; fi
  done

  if [[ -z "$found" ]]; then
    warn "python3.11/3.12 topilmadi — deadsnakes PPA orqali 3.11 o'rnatiladi"
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 \
      || die "deadsnakes PPA qo'shilmadi. python3.11 ni qo'lda o'rnating."
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    found="python3.11"
  else
    apt-get install -y -qq "${found}-venv" "${found}-dev" 2>/dev/null || true
  fi

  echo "$found" > /tmp/.face_python
  ok "Python: $found ($("$found" --version 2>&1))"
}

_ensure_go() {
  local need="$GO_VERSION"
  if command -v go >/dev/null 2>&1; then
    local have; have="$(go version | awk '{print $3}' | sed 's/^go//')"
    # go.mod `go 1.25.0` talab qiladi.
    if [[ "$(printf '%s\n%s\n' "1.25.0" "$have" | sort -V | head -1)" == "1.25.0" ]]; then
      ok "Go: $have (mavjud, yetarli)"
      return
    fi
    warn "Go $have eski — $need o'rnatiladi"
  fi

  local arch; arch="$(uname -m)"
  local goarch="amd64"; [[ "$arch" == "aarch64" ]] && goarch="arm64"
  local tgz="/tmp/go${need}.linux-${goarch}.tar.gz"

  curl -fL --retry 3 -o "$tgz" "https://go.dev/dl/go${need}.linux-${goarch}.tar.gz" \
    || die "Go yuklab olinmadi"
  rm -rf /usr/local/go && tar -C /usr/local -xzf "$tgz" && rm -f "$tgz"
  ln -sf /usr/local/go/bin/go /usr/local/bin/go
  ln -sf /usr/local/go/bin/gofmt /usr/local/bin/gofmt
  ok "Go o'rnatildi: $(go version)"
}

# =============================================================================
#  2 — Foydalanuvchi va papkalar
# =============================================================================
do_user() {
  need_root
  step "Servis foydalanuvchisi va papkalar"

  if id "$FACE_USER" >/dev/null 2>&1; then
    ok "Foydalanuvchi '$FACE_USER' allaqachon bor"
  else
    useradd --system --create-home --home-dir "/home/$FACE_USER" \
            --shell /usr/sbin/nologin "$FACE_USER"
    ok "Foydalanuvchi '$FACE_USER' yaratildi (login qila olmaydi)"
  fi

  # .cache — face-ai.service da HOME va XDG_CACHE_HOME shu yerga yo'naltirilgan
  # (ProtectHome=true bilan /home ko'rinmaydi).
  install -d -o "$FACE_USER" -g "$FACE_USER" -m 755 \
      "$APP_ROOT" "$APP_ROOT/bin" "$APP_ROOT/models" "$APP_ROOT/.cache" "$APP_DIR" \
      "$QDRANT_DIR" "$QDRANT_DATA" "$QDRANT_DATA/storage" "$QDRANT_DATA/snapshots" \
      "$LOG_DIR"
  install -d -o root -g "$FACE_USER" -m 750 "$ENV_DIR"
  ok "Papkalar tayyor"
}

# =============================================================================
#  3 — Manba kodni nusxalash
# =============================================================================
do_code() {
  need_root
  step "Manba kod $APP_DIR ga ko'chirilmoqda"

  # node_modules/venv/.git kerak emas; genproto ESA KERAK (oldindan yaratilgan,
  # protoc o'rnatish shart emas).
  for sub in python-ai-service golang-backend proto; do
    [[ -d "$SRC_DIR/$sub" ]] || die "$SRC_DIR/$sub topilmadi"
    rm -rf "${APP_DIR:?}/$sub.new"
    cp -a "$SRC_DIR/$sub" "$APP_DIR/$sub.new"
    rm -rf "${APP_DIR:?}/$sub"
    mv "$APP_DIR/$sub.new" "$APP_DIR/$sub"
  done
  find "$APP_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  chown -R "$FACE_USER:$FACE_USER" "$APP_DIR"
  ok "Kod ko'chirildi"
}

# =============================================================================
#  4 — PostgreSQL (kalitlar/audit bazasi)
# =============================================================================
do_postgres() {
  need_root
  step "PostgreSQL sozlanmoqda"

  systemctl enable --now postgresql >/dev/null 2>&1 || true
  systemctl is-active --quiet postgresql || die "postgresql ishga tushmadi"

  local db_user="face" db_name="face_auth" db_pass
  db_pass="$(_read_env_value "$ENV_DIR/face-backend.env" POSTGRES_PASSWORD || true)"
  if [[ -z "${db_pass:-}" || "$db_pass" == "CHANGE-ME" ]]; then
    db_pass="$(openssl rand -base64 32 | tr -d '\n=+/' | cut -c1-32)"
    ok "Yangi baza paroli yasaldi"
  else
    ok "Mavjud baza paroli ishlatiladi"
  fi

  if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$db_user'" | grep -q 1; then
    sudo -u postgres psql -qc "ALTER ROLE $db_user WITH LOGIN PASSWORD '$db_pass';"
    ok "Rol '$db_user' yangilandi"
  else
    sudo -u postgres psql -qc "CREATE ROLE $db_user WITH LOGIN PASSWORD '$db_pass';"
    ok "Rol '$db_user' yaratildi"
  fi

  if sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$db_name'" | grep -q 1; then
    ok "Baza '$db_name' allaqachon bor"
  else
    sudo -u postgres createdb -O "$db_user" "$db_name"
    ok "Baza '$db_name' yaratildi"
  fi

  # Jadvallarni face-backend o'zi ishga tushganda yaratadi (embed migratsiyalar).
  echo "$db_pass" > /tmp/.face_db_pass
  chmod 600 /tmp/.face_db_pass
}

# =============================================================================
#  5 — Qdrant (vektor bazasi)
# =============================================================================
do_qdrant() {
  need_root
  step "Qdrant $QDRANT_VERSION o'rnatilmoqda"

  if [[ -x "$QDRANT_DIR/qdrant" ]]; then
    ok "Qdrant binari allaqachon bor: $QDRANT_DIR/qdrant"
  else
    local arch asset
    arch="$(uname -m)"
    if [[ "$arch" == "x86_64" ]]; then
      asset="qdrant-x86_64-unknown-linux-gnu.tar.gz"
    else
      asset="qdrant-aarch64-unknown-linux-musl.tar.gz"
    fi

    curl -fL --retry 3 -o /tmp/qdrant.tar.gz \
      "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${asset}" \
      || die "Qdrant yuklab olinmadi"
    tar -xzf /tmp/qdrant.tar.gz -C "$QDRANT_DIR"
    rm -f /tmp/qdrant.tar.gz
    chmod +x "$QDRANT_DIR/qdrant"
    ok "Qdrant binari o'rnatildi"
  fi

  # Qdrant ish papkasidagi config/config.yaml ni o'qiydi — u bo'lmasa
  # ishga tushmaydi. Aynan shu versiyanikini olamiz.
  if [[ -f "$QDRANT_DIR/config/config.yaml" ]]; then
    ok "config.yaml allaqachon bor"
  else
    install -d "$QDRANT_DIR/config"
    curl -fL --retry 3 -o "$QDRANT_DIR/config/config.yaml" \
      "https://raw.githubusercontent.com/qdrant/qdrant/${QDRANT_VERSION}/config/config.yaml" \
      || die "Qdrant config.yaml yuklab olinmadi"
    ok "config.yaml yuklandi"
  fi

  chown -R "$FACE_USER:$FACE_USER" "$QDRANT_DIR" "$QDRANT_DATA"
}

# =============================================================================
#  6 — Python AI servis
# =============================================================================
do_python() {
  need_root
  step "Python AI servis o'rnatilmoqda (uzoq davom etadi — torch ~900 MB)"

  local py; py="$(cat /tmp/.face_python 2>/dev/null || echo '')"
  if [[ -z "$py" ]] || ! command -v "$py" >/dev/null 2>&1; then
    for v in 3.11 3.12; do
      command -v "python$v" >/dev/null 2>&1 && { py="python$v"; break; }
    done
  fi
  [[ -n "$py" ]] || die "python3.11 yoki 3.12 topilmadi — avval: sudo bash $0 deps"

  if [[ ! -x "$APP_ROOT/venv/bin/python" ]]; then
    "$py" -m venv "$APP_ROOT/venv" || die "venv yaratilmadi"
    ok "venv yaratildi: $APP_ROOT/venv"
  else
    ok "venv allaqachon bor"
  fi

  local pip="$APP_ROOT/venv/bin/pip"
  "$pip" install --quiet --upgrade pip setuptools wheel
  # insightface Cython'siz qurilmaydi, numpy esa undan oldin turishi kerak.
  "$pip" install --quiet "numpy==1.26.4" "cython"
  "$pip" install -r "$APP_DIR/python-ai-service/requirements.txt" \
    || die "pip install muvaffaqiyatsiz — yuqoridagi xatoni o'qing"
  ok "Python bog'liqliklari o'rnatildi"

  step "Anti-spoofing vaznlari"
  bash "$APP_DIR/python-ai-service/scripts/download_models.sh" \
       "$APP_DIR/python-ai-service/resources/anti_spoof_models"

  step "InsightFace buffalo_l yuklab olinmoqda (~300 MB, bir marta)"
  install -d -o "$FACE_USER" -g "$FACE_USER" "$APP_ROOT/models/insightface"
  # `env` orqali: `sudo -u user VAR=val cmd` ba'zi sudo versiyalarida
  # o'zgaruvchini o'tkazmaydi, `env` esa hamma joyda bir xil ishlaydi.
  sudo -u "$FACE_USER" env "INSIGHTFACE_ROOT=$APP_ROOT/models/insightface" \
    "HOME=$APP_ROOT" "XDG_CACHE_HOME=$APP_ROOT/.cache" \
    "$APP_ROOT/venv/bin/python" -c "
from insightface.app import FaceAnalysis
a = FaceAnalysis(name='buffalo_l', root='$APP_ROOT/models/insightface',
                 allowed_modules=['detection','recognition'])
a.prepare(ctx_id=-1, det_size=(640,640))
print('buffalo_l tayyor')
" || die "buffalo_l yuklanmadi — tarmoqni tekshiring"

  chown -R "$FACE_USER:$FACE_USER" "$APP_ROOT/venv" "$APP_ROOT/models" "$APP_DIR"
  ok "AI servis tayyor"
}

# =============================================================================
#  7 — Golang backend
# =============================================================================
do_build() {
  need_root
  step "Golang backend qurilmoqda"

  command -v go >/dev/null 2>&1 || die "go topilmadi — avval: sudo bash $0 deps"

  # Go keshi root'niki bo'lsin, servis foydalanuvchisiga yozish huquqi kerak emas.
  ( cd "$APP_DIR/golang-backend" && \
    GOFLAGS=-mod=mod CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' \
      -o "$APP_ROOT/bin/face-backend" ./cmd/server ) \
    || die "go build muvaffaqiyatsiz"

  chown "$FACE_USER:$FACE_USER" "$APP_ROOT/bin/face-backend"
  chmod 755 "$APP_ROOT/bin/face-backend"
  ok "Binar tayyor: $APP_ROOT/bin/face-backend"
}

# =============================================================================
#  8 — Sozlama fayllari va sirlar
# =============================================================================
_read_env_value() {  # fayl, kalit -> qiymat (yo'q bo'lsa bo'sh)
  [[ -f "$1" ]] || return 1
  grep -E "^$2=" "$1" | head -1 | cut -d= -f2- || true
}

_set_env_value() {   # fayl, kalit, qiymat
  local f="$1" k="$2" v="$3"
  if grep -qE "^$k=" "$f"; then
    # Qiymatda `|` bo'lishi mumkin emas (base64 dan tozalaymiz), shuning uchun
    # ajratgich sifatida xavfsiz.
    sed -i "s|^$k=.*|$k=$v|" "$f"
  else
    printf '%s=%s\n' "$k" "$v" >> "$f"
  fi
}

do_env() {
  need_root
  step "Sozlama fayllari: $ENV_DIR"

  local here="$SRC_DIR/deploy/bare-metal/env"
  for name in qdrant ai-service face-backend; do
    if [[ -f "$ENV_DIR/$name.env" ]]; then
      ok "$name.env allaqachon bor — tegilmadi"
    else
      cp "$here/$name.env.example" "$ENV_DIR/$name.env"
      ok "$name.env yaratildi"
    fi
  done

  chown root:"$FACE_USER" "$ENV_DIR"/*.env
  chmod 640 "$ENV_DIR"/qdrant.env "$ENV_DIR"/ai-service.env
  chmod 640 "$ENV_DIR"/face-backend.env

  local f="$ENV_DIR/face-backend.env"

  # --- Baza paroli ---
  if [[ -f /tmp/.face_db_pass ]]; then
    _set_env_value "$f" POSTGRES_PASSWORD "$(cat /tmp/.face_db_pass)"
    rm -f /tmp/.face_db_pass
    ok "POSTGRES_PASSWORD yozildi"
  fi

  # FACE_ID_PEPPER endi ishlatilmaydi: `person_id` PNFL ning o'zi. Eski
  # o'rnatmalarda .env da qolib ketgan bo'lsa servis uni e'tiborsiz qoldiradi.

  # --- ADMIN_BOOTSTRAP_KEY ---
  local boot; boot="$(_read_env_value "$f" ADMIN_BOOTSTRAP_KEY || true)"
  if [[ -n "${boot:-}" ]]; then
    ok "ADMIN_BOOTSTRAP_KEY allaqachon bor"
  else
    _set_env_value "$f" ADMIN_BOOTSTRAP_KEY \
      "$(openssl rand -base64 48 | tr -d '\n=' | tr '+/' '-_')"
    ok "ADMIN_BOOTSTRAP_KEY yasaldi"
  fi
}

# =============================================================================
#  9 — systemd
# =============================================================================
do_services() {
  need_root
  step "systemd servislari o'rnatilmoqda"

  local here="$SRC_DIR/deploy/bare-metal/systemd"
  for unit in qdrant face-ai face-backend; do
    install -m 644 "$here/$unit.service" "/etc/systemd/system/$unit.service"
    ok "$unit.service o'rnatildi"
  done

  systemctl daemon-reload
  systemctl enable qdrant face-ai face-backend >/dev/null 2>&1
  ok "Avtoyuklashga qo'shildi"

  step "Servislar ishga tushirilmoqda"
  systemctl restart qdrant
  sleep 3
  systemctl restart face-ai
  warn "face-ai modellarni yuklaydi — 30-120 soniya kutilmoqda..."
  sleep 5
  systemctl restart face-backend
  sleep 3

  do_status
}

# =============================================================================
#  10 — Holat va tekshiruv
# =============================================================================
do_status() {
  step "Holat"
  for unit in postgresql qdrant face-ai face-backend; do
    if systemctl is-active --quiet "$unit"; then
      ok "$unit — ishlayapti"
    else
      printf '%s  ✗ %s — ISHLAMAYAPTI  (journalctl -u %s -n 50)%s\n' \
        "$c_err" "$unit" "$unit" "$c_off"
    fi
  done

  step "Endpointlar"
  _probe "Qdrant   " "http://127.0.0.1:6333/healthz"
  _probe "AI (HTTP)" "http://127.0.0.1:8000/health"
  _probe "Backend  " "http://127.0.0.1:8080/health"
}

_probe() {
  local label="$1" url="$2" code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" || echo 000)"
  if [[ "$code" == "200" ]]; then
    ok "$label $url -> 200"
  else
    warn "$label $url -> $code (hali ko'tarilmagan bo'lishi mumkin)"
  fi
}

# =============================================================================
#  11 — Birinchi API kalitni olish (bootstrap)
# =============================================================================
do_bootstrap() {
  need_root
  step "Birinchi API kalit olinmoqda"

  local f="$ENV_DIR/face-backend.env" boot
  boot="$(_read_env_value "$f" ADMIN_BOOTSTRAP_KEY || true)"
  [[ -n "${boot:-}" ]] || die "ADMIN_BOOTSTRAP_KEY bo'sh — kalit allaqachon olingan bo'lsa kerak"

  local base="http://127.0.0.1:$( _read_env_value "$f" PORT || echo 8080 )"

  local client_json client_id
  client_json="$(curl -s -X POST "$base/api/v1/admin/clients" \
      -H "X-API-Key: $boot" -H "Content-Type: application/json" \
      -d '{"name":"kirdi_chiqdi","contact":"admin@example.uz"}')" \
    || die "so'rov yuborilmadi — face-backend ishlayaptimi?"

  client_id="$(printf '%s' "$client_json" | python3 -c \
      'import json,sys; print(json.load(sys.stdin)["client"]["id"])' 2>/dev/null || true)"
  [[ -n "$client_id" ]] || die "Mijoz yaratilmadi. Javob: $client_json"
  ok "Mijoz yaratildi: $client_id"

  # `admin:keys` — VMS ning "Integratsiya" sahifasi uchun SHART. Busiz
  # /admin/* marshrutlari 403 qaytaradi va o'sha sahifa butunlay ishlamaydi:
  # kalitlar ro'yxati ham, rotatsiya ham, tiklash ham. Buni keyin qo'shib
  # bo'lmaydi — qo'shish uchun aynan o'sha sahifa kerak, u esa ishlamaydi.
  #
  # Ro'yxat kirdi_chiqdi backendidagi `recoveryScopes` bilan bir xil bo'lishi
  # kerak (internal/service/face_key_service.go) — u tiklashda aynan shu
  # ro'yxatni so'raydi.
  local key_json secret
  key_json="$(curl -s -X POST "$base/api/v1/admin/keys" \
      -H "X-API-Key: $boot" -H "Content-Type: application/json" \
      -d "{\"client_id\":\"$client_id\",\"label\":\"kirdi_chiqdi prod\",
           \"scopes\":[\"face:verify\",\"face:register\",\"face:read\",\"face:delete\",\"admin:keys\"],
           \"rate_limit_per_min\":600}")"

  secret="$(printf '%s' "$key_json" | python3 -c \
      'import json,sys; print(json.load(sys.stdin)["secret"])' 2>/dev/null || true)"
  [[ -n "$secret" ]] || die "Kalit yaratilmadi. Javob: $key_json"

  printf '\n%s══════════════════════════════════════════════════════════════%s\n' "$c_ok" "$c_off"
  printf '%s  FACE_API_KEY=%s%s\n' "$c_ok" "$secret" "$c_off"
  printf '%s══════════════════════════════════════════════════════════════%s\n' "$c_ok" "$c_off"
  printf '\n  ★ Bu qiymat BOSHQA KO\047RSATILMAYDI. Hoziroq nusxalang:\n'
  printf '    kirdi_chiqdi .env fayliga:  FACE_API_KEY=%s\n\n' "$secret"

  # Bootstrap yo'lini yopamiz — birinchi kalit berilgach u baribir ishlamaydi,
  # lekin sirni faylda saqlashning ma'nosi yo'q.
  _set_env_value "$f" ADMIN_BOOTSTRAP_KEY ""
  systemctl restart face-backend
  ok "ADMIN_BOOTSTRAP_KEY o'chirildi va servis qayta ishga tushirildi"
}

# =============================================================================
#  Yordam va dispetcher
# =============================================================================
usage() {
  cat <<EOF
Face servisni Docker'siz o'rnatish.

  sudo bash $0 [qadam]

Qadamlar (ko'rsatilmasa hammasi tartib bilan bajariladi):
  preflight   server talablarga mos keladimi
  deps        tizim paketlari + Python 3.11/3.12 + Go
  user        'face' foydalanuvchisi va papkalar
  code        manba kodni /opt/face/app ga ko'chirish
  postgres    baza va rol yaratish
  qdrant      Qdrant binari + config
  python      venv + bog'liqliklar + modellar (eng uzun qadam)
  build       golang backend binarini qurish
  env         /etc/face/*.env va sirlar
  services    systemd unitlari + ishga tushirish
  status      holatni ko'rsatish
  bootstrap   birinchi API kalitni olish (servis ishlab turganda)

Muhit o'zgaruvchilari: FACE_USER APP_ROOT QDRANT_DIR ENV_DIR QDRANT_VERSION GO_VERSION
EOF
}

main() {
  case "${1:-all}" in
    -h|--help|help) usage ;;
    preflight) do_preflight ;;
    deps)      do_deps ;;
    user)      do_user ;;
    code)      do_code ;;
    postgres)  do_postgres ;;
    qdrant)    do_qdrant ;;
    python)    do_python ;;
    build)     do_build ;;
    env)       do_env ;;
    services)  do_services ;;
    status)    do_status ;;
    bootstrap) do_bootstrap ;;
    all)
      do_preflight; do_deps; do_user; do_code
      do_postgres; do_qdrant; do_python; do_build
      do_env; do_services
      printf '\n%s✓ O\047rnatish tugadi.%s\n' "$c_ok" "$c_off"
      printf '  Keyingi qadam — birinchi API kalitni oling:\n'
      printf '    sudo bash %s bootstrap\n\n' "$0"
      ;;
    *) die "Noma'lum qadam: $1  (yordam: $0 --help)" ;;
  esac
}

main "$@"
