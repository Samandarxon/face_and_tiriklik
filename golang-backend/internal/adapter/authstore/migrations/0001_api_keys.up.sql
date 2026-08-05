-- Servisdan foydalanuvchi tashqi platformalar.
CREATE TABLE IF NOT EXISTS api_clients (
    id          UUID PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,
    contact     TEXT        NOT NULL DEFAULT '',
    description TEXT        NOT NULL DEFAULT '',
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- API kalitlar.
--
-- `key_hash` — kalitning sirli qismidan olingan SHA-256 (hex). Ochiq matn
-- saqlanmaydi: kalit yo'qolsa uni tiklab bo'lmaydi, faqat yangisini berish
-- mumkin. `prefix` esa yashirin emas — u qidiruv kaliti, shuning uchun
-- unikal va indekslangan.
CREATE TABLE IF NOT EXISTS api_keys (
    id                 UUID PRIMARY KEY,
    client_id          UUID        NOT NULL REFERENCES api_clients (id) ON DELETE CASCADE,
    label              TEXT        NOT NULL DEFAULT '',
    prefix             TEXT        NOT NULL UNIQUE,
    key_hash           TEXT        NOT NULL,
    scopes             TEXT[]      NOT NULL,
    rate_limit_per_min INTEGER     NOT NULL DEFAULT 0,
    expires_at         TIMESTAMPTZ,
    revoked_at         TIMESTAMPTZ,
    last_used_at       TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_client ON api_keys (client_id);
-- Faol kalitlarni sanash uchun (bootstrap kaliti shu asosda o'chiriladi).
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys (revoked_at) WHERE revoked_at IS NULL;

-- So'rovlar jurnali.
--
-- Shaxsiy ma'lumot yozilmaydi: `person_id` allaqachon qaytarilmas hash.
-- Jurnal "kim, qachon, nimani so'radi va natija qanday bo'ldi" savoliga javob
-- beradi — bu yuz bilan ishlaydigan tizim uchun majburiy talab.
CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL PRIMARY KEY,
    client_id  UUID REFERENCES api_clients (id) ON DELETE SET NULL,
    key_id     UUID,
    action     TEXT        NOT NULL,
    person_id  TEXT        NOT NULL DEFAULT '',
    result     TEXT        NOT NULL,
    code       TEXT        NOT NULL DEFAULT '',
    score      REAL        NOT NULL DEFAULT 0,
    latency_ms INTEGER     NOT NULL DEFAULT 0,
    request_id TEXT        NOT NULL DEFAULT '',
    ip         TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_client ON audit_log (client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_person ON audit_log (person_id) WHERE person_id <> '';
