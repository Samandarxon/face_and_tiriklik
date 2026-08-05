package authstore

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Store `domain.APIKeyStore` ni amalga oshiradi.
var _ domain.APIKeyStore = (*Store)(nil)

// Postgres xato kodlari.
const (
	pgUniqueViolation     = "23505"
	pgForeignKeyViolation = "23503"
)

// ============================================================== Mijozlar

func (s *Store) CreateClient(ctx context.Context, client *domain.APIClient) error {
	now := s.now()
	client.CreatedAt, client.UpdatedAt = now, now

	_, err := s.pool.Exec(ctx, `
		INSERT INTO api_clients (id, name, contact, description, is_active, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		client.ID, client.Name, client.Contact, client.Description,
		client.IsActive, client.CreatedAt, client.UpdatedAt)

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == pgUniqueViolation {
		return domain.NewError(domain.CodeConflict, "Bunday nomli mijoz allaqachon mavjud.")
	}
	return wrapStoreErr(err, "mijozni yaratib bo'lmadi")
}

func (s *Store) UpdateClient(ctx context.Context, client *domain.APIClient) error {
	client.UpdatedAt = s.now()

	tag, err := s.pool.Exec(ctx, `
		UPDATE api_clients
		   SET name = $2, contact = $3, description = $4, is_active = $5, updated_at = $6
		 WHERE id = $1`,
		client.ID, client.Name, client.Contact, client.Description,
		client.IsActive, client.UpdatedAt)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == pgUniqueViolation {
			return domain.NewError(domain.CodeConflict, "Bunday nomli mijoz allaqachon mavjud.")
		}
		return wrapStoreErr(err, "mijozni yangilab bo'lmadi")
	}
	if tag.RowsAffected() == 0 {
		return domain.NewError(domain.CodeNotFound, "Mijoz topilmadi.")
	}
	return nil
}

func (s *Store) GetClient(ctx context.Context, id string) (*domain.APIClient, error) {
	var c domain.APIClient
	err := s.pool.QueryRow(ctx, `
		SELECT id::text, name, contact, description, is_active, created_at, updated_at
		  FROM api_clients WHERE id = $1`, id,
	).Scan(&c.ID, &c.Name, &c.Contact, &c.Description, &c.IsActive, &c.CreatedAt, &c.UpdatedAt)

	if errors.Is(err, pgx.ErrNoRows) {
		return nil, domain.NewError(domain.CodeNotFound, "Mijoz topilmadi.")
	}
	if err != nil {
		return nil, wrapStoreErr(err, "mijozni o'qib bo'lmadi")
	}
	return &c, nil
}

func (s *Store) ListClients(ctx context.Context) ([]domain.APIClient, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT id::text, name, contact, description, is_active, created_at, updated_at
		  FROM api_clients ORDER BY created_at DESC`)
	if err != nil {
		return nil, wrapStoreErr(err, "mijozlarni o'qib bo'lmadi")
	}
	defer rows.Close()

	out := make([]domain.APIClient, 0, 16)
	for rows.Next() {
		var c domain.APIClient
		if err := rows.Scan(&c.ID, &c.Name, &c.Contact, &c.Description,
			&c.IsActive, &c.CreatedAt, &c.UpdatedAt); err != nil {
			return nil, wrapStoreErr(err, "mijoz yozuvini o'qib bo'lmadi")
		}
		out = append(out, c)
	}
	return out, wrapStoreErr(rows.Err(), "mijozlarni o'qib bo'lmadi")
}

// DeleteClient — mijoz bilan birga uning barcha kalitlari ham o'chadi
// (ON DELETE CASCADE), audit jurnali esa saqlanib qoladi (SET NULL).
func (s *Store) DeleteClient(ctx context.Context, id string) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM api_clients WHERE id = $1`, id)
	if err != nil {
		return wrapStoreErr(err, "mijozni o'chirib bo'lmadi")
	}
	if tag.RowsAffected() == 0 {
		return domain.NewError(domain.CodeNotFound, "Mijoz topilmadi.")
	}
	return nil
}

// ============================================================== Kalitlar

func (s *Store) CreateKey(ctx context.Context, key *domain.APIKey, hash string) error {
	key.CreatedAt = s.now()

	_, err := s.pool.Exec(ctx, `
		INSERT INTO api_keys
			(id, client_id, label, prefix, key_hash, scopes, rate_limit_per_min, expires_at, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
		key.ID, key.ClientID, key.Label, key.Prefix, hash,
		scopesToStrings(key.Scopes), key.RateLimitPerMin, key.ExpiresAt, key.CreatedAt)

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		switch pgErr.Code {
		case pgUniqueViolation:
			return domain.NewError(domain.CodeConflict, "Kalit prefiksi takrorlandi, qayta urinib ko'ring.")
		case pgForeignKeyViolation:
			return domain.NewError(domain.CodeNotFound, "Mijoz topilmadi.")
		}
	}
	return wrapStoreErr(err, "kalitni saqlab bo'lmadi")
}

// FindKeyByPrefix — autentifikatsiyaning issiq yo'li: prefiks indekslangan,
// shuning uchun bu bitta indeks bo'yicha qidiruv.
func (s *Store) FindKeyByPrefix(ctx context.Context, prefix string) (*domain.APIKey, string, error) {
	var (
		key    domain.APIKey
		hash   string
		scopes []string
	)
	err := s.pool.QueryRow(ctx, `
		SELECT k.id::text, k.client_id::text, c.name, k.label, k.prefix, k.key_hash,
		       k.scopes, k.rate_limit_per_min, k.expires_at, k.revoked_at,
		       k.last_used_at, k.created_at, c.is_active
		  FROM api_keys k
		  JOIN api_clients c ON c.id = k.client_id
		 WHERE k.prefix = $1`, prefix,
	).Scan(&key.ID, &key.ClientID, &key.ClientName, &key.Label, &key.Prefix, &hash,
		&scopes, &key.RateLimitPerMin, &key.ExpiresAt, &key.RevokedAt,
		&key.LastUsedAt, &key.CreatedAt, &key.ClientActive)

	if errors.Is(err, pgx.ErrNoRows) {
		return nil, "", domain.NewError(domain.CodeUnauthenticated, "API kalit noto'g'ri.")
	}
	if err != nil {
		return nil, "", wrapStoreErr(err, "kalitni o'qib bo'lmadi")
	}

	key.Scopes = stringsToScopes(scopes)
	return &key, hash, nil
}

func (s *Store) GetKey(ctx context.Context, id string) (*domain.APIKey, error) {
	var (
		key    domain.APIKey
		scopes []string
	)
	err := s.pool.QueryRow(ctx, `
		SELECT k.id::text, k.client_id::text, c.name, k.label, k.prefix,
		       k.scopes, k.rate_limit_per_min, k.expires_at, k.revoked_at,
		       k.last_used_at, k.created_at, c.is_active
		  FROM api_keys k
		  JOIN api_clients c ON c.id = k.client_id
		 WHERE k.id = $1`, id,
	).Scan(&key.ID, &key.ClientID, &key.ClientName, &key.Label, &key.Prefix,
		&scopes, &key.RateLimitPerMin, &key.ExpiresAt, &key.RevokedAt,
		&key.LastUsedAt, &key.CreatedAt, &key.ClientActive)

	if errors.Is(err, pgx.ErrNoRows) {
		return nil, domain.NewError(domain.CodeNotFound, "Kalit topilmadi.")
	}
	if err != nil {
		return nil, wrapStoreErr(err, "kalitni o'qib bo'lmadi")
	}

	key.Scopes = stringsToScopes(scopes)
	return &key, nil
}

// ListKeys — `clientID` bo'sh bo'lsa barcha kalitlar qaytadi.
func (s *Store) ListKeys(ctx context.Context, clientID string) ([]domain.APIKey, error) {
	const query = `
		SELECT k.id::text, k.client_id::text, c.name, k.label, k.prefix,
		       k.scopes, k.rate_limit_per_min, k.expires_at, k.revoked_at,
		       k.last_used_at, k.created_at, c.is_active
		  FROM api_keys k
		  JOIN api_clients c ON c.id = k.client_id
		 WHERE ($1 = '' OR k.client_id::text = $1)
		 ORDER BY k.created_at DESC`

	rows, err := s.pool.Query(ctx, query, clientID)
	if err != nil {
		return nil, wrapStoreErr(err, "kalitlarni o'qib bo'lmadi")
	}
	defer rows.Close()

	out := make([]domain.APIKey, 0, 16)
	for rows.Next() {
		var (
			key    domain.APIKey
			scopes []string
		)
		if err := rows.Scan(&key.ID, &key.ClientID, &key.ClientName, &key.Label, &key.Prefix,
			&scopes, &key.RateLimitPerMin, &key.ExpiresAt, &key.RevokedAt,
			&key.LastUsedAt, &key.CreatedAt, &key.ClientActive); err != nil {
			return nil, wrapStoreErr(err, "kalit yozuvini o'qib bo'lmadi")
		}
		key.Scopes = stringsToScopes(scopes)
		out = append(out, key)
	}
	return out, wrapStoreErr(rows.Err(), "kalitlarni o'qib bo'lmadi")
}

// RevokeKey — kalitni bekor qiladi. Qayta bekor qilish xato emas (idempotent),
// lekin birinchi bekor qilingan vaqt saqlanib qoladi.
func (s *Store) RevokeKey(ctx context.Context, id string, at time.Time) error {
	tag, err := s.pool.Exec(ctx,
		`UPDATE api_keys SET revoked_at = COALESCE(revoked_at, $2) WHERE id = $1`, id, at)
	if err != nil {
		return wrapStoreErr(err, "kalitni bekor qilib bo'lmadi")
	}
	if tag.RowsAffected() == 0 {
		return domain.NewError(domain.CodeNotFound, "Kalit topilmadi.")
	}
	return nil
}

// PurgeKey — yozuvni butunlay o'chiradi.
//
// Bekor qilishdan farqi: bekor qilingan kalit ishlamaydi, lekin qatori
// qoladi — u qachon berilgani, kimga tegishli bo'lgani va oxirgi marta
// qachon ishlatilgani ko'rinib turadi. Bu dalil. O'chirilgach esa hech
// narsa qolmaydi, shuning uchun bu faqat ro'yxatni tozalash uchun va
// faqat ALLAQACHON bekor qilingan kalitlarga nisbatan qo'llanadi (shartni
// usecase qatlami tekshiradi).
func (s *Store) PurgeKey(ctx context.Context, id string) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM api_keys WHERE id = $1`, id)
	if err != nil {
		return wrapStoreErr(err, "kalitni o'chirib bo'lmadi")
	}
	if tag.RowsAffected() == 0 {
		return domain.NewError(domain.CodeNotFound, "Kalit topilmadi.")
	}
	return nil
}

// TouchKey — oxirgi ishlatilgan vaqtni yangilaydi. Har so'rovda emas, keshning
// muddati tugaganda chaqiriladi (aks holda bu yozuv issiq yo'lni sekinlashtiradi).
func (s *Store) TouchKey(ctx context.Context, id string, at time.Time) error {
	_, err := s.pool.Exec(ctx, `UPDATE api_keys SET last_used_at = $2 WHERE id = $1`, id, at)
	return wrapStoreErr(err, "kalit vaqtini yangilab bo'lmadi")
}

// CountKeys — faol (bekor qilinmagan) kalitlar soni. Bootstrap kaliti faqat
// bu son nolga teng bo'lganda ishlaydi.
func (s *Store) CountKeys(ctx context.Context) (int, error) {
	var n int
	err := s.pool.QueryRow(ctx,
		`SELECT count(*) FROM api_keys WHERE revoked_at IS NULL`).Scan(&n)
	return n, wrapStoreErr(err, "kalitlarni sanab bo'lmadi")
}

// ============================================================== Yordamchilar

func scopesToStrings(scopes []domain.Scope) []string {
	out := make([]string, len(scopes))
	for i, s := range scopes {
		out[i] = string(s)
	}
	return out
}

func stringsToScopes(raw []string) []domain.Scope {
	out := make([]domain.Scope, 0, len(raw))
	for _, s := range raw {
		out = append(out, domain.Scope(s))
	}
	return out
}

// wrapStoreErr — baza xatosini domain xatosiga o'giradi. Domain xatosi
// bo'lsa (yuqorida yasalganlar) o'zgarishsiz o'tadi.
func wrapStoreErr(err error, message string) error {
	if err == nil {
		return nil
	}
	if _, ok := domain.AsError(err); ok {
		return err
	}
	return domain.WrapError(domain.CodeStoreFailure, message+".", err)
}
