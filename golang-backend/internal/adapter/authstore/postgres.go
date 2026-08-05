// Package authstore — kirish nazorati ombori (Postgres).
//
// Bu adapter `domain.APIKeyStore` va `domain.AuditSink` portlarini amalga
// oshiradi. Yuz vektorlari bu yerda EMAS — ular Qdrant'da. Postgres faqat
// "kim so'ray oladi va nima so'radi" ma'lumotini saqlaydi.
package authstore

import (
	"context"
	"embed"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// Config — ulanish sozlamalari.
type Config struct {
	DSN             string
	MaxConns        int32
	MinConns        int32
	MaxConnLifetime time.Duration
	ConnectTimeout  time.Duration
}

// Store — pgxpool ustidagi ombor.
type Store struct {
	pool *pgxpool.Pool
	now  func() time.Time
}

// New — pulni ochadi va aloqani tekshiradi.
func New(ctx context.Context, cfg Config) (*Store, error) {
	poolCfg, err := pgxpool.ParseConfig(cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("postgres DSN xato: %w", err)
	}
	if cfg.MaxConns > 0 {
		poolCfg.MaxConns = cfg.MaxConns
	}
	if cfg.MinConns > 0 {
		poolCfg.MinConns = cfg.MinConns
	}
	if cfg.MaxConnLifetime > 0 {
		poolCfg.MaxConnLifetime = cfg.MaxConnLifetime
	}
	if cfg.ConnectTimeout > 0 {
		poolCfg.ConnConfig.ConnectTimeout = cfg.ConnectTimeout
	}

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("postgres pulini ochib bo'lmadi: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres javob bermayapti: %w", err)
	}

	return &Store{pool: pool, now: func() time.Time { return time.Now().UTC() }}, nil
}

// Close — pulni yopadi.
func (s *Store) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

// Pool — audit yozuvchisi uchun (bir xil ulanish puli ishlatiladi).
func (s *Store) Pool() *pgxpool.Pool { return s.pool }

// Migrate — `migrations/` ichidagi barcha `.up.sql` fayllarni tartib bilan
// qo'llaydi. Fayllar binarga `go:embed` orqali kiritilgan, shuning uchun
// Docker image'ida qo'shimcha hech narsa kerak emas.
func (s *Store) Migrate(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version    TEXT PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`); err != nil {
		return fmt.Errorf("schema_migrations yaratib bo'lmadi: %w", err)
	}

	entries, err := migrationFS.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("migratsiyalarni o'qib bo'lmadi: %w", err)
	}

	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".up.sql") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	for _, name := range names {
		version := strings.TrimSuffix(name, ".up.sql")

		var applied bool
		if err := s.pool.QueryRow(ctx,
			`SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version = $1)`, version,
		).Scan(&applied); err != nil {
			return fmt.Errorf("migratsiya holatini tekshirib bo'lmadi (%s): %w", version, err)
		}
		if applied {
			continue
		}

		body, err := migrationFS.ReadFile("migrations/" + name)
		if err != nil {
			return fmt.Errorf("migratsiya faylini o'qib bo'lmadi (%s): %w", name, err)
		}

		tx, err := s.pool.Begin(ctx)
		if err != nil {
			return fmt.Errorf("tranzaksiya ochib bo'lmadi (%s): %w", version, err)
		}
		if _, err := tx.Exec(ctx, string(body)); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("migratsiya bajarilmadi (%s): %w", version, err)
		}
		if _, err := tx.Exec(ctx,
			`INSERT INTO schema_migrations (version) VALUES ($1)`, version,
		); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("migratsiyani belgilab bo'lmadi (%s): %w", version, err)
		}
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("migratsiya commit bo'lmadi (%s): %w", version, err)
		}
	}

	return nil
}

// Healthy — /health/ready uchun.
func (s *Store) Healthy(ctx context.Context) error {
	if err := s.pool.Ping(ctx); err != nil {
		return fmt.Errorf("postgres: %w", err)
	}
	return nil
}
