package authstore

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// AuditWriter — jurnalni FONDA yozadi.
//
// Nega asinxron: audit yozuvi so'rov javobini kechiktirmasligi kerak — yuz
// tanish oqimida har millisekund ko'rinadi. Nega buferli navbat: baza
// sekinlashsa yoki uzilsa, servis ishlashda davom etadi; navbat to'lsa yozuv
// tashlab yuboriladi va bu haqda ogohlantirish logga chiqadi. Ya'ni audit
// asosiy funksiyani hech qachon to'xtatmaydi.
type AuditWriter struct {
	store   *Store
	queue   chan domain.AuditEntry
	log     *slog.Logger
	wg      sync.WaitGroup
	dropped int64
	mu      sync.Mutex
}

var _ domain.AuditSink = (*AuditWriter)(nil)

// NewAuditWriter — navbat hajmi va yozuvchi gorutinani ishga tushiradi.
func NewAuditWriter(store *Store, queueSize int, log *slog.Logger) *AuditWriter {
	if queueSize <= 0 {
		queueSize = 1024
	}
	w := &AuditWriter{
		store: store,
		queue: make(chan domain.AuditEntry, queueSize),
		log:   log,
	}
	w.wg.Add(1)
	go w.run()
	return w
}

// Record — yozuvni navbatga qo'yadi. Hech qachon bloklamaydi.
func (w *AuditWriter) Record(entry domain.AuditEntry) {
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now().UTC()
	}

	select {
	case w.queue <- entry:
	default:
		w.mu.Lock()
		w.dropped++
		count := w.dropped
		w.mu.Unlock()
		if count%100 == 1 {
			w.log.Warn("audit navbati to'lgan, yozuv tashlab yuborildi", "dropped_total", count)
		}
	}
}

// Close — navbatdagi yozuvlarni oxirigacha yozib, to'xtaydi.
func (w *AuditWriter) Close() {
	close(w.queue)
	w.wg.Wait()
}

func (w *AuditWriter) run() {
	defer w.wg.Done()

	for entry := range w.queue {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		w.write(ctx, entry)
		cancel()
	}
}

func (w *AuditWriter) write(ctx context.Context, entry domain.AuditEntry) {
	// client_id UUID ustuni: bootstrap kaliti kabi holatlarda u bo'sh bo'ladi.
	var clientID, keyID any
	if entry.ClientID != "" {
		clientID = entry.ClientID
	}
	if entry.KeyID != "" && entry.KeyID != "bootstrap" {
		keyID = entry.KeyID
	}

	_, err := w.store.pool.Exec(ctx, `
		INSERT INTO audit_log
			(client_id, key_id, action, person_id, result, code, score, latency_ms, request_id, ip, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
		clientID, keyID, entry.Action, entry.PersonID, entry.Result,
		string(entry.Code), entry.Score, entry.LatencyMs, entry.RequestID, entry.IP, entry.CreatedAt)
	if err != nil {
		w.log.Error("audit yozuvini saqlab bo'lmadi", "action", entry.Action, "error", err)
	}
}

// ListAudit — admin uchun oxirgi yozuvlar.
// CountAudit — filtr bo'yicha jami yozuvlar soni.
//
// Sahifalash uchun kerak: sahifa raqami bilan birga "nechtadan" ham
// ko'rsatilmasa, operator ro'yxat tugaganini faqat bo'sh sahifaga urilib
// bilib oladi.
func (s *Store) CountAudit(ctx context.Context, clientID string) (int, error) {
	var n int
	err := s.pool.QueryRow(ctx, `
		SELECT count(*) FROM audit_log
		 WHERE ($1 = '' OR client_id::text = $1)`, clientID).Scan(&n)
	if err != nil {
		return 0, wrapStoreErr(err, "jurnal sonini o'qib bo'lmadi")
	}
	return n, nil
}

func (s *Store) ListAudit(
	ctx context.Context, clientID string, limit, offset int,
) ([]domain.AuditEntry, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	// LEFT JOIN, chunki mijoz o'chirilgan bo'lishi mumkin — u holda yozuv
	// qoladi va nomsiz ko'rinadi. Ilgari bu qo'shilish umuman yo'q edi va
	// `client_name` hech qachon to'ldirilmasdi: jurnalning "Tizim" ustuni
	// har bir qatorda bo'sh chiqardi, garchi client_id saqlangan bo'lsa ham.
	rows, err := s.pool.Query(ctx, `
		SELECT COALESCE(a.client_id::text, ''), COALESCE(c.name, ''),
		       COALESCE(a.key_id::text, ''), a.action, a.person_id,
		       a.result, a.code, a.score, a.latency_ms, a.request_id, a.ip, a.created_at
		  FROM audit_log a
		  LEFT JOIN api_clients c ON c.id = a.client_id
		 WHERE ($1 = '' OR a.client_id::text = $1)
		 ORDER BY a.created_at DESC
		 LIMIT $2 OFFSET $3`, clientID, limit, offset)
	if err != nil {
		return nil, wrapStoreErr(err, "jurnalni o'qib bo'lmadi")
	}
	defer rows.Close()

	out := make([]domain.AuditEntry, 0, limit)
	for rows.Next() {
		var e domain.AuditEntry
		var code string
		if err := rows.Scan(&e.ClientID, &e.ClientName, &e.KeyID, &e.Action, &e.PersonID,
			&e.Result, &code, &e.Score, &e.LatencyMs, &e.RequestID, &e.IP, &e.CreatedAt); err != nil {
			return nil, wrapStoreErr(err, "jurnal yozuvini o'qib bo'lmadi")
		}
		e.Code = domain.Code(code)
		out = append(out, e)
	}
	return out, wrapStoreErr(rows.Err(), "jurnalni o'qib bo'lmadi")
}
