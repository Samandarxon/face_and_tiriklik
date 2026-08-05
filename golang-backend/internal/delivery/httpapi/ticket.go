package httpapi

import (
	"crypto/rand"
	"encoding/base64"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Nima uchun ticket kerak?
//
// Brauzerdagi `new WebSocket(...)` maxsus sarlavha yubora olmaydi — ya'ni
// `X-API-Key` ni WS ulanishida berish imkonsiz. Kalitni URL query'ga qo'yish
// esa uni brauzer tarixiga, proksi loglariga va Referer sarlavhasiga chiqarib
// yuboradi.
//
// Yechim: mijoz (server tarafda, kalit bilan) qisqa muddatli BIR MARTALIK
// ticket oladi va faqat shuni brauzerga beradi. Ticket 60 soniya yashaydi va
// ishlatilishi bilan o'chadi, shuning uchun uni ushlab qolishdan foyda yo'q.

// TicketTTL — ticketning amal qilish muddati.
const TicketTTL = 60 * time.Second

// ticket — berilgan ruxsatning qisqa muddatli nusxasi.
type ticket struct {
	key     domain.APIKey
	expires time.Time
}

// TicketStore — bir martalik ticketlar ombori (xotirada).
//
// Xotirada bo'lgani uchun servis qayta ishga tushsa ticketlar yo'qoladi —
// bu muammo emas: mijoz yangisini bir so'rovda oladi. Bir nechta nusxa
// ishlatilsa, WS ulanishi ticket berilgan nusxaga tushishi kerak (sticky
// session) — yoki bu omborni Redis'ga ko'chirish kerak bo'ladi.
type TicketStore struct {
	mu      sync.Mutex
	tickets map[string]ticket
	now     func() time.Time
	stop    chan struct{}
}

func NewTicketStore() *TicketStore {
	s := &TicketStore{
		tickets: make(map[string]ticket),
		now:     time.Now,
		stop:    make(chan struct{}),
	}
	go s.collectGarbage()
	return s
}

// Issue — yangi ticket yasaydi.
func (s *TicketStore) Issue(key domain.APIKey) (string, time.Time, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", time.Time{}, domain.WrapError(domain.CodeInternal, "Ticket yasab bo'lmadi.", err)
	}
	token := base64.RawURLEncoding.EncodeToString(raw)
	expires := s.now().Add(TicketTTL)

	s.mu.Lock()
	s.tickets[token] = ticket{key: key, expires: expires}
	s.mu.Unlock()

	return token, expires, nil
}

// Redeem — ticketni ISHLATADI: topilsa qaytaradi va darhol o'chiradi.
func (s *TicketStore) Redeem(token string) (*domain.APIKey, error) {
	if token == "" {
		return nil, domain.NewError(domain.CodeUnauthenticated, "Ticket ko'rsatilmagan.")
	}

	s.mu.Lock()
	t, ok := s.tickets[token]
	delete(s.tickets, token)
	s.mu.Unlock()

	if !ok {
		return nil, domain.NewError(domain.CodeUnauthenticated,
			"Ticket topilmadi yoki allaqachon ishlatilgan.")
	}
	if !t.expires.After(s.now()) {
		return nil, domain.NewError(domain.CodeUnauthenticated, "Ticket muddati tugagan.")
	}

	key := t.key
	return &key, nil
}

// Close — tozalash gorutinasini to'xtatadi.
func (s *TicketStore) Close() { close(s.stop) }

func (s *TicketStore) collectGarbage() {
	ticker := time.NewTicker(TicketTTL)
	defer ticker.Stop()

	for {
		select {
		case <-s.stop:
			return
		case <-ticker.C:
			now := s.now()
			s.mu.Lock()
			for token, t := range s.tickets {
				if !t.expires.After(now) {
					delete(s.tickets, token)
				}
			}
			s.mu.Unlock()
		}
	}
}

// TicketHandler — ticket beruvchi endpoint.
type TicketHandler struct {
	store *TicketStore
}

func NewTicketHandler(store *TicketStore) *TicketHandler {
	return &TicketHandler{store: store}
}

// Issue godoc
//
//	@Summary		WebSocket uchun bir martalik ticket olish
//	@Description	Brauzer WS ulanishida sarlavha yubora olmaydi. Server tarafi shu endpointdan ticket olib, uni brauzerga beradi: `wss://.../api/v1/stream?ticket=...`. Ticket 60 soniya yashaydi va bir marta ishlatiladi.
//	@Tags			stream
//	@Security		ApiKeyAuth
//	@Produce		json
//	@Success		201	{object}	map[string]any
//	@Failure		401	{object}	errorBody
//	@Router			/stream/tickets [post]
func (h *TicketHandler) Issue(c *gin.Context) {
	key := apiKeyOf(c)
	if key == nil {
		respondErrorCode(c, domain.CodeUnauthenticated, "API kalit ko'rsatilmagan.")
		return
	}

	token, expires, err := h.store.Issue(*key)
	if err != nil {
		respondError(c, err)
		return
	}

	respondOK(c, http.StatusCreated, gin.H{
		"ticket":      token,
		"expires_at":  expires.UTC().Format(time.RFC3339),
		"expires_in":  int(TicketTTL.Seconds()),
		"stream_path": "/api/v1/stream",
	})
}
