package httpapi

import (
	"testing"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// ============================================================== rate limit

func TestRateLimitAllowsUpToCapacity(t *testing.T) {
	limiter := NewRateLimiter(0)
	defer limiter.Close()

	for i := range 10 {
		if allowed, _ := limiter.Allow("key-1", 10); !allowed {
			t.Fatalf("%d-so'rov rad etildi — chegara 10 ta edi", i+1)
		}
	}
	if allowed, retry := limiter.Allow("key-1", 10); allowed {
		t.Fatal("chegaradan oshgan so'rov o'tkazildi")
	} else if retry <= 0 {
		t.Error("Retry-After hisoblanmadi")
	}
}

// Har bir kalit o'z hisobiga ega bo'lishi kerak.
func TestRateLimitIsPerKey(t *testing.T) {
	limiter := NewRateLimiter(0)
	defer limiter.Close()

	for range 2 {
		limiter.Allow("key-1", 2)
	}
	if allowed, _ := limiter.Allow("key-1", 2); allowed {
		t.Fatal("birinchi kalit chegarasi ishlamadi")
	}
	if allowed, _ := limiter.Allow("key-2", 2); !allowed {
		t.Fatal("ikkinchi kalit birinchisining chegarasidan zarar ko'rdi")
	}
}

// Vaqt o'tishi bilan tokenlar tiklanishi kerak.
func TestRateLimitRefillsOverTime(t *testing.T) {
	limiter := NewRateLimiter(0)
	defer limiter.Close()

	now := time.Now()
	limiter.now = func() time.Time { return now }

	for range 60 {
		limiter.Allow("key-1", 60)
	}
	if allowed, _ := limiter.Allow("key-1", 60); allowed {
		t.Fatal("chegara ishlamadi")
	}

	// 60/min = sekundiga 1 token.
	now = now.Add(2 * time.Second)
	if allowed, _ := limiter.Allow("key-1", 60); !allowed {
		t.Fatal("vaqt o'tgach token tiklanmadi")
	}
}

func TestRateLimitDisabledWhenZero(t *testing.T) {
	limiter := NewRateLimiter(0)
	defer limiter.Close()

	for i := range 1000 {
		if allowed, _ := limiter.Allow("key-1", 0); !allowed {
			t.Fatalf("cheklov o'chirilgan bo'lsa ham %d-so'rov rad etildi", i+1)
		}
	}
}

// Kalitda o'z chegarasi bo'lmasa servis standarti qo'llanadi.
func TestRateLimitFallsBackToDefault(t *testing.T) {
	limiter := NewRateLimiter(3)
	defer limiter.Close()

	for range 3 {
		limiter.Allow("key-1", 0)
	}
	if allowed, _ := limiter.Allow("key-1", 0); allowed {
		t.Fatal("standart chegara qo'llanmadi")
	}
}

// ================================================================= ticket

func testKey() domain.APIKey {
	return domain.APIKey{
		ID:           "key-1",
		ClientID:     "client-1",
		ClientName:   "kirdi_chiqdi",
		Scopes:       []domain.Scope{domain.ScopeVerify},
		ClientActive: true,
	}
}

// Ticket aynan bir marta ishlashi kerak.
func TestTicketIsSingleUse(t *testing.T) {
	store := NewTicketStore()
	defer store.Close()

	token, _, err := store.Issue(testKey())
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}

	key, err := store.Redeem(token)
	if err != nil {
		t.Fatalf("birinchi ishlatish muvaffaqiyatsiz: %v", err)
	}
	if !key.HasScope(domain.ScopeVerify) {
		t.Error("ticketda huquqlar saqlanmadi")
	}

	if _, err := store.Redeem(token); domain.CodeOf(err) != domain.CodeUnauthenticated {
		t.Fatal("ticket ikkinchi marta ishlatildi")
	}
}

func TestTicketExpires(t *testing.T) {
	store := NewTicketStore()
	defer store.Close()

	now := time.Now()
	store.now = func() time.Time { return now }

	token, _, _ := store.Issue(testKey())
	now = now.Add(TicketTTL + time.Second)

	if _, err := store.Redeem(token); domain.CodeOf(err) != domain.CodeUnauthenticated {
		t.Fatal("muddati o'tgan ticket qabul qilindi")
	}
}

func TestTicketRejectsUnknownToken(t *testing.T) {
	store := NewTicketStore()
	defer store.Close()

	for _, token := range []string{"", "yasama-token"} {
		if _, err := store.Redeem(token); domain.CodeOf(err) != domain.CodeUnauthenticated {
			t.Errorf("%q qabul qilindi", token)
		}
	}
}

func TestTicketsAreUnique(t *testing.T) {
	store := NewTicketStore()
	defer store.Close()

	seen := make(map[string]bool, 50)
	for range 50 {
		token, _, err := store.Issue(testKey())
		if err != nil {
			t.Fatalf("kutilmagan xato: %v", err)
		}
		if seen[token] {
			t.Fatal("ticket takrorlandi")
		}
		seen[token] = true
	}
}
