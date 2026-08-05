package httpapi

import (
	"sync"
	"time"
)

// RateLimiter — kalit bo'yicha "token bucket".
//
// Nega aynan token bucket: u qisqa portlashlarga (bir necha kadr ketma-ket)
// yo'l qo'yadi, lekin o'rtacha tezlikni ushlab turadi. Sanaladigan oyna
// (fixed window) esa oyna chegarasida ikki barobar yukni o'tkazib yuboradi.
//
// Hisoblagichlar xotirada: servis bir nechta nusxada ishlasa, chegara har bir
// nusxa uchun alohida amal qiladi. Bu ataylab — chegara suiiste'molni to'xtatish
// uchun, hisob-kitob uchun emas; markazlashgan hisoblagich (Redis) har so'rovga
// tarmoq safarini qo'shgan bo'lardi.
type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	// defaultPerMin — kalitda o'z chegarasi ko'rsatilmaganda.
	defaultPerMin int
	now           func() time.Time
	stop          chan struct{}
}

type bucket struct {
	tokens   float64
	capacity float64
	perSec   float64
	last     time.Time
}

// NewRateLimiter — `defaultPerMin <= 0` bo'lsa cheklov umuman qo'llanmaydi.
func NewRateLimiter(defaultPerMin int) *RateLimiter {
	rl := &RateLimiter{
		buckets:       make(map[string]*bucket),
		defaultPerMin: defaultPerMin,
		now:           time.Now,
		stop:          make(chan struct{}),
	}
	go rl.collectGarbage()
	return rl
}

// Allow — so'rovga ruxsat bormi. `perMin <= 0` bo'lsa kalitning o'z chegarasi
// yo'q, standart qiymat ishlatiladi; u ham 0 bo'lsa cheklov yo'q.
func (rl *RateLimiter) Allow(keyID string, perMin int) (allowed bool, retryAfter time.Duration) {
	if perMin <= 0 {
		perMin = rl.defaultPerMin
	}
	if perMin <= 0 {
		return true, 0
	}

	now := rl.now()
	capacity := float64(perMin)
	perSec := capacity / 60.0

	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, ok := rl.buckets[keyID]
	if !ok {
		rl.buckets[keyID] = &bucket{tokens: capacity - 1, capacity: capacity, perSec: perSec, last: now}
		return true, 0
	}

	// Chegara admin tomonidan o'zgartirilgan bo'lishi mumkin.
	if b.capacity != capacity {
		b.capacity, b.perSec = capacity, perSec
		if b.tokens > capacity {
			b.tokens = capacity
		}
	}

	elapsed := now.Sub(b.last).Seconds()
	if elapsed > 0 {
		b.tokens = min(b.capacity, b.tokens+elapsed*b.perSec)
		b.last = now
	}

	if b.tokens < 1 {
		// Bitta token to'planishiga qancha qolganini aytamiz.
		wait := time.Duration((1 - b.tokens) / b.perSec * float64(time.Second))
		return false, wait
	}

	b.tokens--
	return true, 0
}

// Close — fon tozalash gorutinasini to'xtatadi.
func (rl *RateLimiter) Close() { close(rl.stop) }

// collectGarbage — uzoq vaqt ishlatilmagan hisoblagichlarni o'chiradi, aks
// holda bekor qilingan kalitlar xotirada abadiy qolib ketardi.
func (rl *RateLimiter) collectGarbage() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-rl.stop:
			return
		case <-ticker.C:
			cutoff := rl.now().Add(-30 * time.Minute)
			rl.mu.Lock()
			for id, b := range rl.buckets {
				if b.last.Before(cutoff) {
					delete(rl.buckets, id)
				}
			}
			rl.mu.Unlock()
		}
	}
}
