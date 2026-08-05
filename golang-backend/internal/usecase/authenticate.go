package usecase

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"log/slog"
	"sync"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Authenticator — taqdim etilgan API kalitni tekshiruvchi stsenariy.
//
// Bu kod HAR BIR so'rovda ishlaydi (real-time oqimda esa har kadrda emas —
// oqim bir marta autentifikatsiya qilinadi), shuning uchun bazaga borish
// qisqa muddatli kesh bilan kamaytiriladi.
type Authenticator struct {
	store    domain.APIKeyStore
	cipher   domain.KeyCipher
	cacheTTL time.Duration
	log      *slog.Logger
	now      func() time.Time

	mu    sync.RWMutex
	cache map[string]cachedKey

	// bootstrap — birinchi kalitni yaratish uchun vaqtinchalik "zaxira eshik".
	bootstrapHash string
}

type cachedKey struct {
	key     domain.APIKey
	hash    string
	expires time.Time
}

// NewAuthenticator — kesh muddati 0 bo'lsa keshlash o'chiriladi.
//
// bootstrapKey — ixtiyoriy. U FAQAT bazada birorta ham faol kalit bo'lmaganda
// ishlaydi: birinchi mijoz va birinchi kalitni yaratib olish uchun. Birinchi
// kalit berilishi bilan bu yo'l avtomatik yopiladi.
func NewAuthenticator(
	store domain.APIKeyStore,
	cipher domain.KeyCipher,
	cacheTTL time.Duration,
	bootstrapKey string,
	log *slog.Logger,
) *Authenticator {
	a := &Authenticator{
		store:    store,
		cipher:   cipher,
		cacheTTL: cacheTTL,
		log:      log,
		now:      func() time.Time { return time.Now().UTC() },
		cache:    make(map[string]cachedKey),
	}
	if bootstrapKey != "" {
		sum := sha256.Sum256([]byte(bootstrapKey))
		a.bootstrapHash = hex.EncodeToString(sum[:])
	}
	return a
}

// BootstrapKeyID — bootstrap kaliti bilan kelgan so'rovlarni auditda ajratish
// uchun ishlatiladigan sun'iy identifikator.
const BootstrapKeyID = "bootstrap"

// Authenticate — xom kalitni tekshiradi va uning metadatasini qaytaradi.
func (a *Authenticator) Authenticate(ctx context.Context, raw string) (*domain.APIKey, error) {
	if raw == "" {
		return nil, domain.NewError(domain.CodeUnauthenticated,
			"API kalit ko'rsatilmagan. `X-API-Key` sarlavhasini yuboring.")
	}

	if a.isBootstrap(raw) {
		return a.bootstrapAccess(ctx)
	}

	prefix, secret, err := a.cipher.Parse(raw)
	if err != nil {
		return nil, domain.NewError(domain.CodeUnauthenticated, "API kalit noto'g'ri.")
	}

	key, hash, err := a.lookup(ctx, prefix)
	if err != nil {
		return nil, err
	}
	if !a.cipher.Matches(secret, hash) {
		return nil, domain.NewError(domain.CodeUnauthenticated, "API kalit noto'g'ri.")
	}
	if err := key.Usable(a.now()); err != nil {
		return nil, err
	}
	return key, nil
}

// lookup — keshdan yoki bazadan kalitni oladi.
func (a *Authenticator) lookup(ctx context.Context, prefix string) (*domain.APIKey, string, error) {
	now := a.now()

	if a.cacheTTL > 0 {
		a.mu.RLock()
		entry, ok := a.cache[prefix]
		a.mu.RUnlock()
		if ok && entry.expires.After(now) {
			key := entry.key
			return &key, entry.hash, nil
		}
	}

	key, hash, err := a.store.FindKeyByPrefix(ctx, prefix)
	if err != nil {
		return nil, "", err
	}

	if a.cacheTTL > 0 {
		a.mu.Lock()
		a.cache[prefix] = cachedKey{key: *key, hash: hash, expires: now.Add(a.cacheTTL)}
		a.mu.Unlock()
	}

	// "Oxirgi ishlatilgan" vaqti keshning muddati tugaganda yangilanadi, har
	// so'rovda emas — aks holda har bir kadr bazaga yozuv qilib turardi.
	a.touchAsync(key.ID, now)

	return key, hash, nil
}

func (a *Authenticator) touchAsync(keyID string, at time.Time) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := a.store.TouchKey(ctx, keyID, at); err != nil {
			a.log.Warn("kalit vaqtini yangilab bo'lmadi", "key_id", keyID, "error", err)
		}
	}()
}

// Invalidate — bitta kalitni keshdan chiqaradi. Bekor qilish yoki rotatsiya
// darhol kuchga kirishi uchun kerak.
func (a *Authenticator) Invalidate(prefix string) {
	a.mu.Lock()
	delete(a.cache, prefix)
	a.mu.Unlock()
}

// InvalidateAll — mijoz o'chirilganda yoki faolsizlantirilganda.
func (a *Authenticator) InvalidateAll() {
	a.mu.Lock()
	a.cache = make(map[string]cachedKey)
	a.mu.Unlock()
}

// isBootstrap — doimiy vaqtli taqqoslash.
func (a *Authenticator) isBootstrap(raw string) bool {
	if a.bootstrapHash == "" {
		return false
	}
	sum := sha256.Sum256([]byte(raw))
	given := hex.EncodeToString(sum[:])
	return subtle.ConstantTimeCompare([]byte(given), []byte(a.bootstrapHash)) == 1
}

// bootstrapAccess — bootstrap kaliti hali amal qiladimi.
func (a *Authenticator) bootstrapAccess(ctx context.Context) (*domain.APIKey, error) {
	count, err := a.store.CountKeys(ctx)
	if err != nil {
		return nil, err
	}
	if count > 0 {
		a.log.Warn("bootstrap kaliti bilan urinish rad etildi — bazada faol kalitlar bor")
		return nil, domain.NewError(domain.CodeUnauthenticated,
			"Bootstrap kaliti endi ishlamaydi: tizimda faol kalitlar mavjud.")
	}

	a.log.Warn("bootstrap kaliti ishlatildi — birinchi kalitni yaratib, uni o'chiring")
	return &domain.APIKey{
		ID:           BootstrapKeyID,
		ClientID:     "",
		ClientName:   "bootstrap",
		Label:        "bootstrap",
		Prefix:       BootstrapKeyID,
		Scopes:       []domain.Scope{domain.ScopeAdmin},
		ClientActive: true,
	}, nil
}
