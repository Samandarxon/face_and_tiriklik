// Package security — kalitlarni yasash va tekshirish uchun kriptografiya.
//
// Bu qatlam ataylab kichik va bir joyda: kalit formati yoki hash algoritmi
// o'zgarsa, faqat shu paket o'zgaradi.
package security

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// KeyPrefixTag — barcha kalitlar shu bilan boshlanadi. Maqsadi ikkita:
// kalitni logda ko'rib darrov tanish va git-secret skanerlari uchun namuna
// berish.
const KeyPrefixTag = "fdsk"

const (
	prefixBytes = 6  // 12 hex belgi — qidiruv kaliti
	secretBytes = 32 // 256 bit entropiya
)

// Cipher — `domain.KeyCipher` portining implementatsiyasi.
type Cipher struct{}

var _ domain.KeyCipher = Cipher{}

// NewCipher — holatsiz, shuning uchun qiymat sifatida qaytariladi.
func NewCipher() Cipher { return Cipher{} }

func (Cipher) Generate() (domain.GeneratedKey, error) { return GenerateAPIKey() }

func (Cipher) Parse(raw string) (prefix, secret string, err error) { return ParseAPIKey(raw) }

func (Cipher) Matches(secret, hash string) bool { return SecretMatches(secret, hash) }

// GenerateAPIKey — "fdsk_<prefix>_<sir>" ko'rinishidagi kalit yasaydi.
//
// Nega parol hashi (bcrypt/argon2) EMAS: bcrypt sekin bo'lishi kerak, chunki
// parol entropiyasi past va uni lug'at bo'yicha topish mumkin. Bu yerda sir —
// 256 bitli tasodifiy qiymat, uni brute-force qilish imkonsiz, sekin hash esa
// har bir so'rovga keraksiz kechikish qo'shgan bo'lardi. Shuning uchun tez
// SHA-256 va doimiy vaqtli taqqoslash yetarli.
func GenerateAPIKey() (domain.GeneratedKey, error) {
	prefixRaw := make([]byte, prefixBytes)
	if _, err := rand.Read(prefixRaw); err != nil {
		return domain.GeneratedKey{}, fmt.Errorf("prefiks yasab bo'lmadi: %w", err)
	}
	secretRaw := make([]byte, secretBytes)
	if _, err := rand.Read(secretRaw); err != nil {
		return domain.GeneratedKey{}, fmt.Errorf("sir yasab bo'lmadi: %w", err)
	}

	prefix := hex.EncodeToString(prefixRaw)
	secret := base64.RawURLEncoding.EncodeToString(secretRaw)

	return domain.GeneratedKey{
		Prefix: prefix,
		Secret: fmt.Sprintf("%s_%s_%s", KeyPrefixTag, prefix, secret),
		Hash:   HashSecret(secret),
	}, nil
}

// ParseAPIKey — taqdim etilgan kalitni prefiks va sirga ajratadi.
func ParseAPIKey(raw string) (prefix, secret string, err error) {
	// SplitN(..., 3) — Split EMAS.
	//
	// Sir base64.RawURLEncoding bilan kodlanadi, uning alifbosi esa
	// "A-Za-z0-9-_" — ya'ni sirning O'ZIDA pastki chiziq bo'lishi mumkin.
	// Oddiy Split bunday kalitni 4 va undan ortiq bo'lakka bo'lib yuborardi,
	// tekshiruv "aynan 3 ta" shartida yiqilardi va kalit "formati noto'g'ri"
	// deb rad etilardi.
	//
	// 43 belgili sirda kamida bitta "_" bo'lish ehtimoli ~49%, ya'ni
	// generatsiya qilingan kalitlarning taxminan YARMI hech qachon
	// ishlamasdi — va sabab "kalit noto'g'ri" degan xabar ostida yashiringan
	// edi, chunki kalitning o'zi haqiqatan bazada turardi.
	//
	// SplitN uchinchi bo'lakdan keyin bo'lishni to'xtatadi, shuning uchun sir
	// butunligicha qoladi.
	parts := strings.SplitN(strings.TrimSpace(raw), "_", 3)
	if len(parts) != 3 || parts[0] != KeyPrefixTag || parts[1] == "" || parts[2] == "" {
		return "", "", fmt.Errorf("kalit formati noto'g'ri")
	}
	return parts[1], parts[2], nil
}

// HashSecret — bazaga yoziladigan hash.
func HashSecret(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])
}

// SecretMatches — doimiy vaqtli taqqoslash: taqqoslash davomiyligi bo'yicha
// kalitni bit-ma-bit topib bo'lmasligi uchun.
func SecretMatches(secret, expectedHash string) bool {
	actual := HashSecret(secret)
	return subtle.ConstantTimeCompare([]byte(actual), []byte(expectedHash)) == 1
}
