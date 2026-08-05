package domain

import (
	"slices"
	"strings"
	"time"
)

// Bu fayl servisning KIRISH NAZORATI modelini ta'riflaydi.
//
// Servis yopiq: har bir so'rov aniq bir mijozga (`APIClient`) tegishli kalit
// bilan keladi. Kalitning o'zi hech qachon saqlanmaydi — bazada faqat uning
// hash'i va qidiruv uchun prefiksi yotadi (xuddi parol kabi, lekin qidirsa
// bo'ladigan qismi ajratilgan holda).

// Scope — kalitga berilgan huquq. Har bir endpoint aniq bitta scope talab
// qiladi, shunda "faqat tekshirish" uchun berilgan kalit bazaga yoza olmaydi.
type Scope string

const (
	// ScopeVerify — rasm bo'yicha tanish (`/persons/verify`, `/stream`).
	ScopeVerify Scope = "face:verify"
	// ScopeRegister — bazaga yuz qo'shish/yangilash.
	ScopeRegister Scope = "face:register"
	// ScopeRead — yozuvlarni o'qish (`GET /persons`, `/persons/:id`, `/resolve`).
	ScopeRead Scope = "face:read"
	// ScopeDelete — yozuvni o'chirish.
	ScopeDelete Scope = "face:delete"
	// ScopeAdmin — mijoz va kalitlarni boshqarish.
	ScopeAdmin Scope = "admin:keys"
)

// AllScopes — validatsiya va hujjat uchun to'liq ro'yxat.
var AllScopes = []Scope{ScopeVerify, ScopeRegister, ScopeRead, ScopeDelete, ScopeAdmin}

// Valid — scope tanilganmi.
func (s Scope) Valid() bool { return slices.Contains(AllScopes, s) }

// ParseScopes — tashqi ro'yxatni tekshirib domain turiga o'giradi.
// Dublikatlar olib tashlanadi, tartib saqlanadi.
func ParseScopes(raw []string) ([]Scope, error) {
	if len(raw) == 0 {
		return nil, NewError(CodeInvalidInput, "Kamida bitta `scope` ko'rsatilishi kerak.")
	}

	seen := make(map[Scope]bool, len(raw))
	out := make([]Scope, 0, len(raw))
	for _, item := range raw {
		scope := Scope(strings.TrimSpace(item))
		if !scope.Valid() {
			return nil, NewError(CodeInvalidInput,
				"Noma'lum scope: "+item+". Ruxsat etilganlar: "+JoinScopes(AllScopes, ", "))
		}
		if seen[scope] {
			continue
		}
		seen[scope] = true
		out = append(out, scope)
	}
	return out, nil
}

// JoinScopes — logga va xato matniga chiqarish uchun.
func JoinScopes(scopes []Scope, sep string) string {
	parts := make([]string, len(scopes))
	for i, s := range scopes {
		parts[i] = string(s)
	}
	return strings.Join(parts, sep)
}

// APIClient — servisdan foydalanuvchi tashqi platforma (masalan `kirdi_chiqdi`).
// Kalitlar aynan mijozga biriktiriladi: mijoz o'chirilsa — kalitlari ham
// ishlamay qoladi, audit jurnalida esa kim so'raganini ko'rish mumkin.
type APIClient struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Contact     string    `json:"contact,omitempty"`
	Description string    `json:"description,omitempty"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// APIKey — mijozga berilgan kalitning METADATASI. Kalitning ochiq matni bu
// yerda yo'q va hech qayerda saqlanmaydi — u faqat yaratilgan paytda bir marta
// javobda qaytariladi.
type APIKey struct {
	ID       string `json:"id"`
	ClientID string `json:"client_id"`
	// ClientName — JOIN natijasi; faqat o'qish uchun to'ldiriladi.
	ClientName string `json:"client_name,omitempty"`
	// Label — kalitni odam tanishi uchun nom ("kirdi_chiqdi prod").
	Label string `json:"label"`
	// Prefix — kalitning yashirin bo'lmagan boshi. Bazada indekslanadi va
	// hash'ni topish uchun ishlatiladi, shu tufayli har bir so'rovda butun
	// jadvalni skanerlash shart emas.
	Prefix string  `json:"prefix"`
	Scopes []Scope `json:"scopes"`
	// RateLimitPerMin — 0 bo'lsa servis sozlamasidagi standart qiymat.
	RateLimitPerMin int        `json:"rate_limit_per_min"`
	ExpiresAt       *time.Time `json:"expires_at,omitempty"`
	RevokedAt       *time.Time `json:"revoked_at,omitempty"`
	LastUsedAt      *time.Time `json:"last_used_at,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	// ClientActive — kalit tekshirilayotganda mijoz holati ham kerak bo'ladi.
	ClientActive bool `json:"-"`
}

// HasScope — kalitda shu huquq bormi.
func (k *APIKey) HasScope(scope Scope) bool { return slices.Contains(k.Scopes, scope) }

// Usable — kalit ayni damda ishlatilishi mumkinmi.
//
// Xato matnlari ataylab umumiy: hujumchiga kalit "mavjud, lekin muddati o'tgan"
// ekanini aytish keraksiz ma'lumot beradi. Aniq sabab logda va auditda qoladi.
func (k *APIKey) Usable(now time.Time) error {
	switch {
	case k.RevokedAt != nil:
		return NewError(CodeUnauthenticated, "API kalit bekor qilingan.")
	case k.ExpiresAt != nil && !k.ExpiresAt.After(now):
		return NewError(CodeUnauthenticated, "API kalit muddati tugagan.")
	case !k.ClientActive:
		return NewError(CodeUnauthenticated, "API kalit faol emas.")
	}
	return nil
}

// GeneratedKey — kriptografiya adapteri yasab bergan xom kalit.
type GeneratedKey struct {
	// Prefix — yashirin emas, bazada indekslanadi va qidiruv uchun ishlatiladi.
	Prefix string
	// Secret — foydalanuvchiga beriladigan to'liq kalit.
	Secret string
	// Hash — bazaga yoziladigan qiymat.
	Hash string
}

// IssuedKey — endigina yaratilgan kalit: metadata + BIR MARTALIK ochiq matn.
type IssuedKey struct {
	Key APIKey `json:"key"`
	// Secret — to'liq kalit ("fdsk_<prefix>_<sir>"). Faqat shu javobda ko'rinadi.
	Secret string `json:"secret"`
}

// AuditEntry — jurnalning bitta yozuvi. Kim, nima qildi, natija qanday bo'ldi.
// Shaxsiy ma'lumot yozilmaydi: person_id allaqachon qaytarilmas hash.
type AuditEntry struct {
	ClientID string
	// ClientName — o'qishda qo'shiladi, yozilmaydi. Usiz jurnaldagi "Tizim"
	// ustuni har doim bo'sh chiqardi: id saqlanardi, nom esa hech qayerdan
	// olinmasdi va javobda umuman yo'q edi.
	ClientName string
	KeyID      string
	Action     string // "verify" | "register" | "delete" | "resolve" | "stream"
	PersonID   string
	Result     string // "ok" | "error"
	Code       Code
	Score      float32
	LatencyMs  int
	RequestID  string
	IP         string
	CreatedAt  time.Time
}
