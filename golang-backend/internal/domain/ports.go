package domain

import (
	"context"
	"time"
)

// Portlar (interfeyslar) DOMAIN qatlamida e'lon qilinadi, implementatsiyalar
// esa `adapter/` da. Shu tufayli use-case'lar Qdrant yoki gRPC haqida hech
// narsa bilmaydi va testda soxta (fake) implementatsiya bilan almashtiriladi.

// FaceAnalyzer — yuzni tahlil qiluvchi AI servis (Python).
type FaceAnalyzer interface {
	// Embed — REGISTER oqimi: liveness'siz, faqat vektor.
	Embed(ctx context.Context, image []byte, requestID string) (*Face, error)

	// VerifyFrame — VERIFY oqimi: avval liveness, keyin vektor.
	// Kadr soxta bo'lsa `CodeSpoofDetected` xatosi qaytadi va vektor
	// UMUMAN hisoblanmaydi.
	VerifyFrame(ctx context.Context, image []byte, requestID string) (*Face, *Liveness, error)

	// CheckLiveness — faqat anti-spoofing (probe).
	CheckLiveness(ctx context.Context, image []byte, requestID string) (*Face, *Liveness, error)

	// OpenStream — uzoq turadigan ikki tomonlama oqim ochadi.
	OpenStream(ctx context.Context) (FrameStream, error)

	// Health — AI servis holati.
	Health(ctx context.Context) (map[string]any, error)
}

// FrameStream — real-time kadrlar oqimi.
//
// Muhim: bitta kadrning xatosi (soxta yuz, yuz yo'q) oqimni UZMAYDI —
// u `FrameResult.Err` da keladi va mijoz keyingi kadrni yuboraveradi.
// `Recv` xato qaytarsa — oqimning o'zi tugagan.
type FrameStream interface {
	Send(frame FrameRequest) error
	Recv() (*FrameResult, error)
	CloseSend() error
	Close() error
}

// FrameRequest — oqimga yuboriladigan bitta kadr.
type FrameRequest struct {
	Image   []byte
	FrameID uint64
	// WantEmbedding — false bo'lsa faqat liveness hisoblanadi (tezroq).
	WantEmbedding bool
}

// FrameResult — bitta kadr uchun natija.
type FrameResult struct {
	FrameID      uint64
	Err          *Error // nil bo'lmasa — shu kadr qayta ishlanmadi
	Face         *Face
	Liveness     *Liveness
	ProcessingMs int
}

// OK — kadr muvaffaqiyatli qayta ishlanganini bildiradi.
func (r *FrameResult) OK() bool { return r.Err == nil }

// PersonRepository — yuz vektorlari ombori (Qdrant).
type PersonRepository interface {
	// EnsureReady — kollektsiya mavjudligini va sozlamalari to'g'riligini
	// kafolatlaydi. Idempotent.
	EnsureReady(ctx context.Context) error

	// Save — vektorni metadata bilan saqlaydi. Bir xil `person.ID` bilan
	// qayta chaqirilsa dublikat yaratmaydi, mavjudini almashtiradi.
	Save(ctx context.Context, person Person, embedding Embedding) (pointID string, err error)

	// SaveSample — mavjud shaxsga QO'SHIMCHA namuna qo'shadi.
	//
	// Asosiy yozuvni almashtirmaydi. Bir xil `sample` nomi bilan qayta
	// chaqirilsa — o'sha namuna yangilanadi, yangisi yaratilmaydi.
	SaveSample(ctx context.Context, person Person, embedding Embedding, sample string) (pointID string, err error)

	// FindNearest — cosine bo'yicha eng yaqin `limit` ta yozuv (chegarasiz).
	// 1:N — "bu kim?" savoliga javob beradi.
	FindNearest(ctx context.Context, embedding Embedding, limit int) ([]Match, error)

	// ScoreAgainst — vektorni AYNAN BITTA yozuv bilan solishtiradi.
	// 1:1 — "bu odam haqiqatan shu odammi?" savoliga javob beradi.
	//
	// Yozuv topilmasa `found=false` qaytadi va bu xato EMAS: chaqiruvchi
	// "ro'yxatdan o'tmagan" javobini beradi.
	ScoreAgainst(ctx context.Context, embedding Embedding, personID string) (match *Match, found bool, err error)

	Get(ctx context.Context, personID string) (*Person, error)
	Delete(ctx context.Context, personID string) error
	List(ctx context.Context, limit int, offset string) (people []Person, next string, err error)
	Count(ctx context.Context) (int64, error)
	Healthy(ctx context.Context) error
}

// IDGenerator — yangi `person_id` manbai.
type IDGenerator interface {
	NewID() (string, error)
}

// SubjectResolver — tashqi identifikatordan (`subject_id`) `person_id` hosil
// qiladi.
//
// `person_id` — bu AYNAN normallashtirilgan `subject_id`, ya'ni PNFL. Hech
// qanday hash yoki maxfiy kalit ishtirok etmaydi: servis o'zi bilan olib
// yuradigan sirga ega emas, shuning uchun uni boshqa serverga ko'chirish
// oddiy nusxalashdan iborat. Sabablari va evaziga nima yo'qolgani —
// `adapter/security/identity.go` da batafsil.
//
// Deterministik: bir xil kirish doim bir xil natija beradi, shu tufayli bitta
// odam butun tizimda bitta person_id ga ega bo'ladi.
type SubjectResolver interface {
	PersonID(subjectID string) (string, error)
}

// KeyCipher — API kalitlarni yasash va tekshirish. Use-case algoritmni
// bilmaydi: format yoki hash o'zgarsa faqat adapter o'zgaradi.
type KeyCipher interface {
	Generate() (GeneratedKey, error)
	// Parse — taqdim etilgan kalitni prefiks va sirga ajratadi.
	Parse(raw string) (prefix, secret string, err error)
	// Matches — sir bazadagi hash'ga mos keladimi (doimiy vaqtda).
	Matches(secret, hash string) bool
}

// APIKeyStore — mijozlar va API kalitlar ombori (Postgres).
type APIKeyStore interface {
	// Mijozlar.
	CreateClient(ctx context.Context, client *APIClient) error
	UpdateClient(ctx context.Context, client *APIClient) error
	GetClient(ctx context.Context, id string) (*APIClient, error)
	ListClients(ctx context.Context) ([]APIClient, error)
	DeleteClient(ctx context.Context, id string) error

	// Kalitlar. `hash` — kalitning sirli qismidan olingan SHA-256, ochiq matn
	// hech qachon uzatilmaydi.
	CreateKey(ctx context.Context, key *APIKey, hash string) error
	// FindKeyByPrefix — prefiks bo'yicha kalit va uning hash'ini qaytaradi.
	FindKeyByPrefix(ctx context.Context, prefix string) (key *APIKey, hash string, err error)
	GetKey(ctx context.Context, id string) (*APIKey, error)
	ListKeys(ctx context.Context, clientID string) ([]APIKey, error)
	RevokeKey(ctx context.Context, id string, at time.Time) error
	// PurgeKey — bekor qilingan kalit yozuvini butunlay o'chiradi.
	PurgeKey(ctx context.Context, id string) error
	TouchKey(ctx context.Context, id string, at time.Time) error
	CountKeys(ctx context.Context) (int, error)

	Healthy(ctx context.Context) error
}

// AuditSink — so'rovlar jurnali. Asosiy oqimni sekinlashtirmasligi uchun
// implementatsiya yozuvni navbatga qo'yib, fonda yozadi.
type AuditSink interface {
	Record(entry AuditEntry)
}
