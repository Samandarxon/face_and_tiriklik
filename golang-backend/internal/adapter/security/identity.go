package security

import (
	"regexp"
	"strings"
	"unicode"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// SubjectResolver — tashqi identifikatordan (`subject_id`) `person_id` hosil
// qiladi. Hosil qilish emas, aslida: `person_id` AYNAN `subject_id` ning
// normallashtirilgan ko'rinishi. PNFL kirsa — PNFL chiqadi.
//
// # Nima uchun hash EMAS
//
// Avval bu yerda maxfiy kalit (FACE_ID_PEPPER) bilan HMAC-SHA256 turardi va
// bazaga faqat hash tushardi. U bitta muammoni yaxshi hal qilardi (xom PNFL
// vektor bazasida yotmaydi), lekin ikkitasini tug'dirardi:
//
//  1. Kalit servisning bir qismiga aylanib qolardi. Uni yo'qotish yoki
//     almashtirish = butun yuz bazasini qayta to'plash. Minglab yozuvdan keyin
//     bu amalda tiklab bo'lmaydigan yo'qotish.
//  2. Baza ko'chma emas edi. Boshqa serverga, boshqa tashkilotga yoki zaxiradan
//     tiklashda kalit ham birga ko'chishi shart edi — kalitsiz Qdrant'dagi
//     million vektor ma'nosiz raqamlar to'plami.
//
// Endi `person_id = PNFL`. Baza o'z-o'zini tushuntiradi: rasm kiradi, PNFL
// chiqadi, va bu PNFL istalgan tizimda (kirdi_chiqdi bazasida, davlat
// reyestrida) to'g'ridan-to'g'ri ishlaydi. Servis hech qanday sirga bog'liq
// emas — uni ko'chirish oddiy fayl nusxalashga aylanadi.
//
// # Buning evaziga nima yo'qoladi
//
// Vektor bazasi endi shaxsiy ma'lumot saqlaydi: "shu yuz — shu PNFL".
// Qdrant'ga kirish huquqini olgan odam buni to'g'ridan-to'g'ri o'qiy oladi,
// avval esa u faqat mazmunsiz hash ko'rardi. Shuning uchun himoya markazi
// identifikatordan INFRATUZILMAGA ko'chadi:
//
//   - Qdrant tashqi tarmoqqa umuman chiqarilmaydi (`127.0.0.1` yoki ichki
//     tarmoq), ustiga `QDRANT__SERVICE__API_KEY` qo'yiladi;
//   - diskdagi ma'lumot shifrlanadi (LUKS/dm-crypt yoki bulut disk shifri);
//   - har bir so'rov API kalit bilan keladi va audit jurnaliga tushadi.
//
// Bu almashuv ataylab qilingan: ma'lumotni butunlay yo'qotish xavfi undan
// sizib chiqish xavfidan og'irroq deb baholandi.
type SubjectResolver struct {
	// strictPNFL — true bo'lsa faqat 14 xonali PNFL qabul qilinadi.
	// false — boshqa formatdagi identifikatorlarga ham ruxsat (tabel raqami
	// va h.k.), lekin baribir normallashtiriladi.
	strictPNFL bool
}

var _ domain.SubjectResolver = (*SubjectResolver)(nil)

// pnflRe — JShShIR (PNFL) aynan 14 ta raqam.
var pnflRe = regexp.MustCompile(`^\d{14}$`)

// MaxSubjectIDLen — erkin formatdagi identifikator uchun yuqori chegara.
const MaxSubjectIDLen = 128

// NewSubjectResolver — `strictPNFL` true bo'lsa faqat PNFL qabul qilinadi.
func NewSubjectResolver(strictPNFL bool) *SubjectResolver {
	return &SubjectResolver{strictPNFL: strictPNFL}
}

// PersonID — `subject_id` ni normallashtirib `person_id` sifatida qaytaradi.
//
// Deterministik va teskari o'girilishi mumkin: bitta odam — bitta PNFL —
// butun tizim bo'ylab bitta `person_id`.
func (r *SubjectResolver) PersonID(subjectID string) (string, error) {
	normalized := NormalizeSubjectID(subjectID)
	if normalized == "" {
		return "", domain.NewError(domain.CodeInvalidInput,
			"`subject_id` bo'sh bo'lishi mumkin emas.")
	}

	if r.strictPNFL {
		if !pnflRe.MatchString(normalized) {
			return "", domain.NewError(domain.CodeInvalidInput,
				"`subject_id` 14 xonali PNFL (JShShIR) bo'lishi kerak.").
				WithDetails(map[string]any{
					"got_length": len(normalized),
					"expected":   "14 ta raqam",
				})
		}
		return normalized, nil
	}

	if len(normalized) > MaxSubjectIDLen {
		return "", domain.NewError(domain.CodeInvalidInput,
			"`subject_id` 128 belgidan oshmasligi kerak.")
	}
	return normalized, nil
}

// StrictPNFL — servis ayni damda qat'iy PNFL rejimidami.
func (r *SubjectResolver) StrictPNFL() bool { return r.strictPNFL }

// NormalizeSubjectID — kirishni bir ko'rinishga keltiradi, aks holda
// " 123-456 789 01234 " va "12345678901234" ikki xil odam bo'lib qolardi.
// Bo'shliqlar, defis, nuqta va pastki chiziq olib tashlanadi, harflar
// kichraytiriladi.
func NormalizeSubjectID(raw string) string {
	var b strings.Builder
	b.Grow(len(raw))
	for _, r := range raw {
		switch {
		case unicode.IsSpace(r), r == '-', r == '.', r == '_':
			continue
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// IsPNFL — normallashtirilgandan keyin 14 xonali PNFL bo'ladimi.
func IsPNFL(raw string) bool { return pnflRe.MatchString(NormalizeSubjectID(raw)) }
