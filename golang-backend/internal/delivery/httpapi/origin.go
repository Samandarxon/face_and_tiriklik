package httpapi

import "strings"

// originPolicy — qaysi brauzer sahifasi bu servisga murojaat qila olishi.
//
// Bu servisning ASOSIY mijozi brauzer emas: chaqiruvchi tizimning backendi
// server-server rejimida, X-API-Key bilan murojaat qiladi. Bunday so'rovda
// Origin sarlavhasi umuman bo'lmaydi va CORS ham qo'llanmaydi — ya'ni bu
// yerdagi siyosat FAQAT brauzerga tegishli va uni qattiq yopish hech qanday
// integratsiyani buzmaydi.
//
// Qabul qilinadigan qiymatlar (CORS_ALLOW_ORIGIN):
//
//	*                       — har qanday sahifa (faqat ishlab chiqish uchun)
//	none                    — hech qanday brauzer emas (PRODUCTION uchun standart tanlov)
//	https://a.uz            — bitta aniq origin
//	https://a.uz,https://b  — bir nechta, vergul bilan
//
// Bo'sh qiymat `*` deb qabul qilinadi — eski sozlamalar bilan mos qolishi
// uchun. Brauzerni butunlay yopish uchun ATAYLAB `none` yozish kerak: bu
// "unutib qoldirilgan" va "ataylab yopilgan" holatlarni farqlaydi.
type originPolicy struct {
	allowAll bool
	allowed  map[string]struct{}
}

func newOriginPolicy(spec string) originPolicy {
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "*" {
		return originPolicy{allowAll: true}
	}
	switch strings.ToLower(spec) {
	case "none", "off", "disabled":
		return originPolicy{}
	}

	allowed := make(map[string]struct{})
	for o := range strings.SplitSeq(spec, ",") {
		if o = normalizeOrigin(o); o != "" {
			allowed[o] = struct{}{}
		}
	}
	return originPolicy{allowed: allowed}
}

// allows — shu Origin'li brauzer so'roviga ruxsat berilishi.
//
// Bo'sh Origin bu yerda HAR DOIM rad etiladi: bu metod brauzer uchun, brauzer
// esa Origin'ni har doim yuboradi. Server-server mijozlarni ajratish
// chaqiruvchi tomonda (`allowsNonBrowser`) qilinadi.
func (p originPolicy) allows(origin string) bool {
	if p.allowAll {
		return true
	}
	origin = normalizeOrigin(origin)
	if origin == "" {
		return false
	}
	_, ok := p.allowed[origin]
	return ok
}

// allowsNonBrowser — WebSocket handshake uchun tekshiruv.
//
// Origin butunlay yo'q bo'lsa, so'rov brauzerdan kelmagan: bu chaqiruvchi
// tizimning proxysi, mobil ilova yoki diagnostika vositasi. Brauzer Origin'ni
// tushirib qoldira olmaydi, ya'ni bu yo'l bilan hujum sahifasi o'tolmaydi —
// va oqim baribir bir martalik ticket bilan himoyalangan.
//
// Bu chekinishsiz `CORS_ALLOW_ORIGIN` ga aniq domen yozilishi bilan
// backend'ning liveness proxysi jimgina 403 ola boshlaydi: gorilla dialer
// Origin yubormaydi.
func (p originPolicy) allowsNonBrowser(origin string) bool {
	if strings.TrimSpace(origin) == "" {
		return true
	}
	return p.allows(origin)
}

// header — javobga qo'yiladigan Access-Control-Allow-Origin qiymati.
// Bo'sh natija "sarlavha umuman yuborilmasin" degani.
//
// Ro'yxat rejimida javob `*` emas, aynan so'rovdagi origin bilan qaytadi:
// `*` va `Allow-Credentials` birga ishlamaydi, bir nechta origin'ni esa
// standart umuman qo'llab-quvvatlamaydi.
func (p originPolicy) header(origin string) string {
	if p.allowAll {
		return "*"
	}
	if p.allows(origin) {
		return origin
	}
	return ""
}

// normalizeOrigin — solishtirishdan oldingi tozalash.
//
// Origin katta-kichik harfga sezgir emas va oxirgi `/` bilan yozilishi mumkin
// (`https://a.uz/`), brauzer esa uni yubormaydi — sozlamadagi shunday kichik
// farq butun integratsiyani to'xtatib qo'yadi.
func normalizeOrigin(o string) string {
	return strings.ToLower(strings.TrimRight(strings.TrimSpace(o), "/"))
}
