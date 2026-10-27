package usecase

import (
	"math"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// IdentityTrack — aktiv tekshiruv davomida kamera oldidagi odam O'ZGARMAGANINI
// kafolatlaydi.
//
// # Qanday hujumni to'sadi
//
// Challenge harakatni (boshni burish) tekshiradi, tanish esa boshqa kadrda
// bo'ladi. Ular bir-biriga bog'lanmasa, hujumchi buyruqni O'Z jonli yuzi
// bilan bajarib, keyin virtual kamerada boshqa odamning videosini ko'rsatardi:
// passiv model videoni to'smaydi, challenge esa allaqachon o'tgan bo'lardi.
//
// Track birinchi vektorni "langar" qilib oladi va keyingi har bir vektor —
// harakat bajarilgan kadr ham, tanish kadri ham — u bilan bir odamga tegishli
// bo'lishi shart.
//
// # Nimani to'smaydi
//
// Butun seans davomida o'sha odamning o'zi ko'rsatilsa (oldindan yozilgan
// chap/o'ng kliplar, real vaqtdagi deepfake) — track buni farqlay olmaydi.
type IdentityTrack struct {
	threshold float32
	anchor    domain.Embedding
}

// DefaultContinuityThreshold — bir odamning turli rakursdagi kadrlari uchun
// kosinus o'xshashlik chegarasi. Tanish chegarasidan (MATCH_THRESHOLD) past:
// bosh 15+ gradusga burilganda bir odamning bali ham pasayadi, begona odamlar
// esa buffalo_l da odatda 0.3 dan past. Haqiqiy FAR/FRR o'lchanib
// sozlanishi kerak.
const DefaultContinuityThreshold = 0.40

func NewIdentityTrack(threshold float32) *IdentityTrack {
	if threshold <= 0 {
		threshold = DefaultContinuityThreshold
	}
	return &IdentityTrack{threshold: threshold}
}

// Same — e langar bilan bir odammi. Birinchi chaqiruv langarni o'rnatadi.
// Bo'sh vektor hech qachon "bir xil" emas.
func (t *IdentityTrack) Same(e domain.Embedding) bool {
	if e.Dim() == 0 {
		return false
	}
	if t.anchor == nil {
		t.anchor = append(domain.Embedding(nil), e...)
		return true
	}
	return cosine(t.anchor, e) >= t.threshold
}

// Anchored — langar o'rnatilganmi.
func (t *IdentityTrack) Anchored() bool { return t.anchor != nil }

func cosine(a, b domain.Embedding) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
