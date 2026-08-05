// Package domain — biznes ob'ektlari va qoidalari.
//
// Clean Architecture: bu paket ENG ICHKI qatlam. U Gin, gRPC, Qdrant yoki
// boshqa hech qanday tashqi kutubxonaga bog'liq emas — faqat standart
// kutubxona. Shu tufayli biznes qoidalarini infratuzilmasiz test qilish mumkin.
package domain

import "time"

// Embedding — yuzning 512 o'lchamli, L2-normalizatsiyalangan vektori.
type Embedding []float32

func (e Embedding) Dim() int { return len(e) }

// BoundingBox — kadrdagi yuz koordinatalari (piksel).
type BoundingBox struct {
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
}

// Liveness — anti-spoofing natijasi.
type Liveness struct {
	IsReal           bool               `json:"is_real"`
	RealProbability  float64            `json:"real_probability"`
	SpoofProbability float64            `json:"spoof_probability"`
	Threshold        float64            `json:"threshold"`
	PerModel         map[string]float64 `json:"per_model,omitempty"`
}

// MinSide — bboxning eng qisqa tomoni (piksel).
//
// Yuz o'lchamini shu bo'yicha baholaymiz: cho'zilgan ramkada uzun tomon
// yuzda qancha tafsilot borligini yashiradi, qisqasi esa yashirmaydi.
func (b BoundingBox) MinSide() int {
	w, h := b.X2-b.X1, b.Y2-b.Y1
	if w < h {
		return w
	}
	return h
}

// Face — kadrdan ajratib olingan yuz.
type Face struct {
	BBox      BoundingBox `json:"bbox"`
	DetScore  float64     `json:"det_score"`
	Embedding Embedding   `json:"-"`
	// Pose — boshning burilishi. Aktiv liveness uchun; passiv oqimlarda nil.
	Pose *HeadPose `json:"pose,omitempty"`
}

// HeadPose — boshning fazodagi burilishi (gradus).
//
// Faqat Yaw qaror qabul qilishda ishlatiladi. Pitch va roll 5 ta kalit
// nuqtadan ancha noaniq chiqadi va ularga tayanish tekshiruvni ishonchsiz
// qilardi — batafsil: python-ai-service/src/infrastructure/headpose.py
type HeadPose struct {
	// Yaw: manfiy = chapga, musbat = o'ngga (kameradan qaraganda).
	Yaw   float64 `json:"yaw"`
	Pitch float64 `json:"pitch"`
	Roll  float64 `json:"roll"`
	// Available false bo'lsa qolgan maydonlar ma'nosiz. Bunday kadr aktiv
	// tekshiruvda hisobga OLINMAYDI: noto'g'ri yaw tekshiruvni aldab
	// o'tkazishi mumkin, "bilmayman" esa faqat sekinlashtiradi.
	Available bool `json:"available"`
}

// Person — bazadagi inson yozuvi.
type Person struct {
	ID         string    `json:"person_id"`
	Name       string    `json:"name"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at,omitempty"`
	DetScore   float64   `json:"det_score,omitempty"`
	SourceFile string    `json:"source_file,omitempty"`
}

// Match — vektor qidiruvi natijasi.
type Match struct {
	Person Person  `json:"person"`
	Score  float32 `json:"score"`
	// Samples — shu shaxsning nechta namunasi nomzodlar orasiga tushdi.
	// Ko'p namuna bilan mos kelish bitta namuna bilan mos kelishdan
	// ishonchliroq, va bu re-ranking uchun foydali signal.
	Samples int `json:"samples,omitempty"`
}

// RegistrationResult — ro'yxatga olish natijasi.
type RegistrationResult struct {
	Person      Person `json:"person"`
	PointID     string `json:"point_id"`
	Updated     bool   `json:"updated"`      // mavjud yozuv yangilandimi
	GeneratedID bool   `json:"generated_id"` // ID'ni server bergani
	Face        Face   `json:"face"`

	// Liveness — faqat `RequireLiveness` bilan ro'yxatga olinganda to'ldiriladi.
	// Arxiv fotosidan ro'yxatga olishda anti-spoofing umuman ishlamaydi, shuning
	// uchun bu maydon nil bo'ladi va javobdan tushib qoladi.
	Liveness *Liveness `json:"liveness,omitempty"`
}

// VerificationResult — muvaffaqiyatli autentifikatsiya natijasi.
type VerificationResult struct {
	Person    Person    `json:"person"`
	Score     float32   `json:"score"`
	Threshold float32   `json:"threshold"`
	Liveness  *Liveness `json:"liveness,omitempty"`
	Face      Face      `json:"face"`
}

// LivenessResult — faqat tiriklik tekshiruvi natijasi (probe).
type LivenessResult struct {
	Face     Face      `json:"face"`
	Liveness *Liveness `json:"liveness"`
}

// MatchThreshold — cosine o'xshashlik chegarasi.
type MatchThreshold float32

// Accepts — berilgan ball chegaradan o'tadimi.
func (t MatchThreshold) Accepts(score float32) bool { return score >= float32(t) }
