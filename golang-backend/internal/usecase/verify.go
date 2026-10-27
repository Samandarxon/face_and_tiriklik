package usecase

import (
	"context"
	"log/slog"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// searchCandidates — vektor bazadan olinadigan XOM nomzodlar soni.
//
// Birinchisi qaror uchun, qolganlari diagnostika uchun: ikkinchi o'rindagi
// ball birinchisiga juda yaqin bo'lsa — chegara past qo'yilganini bildiradi.
//
// Bir shaxsda bir nechta namuna bo'lgani uchun bu son shaxslar soniga teng
// emas: bitta odam ro'yxatning bir necha o'rnini egallashi mumkin va bu
// normal. 3 ta xom natija bilan galereya kattalashganda ro'yxatda amalda
// bitta shaxs qolib ketardi — ikkinchi nomzod umuman ko'rinmasdi.
//
// 32 — galereyada odatda 3-10 namuna bo'lishini hisobga olgan zaxira bilan.
const searchCandidates = 32

// Verifier — kameradagi insonni autentifikatsiya qilish stsenariysi.
type Verifier struct {
	analyzer  domain.FaceAnalyzer
	repo      domain.PersonRepository
	threshold domain.MatchThreshold
	// margin — 1-o'rin 2-o'rindan kamida shuncha oldinda bo'lishi kerak.
	// 0 — tekshirilmaydi. Qarang: SetMargin.
	margin float32
	log    *slog.Logger
}

// SetMargin — 1:N qarorida ikkinchi nomzoddan talab qilinadigan farq.
//
// Chegaradan o'tgan, lekin boshqa odam ham deyarli shunday ball olgan
// natija — taxmin, tanish emas: bunday holatda kim ekanini yaqin turgan
// ikkinchi nomzod emas, tasodif hal qilardi. Login uchun bu soxta qabul.
// Shuning uchun bunday natija "topilmadi" deb rad etiladi.
func (v *Verifier) SetMargin(m float32) {
	if m < 0 {
		m = 0
	}
	v.margin = m
}

func NewVerifier(
	analyzer domain.FaceAnalyzer,
	repo domain.PersonRepository,
	threshold domain.MatchThreshold,
	log *slog.Logger,
) *Verifier {
	return &Verifier{analyzer: analyzer, repo: repo, threshold: threshold, log: log}
}

// Execute — tizimning eng muhim stsenariysi:
//
//	STEP 1  Liveness (Anti-Spoofing) — AI servisda.
//	        Kadr soxta bo'lsa shu yerda TO'XTAYMIZ va omborga MURDJAAT QILMAYMIZ.
//	STEP 2  Embedding — faqat tirik inson tasdiqlangandan keyin.
//	STEP 3  Vektor qidiruv — cosine ball chegaradan o'tishi kerak.
//
// STEP 1 va 2 AI servis ichida ketma-ket bajariladi (bitta gRPC chaqiruvi):
// soxta kadrda embedding umuman hisoblanmaydi, ya'ni og'ir model ishga
// tushmaydi ham.
func (v *Verifier) Execute(ctx context.Context, image []byte, requestID string) (*domain.VerificationResult, error) {
	// --- STEP 1 + STEP 2 ---
	face, liveness, err := v.analyzer.VerifyFrame(ctx, image, requestID)
	if err != nil {
		if domain.IsSpoof(err) {
			v.log.Warn("SPOOF aniqlandi — qidiruv bekor qilindi", "request_id", requestID)
		}
		return nil, err
	}

	// --- STEP 3 ---
	return v.match(ctx, face, liveness, requestID)
}

// MatchEmbedding — vektor allaqachon hisoblangan holat uchun (streaming oqimi).
//
// DIQQAT: chaqiruvchi tiriklik tasdiqlanganiga ISHONCH HOSIL QILGAN bo'lishi
// shart. Shuning uchun bu metod liveness'ni qayta tekshiradi va tirik bo'lmasa
// qidiruvni bajarmaydi — himoya ikki qatlamli.
func (v *Verifier) MatchEmbedding(
	ctx context.Context,
	face *domain.Face,
	liveness *domain.Liveness,
	requestID string,
) (*domain.VerificationResult, error) {
	if liveness == nil || !liveness.IsReal {
		v.log.Warn("MatchEmbedding tirik bo'lmagan kadr bilan chaqirildi — rad etildi",
			"request_id", requestID)
		return nil, domain.NewError(domain.CodeSpoofDetected,
			"Soxta yuz/rasm aniqlandi. Tirik inson kameraga qarashi shart!").
			WithLiveness(liveness)
	}
	return v.match(ctx, face, liveness, requestID)
}

func (v *Verifier) match(
	ctx context.Context,
	face *domain.Face,
	liveness *domain.Liveness,
	requestID string,
) (*domain.VerificationResult, error) {
	matches, err := v.repo.FindNearest(ctx, face.Embedding, searchCandidates)
	if err != nil {
		return nil, err
	}

	// Eng yaqin ball faqat logga yoziladi, javobga emas: mijozga qaytarilsa
	// u "qanchalik yaqinlashdim" orakuliga aylanadi — kadrni shu ball
	// bo'yicha qadamma-qadam o'zgartirib chegaradan o'tkazish mumkin.
	notFound := func(reason string) error {
		logArgs := []any{"request_id", requestID, "reason", reason,
			"threshold", float32(v.threshold)}
		if len(matches) > 0 {
			logArgs = append(logArgs, "best_score", matches[0].Score)
		}
		if len(matches) > 1 {
			logArgs = append(logArgs, "second_score", matches[1].Score)
		}
		v.log.Info("verify: mos keluvchi topilmadi", logArgs...)

		return domain.NewError(domain.CodePersonNotFound, "Bazadan inson topilmadi.").
			WithDetails(map[string]any{"threshold": float32(v.threshold)}).
			WithLiveness(liveness)
	}

	if len(matches) == 0 || !v.threshold.Accepts(matches[0].Score) {
		return nil, notFound("below_threshold")
	}
	if v.margin > 0 && len(matches) > 1 && matches[0].Score-matches[1].Score < v.margin {
		return nil, notFound("ambiguous")
	}

	best := matches[0]
	v.log.Info("verify: inson topildi", "request_id", requestID,
		"person_id", best.Person.ID, "score", best.Score)

	return &domain.VerificationResult{
		Person:    best.Person,
		Score:     best.Score,
		Threshold: float32(v.threshold),
		Liveness:  liveness,
		Face:      *face,
	}, nil
}

// SearchResult — nomzodlar ro'yxati.
//
// Bu QAROR emas, TAXMIN. `Decided` yo'q, `Person` yo'q — faqat ranjirlangan
// ro'yxat va har birining balli. Chegara ma'lumot uchun qaytariladi, lekin
// QO'LLANMAYDI.
type SearchResult struct {
	Candidates []domain.Match   `json:"candidates"`
	Threshold  float32          `json:"threshold"`
	Face       domain.Face      `json:"face"`
	Liveness   *domain.Liveness `json:"liveness,omitempty"`
}

// Search — 1:N nomzodlar ro'yxati, chegarasiz.
//
// # Nima uchun chegara qo'llanmaydi
//
// `Execute` bitta javob beradi va qaror qabul qiladi: bu turniket uchun
// to'g'ri. Tergov qidiruvi esa boshqa vazifa — u yerda maqsad hech kimni
// o'tkazib yubormaslik, aniqlikni esa odam ta'minlaydi.
//
// Agar bu yerda ham chegara qo'llansa, past chegarali qidiruv rejimi
// mavjud bo'lolmasdi: bitta global sozlama ikkita qarama-qarshi vazifaga
// xizmat qilishga majbur bo'lardi. Shuning uchun bu funksiya O'LCHAYDI,
// qaror esa chaqiruvchi tizimda — u yerda kim so'rayotgani va nima uchun
// so'rayotgani ma'lum.
//
// # Nima uchun liveness ixtiyoriy
//
// Arxiv rasmi bo'yicha qidirish — qonuniy stsenariy, va arxiv rasmi
// ta'rifga ko'ra "tirik" emas. Majburiy liveness bu rejimni butunlay
// yopib qo'yardi. Kameradan qidirilganda esa `requireLiveness` yoqiladi.
func (v *Verifier) Search(
	ctx context.Context, image []byte, limit int, requireLiveness bool, requestID string,
) (*SearchResult, error) {
	if limit <= 0 {
		limit = 10
	}
	if limit > 100 {
		limit = 100
	}

	var (
		face     *domain.Face
		liveness *domain.Liveness
		err      error
	)
	if requireLiveness {
		face, liveness, err = v.analyzer.VerifyFrame(ctx, image, requestID)
	} else {
		face, err = v.analyzer.Embed(ctx, image, requestID)
	}
	if err != nil {
		return nil, err
	}

	// Xom nomzodlar soni ko'proq: bir shaxs galereyasidagi bir necha namuna
	// ro'yxatning bir necha o'rnini egallashi mumkin, va ular yig'ilgach
	// natija kutilganidan qisqa chiqadi.
	raw := limit * 4
	if raw < searchCandidates {
		raw = searchCandidates
	}
	matches, err := v.repo.FindNearest(ctx, face.Embedding, raw)
	if err != nil {
		return nil, err
	}
	if len(matches) > limit {
		matches = matches[:limit]
	}

	v.log.Info("qidiruv: nomzodlar qaytarildi",
		"request_id", requestID, "count", len(matches))

	return &SearchResult{
		Candidates: matches,
		Threshold:  float32(v.threshold),
		Face:       *face,
		Liveness:   liveness,
	}, nil
}

// LivenessProbe — faqat tiriklik tekshiruvi (ombor ishlatilmaydi).
type LivenessProbe struct {
	analyzer domain.FaceAnalyzer
}

func NewLivenessProbe(analyzer domain.FaceAnalyzer) *LivenessProbe {
	return &LivenessProbe{analyzer: analyzer}
}

// Execute — soxta kadr ham XATO EMAS: bu tekshiruv endpointi, avtorizatsiya
// qarori emas. Shuning uchun natija `Liveness.IsReal=false` bilan qaytadi.
func (p *LivenessProbe) Execute(ctx context.Context, image []byte, requestID string) (*domain.LivenessResult, error) {
	face, liveness, err := p.analyzer.CheckLiveness(ctx, image, requestID)
	if err != nil {
		if domainErr, ok := domain.AsError(err); ok && domainErr.Code == domain.CodeSpoofDetected {
			result := &domain.LivenessResult{Liveness: domainErr.Liveness}
			if result.Liveness == nil {
				result.Liveness = &domain.Liveness{IsReal: false}
			}
			return result, nil
		}
		return nil, err
	}
	return &domain.LivenessResult{Face: *face, Liveness: liveness}, nil
}

// ============================================================ 1:1 VERIFICATION

// ConfirmResult — 1:1 solishtirish natijasi.
//
// 1:N dan farqi: bu yerda "topilmadi" xato emas, javob. Chaqiruvchi kimni
// tekshirayotganini oldindan bilgani uchun natija — sof "ha/yo'q" + ball.
type ConfirmResult struct {
	PersonID  string           `json:"person_id"`
	Matched   bool             `json:"matched"`
	Score     float32          `json:"score"`
	Threshold float32          `json:"threshold"`
	Person    *domain.Person   `json:"person,omitempty"`
	Liveness  *domain.Liveness `json:"liveness,omitempty"`
	Face      domain.Face      `json:"face"`
}

// Confirmer — FACE VERIFICATION (1:1) stsenariysi.
//
// "Bu odam haqiqatan shu odammi?" — chaqiruvchi `person_id` (ya'ni PNFL) ni
// oldindan biladi va faqat tasdiqlashni so'raydi.
//
// Nima uchun 1:N (Verifier) yetarli emas:
//
//   - 1:N butun baza bo'ylab qidiradi. Baza kattalashgan sari eng yaqin
//     begona odam tasodifan chegaradan o'tib ketishi ehtimoli ortadi
//     (false accept). 1:1 da nomzod bitta — bu xavf umuman yo'q.
//   - Chek/karta/PNFL allaqachon ma'lum bo'lgan oqimda (masalan chiqishda
//     chekni tasdiqlash) 1:N ortiqcha ish va ortiqcha xavf.
//
// Liveness bu yerda ham MAJBURIY va birinchi bajariladi — soxta kadr
// omborga umuman yetib bormaydi.
type Confirmer struct {
	analyzer  domain.FaceAnalyzer
	repo      domain.PersonRepository
	threshold domain.MatchThreshold
	log       *slog.Logger
}

func NewConfirmer(
	analyzer domain.FaceAnalyzer,
	repo domain.PersonRepository,
	threshold domain.MatchThreshold,
	log *slog.Logger,
) *Confirmer {
	return &Confirmer{analyzer: analyzer, repo: repo, threshold: threshold, log: log}
}

// Execute — STEP 1 liveness -> STEP 2 embedding -> STEP 3 bitta yozuv bilan
// solishtirish.
func (c *Confirmer) Execute(
	ctx context.Context,
	personID string,
	image []byte,
	requestID string,
) (*ConfirmResult, error) {
	if personID == "" {
		return nil, domain.NewError(domain.CodeInvalidInput, "`person_id` majburiy.")
	}

	// --- STEP 1 + STEP 2 ---
	face, liveness, err := c.analyzer.VerifyFrame(ctx, image, requestID)
	if err != nil {
		if domain.IsSpoof(err) {
			c.log.Warn("SPOOF aniqlandi — 1:1 solishtirish bekor qilindi",
				"request_id", requestID, "person_id", personID)
		}
		return nil, err
	}

	// --- STEP 3 ---
	match, found, err := c.repo.ScoreAgainst(ctx, face.Embedding, personID)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, domain.NewError(domain.CodePersonNotFound,
			"Bu `person_id` bilan ro'yxatdan o'tgan yuz topilmadi.").
			WithDetails(map[string]any{"person_id": personID}).
			WithLiveness(liveness)
	}

	matched := c.threshold.Accepts(match.Score)
	c.log.Info("confirm (1:1)", "request_id", requestID, "person_id", personID,
		"score", match.Score, "matched", matched)

	person := match.Person
	return &ConfirmResult{
		PersonID:  personID,
		Matched:   matched,
		Score:     match.Score,
		Threshold: float32(c.threshold),
		Person:    &person,
		Liveness:  liveness,
		Face:      *face,
	}, nil
}
