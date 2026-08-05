package usecase

import (
	"context"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// ============================================================================
//  ENG MUHIM QOIDA: soxta kadrda vektor bazaga so'rov YUBORILMAYDI.
// ============================================================================

func TestVerifyStopsAtSpoofAndNeverSearches(t *testing.T) {
	spoofLiveness := &domain.Liveness{
		IsReal:           false,
		RealProbability:  0.0156,
		SpoofProbability: 0.9844,
		Threshold:        0.75,
		PerModel:         map[string]float64{"v2": 0.0061, "v1se": 0.025},
	}

	analyzer := &fakeAnalyzer{
		err: domain.NewError(domain.CodeSpoofDetected,
			"Soxta yuz/rasm aniqlandi. Tirik inson kameraga qarashi shart!").
			WithLiveness(spoofLiveness),
	}
	repo := newFakeRepo()
	verifier := NewVerifier(analyzer, repo, 0.65, testLogger())

	result, err := verifier.Execute(context.Background(), []byte("frame"), "req-1")

	if result != nil {
		t.Fatal("soxta kadrda natija qaytmasligi kerak")
	}
	if !domain.IsSpoof(err) {
		t.Fatalf("spoof_detected kutilgan, olindi: %v", err)
	}
	if n := repo.searchCalls.Load(); n != 0 {
		t.Fatalf("SPOOF holatida %d ta qidiruv bajarildi — 0 bo'lishi SHART!", n)
	}

	// Liveness ballari mijozga yetib borishi kerak (audit uchun).
	domainErr, _ := domain.AsError(err)
	if domainErr.Liveness == nil || domainErr.Liveness.RealProbability != 0.0156 {
		t.Fatalf("liveness tafsilotlari yo'qoldi: %+v", domainErr.Liveness)
	}
}

// Streaming yo'lida ham himoya bo'lishi kerak: chaqiruvchi xato qilib tirik
// bo'lmagan kadr bilan chaqirsa ham qidiruv bajarilmasligi lozim.
func TestMatchEmbeddingRejectsNonLiveFrame(t *testing.T) {
	repo := newFakeRepo()
	verifier := NewVerifier(&fakeAnalyzer{}, repo, 0.65, testLogger())

	cases := map[string]*domain.Liveness{
		"nil liveness":  nil,
		"is_real=false": {IsReal: false, RealProbability: 0.02},
	}

	for name, liveness := range cases {
		t.Run(name, func(t *testing.T) {
			repo.searchCalls.Store(0)

			_, err := verifier.MatchEmbedding(context.Background(), liveFace(), liveness, "req")
			if !domain.IsSpoof(err) {
				t.Fatalf("spoof_detected kutilgan, olindi: %v", err)
			}
			if n := repo.searchCalls.Load(); n != 0 {
				t.Fatalf("%d ta qidiruv bajarildi — 0 kutilgan", n)
			}
		})
	}
}

// ============================================================================

func TestVerifyReturnsPersonAboveThreshold(t *testing.T) {
	liveness := &domain.Liveness{IsReal: true, RealProbability: 0.9977, Threshold: 0.75}
	analyzer := &fakeAnalyzer{face: liveFace(), liveness: liveness}

	repo := newFakeRepo()
	repo.matches = []domain.Match{
		{Person: domain.Person{ID: "emp-001", Name: "Ali Valiyev"}, Score: 0.81},
		{Person: domain.Person{ID: "emp-002", Name: "Vali Aliyev"}, Score: 0.42},
	}

	verifier := NewVerifier(analyzer, repo, 0.65, testLogger())
	result, err := verifier.Execute(context.Background(), []byte("frame"), "req")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}

	if result.Person.ID != "emp-001" || result.Person.Name != "Ali Valiyev" {
		t.Fatalf("noto'g'ri inson: %+v", result.Person)
	}
	if result.Score != 0.81 || result.Threshold != 0.65 {
		t.Fatalf("score=%v threshold=%v", result.Score, result.Threshold)
	}
	if result.Liveness == nil || !result.Liveness.IsReal {
		t.Fatal("javobda liveness bo'lishi kerak")
	}
	if n := repo.searchCalls.Load(); n != 1 {
		t.Fatalf("tirik yuz uchun 1 ta qidiruv kutilgan, %d bajarildi", n)
	}
}

func TestVerifyReturnsNotFoundBelowThreshold(t *testing.T) {
	liveness := &domain.Liveness{IsReal: true, RealProbability: 0.95}
	analyzer := &fakeAnalyzer{face: liveFace(), liveness: liveness}

	repo := newFakeRepo()
	repo.matches = []domain.Match{
		{Person: domain.Person{ID: "emp-001"}, Score: 0.6499}, // chegaradan pastroq
	}

	verifier := NewVerifier(analyzer, repo, 0.65, testLogger())
	_, err := verifier.Execute(context.Background(), []byte("frame"), "req")

	if domain.CodeOf(err) != domain.CodePersonNotFound {
		t.Fatalf("person_not_found kutilgan, olindi: %v", err)
	}

	domainErr, _ := domain.AsError(err)
	// Chegara sozlashda yordam berishi uchun eng yaqin ball qaytariladi.
	if domainErr.Details["best_score"] != float32(0.6499) {
		t.Fatalf("best_score = %v", domainErr.Details["best_score"])
	}
	// Tirik inson bo'lgani uchun liveness ham qaytadi.
	if domainErr.Liveness == nil || !domainErr.Liveness.IsReal {
		t.Fatal("404 javobida ham liveness bo'lishi kerak")
	}
}

func TestVerifyReturnsNotFoundOnEmptyDatabase(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     liveFace(),
		liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99},
	}
	verifier := NewVerifier(analyzer, newFakeRepo(), 0.65, testLogger())

	if _, err := verifier.Execute(context.Background(), []byte("f"), "req"); domain.CodeOf(err) != domain.CodePersonNotFound {
		t.Fatalf("person_not_found kutilgan, olindi: %v", err)
	}
}

// Chegara chegaraviy qiymatda: score == threshold qabul qilinishi kerak.
func TestThresholdIsInclusive(t *testing.T) {
	threshold := domain.MatchThreshold(0.65)
	if !threshold.Accepts(0.65) {
		t.Error("score == threshold qabul qilinishi kerak")
	}
	if threshold.Accepts(0.6499) {
		t.Error("score < threshold rad etilishi kerak")
	}
}

// --------------------------------------------------------- LivenessProbe

// Probe uchun soxta kadr XATO EMAS — bu tekshiruv endpointi.
func TestLivenessProbeReturnsResultInsteadOfError(t *testing.T) {
	spoof := &domain.Liveness{IsReal: false, RealProbability: 0.0156}
	analyzer := &fakeAnalyzer{
		err: domain.NewError(domain.CodeSpoofDetected, "soxta").WithLiveness(spoof),
	}

	result, err := NewLivenessProbe(analyzer).Execute(context.Background(), []byte("f"), "req")
	if err != nil {
		t.Fatalf("probe xato qaytarmasligi kerak: %v", err)
	}
	if result.Liveness == nil || result.Liveness.IsReal {
		t.Fatalf("is_real=false kutilgan: %+v", result.Liveness)
	}
	if result.Liveness.RealProbability != 0.0156 {
		t.Fatalf("real_probability yo'qoldi: %v", result.Liveness.RealProbability)
	}
}

// Boshqa xatolar (yuz yo'q, servis o'chiq) probe uchun ham xato bo'lib qoladi.
func TestLivenessProbePropagatesRealErrors(t *testing.T) {
	analyzer := &fakeAnalyzer{err: domain.NewError(domain.CodeNoFaceDetected, "yuz yo'q")}

	if _, err := NewLivenessProbe(analyzer).Execute(context.Background(), []byte("f"), "req"); domain.CodeOf(err) != domain.CodeNoFaceDetected {
		t.Fatalf("no_face_detected kutilgan, olindi: %v", err)
	}
}
