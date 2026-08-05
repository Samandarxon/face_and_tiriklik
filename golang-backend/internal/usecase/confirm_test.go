package usecase

import (
	"context"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// ============================================================================
//  FACE VERIFICATION (1:1) — "bu odam haqiqatan shu odammi?"
// ============================================================================

func confirmerWith(analyzer *fakeAnalyzer, repo *fakeRepo) *Confirmer {
	return NewConfirmer(analyzer, repo, 0.65, testLogger())
}

// 1:N dagi bilan bir xil eng muhim qoida: soxta kadrda ombor UMUMAN
// bezovta qilinmaydi.
func TestConfirmStopsAtSpoofAndNeverSearches(t *testing.T) {
	analyzer := &fakeAnalyzer{
		err: domain.NewError(domain.CodeSpoofDetected, "Soxta yuz.").
			WithLiveness(&domain.Liveness{IsReal: false, RealProbability: 0.01}),
	}
	repo := newFakeRepo()

	result, err := confirmerWith(analyzer, repo).
		Execute(context.Background(), "12345678901234", []byte("frame"), "req-1")

	if result != nil {
		t.Fatal("soxta kadrda natija qaytmasligi kerak")
	}
	if !domain.IsSpoof(err) {
		t.Fatalf("spoof_detected kutilgan, olindi: %v", err)
	}
	if n := repo.searchCalls.Load(); n != 0 {
		t.Fatalf("SPOOF holatida %d ta solishtirish bajarildi — 0 bo'lishi SHART!", n)
	}
}

// Ball chegaradan o'tsa — tasdiqlangan.
func TestConfirmMatchesAboveThreshold(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     liveFace(),
		liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99, Threshold: 0.75},
	}
	repo := newFakeRepo()
	repo.matches = []domain.Match{
		{Person: domain.Person{ID: "12345678901234", Name: "Karimov A."}, Score: 0.82},
	}

	result, err := confirmerWith(analyzer, repo).
		Execute(context.Background(), "12345678901234", []byte("frame"), "req")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if !result.Matched {
		t.Fatalf("0.82 ball 0.65 chegaradan o'tishi kerak edi")
	}
	if result.PersonID != "12345678901234" {
		t.Fatalf("person_id o'zgarib ketdi: %q", result.PersonID)
	}
	if result.Liveness == nil || !result.Liveness.IsReal {
		t.Fatal("liveness natijasi javobda bo'lishi kerak")
	}
}

// Ball past bo'lsa — bu XATO EMAS, javob. 1:N dan asosiy farqi shu:
// chaqiruvchi kimni tekshirayotganini biladi va "yo'q" javobini kutadi.
func TestConfirmReturnsFalseNotErrorBelowThreshold(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     liveFace(),
		liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99},
	}
	repo := newFakeRepo()
	repo.matches = []domain.Match{
		{Person: domain.Person{ID: "12345678901234"}, Score: 0.41},
	}

	result, err := confirmerWith(analyzer, repo).
		Execute(context.Background(), "12345678901234", []byte("frame"), "req")
	if err != nil {
		t.Fatalf("past ball xato bo'lmasligi kerak: %v", err)
	}
	if result.Matched {
		t.Fatal("0.41 ball 0.65 chegaradan o'tmasligi kerak")
	}
	if result.Score != 0.41 {
		t.Fatalf("ball yo'qoldi: %v", result.Score)
	}
}

// Boshqa odamning yuzi ko'rsatilgan yozuv bilan solishtirilmasligi kerak:
// fake faqat so'ralgan person_id ni qaytaradi, ya'ni 1:N ga "sirg'alib"
// o'tish imkoni yo'q.
func TestConfirmOnlyComparesAgainstRequestedPerson(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     liveFace(),
		liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99},
	}
	repo := newFakeRepo()
	repo.matches = []domain.Match{
		{Person: domain.Person{ID: "99999999999999"}, Score: 0.95}, // begona, ball yuqori
		{Person: domain.Person{ID: "12345678901234"}, Score: 0.30}, // so'ralgan odam
	}

	result, err := confirmerWith(analyzer, repo).
		Execute(context.Background(), "12345678901234", []byte("frame"), "req")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if result.Score != 0.30 {
		t.Fatalf("begona yozuvning bali olindi: %v", result.Score)
	}
	if result.Matched {
		t.Fatal("begona odamning yuqori bali tasdiqlashga sabab bo'ldi")
	}
}

// Yozuv umuman yo'q bo'lsa — 404.
func TestConfirmUnknownPersonID(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     liveFace(),
		liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99},
	}

	_, err := confirmerWith(analyzer, newFakeRepo()).
		Execute(context.Background(), "12345678901234", []byte("frame"), "req")

	if domain.CodeOf(err) != domain.CodePersonNotFound {
		t.Fatalf("person_not_found kutilgan, olindi: %v", err)
	}
}

func TestConfirmRequiresPersonID(t *testing.T) {
	_, err := confirmerWith(&fakeAnalyzer{}, newFakeRepo()).
		Execute(context.Background(), "", []byte("frame"), "req")

	if domain.CodeOf(err) != domain.CodeInvalidInput {
		t.Fatalf("invalid_request kutilgan, olindi: %v", err)
	}
}
