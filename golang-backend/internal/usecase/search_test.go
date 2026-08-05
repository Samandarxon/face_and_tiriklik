package usecase

import (
	"context"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

func searchRepo(matches ...domain.Match) *fakeRepo {
	r := newFakeRepo()
	r.matches = matches
	return r
}

// Qidiruv CHEGARA QO'LLAMAYDI.
//
// Bu butun rejimning mohiyati: `Execute` qaror qabul qiladi va chegaradan
// pastini rad etadi, `Search` esa o'lchaydi va hammasini qaytaradi. Agar
// bu yerda ham chegara qo'llansa, past chegarali tergov qidiruvi mavjud
// bo'lolmasdi — bitta global sozlama ikkita qarama-qarshi vazifaga xizmat
// qilishga majbur bo'lardi.
func TestSearchReturnsCandidatesBelowTheThreshold(t *testing.T) {
	repo := searchRepo(
		domain.Match{Person: domain.Person{ID: "aaa"}, Score: 0.58},
		domain.Match{Person: domain.Person{ID: "bbb"}, Score: 0.41},
		domain.Match{Person: domain.Person{ID: "ccc"}, Score: 0.22},
	)
	v := NewVerifier(&fakeAnalyzer{face: faceWith(0.9, 120)}, repo, 0.65, testLogger())

	got, err := v.Search(context.Background(), []byte("kadr"), 10, false, "req")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got.Candidates) != 3 {
		t.Fatalf("%d nomzod qaytdi, 3 kutilgan — chegara qo'llanyapti",
			len(got.Candidates))
	}
	if got.Threshold != 0.65 {
		t.Fatalf("chegara ma'lumot sifatida qaytmadi: %v", got.Threshold)
	}
}

// Arxiv rasmi bo'yicha qidirish qonuniy stsenariy, va arxiv rasmi
// ta'rifga ko'ra "tirik" emas — majburiy liveness bu rejimni yopib
// qo'yardi.
func TestSearchWorksWithoutLiveness(t *testing.T) {
	analyzer := &fakeAnalyzer{face: faceWith(0.9, 120)}
	v := NewVerifier(analyzer, searchRepo(), 0.65, testLogger())

	if _, err := v.Search(context.Background(), []byte("arxiv"), 5, false, "req"); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if analyzer.verifyCalls.Load() != 0 {
		t.Fatal("liveness so'ralmagan bo'lsa ham chaqirildi")
	}
	if analyzer.embedCalls.Load() != 1 {
		t.Fatal("embedding hisoblanmadi")
	}
}

// Kameradan qidirilganda liveness majburiy bo'lishi mumkin.
func TestSearchHonoursRequireLiveness(t *testing.T) {
	analyzer := &fakeAnalyzer{
		face:     faceWith(0.9, 120),
		liveness: &domain.Liveness{IsReal: true},
	}
	v := NewVerifier(analyzer, searchRepo(), 0.65, testLogger())

	if _, err := v.Search(context.Background(), []byte("kadr"), 5, true, "req"); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if analyzer.verifyCalls.Load() != 1 {
		t.Fatal("liveness so'ralgan, lekin tekshirilmadi")
	}
}

func TestSearchClampsTheLimit(t *testing.T) {
	many := make([]domain.Match, 50)
	for i := range many {
		many[i] = domain.Match{Person: domain.Person{ID: string(rune('a' + i%26))}, Score: 0.5}
	}
	v := NewVerifier(&fakeAnalyzer{face: faceWith(0.9, 120)}, searchRepo(many...), 0.65, testLogger())

	got, err := v.Search(context.Background(), []byte("kadr"), 5, false, "req")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got.Candidates) != 5 {
		t.Fatalf("%d qaytdi, 5 so'ralgan edi", len(got.Candidates))
	}

	// 0 yoki manfiy — standart qiymat, bo'sh ro'yxat emas.
	got, _ = v.Search(context.Background(), []byte("kadr"), 0, false, "req")
	if len(got.Candidates) == 0 {
		t.Fatal("limit=0 bo'sh natija berdi")
	}
}
