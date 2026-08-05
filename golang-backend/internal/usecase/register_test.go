package usecase

import (
	"context"
	"strings"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

func newRegistrar(analyzer *fakeAnalyzer, repo *fakeRepo) *Registrar {
	// Sifat darvozasi o'chirilgan: bu testlar boshqa xatti-harakatni
	// tekshiradi. Darvozaning o'zi TestEnrollGate* da sinaladi.
	return newRegistrarWithGate(analyzer, repo, EnrollGate{})
}

func newRegistrarWithGate(analyzer *fakeAnalyzer, repo *fakeRepo, gate EnrollGate) *Registrar {
	return NewRegistrar(analyzer, repo, &fakeIDs{}, fakeSubjects{}, 512, gate, testLogger())
}

// person_id yuborilmasa backend uni o'zi generatsiya qilishi kerak.
func TestRegisterGeneratesIDWhenOmitted(t *testing.T) {
	analyzer := &fakeAnalyzer{face: liveFace()}
	repo := newFakeRepo()

	result, err := newRegistrar(analyzer, repo).Execute(context.Background(), RegisterCommand{
		Name:  "Ali Valiyev",
		Image: []byte("frame"),
	})
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}

	if !result.GeneratedID {
		t.Fatal("generated_id=true kutilgan")
	}
	if !strings.HasPrefix(result.Person.ID, "generated-") {
		t.Fatalf("ID generatordan olinmadi: %q", result.Person.ID)
	}
	if result.Updated {
		t.Fatal("yangi ID uchun updated=false bo'lishi kerak")
	}
	if _, ok := repo.stored[result.Person.ID]; !ok {
		t.Fatal("yozuv omborga saqlanmadi")
	}
	if result.Person.CreatedAt.IsZero() {
		t.Fatal("created_at to'ldirilmagan")
	}
}

// Mijoz o'z ID sini bersa — u o'zgartirilmaydi va mavjud yozuv YANGILANADI.
func TestRegisterHonoursClientIDAndUpdates(t *testing.T) {
	analyzer := &fakeAnalyzer{face: liveFace()}
	repo := newFakeRepo()
	registrar := newRegistrar(analyzer, repo)

	first, err := registrar.Execute(context.Background(), RegisterCommand{
		PersonID: "emp-001", Name: "Ali", Image: []byte("f1"),
	})
	if err != nil {
		t.Fatalf("birinchi register: %v", err)
	}
	if first.GeneratedID || first.Updated || first.Person.ID != "emp-001" {
		t.Fatalf("birinchi register kutilmagan: %+v", first)
	}

	second, err := registrar.Execute(context.Background(), RegisterCommand{
		PersonID: "emp-001", Name: "Ali Valiyev", Image: []byte("f2"),
	})
	if err != nil {
		t.Fatalf("ikkinchi register: %v", err)
	}
	if !second.Updated {
		t.Fatal("mavjud ID uchun updated=true kutilgan")
	}
	// Dublikat emas, almashtirish.
	if len(repo.stored) != 1 {
		t.Fatalf("omborda %d ta yozuv — 1 kutilgan (dublikat yaratilgan!)", len(repo.stored))
	}
	// created_at saqlanadi, updated_at qo'shiladi.
	if !second.Person.CreatedAt.Equal(first.Person.CreatedAt) {
		t.Error("created_at o'zgarmasligi kerak")
	}
	if second.Person.UpdatedAt.IsZero() {
		t.Error("updated_at to'ldirilishi kerak")
	}
}

func TestRegisterRejectsInvalidInput(t *testing.T) {
	cases := map[string]RegisterCommand{
		"ism juda uzun":            {Name: strings.Repeat("a", 257), Image: []byte("f")},
		"subject_id juda uzun":     {SubjectID: strings.Repeat("1", 129), Image: []byte("f")},
		"person_id juda uzun":      {PersonID: strings.Repeat("1", 129), Image: []byte("f")},
		"ikkala identifikator ham": {SubjectID: "12345", PersonID: "abc", Image: []byte("f")},
	}

	for name, cmd := range cases {
		t.Run(name, func(t *testing.T) {
			analyzer := &fakeAnalyzer{face: liveFace()}
			registrar := newRegistrar(analyzer, newFakeRepo())

			_, err := registrar.Execute(context.Background(), cmd)
			if domain.CodeOf(err) != domain.CodeInvalidInput {
				t.Fatalf("invalid_request kutilgan, olindi: %v", err)
			}
			// Validatsiya yiqilganda AI servisga so'rov ketmasligi kerak.
			if n := analyzer.embedCalls.Load(); n != 0 {
				t.Fatalf("AI servisga %d ta so'rov yuborildi — 0 kutilgan", n)
			}
		})
	}
}

// `name` ataylab ixtiyoriy: servis bir nechta platformaga xizmat qiladi va
// unga F.I.SH yubormaslik afzal.
func TestRegisterAllowsEmptyName(t *testing.T) {
	analyzer := &fakeAnalyzer{face: liveFace()}
	repo := newFakeRepo()

	result, err := newRegistrar(analyzer, repo).Execute(context.Background(), RegisterCommand{
		SubjectID: "12345678901234",
		Image:     []byte("frame"),
	})
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if result.Person.Name != "" {
		t.Fatalf("bo'sh ism kutilgan, olindi: %q", result.Person.Name)
	}
}

// subject_id berilganda person_id undan hisoblanadi, generator ishlatilmaydi.
func TestRegisterHashesSubjectID(t *testing.T) {
	analyzer := &fakeAnalyzer{face: liveFace()}
	repo := newFakeRepo()

	result, err := newRegistrar(analyzer, repo).Execute(context.Background(), RegisterCommand{
		SubjectID: "12345678901234",
		Image:     []byte("frame"),
	})
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if result.GeneratedID {
		t.Error("subject_id berilganda ID generatsiya qilinmasligi kerak")
	}
	// person_id AYNAN PNFL bo'lishi kerak — servisning ko'chma bo'lishi
	// shunga bog'liq.
	if result.Person.ID != "12345678901234" {
		t.Fatalf("person_id PNFL ga teng emas: %q", result.Person.ID)
	}
	if _, ok := repo.stored["12345678901234"]; !ok {
		t.Fatal("yozuv PNFL bo'yicha saqlanmadi")
	}
}

// Register liveness tekshirmaydi — arxiv fotolaridan ro'yxatga olish uchun.
func TestRegisterDoesNotRunLivenessCheck(t *testing.T) {
	analyzer := &fakeAnalyzer{face: liveFace()}

	if _, err := newRegistrar(analyzer, newFakeRepo()).Execute(context.Background(), RegisterCommand{
		Name: "Ali", Image: []byte("skan"),
	}); err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}

	if analyzer.verifyCalls.Load() != 0 {
		t.Error("register liveness'li yo'lni chaqirmasligi kerak")
	}
	if analyzer.embedCalls.Load() != 1 {
		t.Errorf("Embed 1 marta chaqirilishi kerak, %d marta chaqirildi", analyzer.embedCalls.Load())
	}
}

// AI servis noto'g'ri o'lchamdagi vektor qaytarsa — omborga yozilmasligi kerak.
func TestRegisterRejectsWrongEmbeddingDimension(t *testing.T) {
	analyzer := &fakeAnalyzer{face: &domain.Face{Embedding: testEmbedding(0.1)[:128]}}
	repo := newFakeRepo()

	_, err := newRegistrar(analyzer, repo).Execute(context.Background(), RegisterCommand{
		Name: "Ali", Image: []byte("f"),
	})
	if domain.CodeOf(err) != domain.CodeInternal {
		t.Fatalf("internal_error kutilgan, olindi: %v", err)
	}
	if repo.saveCalls.Load() != 0 {
		t.Fatal("noto'g'ri vektor omborga yozilmasligi kerak")
	}
}
