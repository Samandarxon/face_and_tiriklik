package usecase

import (
	"context"
	"strings"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

func faceWith(det float64, side int) *domain.Face {
	emb := make(domain.Embedding, 512)
	emb[0] = 1
	return &domain.Face{
		DetScore:  det,
		BBox:      domain.BoundingBox{X1: 0, Y1: 0, X2: side, Y2: side},
		Embedding: emb,
	}
}

// Bazaga yozish qidiruvdan qattiqroq talab qiladi.
//
// Farq assimetrik: sifatsiz YOZUV bazada umrbod qoladi va har bir keyingi
// qidiruvga zarar yetkazadi; sifatsiz SO'ROV esa faqat bitta urinishni
// buzadi va qayta urinib ko'rish mumkin.
func TestEnrollGateRejectsLowQuality(t *testing.T) {
	gate := EnrollGate{MinDetScore: 0.70, MinFaceSize: 80}

	cases := []struct {
		name   string
		face   *domain.Face
		reject bool
		reason string
	}{
		{"yaxshi kadr", faceWith(0.85, 120), false, ""},
		{"aynan chegarada", faceWith(0.70, 80), false, ""},
		{"det_score past", faceWith(0.62, 120), true, "det_score"},
		{"yuz kichik", faceWith(0.90, 55), true, "face_size"},
		{"ikkalasi ham yomon", faceWith(0.40, 30), true, "det_score"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := gate.check(tc.face)
			if tc.reject && err == nil {
				t.Fatal("sifatsiz kadr bazaga yozilishga ruxsat oldi")
			}
			if !tc.reject && err != nil {
				t.Fatalf("yaxshi kadr rad etildi: %v", err)
			}
			// Sabab aytilishi shart: "rad etildi" o'zi operatorga nima
			// qilishni aytmaydi, "yuz kichik" esa aytadi.
			if tc.reject && !strings.Contains(err.Error(), "") {
				t.Fatal("sabab yo'q")
			}
		})
	}
}

// Nol qiymat — darvoza yo'q. Mavjud o'rnatmalar o'zgarmasligi uchun.
func TestEnrollGateDisabledByDefault(t *testing.T) {
	if err := (EnrollGate{}).check(faceWith(0.01, 5)); err != nil {
		t.Fatalf("o'chirilgan darvoza rad etdi: %v", err)
	}
}

// Darvoza YOZISHDAN OLDIN ishlashi kerak — keyin tekshirish ma'nosiz,
// yozuv allaqachon bazada bo'lardi.
func TestEnrollGateBlocksBeforeTheWrite(t *testing.T) {
	repo := newFakeRepo()
	analyzer := &fakeAnalyzer{face: faceWith(0.30, 40)}
	r := newRegistrarWithGate(analyzer, repo, EnrollGate{MinDetScore: 0.70, MinFaceSize: 80})

	_, err := r.Execute(context.Background(), RegisterCommand{
		PersonID: "52306006180076",
		Image:    []byte("kadr"),
	})
	if err == nil {
		t.Fatal("sifatsiz kadr yozildi")
	}
	if domain.CodeOf(err) != domain.CodeLowQualityFace {
		t.Fatalf("kod = %v, low_quality_face kutilgan", domain.CodeOf(err))
	}
	if n := repo.saveCalls.Load(); n != 0 {
		t.Fatalf("bazaga %d marta yozildi — darvoza kech ishladi", n)
	}
}
