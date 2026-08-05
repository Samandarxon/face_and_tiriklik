package vectordb

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Haqiqiy Qdrant instansiyasiga qarshi ishlaydi:
//
//	QDRANT_TEST_URL=http://localhost:6333 go test ./internal/adapter/vectordb -v
//
// O'zgaruvchi berilmasa testlar o'tkazib yuboriladi.
func testRepo(t *testing.T) *QdrantRepository {
	t.Helper()
	url := os.Getenv("QDRANT_TEST_URL")
	if url == "" {
		t.Skip("QDRANT_TEST_URL berilmagan — integratsion test o'tkazib yuborildi")
	}

	collection := "faces_test_" + time.Now().Format("150405.000000")
	repo := NewQdrant(Config{
		URL:        url,
		APIKey:     os.Getenv("QDRANT_TEST_API_KEY"),
		Collection: sanitize(collection),
		VectorSize: 512,
		Timeout:    15 * time.Second,
	})

	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = repo.do(ctx, "DELETE", "/collections/"+repo.collection, nil, nil)
	})
	return repo
}

func sanitize(name string) string {
	out := make([]rune, 0, len(name))
	for _, r := range name {
		if r == '.' {
			continue
		}
		out = append(out, r)
	}
	return string(out)
}

// randomEmbedding — InsightFace kabi L2-normalizatsiyalangan vektor.
func randomEmbedding(seed int64) domain.Embedding {
	rng := rand.New(rand.NewSource(seed))
	v := make(domain.Embedding, 512)
	var sum float64
	for i := range v {
		v[i] = float32(rng.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	norm := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= norm
	}
	return v
}

// nudge — har bir komponentga sigma=amount shovqin qo'shadi.
// 512 o'lchamda cosine ≈ 1/sqrt(1 + amount²·512), ya'ni 0.03 ≈ 0.83 score.
func nudge(v domain.Embedding, amount float32, seed int64) domain.Embedding {
	rng := rand.New(rand.NewSource(seed))
	out := make(domain.Embedding, len(v))
	var sum float64
	for i := range v {
		out[i] = v[i] + amount*float32(rng.NormFloat64())
		sum += float64(out[i]) * float64(out[i])
	}
	norm := float32(math.Sqrt(sum))
	for i := range out {
		out[i] /= norm
	}
	return out
}

func TestIntegrationFullLifecycle(t *testing.T) {
	repo := testRepo(t)
	ctx := context.Background()

	// --- Kollektsiya yaratish (512-dim, Cosine) + idempotentlik ---
	if err := repo.EnsureReady(ctx); err != nil {
		t.Fatalf("EnsureReady: %v", err)
	}
	if err := repo.EnsureReady(ctx); err != nil {
		t.Fatalf("takroriy EnsureReady xato berdi: %v", err)
	}

	// --- Save ---
	ali := randomEmbedding(1)
	created := time.Now().UTC().Truncate(time.Second)

	if _, err := repo.Save(ctx, domain.Person{
		ID: "emp-001", Name: "Ali Valiyev", CreatedAt: created, DetScore: 0.91,
	}, ali); err != nil {
		t.Fatalf("Save(emp-001): %v", err)
	}
	if _, err := repo.Save(ctx, domain.Person{
		ID: "emp-002", Name: "Vali Aliyev", CreatedAt: created,
	}, randomEmbedding(2)); err != nil {
		t.Fatalf("Save(emp-002): %v", err)
	}

	if count, err := repo.Count(ctx); err != nil || count != 2 {
		t.Fatalf("Count = %d, err=%v; kutilgan 2", count, err)
	}

	// --- Get ---
	person, err := repo.Get(ctx, "emp-001")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if person == nil || person.Name != "Ali Valiyev" || person.DetScore != 0.91 {
		t.Fatalf("Get(emp-001) = %+v", person)
	}
	if !person.CreatedAt.Equal(created) {
		t.Fatalf("created_at aylanishda buzildi: %v != %v", person.CreatedAt, created)
	}
	if missing, err := repo.Get(ctx, "yoq-odam"); err != nil || missing != nil {
		t.Fatalf("mavjud bo'lmagan odam uchun (nil, nil) kutilgan, olindi (%+v, %v)", missing, err)
	}

	// --- FindNearest: aynan o'sha vektor ~1.0 score berishi kerak ---
	matches, err := repo.FindNearest(ctx, ali, 3)
	if err != nil {
		t.Fatalf("FindNearest: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("2 ta natija kutilgan, olindi %d", len(matches))
	}
	if matches[0].Person.ID != "emp-001" || matches[0].Score < 0.999 {
		t.Fatalf("eng yaqin: %q score=%.4f", matches[0].Person.ID, matches[0].Score)
	}
	// Begona vektor past ball berishi kerak (tasodifiy 512-dim vektorlar ~0).
	if matches[1].Score > 0.3 {
		t.Fatalf("begona vektor score %.4f — juda yuqori", matches[1].Score)
	}

	// --- "O'sha odamning boshqa kadri" chegaradan o'tishi kerak ---
	matches, err = repo.FindNearest(ctx, nudge(ali, 0.03, 42), 1)
	if err != nil {
		t.Fatalf("FindNearest(similar): %v", err)
	}
	if matches[0].Person.ID != "emp-001" || matches[0].Score < 0.65 {
		t.Fatalf("o'xshash kadr: %q %.4f", matches[0].Person.ID, matches[0].Score)
	}

	// --- Qayta saqlash dublikat yaratmasligi kerak ---
	if _, err := repo.Save(ctx, domain.Person{
		ID: "emp-001", Name: "Ali Valiyev", CreatedAt: created, UpdatedAt: time.Now().UTC(),
	}, nudge(ali, 0.02, 7)); err != nil {
		t.Fatalf("qayta Save: %v", err)
	}
	if count, err := repo.Count(ctx); err != nil || count != 2 {
		t.Fatalf("qayta save'dan keyin Count = %d, kutilgan 2 (dublikat yaratilgan!)", count)
	}

	// --- List ---
	people, _, err := repo.List(ctx, 50, "")
	if err != nil || len(people) != 2 {
		t.Fatalf("List = %d ta, err=%v; kutilgan 2", len(people), err)
	}

	// --- Delete ---
	if err := repo.Delete(ctx, "emp-002"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if gone, err := repo.Get(ctx, "emp-002"); err != nil || gone != nil {
		t.Fatalf("o'chirilgandan keyin (nil, nil) kutilgan, olindi (%+v, %v)", gone, err)
	}
	if count, err := repo.Count(ctx); err != nil || count != 1 {
		t.Fatalf("o'chirilgandan keyin Count = %d, kutilgan 1", count)
	}
}

// API kaliti yoqilgan Qdrant'da kalitsiz mijoz `ErrUnauthorized` olishi kerak —
// bu "hali tayyor emas" dan farq qiladi va qayta urinish befoyda.
func TestIntegrationUnauthorizedIsDistinguishable(t *testing.T) {
	url := os.Getenv("QDRANT_AUTH_TEST_URL")
	key := os.Getenv("QDRANT_AUTH_TEST_KEY")
	if url == "" || key == "" {
		t.Skip("QDRANT_AUTH_TEST_URL / QDRANT_AUTH_TEST_KEY berilmagan")
	}

	ctx := context.Background()
	collection := "faces_auth_test_" + time.Now().Format("150405")

	noKey := NewQdrant(Config{URL: url, Collection: collection, VectorSize: 512, Timeout: 10 * time.Second})
	err := noKey.EnsureReady(ctx)
	if err == nil {
		t.Fatal("kalitsiz so'rov muvaffaqiyatli bo'ldi — auth ishlamayapti")
	}
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("ErrUnauthorized kutilgan, olindi: %v", err)
	}

	withKey := NewQdrant(Config{URL: url, APIKey: key, Collection: collection,
		VectorSize: 512, Timeout: 10 * time.Second})
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = withKey.do(c, "DELETE", "/collections/"+collection, nil, nil)
	})

	if err := withKey.EnsureReady(ctx); err != nil {
		t.Fatalf("to'g'ri kalit bilan ham xato: %v", err)
	}
	if _, err := withKey.Save(ctx, domain.Person{ID: "emp-001", Name: "Ali"}, randomEmbedding(1)); err != nil {
		t.Fatalf("kalit bilan Save: %v", err)
	}
}

// Noto'g'ri o'lchamli vektor Qdrant'ga yetib bormasdan rad etilishi kerak.
func TestRejectsWrongDimension(t *testing.T) {
	repo := NewQdrant(Config{URL: "http://invalid.local", Collection: "faces",
		VectorSize: 512, Timeout: time.Second})

	if _, err := repo.Save(context.Background(), domain.Person{ID: "x"}, make(domain.Embedding, 128)); err == nil {
		t.Error("128 o'lchamli vektor rad etilishi kerak edi")
	}
	if _, err := repo.FindNearest(context.Background(), make(domain.Embedding, 256), 1); err == nil {
		t.Error("256 o'lchamli vektor rad etilishi kerak edi")
	}
}
