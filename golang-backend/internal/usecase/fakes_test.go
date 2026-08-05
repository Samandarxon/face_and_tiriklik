package usecase

import (
	"context"
	"io"
	"log/slog"
	"sync/atomic"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Portlar tufayli use-case'larni HECH QANDAY infratuzilmasiz test qilamiz:
// gRPC ham, Qdrant ham, Docker ham kerak emas.

func testLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func testEmbedding(value float32) domain.Embedding {
	embedding := make(domain.Embedding, 512)
	for i := range embedding {
		embedding[i] = value
	}
	return embedding
}

func liveFace() *domain.Face {
	return &domain.Face{
		BBox:      domain.BoundingBox{X1: 10, Y1: 10, X2: 110, Y2: 110},
		DetScore:  0.91,
		Embedding: testEmbedding(0.0442),
	}
}

// ------------------------------------------------------------- fake analyzer

type fakeAnalyzer struct {
	face     *domain.Face
	liveness *domain.Liveness
	err      error

	embedCalls  atomic.Int32
	verifyCalls atomic.Int32
}

var _ domain.FaceAnalyzer = (*fakeAnalyzer)(nil)

func (f *fakeAnalyzer) Embed(context.Context, []byte, string) (*domain.Face, error) {
	f.embedCalls.Add(1)
	if f.err != nil {
		return nil, f.err
	}
	return f.face, nil
}

func (f *fakeAnalyzer) VerifyFrame(context.Context, []byte, string) (*domain.Face, *domain.Liveness, error) {
	f.verifyCalls.Add(1)
	if f.err != nil {
		return nil, nil, f.err
	}
	return f.face, f.liveness, nil
}

func (f *fakeAnalyzer) CheckLiveness(context.Context, []byte, string) (*domain.Face, *domain.Liveness, error) {
	if f.err != nil {
		return nil, nil, f.err
	}
	return f.face, f.liveness, nil
}

func (f *fakeAnalyzer) OpenStream(context.Context) (domain.FrameStream, error) { return nil, nil }
func (f *fakeAnalyzer) Health(context.Context) (map[string]any, error)         { return nil, nil }

// ----------------------------------------------------------------- fake repo

type fakeRepo struct {
	stored  map[string]domain.Person
	vectors map[string]domain.Embedding
	matches []domain.Match
	saveErr error

	// searchCalls — ENG MUHIM hisoblagich: soxta kadrda 0 bo'lishi shart.
	searchCalls atomic.Int32
	saveCalls   atomic.Int32
}

var _ domain.PersonRepository = (*fakeRepo)(nil)

func newFakeRepo() *fakeRepo {
	return &fakeRepo{
		stored:  make(map[string]domain.Person),
		vectors: make(map[string]domain.Embedding),
	}
}

func (r *fakeRepo) EnsureReady(context.Context) error { return nil }
func (r *fakeRepo) Healthy(context.Context) error     { return nil }

func (r *fakeRepo) Save(_ context.Context, person domain.Person, embedding domain.Embedding) (string, error) {
	r.saveCalls.Add(1)
	if r.saveErr != nil {
		return "", r.saveErr
	}
	r.stored[person.ID] = person
	r.vectors[person.ID] = embedding
	return "point-" + person.ID, nil
}

func (r *fakeRepo) SaveSample(
	_ context.Context, person domain.Person, embedding domain.Embedding, sample string,
) (string, error) {
	r.saveCalls.Add(1)
	if r.saveErr != nil {
		return "", r.saveErr
	}
	key := person.ID + "/" + sample
	r.stored[key] = person
	r.vectors[key] = embedding
	return "point-" + key, nil
}

func (r *fakeRepo) FindNearest(context.Context, domain.Embedding, int) ([]domain.Match, error) {
	r.searchCalls.Add(1)
	return r.matches, nil
}

// ScoreAgainst — 1:1. `matches` dagi ro'yxatdan aynan shu odamni qidiradi,
// shunda test kutilgan ballni oldindan bera oladi.
func (r *fakeRepo) ScoreAgainst(
	_ context.Context, _ domain.Embedding, personID string,
) (*domain.Match, bool, error) {
	r.searchCalls.Add(1)
	for _, m := range r.matches {
		if m.Person.ID == personID {
			return &m, true, nil
		}
	}
	return nil, false, nil
}

func (r *fakeRepo) Get(_ context.Context, personID string) (*domain.Person, error) {
	if person, ok := r.stored[personID]; ok {
		return &person, nil
	}
	return nil, nil
}

func (r *fakeRepo) Delete(_ context.Context, personID string) error {
	delete(r.stored, personID)
	return nil
}

func (r *fakeRepo) List(context.Context, int, string) ([]domain.Person, string, error) {
	people := make([]domain.Person, 0, len(r.stored))
	for _, person := range r.stored {
		people = append(people, person)
	}
	return people, "", nil
}

func (r *fakeRepo) Count(context.Context) (int64, error) { return int64(len(r.stored)), nil }

// ------------------------------------------------------------ fake generator

// fakeSubjects — production bilan BIR XIL semantika: `person_id` bu
// `subject_id` ning o'zi. Test soxta prefiks qo'shmasligi muhim, aks holda
// "PNFL o'zi ID bo'ladi" shartini test tasdiqlamay qo'yardi.
type fakeSubjects struct{}

var _ domain.SubjectResolver = fakeSubjects{}

func (fakeSubjects) PersonID(subjectID string) (string, error) {
	if subjectID == "" {
		return "", domain.NewError(domain.CodeInvalidInput, "`subject_id` bo'sh.")
	}
	// Uzunlik tekshiruvi endi resolver ichida — fake ham xuddi shunday
	// chegara qo'yadi, aks holda test production'da bor himoyani sinamaydi.
	if len(subjectID) > 128 {
		return "", domain.NewError(domain.CodeInvalidInput, "`subject_id` juda uzun.")
	}
	return subjectID, nil
}

type fakeIDs struct{ next int }

var _ domain.IDGenerator = (*fakeIDs)(nil)

func (g *fakeIDs) NewID() (string, error) {
	g.next++
	return "generated-" + string(rune('0'+g.next)), nil
}
