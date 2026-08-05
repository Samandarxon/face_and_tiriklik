package usecase

import (
	"context"
	"strings"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Resolver — `subject_id` (PNFL, tabel raqami) dan `person_id` ni hisoblab
// beradi va yozuv bazada bor-yo'qligini aytadi.
//
// Nima uchun kerak: yuz orqali tanish har doim ham ishlamaydi — yorug'lik
// yomon, odam niqobda, kamera ishlamayapti. Bunday holatda operator odamning
// PNFL sini kiritadi, mijoz shu endpoint orqali `person_id` ni oladi va o'z
// bazasidan yozuvni topadi. Ya'ni yuz — qulay yo'l, `subject_id` — kafolatli
// zaxira yo'l.
//
// Rasm talab qilinmaydi va hech qanday vektor qidiruvi bo'lmaydi: bu sof
// hisoblash + bitta o'qish.
type Resolver struct {
	subjects domain.SubjectResolver
	repo     domain.PersonRepository
}

func NewResolver(subjects domain.SubjectResolver, repo domain.PersonRepository) *Resolver {
	return &Resolver{subjects: subjects, repo: repo}
}

// ResolveResult — hisoblangan ID va uning bazadagi holati.
type ResolveResult struct {
	PersonID   string         `json:"person_id"`
	Registered bool           `json:"registered"`
	Person     *domain.Person `json:"person,omitempty"`
}

func (r *Resolver) Execute(ctx context.Context, subjectID string) (*ResolveResult, error) {
	subjectID = strings.TrimSpace(subjectID)
	if subjectID == "" {
		return nil, domain.NewError(domain.CodeInvalidInput, "`subject_id` majburiy.")
	}
	if len(subjectID) > 128 {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`subject_id` 128 belgidan oshmasligi kerak.")
	}

	personID, err := r.subjects.PersonID(subjectID)
	if err != nil {
		return nil, err
	}

	// Yozuv topilmasligi XATO EMAS: mijoz "hali ro'yxatdan o'tmagan" javobini
	// oladi va uni ro'yxatga olishi mumkin.
	person, err := r.repo.Get(ctx, personID)
	if err != nil {
		return nil, err
	}

	return &ResolveResult{
		PersonID:   personID,
		Registered: person != nil,
		Person:     person,
	}, nil
}
