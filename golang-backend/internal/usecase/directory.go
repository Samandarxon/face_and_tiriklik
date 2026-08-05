package usecase

import (
	"context"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Directory — ro'yxatdan o'tganlar ustidagi CRUD stsenariylari.
type Directory struct {
	repo domain.PersonRepository
}

func NewDirectory(repo domain.PersonRepository) *Directory {
	return &Directory{repo: repo}
}

func (d *Directory) Get(ctx context.Context, personID string) (*domain.Person, error) {
	if personID == "" {
		return nil, domain.NewError(domain.CodeInvalidInput, "`person_id` majburiy.")
	}

	person, err := d.repo.Get(ctx, personID)
	if err != nil {
		return nil, err
	}
	if person == nil {
		return nil, domain.NewError(domain.CodePersonNotFound,
			"Bunday person_id bilan inson topilmadi.").
			WithDetails(map[string]any{"person_id": personID})
	}
	return person, nil
}

func (d *Directory) Delete(ctx context.Context, personID string) error {
	// Avval mavjudligini tekshiramiz — aks holda mijoz o'chirilmagan yozuvni
	// o'chirildi deb o'ylaydi (Qdrant yo'q ID uchun ham xato bermaydi).
	if _, err := d.Get(ctx, personID); err != nil {
		return err
	}
	return d.repo.Delete(ctx, personID)
}

type PersonPage struct {
	People     []domain.Person `json:"persons"`
	NextOffset string          `json:"next_offset,omitempty"`
}

func (d *Directory) List(ctx context.Context, limit int, offset string) (*PersonPage, error) {
	people, next, err := d.repo.List(ctx, limit, offset)
	if err != nil {
		return nil, err
	}
	return &PersonPage{People: people, NextOffset: next}, nil
}

func (d *Directory) Count(ctx context.Context) (int64, error) {
	return d.repo.Count(ctx)
}
