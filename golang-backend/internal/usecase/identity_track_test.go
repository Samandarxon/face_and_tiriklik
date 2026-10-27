package usecase

import (
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

func unit(dim, hot int) domain.Embedding {
	e := make(domain.Embedding, dim)
	e[hot] = 1
	return e
}

func TestIdentityTrackAnchorsAndAcceptsSamePerson(t *testing.T) {
	tr := NewIdentityTrack(0.4)
	a := unit(8, 0)
	if !tr.Same(a) || !tr.Anchored() {
		t.Fatal("birinchi vektor langar bo'lishi kerak")
	}
	near := domain.Embedding{0.9, 0.3, 0, 0, 0, 0, 0, 0}
	if !tr.Same(near) {
		t.Fatal("o'sha odamning boshqa rakursi rad etildi")
	}
}

// Challenge'ni bir odam bajarib, tanishga boshqasi qo'yilsa — rad.
func TestIdentityTrackRejectsSwappedPerson(t *testing.T) {
	tr := NewIdentityTrack(0.4)
	tr.Same(unit(8, 0))
	if tr.Same(unit(8, 1)) {
		t.Fatal("boshqa odam 'bir xil' deb qabul qilindi")
	}
}

func TestIdentityTrackRejectsEmptyEmbedding(t *testing.T) {
	tr := NewIdentityTrack(0)
	if tr.Same(nil) || tr.Anchored() {
		t.Fatal("bo'sh vektor langar bo'lmasligi kerak")
	}
}
