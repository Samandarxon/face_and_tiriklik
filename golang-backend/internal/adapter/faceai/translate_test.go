package faceai

import (
	"errors"
	"math"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	facev1 "github.com/face_detecting_microservice/golang-backend/genproto/face/v1"
	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// gRPC xatolari domain kodlariga to'g'ri o'girilishi kerak — butun delivery
// qatlami AYNAN shu kodlarga tayanadi.
func TestTranslateErrorMapsFaceErrorDetails(t *testing.T) {
	cases := []struct {
		name     string
		proto    facev1.ErrorCode
		grpcCode codes.Code
		want     domain.Code
	}{
		{"spoof", facev1.ErrorCode_ERROR_CODE_SPOOF_DETECTED, codes.PermissionDenied, domain.CodeSpoofDetected},
		{"yuz yo'q", facev1.ErrorCode_ERROR_CODE_NO_FACE_DETECTED, codes.FailedPrecondition, domain.CodeNoFaceDetected},
		{"ko'p yuz", facev1.ErrorCode_ERROR_CODE_MULTIPLE_FACES, codes.FailedPrecondition, domain.CodeMultipleFaces},
		{"past sifat", facev1.ErrorCode_ERROR_CODE_LOW_QUALITY_FACE, codes.FailedPrecondition, domain.CodeLowQualityFace},
		{"noto'g'ri rasm", facev1.ErrorCode_ERROR_CODE_INVALID_IMAGE, codes.InvalidArgument, domain.CodeInvalidInput},
		{"liveness yo'q", facev1.ErrorCode_ERROR_CODE_LIVENESS_UNAVAILABLE, codes.Unavailable, domain.CodeAIUnavailable},
		{"model tayyor emas", facev1.ErrorCode_ERROR_CODE_MODEL_NOT_READY, codes.Unavailable, domain.CodeAIUnavailable},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st, err := status.New(tc.grpcCode, "xato").WithDetails(&facev1.FaceError{
				Code:    tc.proto,
				Message: "test xabari",
			})
			if err != nil {
				t.Fatalf("WithDetails: %v", err)
			}

			got := translateError(st.Err())
			if domain.CodeOf(got) != tc.want {
				t.Fatalf("kod = %v, kutilgan %v", domain.CodeOf(got), tc.want)
			}

			domainErr, _ := domain.AsError(got)
			if domainErr.Message != "test xabari" {
				t.Fatalf("xabar yo'qoldi: %q", domainErr.Message)
			}
		})
	}
}

// Spoof xatosida liveness ballari ham o'tishi kerak — ularsiz mijoz nima
// uchun rad etilganini ko'rsata olmaydi.
func TestTranslateErrorCarriesLivenessScores(t *testing.T) {
	st, err := status.New(codes.PermissionDenied, "soxta").WithDetails(&facev1.FaceError{
		Code:    facev1.ErrorCode_ERROR_CODE_SPOOF_DETECTED,
		Message: "Soxta yuz/rasm aniqlandi.",
		Details: map[string]string{"hint": "telefon ekrani"},
		Liveness: &facev1.Liveness{
			IsReal:           false,
			RealProbability:  0.0156,
			SpoofProbability: 0.9844,
			Threshold:        0.75,
			PerModel:         map[string]float32{"v2": 0.0061, "v1se": 0.025},
		},
	})
	if err != nil {
		t.Fatalf("WithDetails: %v", err)
	}

	translated := translateError(st.Err())
	if !domain.IsSpoof(translated) {
		t.Fatalf("spoof kutilgan: %v", translated)
	}

	domainErr, _ := domain.AsError(translated)
	if domainErr.Liveness == nil {
		t.Fatal("liveness yo'qoldi")
	}
	if domainErr.Liveness.IsReal {
		t.Error("is_real=false kutilgan")
	}
	// proto float32 -> domain float64: aniq tenglik emas, tolerantlik bilan.
	if math.Abs(domainErr.Liveness.RealProbability-0.0156) > 1e-6 {
		t.Errorf("real_probability = %v, kutilgan ~0.0156", domainErr.Liveness.RealProbability)
	}
	if len(domainErr.Liveness.PerModel) != 2 {
		t.Errorf("per_model = %v", domainErr.Liveness.PerModel)
	}
	if domainErr.Details["hint"] != "telefon ekrani" {
		t.Errorf("details = %v", domainErr.Details)
	}
}

// Detalsiz gRPC xatosi — transport muammosi, biznes xatosi emas.
func TestTranslateErrorWithoutDetailsIsInfrastructure(t *testing.T) {
	cases := map[codes.Code]domain.Code{
		codes.Unavailable:       domain.CodeAIUnavailable,
		codes.DeadlineExceeded:  domain.CodeAIUnavailable,
		codes.Canceled:          domain.CodeAIUnavailable,
		codes.ResourceExhausted: domain.CodeInvalidInput,
		codes.Unknown:           domain.CodeInternal,
	}

	for grpcCode, want := range cases {
		t.Run(grpcCode.String(), func(t *testing.T) {
			got := translateError(status.Error(grpcCode, "boom"))
			if domain.CodeOf(got) != want {
				t.Fatalf("kod = %v, kutilgan %v", domain.CodeOf(got), want)
			}
		})
	}
}

func TestTranslateErrorPassesThroughNil(t *testing.T) {
	if translateError(nil) != nil {
		t.Fatal("nil xato nil qolishi kerak")
	}
}

// gRPC bo'lmagan xato ham domain xatosiga aylanishi kerak (panic emas).
func TestTranslateErrorHandlesPlainError(t *testing.T) {
	got := translateError(errors.New("tarmoq yiqildi"))
	if domain.CodeOf(got) != domain.CodeInternal && domain.CodeOf(got) != domain.CodeAIUnavailable {
		t.Fatalf("kutilmagan kod: %v", domain.CodeOf(got))
	}
}
