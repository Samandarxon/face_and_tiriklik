package domain

import (
	"errors"
	"fmt"
)

// Code — barqaror xato kodi. Mijoz AYNAN shu qiymatga tayanadi, matnga emas.
// Delivery qatlami uni HTTP statusiga o'giradi, adapterlar esa gRPC/REST
// xatolarini shu kodlarga keltiradi.
type Code string

const (
	CodeInvalidInput    Code = "invalid_request"
	CodeNoFaceDetected  Code = "no_face_detected"
	CodeMultipleFaces   Code = "multiple_faces"
	CodeLowQualityFace  Code = "low_quality_face"
	CodeSpoofDetected   Code = "spoof_detected"
	CodePersonNotFound  Code = "person_not_found"
	CodeAIUnavailable   Code = "ai_service_unavailable"
	CodeVectorDBFailure Code = "vector_db_error"
	CodeInternal        Code = "internal_error"

	// Kirish nazorati kodlari.

	// CodeUnauthenticated — kalit yo'q, noto'g'ri, bekor qilingan yoki
	// muddati o'tgan (401).
	CodeUnauthenticated Code = "unauthenticated"
	// CodeForbidden — kalit haqiqiy, lekin bu amal uchun scope yo'q (403).
	CodeForbidden Code = "forbidden"
	// CodeRateLimited — kalit uchun belgilangan chastota oshib ketdi (429).
	CodeRateLimited Code = "rate_limited"
	// CodeNotFound — mijoz yoki kalit topilmadi (404).
	CodeNotFound Code = "not_found"
	// CodeConflict — bir xil nomli mijoz allaqachon bor (409).
	CodeConflict Code = "already_exists"
	// CodeStoreFailure — boshqaruv bazasi (Postgres) bilan aloqa uzildi (503).
	CodeStoreFailure Code = "auth_store_error"
)

// Error — domain xatosi.
type Error struct {
	Code    Code
	Message string
	Details map[string]any
	// Liveness — faqat `CodeSpoofDetected` da to'ldiriladi. Mijozga soxta
	// deb topilish sababini ko'rsatish va audit uchun kerak.
	Liveness *Liveness
	cause    error
}

func (e *Error) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error { return e.cause }

// NewError — yangi domain xatosi.
func NewError(code Code, message string) *Error {
	return &Error{Code: code, Message: message}
}

// WrapError — tashqi xatoni domain xatosiga o'raydi (sabab saqlanadi).
func WrapError(code Code, message string, cause error) *Error {
	return &Error{Code: code, Message: message, cause: cause}
}

// WithDetails — zanjirlab ishlatish uchun.
func (e *Error) WithDetails(details map[string]any) *Error {
	e.Details = details
	return e
}

// WithLiveness — spoof xatosiga liveness ballarini biriktiradi.
func (e *Error) WithLiveness(liveness *Liveness) *Error {
	e.Liveness = liveness
	return e
}

// CodeOf — xatodan domain kodini ajratib oladi. Domain xatosi bo'lmasa
// `CodeInternal` qaytadi.
func CodeOf(err error) Code {
	var domainErr *Error
	if errors.As(err, &domainErr) {
		return domainErr.Code
	}
	return CodeInternal
}

// AsError — xatoni `*domain.Error` ga keltiradi (bo'lmasa nil, false).
func AsError(err error) (*Error, bool) {
	var domainErr *Error
	if errors.As(err, &domainErr) {
		return domainErr, true
	}
	return nil, false
}

// IsSpoof — kadr soxta deb topilganini bildiradi. Bu tizimning eng muhim
// xato holati: bunda vektor bazaga so'rov YUBORILMAYDI.
func IsSpoof(err error) bool { return CodeOf(err) == CodeSpoofDetected }
