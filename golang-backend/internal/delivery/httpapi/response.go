// Package httpapi — delivery qatlami: HTTP va WebSocket adapterlari.
//
// Bu qatlamning yagona vazifasi — tashqi so'rovni use-case buyrug'iga o'girish
// va natijani JSON'ga aylantirish. Biznes qarorlari bu yerda QABUL QILINMAYDI.
package httpapi

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// statusByCode — domain kodini HTTP statusiga o'giradi. Yagona joyda saqlanadi,
// shunda handlerlar status kodini takrorlab yurmaydi.
var statusByCode = map[domain.Code]int{
	domain.CodeInvalidInput:    http.StatusBadRequest,
	domain.CodeSpoofDetected:   http.StatusBadRequest,
	domain.CodeNoFaceDetected:  http.StatusUnprocessableEntity,
	domain.CodeMultipleFaces:   http.StatusUnprocessableEntity,
	domain.CodeLowQualityFace:  http.StatusUnprocessableEntity,
	domain.CodePersonNotFound:  http.StatusNotFound,
	domain.CodeAIUnavailable:   http.StatusServiceUnavailable,
	domain.CodeVectorDBFailure: http.StatusServiceUnavailable,
	domain.CodeInternal:        http.StatusInternalServerError,

	// Kirish nazorati kodlari. Ular ro'yxatda YO'Q edi va shuning uchun
	// httpStatus() ning standart tarmog'iga tushib, 500 qaytarardi.
	//
	// Bu shunchaki noto'g'ri raqam emas: mijoz tomon 500 ni "servis yiqildi"
	// deb talqin qiladi va qayta uradi, holbuki muammo kalitda va qayta
	// urinish hech qachon yordam bermaydi. Ulangan tizim esa haqiqiy sababni
	// ("kalit rad etildi") umuman ko'rmasdi.
	domain.CodeUnauthenticated: http.StatusUnauthorized,
	domain.CodeForbidden:       http.StatusForbidden,
	domain.CodeRateLimited:     http.StatusTooManyRequests,
}

func httpStatus(code domain.Code) int {
	if status, ok := statusByCode[code]; ok {
		return status
	}
	return http.StatusInternalServerError
}

// errorBody — mijoz ko'radigan xato formati.
type errorBody struct {
	Success  bool             `json:"success"`
	Code     domain.Code      `json:"code"`
	Message  string           `json:"message"`
	Details  map[string]any   `json:"details,omitempty"`
	Liveness *domain.Liveness `json:"liveness,omitempty"`
}

// respondError — har qanday xatoni yagona formatda qaytaradi.
func respondError(c *gin.Context, err error) {
	domainErr, ok := domain.AsError(err)
	if !ok {
		// Kutilmagan xato — tafsilotni mijozga oshkor qilmaymiz.
		c.AbortWithStatusJSON(http.StatusInternalServerError, errorBody{
			Code:    domain.CodeInternal,
			Message: "Ichki server xatosi.",
		})
		return
	}

	c.AbortWithStatusJSON(httpStatus(domainErr.Code), errorBody{
		Code:     domainErr.Code,
		Message:  domainErr.Message,
		Details:  domainErr.Details,
		Liveness: domainErr.Liveness,
	})
}

func respondErrorCode(c *gin.Context, code domain.Code, message string) {
	respondError(c, domain.NewError(code, message))
}

func respondOK(c *gin.Context, status int, payload gin.H) {
	if payload == nil {
		payload = gin.H{}
	}
	payload["success"] = true
	c.JSON(status, payload)
}
