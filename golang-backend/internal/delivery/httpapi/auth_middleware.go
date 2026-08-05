package httpapi

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// Kontekst kalitlari — sehrli satrlar handler'lar bo'ylab tarqalmasligi uchun.
const (
	ctxAPIKey = "api_key"
)

// APIKeyHeader — kalit yuboriladigan asosiy sarlavha.
const APIKeyHeader = "X-API-Key"

// authMiddleware — har bir yopiq endpoint oldida turadi.
//
// Kalit ikki ko'rinishda qabul qilinadi: `X-API-Key: fdsk_...` yoki
// `Authorization: Bearer fdsk_...`. Query parametrida kalit QABUL QILINMAYDI —
// u brauzer tarixida, proksi loglarida va Referer sarlavhasida qolib ketadi.
// Brauzerdagi WebSocket uchun buning o'rniga bir martalik ticket bor
// (`/stream/tickets`).
func authMiddleware(auth *usecase.Authenticator) gin.HandlerFunc {
	return func(c *gin.Context) {
		raw := extractAPIKey(c)

		key, err := auth.Authenticate(c.Request.Context(), raw)
		if err != nil {
			// 401 da mijozga qanday sxema kutilayotganini aytamiz.
			c.Header("WWW-Authenticate", `Bearer realm="face-api"`)
			respondError(c, err)
			return
		}

		c.Set(ctxAPIKey, key)
		c.Next()
	}
}

// extractAPIKey — so'rovdan xom kalitni ajratib oladi.
func extractAPIKey(c *gin.Context) string {
	if key := strings.TrimSpace(c.GetHeader(APIKeyHeader)); key != "" {
		return key
	}
	authz := strings.TrimSpace(c.GetHeader("Authorization"))
	if after, found := strings.CutPrefix(authz, "Bearer "); found {
		return strings.TrimSpace(after)
	}
	return ""
}

// requireScope — kalitda kerakli huquq borligini tekshiradi.
//
// Autentifikatsiya (kimsan?) va avtorizatsiya (nima qila olasan?) ataylab
// ikki alohida middleware: "faqat tekshirish" uchun berilgan kalit bazaga
// yozishga urinsa, so'rov marshrutgacha yetib bormaydi.
func requireScope(scope domain.Scope) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := apiKeyOf(c)
		if key == nil {
			respondErrorCode(c, domain.CodeUnauthenticated, "API kalit ko'rsatilmagan.")
			return
		}
		if !key.HasScope(scope) {
			respondError(c, domain.NewError(domain.CodeForbidden,
				"Bu amal uchun `"+string(scope)+"` huquqi kerak.").WithDetails(map[string]any{
				"required_scope": string(scope),
				"granted_scopes": domain.JoinScopes(key.Scopes, ","),
			}))
			return
		}
		c.Next()
	}
}

// rateLimitMiddleware — kalit bo'yicha chastota cheklovi.
func rateLimitMiddleware(limiter *RateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := apiKeyOf(c)
		if key == nil {
			c.Next()
			return
		}

		allowed, retryAfter := limiter.Allow(key.ID, key.RateLimitPerMin)
		if !allowed {
			seconds := int(retryAfter.Seconds()) + 1
			c.Header("Retry-After", strconv.Itoa(seconds))
			c.AbortWithStatusJSON(http.StatusTooManyRequests, errorBody{
				Code:    domain.CodeRateLimited,
				Message: "So'rovlar chastotasi oshib ketdi. Biroz kutib qayta urinib ko'ring.",
				Details: map[string]any{"retry_after_seconds": seconds},
			})
			return
		}
		c.Next()
	}
}

// apiKeyOf — kontekstdagi kalit (autentifikatsiyadan keyin doim mavjud).
func apiKeyOf(c *gin.Context) *domain.APIKey {
	value, ok := c.Get(ctxAPIKey)
	if !ok {
		return nil
	}
	key, _ := value.(*domain.APIKey)
	return key
}

// auditContext — audit yozuvi va metrikalar uchun so'rov konteksti.
type auditContext struct {
	ClientID   string
	ClientName string
	KeyID      string
	RequestID  string
	IP         string
	Started    time.Time
}

func auditContextOf(c *gin.Context) auditContext {
	ctx := auditContext{
		RequestID: requestIDOf(c),
		IP:        c.ClientIP(),
		Started:   time.Now(),
	}
	if key := apiKeyOf(c); key != nil {
		ctx.ClientID, ctx.ClientName, ctx.KeyID = key.ClientID, key.ClientName, key.ID
	}
	return ctx
}
