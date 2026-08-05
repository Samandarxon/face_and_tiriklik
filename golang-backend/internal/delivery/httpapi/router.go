package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	_ "github.com/face_detecting_microservice/golang-backend/docs" // generated swagger spec
	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// Deps — routerga kerakli barcha bog'liqliklar.
type Deps struct {
	Person  *PersonHandler
	Health  *HealthHandler
	Stream  *StreamHandler
	Admin   *AdminHandler
	Ticket  *TicketHandler
	Auth    *usecase.Authenticator
	Limiter *RateLimiter
	Metrics *Metrics
	Log     *slog.Logger

	MaxMultipartMemory int64
	CORSAllowOrigin    string
	WebDir             string // statik demo interfeys papkasi ("" -> o'chirilgan)
	SwaggerEnabled     bool
}

// NewRouter — barcha marshrutlarni yig'adi.
//
// Xavfsizlik modeli:
//
//	/health                 — ochiq (orkestrator uchun, hech qanday ma'lumot bermaydi)
//	/api/v1/stream          — bir martalik TICKET bilan (brauzer sarlavha yubora olmaydi)
//	qolgan barcha marshrut  — API kalit + aniq scope + chastota cheklovi
//
// Ya'ni kalitsiz kirish mumkin bo'lgan yagona joy — holat tekshiruvi.
func NewRouter(deps Deps) *gin.Engine {
	router := gin.New()
	router.Use(
		recoveryMiddleware(deps.Log),
		requestIDMiddleware(),
		loggingMiddleware(deps.Log),
		corsMiddleware(deps.CORSAllowOrigin),
	)
	router.MaxMultipartMemory = deps.MaxMultipartMemory

	// Holat tekshiruvi ochiq: Docker/Kubernetes probe'lari kalit yubora olmaydi,
	// javobda esa hech qanday nozik ma'lumot yo'q.
	router.GET("/health", deps.Health.Check)
	router.GET("/health/ready", deps.Health.Ready)

	api := router.Group("/api/v1")

	// Real-time oqim — autentifikatsiya ticket orqali, handler ichida.
	api.GET("/stream", deps.Stream.Handle)

	// Qolgan hamma narsa yopiq.
	guarded := api.Group("", authMiddleware(deps.Auth), rateLimitMiddleware(deps.Limiter))
	{
		guarded.POST("/stream/tickets", requireScope(domain.ScopeVerify), deps.Ticket.Issue)

		guarded.POST("/liveness/check", requireScope(domain.ScopeVerify), deps.Person.CheckLiveness)
		guarded.POST("/persons/verify", requireScope(domain.ScopeVerify), deps.Person.Verify)
		// 1:1 tasdiqlash. Yo'l `/persons/:person_id/...` ostida, shuning uchun
		// `/persons/verify` dan OLDIN e'lon qilinmasligi kerak emas — gin
		// statik segmentni parametrdan ustun qo'yadi.
		guarded.POST("/persons/:person_id/verify", requireScope(domain.ScopeVerify), deps.Person.Confirm)
		// 1:N nomzodlar ro'yxati — chegara QO'LLANMAYDI, qaror chaqiruvchida.
		// `read` scope: bu o'qish amali, hech narsani o'zgartirmaydi.
		guarded.POST("/persons/search", requireScope(domain.ScopeRead), deps.Person.Search)

		guarded.POST("/persons/register", requireScope(domain.ScopeRegister), deps.Person.Register)

		// Galereyaga namuna qo'shish — `register` dan farqi shundaki, u
		// asosiy vektorni almashtirmaydi, yoniga qo'shadi. Bir shaxsning
		// turli yosh va rakursdagi namunalari shu yerda to'planadi.
		guarded.POST("/persons/:person_id/samples",
			requireScope(domain.ScopeRegister), deps.Person.AddSample)
		guarded.POST("/persons/resolve", requireScope(domain.ScopeRead), deps.Person.Resolve)
		guarded.GET("/persons", requireScope(domain.ScopeRead), deps.Person.List)
		guarded.GET("/persons/:person_id", requireScope(domain.ScopeRead), deps.Person.Get)
		guarded.DELETE("/persons/:person_id", requireScope(domain.ScopeDelete), deps.Person.Delete)

		// Boshqaruv: mijozlar, kalitlar, jurnal.
		admin := guarded.Group("/admin", requireScope(domain.ScopeAdmin))
		{
			admin.GET("/scopes", deps.Admin.Scopes)
			admin.GET("/audit", deps.Admin.Audit)

			admin.POST("/clients", deps.Admin.CreateClient)
			admin.GET("/clients", deps.Admin.ListClients)
			admin.GET("/clients/:client_id", deps.Admin.GetClient)
			admin.PUT("/clients/:client_id", deps.Admin.UpdateClient)
			admin.DELETE("/clients/:client_id", deps.Admin.DeleteClient)

			admin.POST("/keys", deps.Admin.IssueKey)
			admin.GET("/keys", deps.Admin.ListKeys)
			admin.POST("/keys/:key_id/rotate", deps.Admin.RotateKey)
			admin.DELETE("/keys/:key_id", deps.Admin.RevokeKey)
			// Bekor qilishdan keyingi qadam: yozuvni butunlay o'chirish.
			// Alohida yo'l, chunki bu boshqa amal — biri to'xtatadi, ikkinchisi
			// izni ham yo'q qiladi.
			admin.DELETE("/keys/:key_id/purge", deps.Admin.PurgeKey)
		}
	}

	// Metrikalar ham yopiq: ular kim qancha so'rov yuborayotganini oshkor qiladi.
	// Prometheus konfiguratsiyasida `authorization` sarlavhasini ko'rsating.
	if deps.Metrics != nil {
		router.GET("/metrics",
			authMiddleware(deps.Auth), requireScope(domain.ScopeAdmin),
			gin.WrapH(deps.Metrics.HTTPHandler()))
	}

	// Swagger UI. The spec documents a closed API — reading it grants nothing,
	// and integrators need it before they have a key.
	if deps.SwaggerEnabled {
		router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	if deps.WebDir != "" {
		// no-store — bu demo interfeys ishlab chiqish paytida o'zgarib turadi.
		//
		// Kesh bilan bir marta kuydik: index.html yangilanib, app.js esa
		// brauzer keshida eski bo'lib qoldi. Natijada sahifada yangi tugmalar
		// ko'rindi, lekin ular ortidagi kod eskicha ishladi — WebSocket
		// ticketsiz ulanib, 401 oldi va sabab hech qayerda ko'rinmadi.
		//
		// Bu sahifa kuniga bir necha marta ochiladigan debug vositasi, ya'ni
		// keshdan yutuq deyarli yo'q; noto'g'ri kesh esa soatlab chalkashtiradi.
		noCache := func(c *gin.Context) {
			c.Header("Cache-Control", "no-store, must-revalidate")
			c.Next()
		}
		router.Group("/static", noCache).Static("", deps.WebDir)
		router.GET("/", noCache, func(c *gin.Context) { c.File(deps.WebDir + "/index.html") })
	}

	router.NoRoute(func(c *gin.Context) {
		respondErrorCode(c, domain.CodeInvalidInput, "Bunday endpoint mavjud emas.")
	})

	return router
}

// ================================================================ middleware

func requestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.GetHeader("X-Request-ID")
		if id == "" {
			buf := make([]byte, 6)
			if _, err := rand.Read(buf); err == nil {
				id = hex.EncodeToString(buf)
			}
		}
		c.Set("request_id", id)
		c.Header("X-Request-ID", id)
		c.Next()
	}
}

func loggingMiddleware(log *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()

		// Statik fayllarni logga yozmaymiz.
		if strings.HasPrefix(c.Request.URL.Path, "/static") {
			return
		}
		log.Info("http",
			"request_id", c.GetString("request_id"),
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"status", c.Writer.Status(),
			"duration_ms", time.Since(start).Milliseconds(),
			"client_ip", c.ClientIP())
	}
}

func recoveryMiddleware(log *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("panic", "request_id", c.GetString("request_id"),
					"path", c.Request.URL.Path, "panic", r)
				respondErrorCode(c, domain.CodeInternal, "Ichki server xatosi.")
			}
		}()
		c.Next()
	}
}

// corsMiddleware — demo interfeys va tashqi frontend uchun.
// PRODUCTION: `*` o'rniga aniq domenlarni ko'rsating.
func corsMiddleware(allowed string) gin.HandlerFunc {
	if allowed == "" {
		allowed = "*"
	}
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", allowed)
		c.Header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}
