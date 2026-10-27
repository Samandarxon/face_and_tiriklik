// Composition root — barcha qatlamlar shu yerda bir-biriga ulanadi.
//
// Clean Architecture'da bog'liqliklar faqat shu faylda "simlanadi": domain va
// use-case qatlamlari hech qachon konkret implementatsiyani ko'rmaydi.
package main

//	@title			Face Authentication API
//	@version		2.0
//	@description	Anti-spoofing himoyasiga ega yuz orqali autentifikatsiya servisi.
//	@description
//	@description	Servis YOPIQ: `/health` dan tashqari har bir endpoint API kalit talab qiladi.
//	@description	Kalitlar mijozlarga (platformalarga) beriladi va scope'lar bilan cheklanadi:
//	@description	`face:verify`, `face:register`, `face:read`, `face:delete`, `admin:keys`.
//	@description
//	@description	`person_id` — bu normallashtirilgan PNFL ning O'ZI. Maxfiy kalit yoki hash yo'q:
//	@description	baza ko'chma bo'lishi uchun servis hech qanday sirga bog'lanmaydi. Evaziga
//	@description	vektor bazasi shaxsiy ma'lumot saqlaydi — uni tashqi tarmoqqa chiqarmang.
//
//	@BasePath	/api/v1
//
//	@securityDefinitions.apikey	ApiKeyAuth
//	@in							header
//	@name						X-API-Key
//	@description				Mijozga berilgan kalit: `fdsk_<prefix>_<sir>`

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/adapter/authstore"
	"github.com/face_detecting_microservice/golang-backend/internal/adapter/faceai"
	"github.com/face_detecting_microservice/golang-backend/internal/adapter/security"
	"github.com/face_detecting_microservice/golang-backend/internal/adapter/vectordb"
	"github.com/face_detecting_microservice/golang-backend/internal/config"
	"github.com/face_detecting_microservice/golang-backend/internal/delivery/httpapi"
	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(log)

	if err := run(log); err != nil {
		log.Error("ishga tushirib bo'lmadi", "error", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("konfiguratsiya xatosi: %w", err)
	}

	// --- Kirish nazorati ombori (Postgres) ---
	//
	// Eng birinchi ulanadi: kalitlar bazasi bo'lmasa servis umuman yopiq
	// bo'la olmaydi, ya'ni "ochiq holda ishga tushib ketish" ehtimoli yo'q.
	storeCtx, storeCancel := context.WithTimeout(context.Background(), 30*time.Second)
	store, err := authstore.New(storeCtx, authstore.Config{
		DSN:            cfg.PostgresDSN,
		MaxConns:       int32(cfg.PostgresMaxConns),
		ConnectTimeout: cfg.PostgresConnectTimeout,
	})
	storeCancel()
	if err != nil {
		return err
	}
	defer store.Close()

	migrateCtx, migrateCancel := context.WithTimeout(context.Background(), 60*time.Second)
	err = store.Migrate(migrateCtx)
	migrateCancel()
	if err != nil {
		return fmt.Errorf("migratsiya xatosi: %w", err)
	}
	log.Info("kirish nazorati bazasi tayyor")

	audit := authstore.NewAuditWriter(store, cfg.AuditQueueSize, log)
	defer audit.Close()

	// `person_id` = normallashtirilgan PNFL. Maxfiy kalit yo'q, shuning uchun
	// bu yerda xato ham qaytmaydi.
	subjects := security.NewSubjectResolver(cfg.StrictPNFL)

	// --- Adapterlar (tashqi dunyo) ---
	aiTLS, err := faceai.LoadTLS(cfg.AITLSCAFile, cfg.AITLSCertFile, cfg.AITLSKeyFile, cfg.AITLSServerName)
	if err != nil {
		return err
	}
	if aiTLS == nil {
		log.Warn("AI servis bilan gRPC TLS'siz — faqat yopiq ichki tarmoqda qabul qilinadi",
			"hint", "AI_GRPC_TLS_CA_FILE (va mTLS uchun AI_GRPC_TLS_CERT_FILE/KEY_FILE)")
	}
	analyzer, err := faceai.New(faceai.Config{
		Target:         cfg.AIServiceTarget,
		Timeout:        cfg.AITimeout,
		MaxMessageSize: int(cfg.MaxUploadBytes) + (8 << 20),
		TLS:            aiTLS,
	})
	if err != nil {
		return err
	}
	defer analyzer.Close()

	repo := vectordb.NewQdrant(vectordb.Config{
		URL:        cfg.QdrantURL,
		APIKey:     cfg.QdrantAPIKey,
		Collection: cfg.QdrantCollection,
		VectorSize: cfg.VectorSize,
		Timeout:    cfg.QdrantTimeout,
	})

	if err := waitForRepository(repo, cfg, log); err != nil {
		return err
	}

	// --- Use-case'lar (biznes stsenariylari) ---
	cipher := security.NewCipher()
	registrar := usecase.NewRegistrar(analyzer, repo, vectordb.UUIDGenerator{}, subjects, cfg.VectorSize,
		usecase.EnrollGate{
			MinDetScore: cfg.EnrollDetScore,
			MinFaceSize: cfg.EnrollMinFaceSize,
		}, log)
	verifier := usecase.NewVerifier(analyzer, repo, domain.MatchThreshold(cfg.MatchThreshold), log)
	verifier.SetMargin(cfg.MatchMargin)
	// Aktiv liveness ixtiyoriy: o'chiq bo'lsa nil uzatiladi va oqim avvalgidek
	// faqat passiv tekshiruv bilan ishlaydi.
	var challenges *usecase.ChallengeStore
	if cfg.ActiveLivenessEnabled {
		challenges = usecase.NewChallengeStore()
		defer challenges.Close()
		log.Info("aktiv liveness yoqilgan", "ttl", usecase.ChallengeTTL)
	}

	// 1:1 o'z chegarasi bilan: nomzod bitta va u chaqiruvchi tomonidan
	// hujjatdan olingan, shuning uchun bu yerda qattiqlik xavfsizlikni
	// oshirmaydi — faqat haqiqiy odamni rad etadi.
	confirmThreshold := cfg.ConfirmThreshold
	if confirmThreshold == 0 {
		confirmThreshold = cfg.MatchThreshold - config.ConfirmThresholdGap
	}
	log.Info("tanish chegaralari",
		"match_1n", cfg.MatchThreshold, "confirm_1to1", confirmThreshold)
	confirmer := usecase.NewConfirmer(analyzer, repo, domain.MatchThreshold(confirmThreshold), log)
	probe := usecase.NewLivenessProbe(analyzer)
	directory := usecase.NewDirectory(repo)
	resolver := usecase.NewResolver(subjects, repo)
	keyManager := usecase.NewKeyManager(store, cipher)
	authenticator := usecase.NewAuthenticator(store, cipher, cfg.AuthCacheTTL, cfg.AdminBootstrapKey, log)

	if cfg.AdminBootstrapKey != "" {
		if count, err := store.CountKeys(context.Background()); err == nil && count == 0 {
			log.Warn("BOOTSTRAP REJIMI: bazada faol kalit yo'q. " +
				"ADMIN_BOOTSTRAP_KEY bilan mijoz va kalit yarating, so'ng uni o'chiring.")
		}
	}

	// --- Delivery (HTTP + WebSocket) ---
	metrics := httpapi.NewMetrics()
	tickets := httpapi.NewTicketStore()
	defer tickets.Close()
	limiter := httpapi.NewRateLimiter(cfg.DefaultRateLimitPerMin)
	defer limiter.Close()

	gin.SetMode(cfg.GinMode)
	routerDeps := httpapi.Deps{
		Person: httpapi.NewPersonHandler(
			registrar, verifier, confirmer, probe, directory, resolver,
			audit, metrics, cfg.MaxUploadBytes),
		Health:  httpapi.NewHealthHandler(analyzer, repo, store, cfg.QdrantCollection),
		Stream:  httpapi.NewStreamHandler(analyzer, verifier, tickets, audit, metrics, cfg.CORSAllowOrigin, log, challenges),
		Admin:   httpapi.NewAdminHandler(keyManager, authenticator, store, audit),
		Ticket:  httpapi.NewTicketHandler(tickets),
		Auth:    authenticator,
		Limiter: limiter,
		Log:     log,

		MaxMultipartMemory: cfg.MaxUploadBytes,
		CORSAllowOrigin:    cfg.CORSAllowOrigin,
		WebDir:             cfg.WebDir,
		SwaggerEnabled:     cfg.SwaggerEnabled,
	}
	if cfg.MetricsEnabled {
		routerDeps.Metrics = metrics
	}
	router := httpapi.NewRouter(routerDeps)

	server := &http.Server{
		Addr:        ":" + cfg.Port,
		Handler:     router,
		ReadTimeout: cfg.ReadTimeout,
		// WriteTimeout WebSocket uchun 0 bo'lishi kerak — aks holda uzoq
		// turadigan oqim majburan uziladi. Har bir yozuvda alohida deadline
		// qo'yiladi (`stream_handler.go`).
		WriteTimeout: 0,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Info("server ishga tushdi",
			"port", cfg.Port,
			"ai_service", cfg.AIServiceTarget,
			"ai_transport", "grpc",
			"qdrant", cfg.QdrantURL,
			"collection", cfg.QdrantCollection,
			"vector_size", cfg.VectorSize,
			"match_threshold", cfg.MatchThreshold)

		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server ishga tushmadi", "error", err)
			os.Exit(1)
		}
	}()

	// --- Graceful shutdown ---
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("server to'xtatilmoqda...")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Error("shutdown xatosi", "error", err)
	}
	log.Info("server to'xtadi")
	return nil
}

// waitForRepository — konteynerlar navbatma-navbat ko'tarilishini hisobga oladi.
//
// Faqat "hali ko'tarilmagan" xatolarida qayta urinadi. Autentifikatsiya xatosi
// (401/403) — sozlama xatosi, uni kutish bilan tuzatib bo'lmaydi.
func waitForRepository(repo *vectordb.QdrantRepository, cfg *config.Config, log *slog.Logger) error {
	const attempts = 30
	var lastErr error

	for i := 1; i <= attempts; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), cfg.QdrantTimeout)
		err := repo.EnsureReady(ctx)
		cancel()

		if err == nil {
			log.Info("qdrant tayyor",
				"collection", cfg.QdrantCollection, "size", cfg.VectorSize, "distance", "Cosine")
			return nil
		}

		if errors.Is(err, vectordb.ErrUnauthorized) {
			return fmt.Errorf(
				"Qdrant API kaliti mos kelmadi (%w).\n"+
					"  Backend kaliti: %s\n"+
					"  Tuzatish: .env dagi QDRANT__SERVICE__API_KEY qatorini tekshiring.\n"+
					"  BO'SH qiymat (`QDRANT__SERVICE__API_KEY=`) qoldirmang — Qdrant auth'ni\n"+
					"  yoqadi, lekin hech qanday kalit mos kelmaydi. Kalit kerak bo'lmasa,\n"+
					"  o'sha qatorni butunlay izohga oling, so'ng: docker compose up -d --force-recreate",
				err, describeKey(cfg.QdrantAPIKey))
		}

		lastErr = err
		log.Warn("qdrant kutilmoqda", "attempt", i, "error", err)
		time.Sleep(2 * time.Second)
	}
	return lastErr
}

// describeKey — kalitni logga chiqarmasdan holatini tasvirlaydi.
func describeKey(key string) string {
	if key == "" {
		return "o'rnatilmagan (bo'sh)"
	}
	return fmt.Sprintf("o'rnatilgan (%d belgi)", len(key))
}
