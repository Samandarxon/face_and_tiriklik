// Package config — environment o'zgaruvchilaridan sozlamalarni o'qiydi.
//
// Faqat composition root (`cmd/server`) shu paketni ishlatadi; use-case va
// domain qatlamlari sozlamalarni parametr sifatida oladi.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// ConfirmThresholdGap — 1:1 chegarasi 1:N dan qancha past turadi.
//
// `CONFIRM_THRESHOLD` sozlanmagan bo'lsa shu ayirma ishlatiladi. Aniq son
// emas, ayirma bo'lgani muhim: 1:1 HAR DOIM 1:N dan yumshoqroq bo'lishi
// kerak, `MATCH_THRESHOLD` qanday qo'yilganidan qat'i nazar.
//
// 0.10 — arxiv fotosining bir necha yillik bo'lishi odatda olib keladigan
// yo'qotish. Undan kattasi 1:1 ni shunchalik yumshatardiki, chegaraning
// o'zi ma'nosini yo'qotardi.
const ConfirmThresholdGap float32 = 0.10

type Config struct {
	// Server
	Port         string
	GinMode      string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	WebDir       string

	// Python AI service (gRPC)
	AIServiceTarget string
	AITimeout       time.Duration
	// AI gRPC TLS (ixtiyoriy). CA bo'sh — TLS o'chiq. Qarang: faceai.LoadTLS.
	AITLSCAFile     string
	AITLSCertFile   string
	AITLSKeyFile    string
	AITLSServerName string

	// Qdrant
	QdrantURL        string
	QdrantAPIKey     string
	QdrantCollection string
	VectorSize       int
	QdrantTimeout    time.Duration

	// Biznes qoidalari
	MatchThreshold float32
	// MatchMargin — 1:N da 1-o'rin 2-o'rindan kamida shuncha oldinda
	// bo'lmasa natija noaniq deb rad etiladi (MATCH_MARGIN). 0 — o'chiq.
	MatchMargin float32

	// ConfirmThreshold — 1:1 tasdiqlash (`/persons/{id}/verify`) uchun
	// ALOHIDA chegara.
	//
	// # Nima uchun 1:N bilan bir xil bo'lmasligi kerak
	//
	// Ikkalasi bir xil ballni o'lchaydi, lekin butunlay boshqa xavfni
	// boshqaradi:
	//
	//   1:N — nomzod butun baza. Chegara pasaysa, HAR BIR begona yozuv
	//         tasodifan birinchi o'ringa chiqish imkoniga ega bo'ladi, va
	//         bu ehtimol baza kattalashgani sari ortadi. Shuning uchun
	//         qattiq bo'lishi shart.
	//
	//   1:1 — nomzod bitta, va u chaqiruvchi tomonidan HUJJATDAN olingan.
	//         Begona bilan tasodifan chalkashish xavfi yo'q. Bu yerda
	//         qattiq chegara faqat bitta narsani beradi: haqiqiy odamni
	//         rad etish (false reject) — masalan arxiv fotosi bir necha
	//         yillik bo'lgani uchun.
	//
	// Ya'ni 1:N da qattiqlik xavfsizlikni oshiradi, 1:1 da esa faqat
	// xizmatni buzadi.
	//
	// # Nima uchun standart qiymat ALOHIDA SON EMAS
	//
	// Sozlanmagan bo'lsa `MatchThreshold - ConfirmThresholdGap` ishlatiladi,
	// ya'ni qiymat 1:N ga NISBATAN belgilanadi.
	//
	// Aniq son (masalan 0.55) yozilsa, u faqat bitta `MATCH_THRESHOLD` uchun
	// to'g'ri bo'lardi. 0.65 da u yumshoqroq, 0.50 da esa QATTIQROQ bo'lib
	// qolardi — ya'ni butun mantiq teskarisiga aylanardi va buni hech kim
	// sezmasdi, chunki ikkala son ham "oqilona" ko'rinadi.
	ConfirmThreshold float32

	// EnrollDetScore, EnrollMinFaceSize — faqat ro'yxatga olish uchun.
	// 0 bo'lsa tekshiruv o'tkazilmaydi (eski xatti-harakat).
	EnrollDetScore    float32
	EnrollMinFaceSize int
	MaxUploadBytes    int64

	// Xavfsizlik
	CORSAllowOrigin string

	// Kirish nazorati ombori (Postgres).
	PostgresDSN            string
	PostgresMaxConns       int
	PostgresConnectTimeout time.Duration

	// StrictPNFL — `subject_id` faqat 14 xonali PNFL bo'lishi shartmi.
	//
	// Standart holatda TRUE: `person_id` endi PNFL ning o'zi, va aralash
	// formatlar (tabel raqami, ichki kod) bazani "bir odam — bir ID" qoidasidan
	// mahrum qiladi. Boshqa formatdagi identifikatorlar bilan ishlash kerak
	// bo'lgan alohida o'rnatmalarda `SUBJECT_ID_STRICT_PNFL=false` qilinadi.
	StrictPNFL bool

	// AdminBootstrapKey — birinchi mijoz va kalitni yaratish uchun vaqtinchalik
	// kalit. Bazada faol kalit paydo bo'lishi bilan avtomatik ishlamay qoladi.
	AdminBootstrapKey string

	// AuthCacheTTL — tekshirilgan kalit shuncha vaqt keshda turadi.
	AuthCacheTTL time.Duration

	// DefaultRateLimitPerMin — kalitda o'z chegarasi ko'rsatilmaganda.
	// 0 — cheklov yo'q.
	DefaultRateLimitPerMin int

	// AuditQueueSize — audit navbatining hajmi.
	AuditQueueSize int

	// MetricsEnabled — `/metrics` endpointi (admin scope bilan himoyalangan).
	MetricsEnabled bool

	// SwaggerEnabled — `/swagger/index.html` interfeysi.
	SwaggerEnabled bool

	// ActiveLivenessEnabled — server buyruq beradigan aktiv tekshiruv.
	//
	// Passiv anti-spoofing (MiniFASNet) bosma surat va ekranni to'sadi, lekin
	// OLDINDAN YOZILGAN videoni emas — u haqiqiy yuzdan olingan. Aktiv
	// tekshiruv har safar tasodifiy buyruq berib shu bo'shliqni yopadi.
	//
	// Standart holatda O'CHIQ: u foydalanuvchidan qo'shimcha harakat talab
	// qiladi va kiosk oqimini sekinlashtiradi, shuning uchun qaror
	// operatorniki bo'lishi kerak. Yoqilganda tanish MAJBURIY ravishda
	// buyruq bajarilishini kutadi.
	ActiveLivenessEnabled bool
}

func Load() (*Config, error) {
	cfg := &Config{
		Port:         env("PORT", "8080"),
		GinMode:      env("GIN_MODE", "release"),
		ReadTimeout:  envDuration("READ_TIMEOUT", 30*time.Second),
		WriteTimeout: envDuration("WRITE_TIMEOUT", 60*time.Second),
		WebDir:       env("WEB_DIR", "./web"),

		// gRPC manzili: "host:port" (sxemasiz).
		AIServiceTarget: normalizeTarget(env("AI_SERVICE_TARGET", "ai-service:50051")),
		AITimeout:       envDuration("AI_TIMEOUT", 30*time.Second),
		AITLSCAFile:     env("AI_GRPC_TLS_CA_FILE", ""),
		AITLSCertFile:   env("AI_GRPC_TLS_CERT_FILE", ""),
		AITLSKeyFile:    env("AI_GRPC_TLS_KEY_FILE", ""),
		AITLSServerName: env("AI_GRPC_TLS_SERVER_NAME", ""),

		QdrantURL:        strings.TrimRight(env("QDRANT_URL", "http://qdrant:6333"), "/"),
		QdrantAPIKey:     env("QDRANT_API_KEY", ""),
		QdrantCollection: env("QDRANT_COLLECTION", "faces"),
		VectorSize:       envInt("VECTOR_SIZE", 512),
		QdrantTimeout:    envDuration("QDRANT_TIMEOUT", 15*time.Second),

		MatchThreshold: float32(envFloat("MATCH_THRESHOLD", 0.65)),
		// 0.05 — buffalo_l kosinus ballida yaqin "egizak" holatlarni
		// ajratadigan ehtiyotkor qiymat. Haqiqiy FAR/FRR bazada o'lchanib
		// sozlanishi kerak.
		MatchMargin: float32(envFloat("MATCH_MARGIN", 0.05)),

		// 0 = sozlanmagan. Aniq son emas: standart qiymat `MatchThreshold`
		// ga NISBATAN hisoblanadi — sababi maydon izohida.
		ConfirmThreshold: float32(envFloat("CONFIRM_THRESHOLD", 0)),

		// Bazaga YOZISH uchun qattiqroq talab.
		//
		// Qidiruvda bizda tanlov yo'q — kamerada nima bo'lsa shu. Yozishda
		// esa bor: sifatsiz yozuv bazada UMRBOD qoladi va har bir keyingi
		// qidiruvga zarar yetkazadi. Shuning uchun ikki xil chegara.
		EnrollDetScore:    float32(envFloat("ENROLL_DET_SCORE", 0.70)),
		EnrollMinFaceSize: envInt("ENROLL_MIN_FACE_SIZE", 80),
		MaxUploadBytes:    int64(envInt("MAX_UPLOAD_BYTES", 10*1024*1024)),

		CORSAllowOrigin: env("CORS_ALLOW_ORIGIN", "*"),

		PostgresDSN:            postgresDSN(),
		PostgresMaxConns:       envInt("POSTGRES_MAX_CONNS", 10),
		PostgresConnectTimeout: envDuration("POSTGRES_CONNECT_TIMEOUT", 10*time.Second),

		StrictPNFL:        envBool("SUBJECT_ID_STRICT_PNFL", true),
		AdminBootstrapKey: env("ADMIN_BOOTSTRAP_KEY", ""),

		AuthCacheTTL:           envDuration("AUTH_CACHE_TTL", 30*time.Second),
		DefaultRateLimitPerMin: envInt("DEFAULT_RATE_LIMIT_PER_MIN", 600),
		AuditQueueSize:         envInt("AUDIT_QUEUE_SIZE", 2048),
		MetricsEnabled:         envBool("METRICS_ENABLED", true),
		SwaggerEnabled:         envBool("SWAGGER_ENABLED", true),
		// Standart yoqiq: yozib olingan videoni faqat aktiv challenge to'sadi.
		ActiveLivenessEnabled: envBool("ACTIVE_LIVENESS_ENABLED", true),
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	if c.VectorSize <= 0 {
		return fmt.Errorf("VECTOR_SIZE musbat bo'lishi kerak, olindi: %d", c.VectorSize)
	}
	if c.MatchThreshold < -1 || c.MatchThreshold > 1 {
		return fmt.Errorf("MATCH_THRESHOLD [-1..1] oralig'ida bo'lishi kerak, olindi: %f", c.MatchThreshold)
	}
	if c.ConfirmThreshold < -1 || c.ConfirmThreshold > 1 {
		return fmt.Errorf("CONFIRM_THRESHOLD [-1..1] oralig'ida bo'lishi kerak, olindi: %f", c.ConfirmThreshold)
	}
	if c.EnrollDetScore < 0 || c.EnrollDetScore > 1 {
		return fmt.Errorf("ENROLL_DET_SCORE [0..1] oralig'ida bo'lishi kerak, olindi: %f", c.EnrollDetScore)
	}
	if c.EnrollMinFaceSize < 0 {
		return fmt.Errorf("ENROLL_MIN_FACE_SIZE manfiy bo'lishi mumkin emas")
	}
	if c.QdrantCollection == "" {
		return fmt.Errorf("QDRANT_COLLECTION bo'sh bo'lishi mumkin emas")
	}
	if c.AIServiceTarget == "" {
		return fmt.Errorf("AI_SERVICE_TARGET bo'sh bo'lishi mumkin emas")
	}

	// Quyidagi ikkitasi xavfsizlik uchun kritik, shuning uchun standart qiymat
	// BERILMAYDI: e'tibordan chetda qolgan sozlama sukut bo'yicha zaif tizim
	// yaratib qo'ymasligi kerak.
	if c.PostgresDSN == "" {
		return fmt.Errorf(
			"POSTGRES_DSN (yoki POSTGRES_HOST/USER/PASSWORD/DB) ko'rsatilishi shart — " +
				"API kalitlar shu bazada saqlanadi")
	}
	// FACE_ID_PEPPER ataylab tekshirilmaydi va umuman o'qilmaydi: `person_id`
	// endi PNFL ning o'zi, hech qanday maxfiy kalit ishtirok etmaydi. Eski
	// o'rnatmalarda bu o'zgaruvchi .env da qolib ketgan bo'lishi mumkin — u
	// shunchaki e'tiborsiz qoldiriladi, xato berilmaydi.
	if c.AdminBootstrapKey != "" && len(c.AdminBootstrapKey) < 32 {
		return fmt.Errorf("ADMIN_BOOTSTRAP_KEY kamida 32 belgi bo'lishi kerak")
	}

	return nil
}

// postgresDSN — to'liq DSN yoki alohida POSTGRES_* o'zgaruvchilari.
//
// Ikki ko'rinish ham qo'llab-quvvatlanadi: bulutli muhitlar odatda tayyor
// `DATABASE_URL` beradi, docker-compose'da esa alohida o'zgaruvchilar qulayroq.
func postgresDSN() string {
	if dsn := env("POSTGRES_DSN", env("DATABASE_URL", "")); dsn != "" {
		return dsn
	}

	host := env("POSTGRES_HOST", "")
	if host == "" {
		return ""
	}
	return fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s",
		env("POSTGRES_USER", "face"),
		env("POSTGRES_PASSWORD", ""),
		host,
		env("POSTGRES_PORT", "5432"),
		env("POSTGRES_DB", "face_auth"),
		env("POSTGRES_SSLMODE", "disable"))
}

// normalizeTarget — eski HTTP manzili berilib qolsa ham ishlashi uchun
// "http://host:8000" ko'rinishidan sxemani olib tashlaydi.
func normalizeTarget(target string) string {
	target = strings.TrimSpace(target)
	for _, prefix := range []string{"http://", "https://", "grpc://"} {
		target = strings.TrimPrefix(target, prefix)
	}
	return strings.TrimRight(target, "/")
}

func env(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v, err := strconv.Atoi(env(key, "")); err == nil {
		return v
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v, err := strconv.ParseFloat(env(key, ""), 64); err == nil {
		return v
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	if v, err := strconv.ParseBool(env(key, "")); err == nil {
		return v
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if v, err := time.ParseDuration(env(key, "")); err == nil {
		return v
	}
	return fallback
}
