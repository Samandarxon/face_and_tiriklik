package usecase

import (
	"math"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// PASSIV-BIRINCHI LIVENESS — buyruqsiz, bir qarashda tekshiruv.
//
// Qarorni FAQAT server chiqaradi: mijoz kadr yuborishdan boshqa narsa qila
// olmaydi, mijozdagi signalni esa soxtalashtirish mumkin.
//
// O'tish uchun BIR ODAMNING kadrlarida hammasi birga bajarilishi shart:
//
//	1. MinFrames ta sifatli kadr: ikkala model ham MinScore dan yuqori, frontal.
//	2. Uch o'lchamli dalil: nuqtalar tekislikdek harakat qilmaydi (planarity.go).
//
// Noaniqlik hech qachon o'tish emas. Vaqt tugasa yoki dalil yetmasa — PassiveFallback:
// challenge'ga o'tiladi (u o'chiq bo'lsa, rad etiladi).

// PassiveDecision — bitta kadrdan keyingi holat.
type PassiveDecision string

const (
	PassivePending  PassiveDecision = "pending"
	PassivePassed   PassiveDecision = "passed"
	PassiveFallback PassiveDecision = "fallback"
)

// PassiveConfig — sozlamalar. Standartlar KALIBRLANMAGAN: `.env.example` ga qarang.
type PassiveConfig struct {
	Enabled   bool
	MinFrames int
	MinScore  float64
	Budget    time.Duration
	Planarity PlanarityThresholds
}

// DefaultPassiveConfig — o'chiq standart qiymatlar.
func DefaultPassiveConfig() PassiveConfig {
	return PassiveConfig{
		MinFrames: 6,
		MinScore:  0.90,
		Budget:    2500 * time.Millisecond,
		Planarity: PlanarityThresholds{
			MinMotion:    0.02,
			MinResidual:  0.012,
			MinNonplanar: 0.70,
			MaxEpipolar:  0.25,
		},
	}
}

const (
	// passiveWindow — oynadagi eng ko'p kadr (~2.4 s, 5 kadr/s). Hisob juftlar
	// soniga kvadratik bog'liq, shuning uchun chegaralangan.
	passiveWindow = 12
	// passiveFrontalYaw — "frontal" deb hisoblanadigan yaw (gradus).
	passiveFrontalYaw = 20.0
	// passiveMinDetScore — detektor ishonchining quyi chegarasi.
	passiveMinDetScore = 0.70
	// passiveMinLandmarks — 5 ta kalit nuqta tekislikni tekshirishga yetmaydi.
	passiveMinLandmarks = 20
	// passiveMaxWeakRun — ketma-ket shuncha yaroqsiz kadrdan keyin dalil
	// tashlanadi. "Deyarli ketma-ket": pirpirash bitta kadrni buzishi mumkin.
	passiveMaxWeakRun = 2
	// passiveFaceGap — yuz shuncha vaqt ko'rinmasa, yangi urinish boshlanadi:
	// kamera oldidan o'tib ketgan odam keyingisining vaqtini yemasin.
	passiveFaceGap = time.Second
	// passiveMaxRestarts — yangi urinishlar soni. Cheksiz bo'lsa, kamerani
	// yopib-ochib bitta oqimda cheksiz tasodifiy urinish qilish mumkin edi.
	passiveMaxRestarts = 2
)

// PassiveFrame — bitta kadr, baholash uchun.
type PassiveFrame struct {
	Face     *domain.Face
	Liveness *domain.Liveness
}

// PassiveStats — sozlash uchun o'lchovlar. Tasvir ham, PNFL ham yo'q.
type PassiveStats struct {
	Decision  PassiveDecision
	Reason    string
	Good      int
	Seen      int
	Resets    int
	Restarts  int
	MinScore  float64
	Planarity PlanarityResult
	Elapsed   time.Duration
}

// PassiveEvaluator — bitta oqimning passiv bosqichi. Bitta goroutine'dan ishlatiladi.
type PassiveEvaluator struct {
	cfg   PassiveConfig
	track *IdentityTrack
	now   func() time.Time

	startedAt  time.Time
	lastFaceAt time.Time
	frames     [][]domain.Point
	weakRun    int
	scoreFloor float64
	stats      PassiveStats
}

// NewPassiveEvaluator — track shaxsni kuzatadi va o'tgandan keyin tanish
// bosqichida ham shu langar ishlatiladi: kadrlar va tanish bir odamga bog'lanadi.
func NewPassiveEvaluator(cfg PassiveConfig, track *IdentityTrack) *PassiveEvaluator {
	return &PassiveEvaluator{cfg: cfg, track: track, now: time.Now, scoreFloor: 1,
		stats: PassiveStats{Decision: PassivePending, MinScore: 1}}
}

// SetClock — testlar uchun.
func (e *PassiveEvaluator) SetClock(now func() time.Time) { e.now = now }

// Stats — joriy o'lchovlar.
func (e *PassiveEvaluator) Stats() PassiveStats {
	s := e.stats
	if !e.startedAt.IsZero() {
		s.Elapsed = e.now().Sub(e.startedAt)
	}
	return s
}

// Progress — 0..1, faqat kadrlar soni bo'yicha. Geometriya o'lchovlari
// ataylab berilmaydi: ular hujumchiga suratni qanday qimirlatishni o'rgatardi.
func (e *PassiveEvaluator) Progress() float64 {
	if e.cfg.MinFrames <= 0 {
		return 0
	}
	return math.Min(0.95, float64(len(e.frames))/float64(e.cfg.MinFrames))
}

// Observe — sog'lom javob kelgan kadrni baholaydi.
func (e *PassiveEvaluator) Observe(frame PassiveFrame) PassiveDecision {
	e.faceSeen()
	e.stats.Seen++

	points, reason := e.usable(frame)
	if reason != "" {
		e.weak(reason)
		return e.finishIfExpired()
	}
	if !e.track.Same(frame.Face.Embedding) {
		// Oyna faqat bitta odamning kadrlaridan iborat bo'lishi shart.
		e.track.Reset()
		e.track.Same(frame.Face.Embedding)
		e.resetEvidence("shaxs_almashdi")
	}

	e.weakRun = 0
	e.frames = append(e.frames, points)
	if len(e.frames) > passiveWindow {
		e.frames = e.frames[1:]
	}
	e.scoreFloor = math.Min(e.scoreFloor, frameScore(frame.Liveness))
	e.stats.Good = len(e.frames)
	e.stats.MinScore = e.scoreFloor

	if len(e.frames) >= e.cfg.MinFrames {
		e.stats.Planarity = EvaluatePlanarity(e.frames, e.cfg.Planarity)
		if e.stats.Planarity.NonPlanar {
			e.stats.Decision = PassivePassed
			e.stats.Reason = "dalil_yetarli"
			return PassivePassed
		}
	}
	return e.finishIfExpired()
}

// ObserveError — kadr darajasidagi xato. Faqat yuz bor bo'lgan xatolar vaqtni
// boshlaydi: `no_face_detected` da odam hali kelmagan.
func (e *PassiveEvaluator) ObserveError(code domain.Code) PassiveDecision {
	switch code {
	case domain.CodeSpoofDetected, domain.CodeMultipleFaces:
		e.faceSeen()
		e.stats.Seen++
		e.resetEvidence(string(code))
	case domain.CodeLowQualityFace:
		e.faceSeen()
		e.stats.Seen++
		e.weak(string(code))
	default:
		// Yuz yo'q yoki servis xatosi: vaqt shu kadr hisobiga tugamaydi.
		return PassivePending
	}
	return e.finishIfExpired()
}

// usable — kadr dalil bo'la oladimi. Bo'sh qator — ha.
func (e *PassiveEvaluator) usable(frame PassiveFrame) ([]domain.Point, string) {
	switch {
	case frame.Face == nil || frame.Liveness == nil || !frame.Liveness.IsReal:
		return nil, "tirik_emas"
	case frame.Face.Embedding.Dim() == 0:
		return nil, "vektor_yoq"
	case len(frame.Face.Landmarks) < passiveMinLandmarks:
		return nil, "nuqtalar_yoq"
	case frameScore(frame.Liveness) < e.cfg.MinScore:
		return nil, "past_ball"
	case frame.Face.DetScore < passiveMinDetScore:
		return nil, "past_detektor"
	case frame.Face.Pose == nil || !frame.Face.Pose.Available ||
		math.Abs(frame.Face.Pose.Yaw) > passiveFrontalYaw:
		return nil, "frontal_emas"
	}
	return frame.Face.Landmarks, ""
}

func (e *PassiveEvaluator) faceSeen() {
	now := e.now()
	if e.startedAt.IsZero() {
		e.startedAt = now
	} else if now.Sub(e.lastFaceAt) > passiveFaceGap {
		// Vaqt faqat chegaragacha yangilanadi, dalil esa har doim tashlanadi:
		// uzilish oldidagi va keyingi kadrlar bitta dalil bo'la olmaydi.
		if e.stats.Restarts < passiveMaxRestarts {
			e.stats.Restarts++
			e.startedAt = now
		}
		e.track.Reset()
		e.resetEvidence("yuz_yoqoldi")
	}
	e.lastFaceAt = now
}

func (e *PassiveEvaluator) weak(reason string) {
	e.stats.Reason = reason
	e.weakRun++
	if e.weakRun > passiveMaxWeakRun {
		e.resetEvidence(reason)
	}
}

func (e *PassiveEvaluator) resetEvidence(reason string) {
	if len(e.frames) > 0 {
		e.stats.Resets++
	}
	e.frames = nil
	e.weakRun = 0
	e.scoreFloor = 1
	e.stats.Reason = reason
	e.stats.Good = 0
}

func (e *PassiveEvaluator) finishIfExpired() PassiveDecision {
	if e.startedAt.IsZero() || e.now().Sub(e.startedAt) < e.cfg.Budget {
		return PassivePending
	}
	e.stats.Decision = PassiveFallback
	switch {
	case e.stats.Good < e.cfg.MinFrames:
		// Reason — kadrni yaroqsiz qilgan oxirgi sabab, shu o'zi foydali.
		if e.stats.Reason == "" {
			e.stats.Reason = "kadr_yetmadi"
		}
	case e.stats.Planarity.Pairs < minPlanarPairs:
		e.stats.Reason = "harakat_kam"
	default:
		e.stats.Reason = "tekislik_inkor_etilmadi"
	}
	return PassiveFallback
}

// frameScore — ansambldagi eng past ball: "ikkala model ham" sharti. Har bir
// model bahosi kelmasa umumiy ehtimol ishlatiladi.
func frameScore(l *domain.Liveness) float64 {
	if l == nil {
		return 0
	}
	score := l.RealProbability
	for _, s := range l.PerModel {
		score = math.Min(score, s)
	}
	return score
}
