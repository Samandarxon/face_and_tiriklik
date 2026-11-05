package usecase

import (
	"math/rand"
	"testing"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

type passiveClock struct{ t time.Time }

func (c *passiveClock) now() time.Time { return c.t }
func (c *passiveClock) tick()          { c.t = c.t.Add(200 * time.Millisecond) }

func newPassive(t *testing.T) (*PassiveEvaluator, *passiveClock) {
	t.Helper()
	clock := &passiveClock{t: time.Unix(1000, 0)}
	cfg := DefaultPassiveConfig()
	cfg.Enabled = true
	e := NewPassiveEvaluator(cfg, NewIdentityTrack(0.4))
	e.SetClock(clock.now)
	return e, clock
}

func passiveEmbedding(hot int) domain.Embedding {
	v := make(domain.Embedding, 8)
	v[hot] = 1
	return v
}

func passiveFrameOf(points []domain.Point, who int, score float64) PassiveFrame {
	return PassiveFrame{
		Face: &domain.Face{
			DetScore:  0.9,
			Embedding: passiveEmbedding(who),
			Landmarks: points,
			Pose:      &domain.HeadPose{Available: true},
		},
		Liveness: &domain.Liveness{
			IsReal: true, RealProbability: score,
			PerModel: map[string]float64{"a": score, "b": score},
		},
	}
}

// feed — kadrlarni 5 kadr/s tezlikda beradi, birinchi yakuniy qarorni qaytaradi.
func feed(e *PassiveEvaluator, c *passiveClock, frames [][]domain.Point, who int, score float64) PassiveDecision {
	for _, points := range frames {
		if d := e.Observe(passiveFrameOf(points, who, score)); d != PassivePending {
			return d
		}
		c.tick()
	}
	return PassivePending
}

func TestPassivePassesWithParallax(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(1))
	if d := feed(e, c, liveSway(rng, 10, 4, 0.2), 0, 0.97); d != PassivePassed {
		t.Fatalf("o'tish kutilgan: %v %+v", d, e.Stats())
	}
	if st := e.Stats(); st.Good < 6 || st.MinScore != 0.97 {
		t.Fatalf("o'lchovlar noto'g'ri: %+v", st)
	}
}

// ASOSIY TEST: ball 0.99 bo'lsa ham tekislik o'tmaydi, faqat challenge'ga boradi.
func TestPassiveFlatPictureNeverPassesEvenWithPerfectScore(t *testing.T) {
	for seed := int64(0); seed < 30; seed++ {
		for _, strength := range []float64{0.3, 1, 2} {
			for _, jitter := range []float64{0, 0.5, 1.5} {
				e, c := newPassive(t)
				rng := rand.New(rand.NewSource(seed))
				d := feed(e, c, flatSequence(rng, 40, strength, jitter), 0, 0.99)
				if d != PassiveFallback {
					t.Fatalf("tekislik: %v (seed=%d kuch=%v titrash=%v) %+v", d, seed, strength, jitter, e.Stats())
				}
			}
		}
	}
}

func TestPassiveTooLittleMotionFallsBack(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(2))
	if d := feed(e, c, liveSway(rng, 30, 0.05, 0), 0, 0.99); d != PassiveFallback {
		t.Fatalf("challenge kutilgan: %v", d)
	}
	if st := e.Stats(); st.Reason != "harakat_kam" {
		t.Fatalf("sabab noto'g'ri: %+v", st)
	}
}

func TestPassiveLowScoreDoesNotPassEvenWithParallax(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(3))
	if d := feed(e, c, liveSway(rng, 30, 4, 0.2), 0, 0.85); d != PassiveFallback {
		t.Fatalf("challenge kutilgan: %v", d)
	}
}

// Ikkala model ham chegaradan o'tishi shart, bittasi yetmaydi.
func TestPassiveEveryModelMustClearTheScore(t *testing.T) {
	e, _ := newPassive(t)
	frame := passiveFrameOf(make([]domain.Point, 106), 0, 0.99)
	frame.Liveness.PerModel["b"] = 0.8
	e.Observe(frame)
	if st := e.Stats(); st.Good != 0 || st.Reason != "past_ball" {
		t.Fatalf("bitta past model hisobga olinmadi: %+v", st)
	}
}

func TestPassiveMissingLandmarksFallBack(t *testing.T) {
	e, c := newPassive(t)
	if d := feed(e, c, make([][]domain.Point, 30), 0, 0.99); d != PassiveFallback {
		t.Fatalf("challenge kutilgan: %v", d)
	}
	if st := e.Stats(); st.Reason != "nuqtalar_yoq" {
		t.Fatalf("sabab noto'g'ri: %+v", st)
	}
}

func TestPassivePersonSwitchResetsWindow(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(4))
	frames := liveSway(rng, 10, 4, 0.2)
	if d := feed(e, c, frames[:5], 0, 0.99); d != PassivePending {
		t.Fatalf("hali qaror bo'lmasligi kerak: %v", d)
	}
	e.Observe(passiveFrameOf(frames[5], 1, 0.99))
	st := e.Stats()
	if st.Good != 1 || st.Resets != 1 || st.Reason != "shaxs_almashdi" {
		t.Fatalf("oyna tashlanmadi: %+v", st)
	}
	// Langar yangi odamga o'tdi: tanish bosqichi ham uni talab qiladi.
	if e.track.Same(passiveEmbedding(0)) {
		t.Fatal("eski odam langar bo'lib qoldi")
	}
}

func TestPassiveSpoofResetsEvidence(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(5))
	feed(e, c, liveSway(rng, 5, 4, 0.2), 0, 0.99)
	e.ObserveError(domain.CodeSpoofDetected)
	if st := e.Stats(); st.Good != 0 || st.Resets != 1 {
		t.Fatalf("spoof dalilni tashlamadi: %+v", st)
	}
}

func TestPassiveBudgetExpiryFallsBack(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(6))
	flat := flatSequence(rng, 2, 0, 0)
	budget := DefaultPassiveConfig().Budget
	started := c.t
	for c.t.Sub(started) < budget {
		if d := e.Observe(passiveFrameOf(flat[0], 0, 0.99)); d != PassivePending {
			t.Fatalf("vaqt tugamasdan qaror: %v", d)
		}
		c.tick()
	}
	if d := e.Observe(passiveFrameOf(flat[1], 0, 0.99)); d != PassiveFallback {
		t.Fatalf("vaqt tugadi, challenge kutilgan: %v", d)
	}
}

// Vaqt yuz ko'ringanda boshlanadi: bo'sh kamera oldida o'tgan vaqt hisob emas.
func TestPassiveBudgetStartsAtFirstFace(t *testing.T) {
	e, c := newPassive(t)
	if d := e.ObserveError(domain.CodeNoFaceDetected); d != PassivePending {
		t.Fatalf("yuzsiz kadr: %v", d)
	}
	c.t = c.t.Add(time.Minute)
	if d := e.ObserveError(domain.CodeNoFaceDetected); d != PassivePending {
		t.Fatalf("yuzsiz kadrda vaqt ketmasligi kerak: %v", d)
	}
	rng := rand.New(rand.NewSource(7))
	if d := feed(e, c, liveSway(rng, 10, 4, 0.2), 0, 0.99); d != PassivePassed {
		t.Fatalf("yuz kech kelgan odam o'tishi kerak: %v %+v", d, e.Stats())
	}
}

// Yuz yo'qolib qaytsa yangi urinish, lekin soni cheklangan: aks holda bitta
// oqimda cheksiz tasodifiy urinish bo'lardi.
func TestPassiveRestartsAreBounded(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(8))
	flat := flatSequence(rng, 2, 0, 0)
	for i := 0; i < passiveMaxRestarts; i++ {
		e.Observe(passiveFrameOf(flat[0], 0, 0.99))
		c.t = c.t.Add(2 * time.Second)
	}
	if d := e.Observe(passiveFrameOf(flat[0], 0, 0.99)); d != PassivePending {
		t.Fatalf("oxirgi ruxsat etilgan urinish: %v", d)
	}
	if st := e.Stats(); st.Restarts != passiveMaxRestarts {
		t.Fatalf("urinishlar: %+v", st)
	}
	c.t = c.t.Add(5 * time.Second)
	if d := e.Observe(passiveFrameOf(flat[0], 0, 0.99)); d != PassiveFallback {
		t.Fatalf("chegaradan keyin vaqt qayta boshlanmasligi kerak: %v", d)
	}
}

func TestPassiveProgressNeverReachesFull(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(9))
	feed(e, c, flatSequence(rng, 11, 0, 0), 0, 0.99)
	if p := e.Progress(); p <= 0 || p >= 1 {
		t.Fatalf("taraqqiyot 1 ga yetmasligi kerak (u o'tish degani emas): %v", p)
	}
}

// Urinishlar chegarasidan keyin ham uzilish oldidagi kadrlar dalil bo'lib
// qolmasligi kerak: aks holda bir suratni yashirib, boshqa rakursdagi suratni
// ko'rsatish orqali dalil yig'ish mumkin edi.
func TestPassiveEvidenceDoesNotSurviveGapAfterRestartCap(t *testing.T) {
	e, c := newPassive(t)
	rng := rand.New(rand.NewSource(10))
	flat := flatSequence(rng, 2, 0, 0)
	for i := 0; i <= passiveMaxRestarts; i++ {
		e.Observe(passiveFrameOf(flat[0], 0, 0.99))
		c.t = c.t.Add(2 * time.Second)
	}
	e.Observe(passiveFrameOf(flat[1], 0, 0.99))
	if st := e.Stats(); st.Restarts != passiveMaxRestarts || st.Good > 1 {
		t.Fatalf("uzilishdan keyin faqat yangi kadr qolishi kerak: %+v", st)
	}
}
