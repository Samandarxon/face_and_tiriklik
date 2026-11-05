package httpapi

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// Passiv-birinchi oqim handleResult orqali, haqiqiy ChallengeStore va WebSocket
// juftligi bilan yuritiladi. Vaqt evaluatorda soxta soat bilan boshqariladi.

type passiveFixture struct {
	s      *streamSession
	client *websocket.Conn
	repo   *oneMatchRepo
	clock  time.Time
	frame  uint64
}

func newPassiveFixture(t *testing.T, active bool) *passiveFixture {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	repo := &oneMatchRepo{}
	verifier := usecase.NewVerifier(nil, repo, 0.65, log)
	var challenges *usecase.ChallengeStore
	if active {
		challenges = usecase.NewChallengeStore()
		t.Cleanup(challenges.Close)
	}
	h := NewStreamHandler(nil, verifier, nil, nil, nil, "", log, challenges)
	cfg := usecase.DefaultPassiveConfig()
	cfg.Enabled = true
	h.SetPassiveLiveness(cfg)

	conn, client := wsPair(t)
	track := usecase.NewIdentityTrack(0.4)
	f := &passiveFixture{client: client, repo: repo, clock: time.Unix(1000, 0)}
	eval := usecase.NewPassiveEvaluator(cfg, track)
	eval.SetClock(func() time.Time { return f.clock })
	f.s = &streamSession{handler: h, conn: conn, requestID: "test", track: track, passive: eval}
	return f
}

func (f *passiveFixture) read(t *testing.T) event {
	t.Helper()
	_ = f.client.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, data, err := f.client.ReadMessage()
	if err != nil {
		t.Fatalf("hodisa kelmadi: %v", err)
	}
	var evt event
	if err := json.Unmarshal(data, &evt); err != nil {
		t.Fatal(err)
	}
	return evt
}

func (f *passiveFixture) send(t *testing.T, who domain.Embedding, points []domain.Point, yaw, score float64) (event, bool) {
	t.Helper()
	f.frame++
	f.clock = f.clock.Add(200 * time.Millisecond)
	keep := f.s.handleResult(context.Background(), &domain.FrameResult{
		FrameID:  f.frame,
		Liveness: &domain.Liveness{IsReal: true, RealProbability: score},
		Face: &domain.Face{
			DetScore: 0.9, Embedding: who, Landmarks: points,
			Pose: &domain.HeadPose{Yaw: yaw, Available: true},
		},
	})
	return f.read(t), keep
}

// headFrames — parallaksli (tirik bosh) nuqtalar.
func headFrames(seed int64, frames int) [][]domain.Point {
	rng := rand.New(rand.NewSource(seed))
	head := make([][3]float64, 106)
	for i := range head {
		x, y := rng.Float64()*2-1, rng.Float64()*2-1
		head[i] = [3]float64{70 * x, 90 * y, 35*math.Exp(-(x*x+y*y)/0.08) - 55*x*x - 25*y*y}
	}
	out := make([][]domain.Point, frames)
	for n := range out {
		a := 4 * math.Pi / 180 * math.Sin(2*math.Pi*float64(n)/float64(frames-1))
		out[n] = make([]domain.Point, len(head))
		for i, p := range head {
			out[n][i] = domain.Point{
				X: 320 + p[0]*math.Cos(a) + p[2]*math.Sin(a) + rng.NormFloat64()*0.2,
				Y: 240 + p[1] + rng.NormFloat64()*0.2,
			}
		}
	}
	return out
}

// flatFrames — tekislik: har kadr bitta gomografiya bilan o'tkazilgan.
func flatFrames(seed int64, frames int) [][]domain.Point {
	base := headFrames(seed, 2)[0]
	rng := rand.New(rand.NewSource(seed))
	out := make([][]domain.Point, frames)
	for n := range out {
		g, tx := rng.NormFloat64()*4e-4, rng.NormFloat64()*8
		out[n] = make([]domain.Point, len(base))
		for i, p := range base {
			w := g*p.X + 1
			out[n][i] = domain.Point{X: (p.X + tx) / w, Y: p.Y / w}
		}
	}
	return out
}

func TestPassiveStreamRecognisesOnlyAfterPass(t *testing.T) {
	f := newPassiveFixture(t, true)
	var last event
	for _, points := range headFrames(1, 12) {
		evt, keep := f.send(t, person(0), points, 0, 0.99)
		if !keep {
			t.Fatalf("oqim yopildi: %+v", evt)
		}
		if f.s.passive != nil && f.repo.calls != 0 {
			t.Fatal("passiv bosqich tugamasdan bazaga murojaat qilindi")
		}
		last = evt
		if evt.Type == "recognized" {
			break
		}
		if evt.ChallengeState != usecase.ChallengePassive || evt.PassiveProgress <= 0 {
			t.Fatalf("passiv holat kutilgan: %+v", evt)
		}
		if evt.Challenge != nil {
			t.Fatal("o'tish yo'lida buyruq berilmasligi kerak")
		}
	}
	if last.Type != "recognized" || f.repo.calls != 1 {
		t.Fatalf("tanish kutilgan: %+v calls=%d", last, f.repo.calls)
	}
	if !f.s.challengePassed || f.s.lastChallenge != usecase.ChallengePassed {
		t.Fatal("holat yangilanmadi")
	}
}

// Tanishdan keyin kamera oldidagi odam almashsa — xuddi challenge'dagidek rad.
func TestPassiveStreamBindsLaterFramesToSamePerson(t *testing.T) {
	f := newPassiveFixture(t, true)
	frames := headFrames(2, 12)
	for _, points := range frames {
		if evt, _ := f.send(t, person(0), points, 0, 0.99); evt.Type == "recognized" {
			break
		}
	}
	calls := f.repo.calls
	evt, keep := f.send(t, person(1), frames[0], 0, 0.99)
	if evt.Type != "spoof" || keep || f.repo.calls != calls {
		t.Fatalf("almashish rad etilmadi: %+v keep=%v", evt, keep)
	}
}

// ASOSIY TEST: surat 0.99 ball bilan ham tanishgacha yetmaydi, buyruq oladi.
func TestPassiveStreamFlatPictureGetsChallengeNotRecognition(t *testing.T) {
	f := newPassiveFixture(t, true)
	var evt event
	for _, points := range flatFrames(3, 30) {
		evt, _ = f.send(t, person(0), points, 0, 0.99)
		if evt.Type == "ready" {
			break
		}
		if evt.Type == "recognized" || f.repo.calls != 0 {
			t.Fatalf("surat tanishga yetdi: %+v", evt)
		}
	}
	if evt.Type != "ready" || evt.Challenge == nil || evt.ChallengeState != usecase.ChallengeAwaitingCentre {
		t.Fatalf("buyruq kutilgan: %+v", evt)
	}
	if f.s.passive != nil || f.s.challengeNonce == "" {
		t.Fatal("oqim challenge bosqichiga o'tmadi")
	}
}

// Challenge'ga o'tgach eski yo'l ishlaydi, langar esa o'sha odamda qoladi.
func TestPassiveStreamChallengeAfterFallback(t *testing.T) {
	f := newPassiveFixture(t, true)
	flat := flatFrames(4, 30)
	for _, points := range flat {
		if evt, _ := f.send(t, person(0), points, 0, 0.99); evt.Type == "ready" {
			break
		}
	}
	if f.s.challengeNonce == "" {
		t.Fatal("challenge berilmadi")
	}
	f.send(t, person(0), flat[0], 0, 0.99) // markaz
	passed := false
	for _, yaw := range []float64{-30, 30} {
		if evt, _ := f.send(t, person(0), flat[0], yaw, 0.99); evt.ChallengeState == usecase.ChallengePassed {
			passed = true
			break
		}
		f.send(t, person(0), flat[0], 0, 0.99)
	}
	if !passed {
		t.Fatal("challenge o'tmadi")
	}
	if f.repo.calls != 0 {
		t.Fatal("tanish kadridan oldin bazaga murojaat qilindi")
	}
	if evt, _ := f.send(t, person(0), flat[0], 0, 0.99); evt.Type != "recognized" {
		t.Fatalf("tanish kutilgan: %+v", evt)
	}
}

// Challenge davomida odam almashsa oqim yopiladi (passivdagi langar saqlanadi).
func TestPassiveStreamPersonSwapAfterFallbackIsRejected(t *testing.T) {
	f := newPassiveFixture(t, true)
	flat := flatFrames(5, 30)
	for _, points := range flat {
		if evt, _ := f.send(t, person(0), points, 0, 0.99); evt.Type == "ready" {
			break
		}
	}
	if evt, keep := f.send(t, person(1), flat[0], 0, 0.99); evt.Type != "spoof" || keep {
		t.Fatalf("almashish rad etilmadi: %+v keep=%v", evt, keep)
	}
}

// Challenge o'chiq bo'lsa noaniqlik rad, o'tish emas.
func TestPassiveStreamWithoutActiveKeepsDenying(t *testing.T) {
	f := newPassiveFixture(t, false)
	var evt event
	keep := true
	for _, points := range flatFrames(6, 30) {
		evt, keep = f.send(t, person(0), points, 0, 0.99)
		if !keep {
			break
		}
	}
	if keep || evt.Type != "error" || evt.Code != domain.CodeSpoofDetected ||
		evt.ChallengeState != usecase.ChallengeExpired {
		t.Fatalf("rad kutilgan: %+v keep=%v", evt, keep)
	}
	if f.repo.calls != 0 {
		t.Fatal("rad etilgan oqimda bazaga murojaat qilindi")
	}
}

// Spoof hodisasi avvalgidek chiqadi va oqimni yopmaydi; vaqt tugasa buyruq beriladi.
func TestPassiveStreamSpoofFramesBehaveAsBefore(t *testing.T) {
	f := newPassiveFixture(t, true)
	spoof := &domain.FrameResult{FrameID: 1, Err: &domain.Error{Code: domain.CodeSpoofDetected, Message: "soxta"}}
	if !f.s.handleResult(context.Background(), spoof) {
		t.Fatal("spoof oqimni yopmasligi kerak")
	}
	if evt := f.read(t); evt.Type != "spoof" {
		t.Fatalf("spoof kutilgan: %+v", evt)
	}
	started := f.clock
	for f.clock.Sub(started) < usecase.DefaultPassiveConfig().Budget {
		f.clock = f.clock.Add(200 * time.Millisecond)
		if !f.s.handleResult(context.Background(), spoof) {
			t.Fatal("spoof oqimni yopmasligi kerak")
		}
		if evt := f.read(t); evt.Type != "spoof" {
			t.Fatalf("spoof kutilgan: %+v", evt)
		}
	}
	if evt := f.read(t); evt.Type != "ready" || evt.Challenge == nil {
		t.Fatalf("buyruq kutilgan: %+v", evt)
	}
	if f.repo.calls != 0 {
		t.Fatal("spoof uchun bazaga murojaat qilindi")
	}
}

// --- run(): ready hodisasi ---

type idleStream struct{ closed chan struct{} }

func (s *idleStream) Send(domain.FrameRequest) error { return nil }
func (s *idleStream) Recv() (*domain.FrameResult, error) {
	<-s.closed
	return nil, io.EOF
}
func (s *idleStream) CloseSend() error { return nil }
func (s *idleStream) Close() error     { return nil }

func readyFrame(t *testing.T, passive bool) string {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	challenges := usecase.NewChallengeStore()
	t.Cleanup(challenges.Close)
	h := NewStreamHandler(nil, nil, nil, nil, nil, "", log, challenges)
	if passive {
		cfg := usecase.DefaultPassiveConfig()
		cfg.Enabled = true
		h.SetPassiveLiveness(cfg)
	}
	conn, client := wsPair(t)
	stream := &idleStream{closed: make(chan struct{})}
	t.Cleanup(func() { close(stream.closed) })
	s := &streamSession{handler: h, conn: conn, stream: stream, requestID: "test"}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go s.run(ctx, cancel)

	_ = client.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, data, err := client.ReadMessage()
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// Flag o'chiq: `ready` avvalgi shaklda — buyruq bilan, yangi maydonlarsiz.
func TestReadyEventUnchangedWhenPassiveOff(t *testing.T) {
	raw := readyFrame(t, false)
	if !strings.Contains(raw, `"challenge":{`) || !strings.Contains(raw, `"challenge_state":"awaiting_centre"`) {
		t.Fatalf("eski shakl buzildi: %s", raw)
	}
	if strings.Contains(raw, "passive") {
		t.Fatalf("flag o'chiq, lekin passiv maydon chiqdi: %s", raw)
	}
}

// Flag yoqiq: buyruq oldindan OSHKOR QILINMAYDI.
func TestReadyEventWithholdsChallengeWhenPassiveOn(t *testing.T) {
	raw := readyFrame(t, true)
	if strings.Contains(raw, `"challenge":`) {
		t.Fatalf("buyruq oldindan berildi: %s", raw)
	}
	if !strings.Contains(raw, `"challenge_state":"passive"`) {
		t.Fatalf("passiv holat kutilgan: %s", raw)
	}
}
