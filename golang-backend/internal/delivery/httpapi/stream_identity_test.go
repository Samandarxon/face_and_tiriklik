package httpapi

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// Aktiv liveness'da challenge'ni bajargan odam bilan tanilgan odam bir
// bo'lishi shart. Bu testlar handleResult ni haqiqiy WebSocket juftligi va
// haqiqiy ChallengeStore bilan yuritadi.

type oneMatchRepo struct {
	domain.PersonRepository
	calls int
}

func (r *oneMatchRepo) FindNearest(context.Context, domain.Embedding, int) ([]domain.Match, error) {
	r.calls++
	return []domain.Match{{Person: domain.Person{ID: "12345678901234"}, Score: 0.9}}, nil
}

func wsPair(t *testing.T) (server *websocket.Conn, client *websocket.Conn) {
	t.Helper()
	ch := make(chan *websocket.Conn, 1)
	up := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := up.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		ch <- c
	}))
	t.Cleanup(srv.Close)
	cl, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { cl.Close() })
	return <-ch, cl
}

type identityFixture struct {
	s      *streamSession
	client *websocket.Conn
	repo   *oneMatchRepo
	frame  uint64
}

func newIdentityFixture(t *testing.T) *identityFixture {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	repo := &oneMatchRepo{}
	verifier := usecase.NewVerifier(nil, repo, 0.65, log)
	challenges := usecase.NewChallengeStore()
	t.Cleanup(challenges.Close)
	h := NewStreamHandler(nil, verifier, nil, nil, nil, "", log, challenges)

	conn, client := wsPair(t)
	ch, err := challenges.Issue()
	if err != nil {
		t.Fatal(err)
	}
	s := &streamSession{handler: h, conn: conn, requestID: "test",
		challengeNonce: ch.Nonce, track: usecase.NewIdentityTrack(0.4)}
	return &identityFixture{s: s, client: client, repo: repo}
}


func person(hot int) domain.Embedding {
	e := make(domain.Embedding, 8)
	e[hot] = 1
	return e
}

// send — bitta kadr natijasi; javob hodisasini qaytaradi.
func (f *identityFixture) send(t *testing.T, emb domain.Embedding, yaw float64) (event, bool) {
	t.Helper()
	f.frame++
	res := &domain.FrameResult{
		FrameID:  f.frame,
		Liveness: &domain.Liveness{IsReal: true, RealProbability: 0.99},
		Face: &domain.Face{
			Embedding: emb,
			Pose:      &domain.HeadPose{Yaw: yaw, Available: true},
		},
	}
	keep := f.s.handleResult(context.Background(), res)
	_ = f.client.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, data, err := f.client.ReadMessage()
	if err != nil {
		t.Fatalf("hodisa kelmadi: %v", err)
	}
	var evt event
	if err := json.Unmarshal(data, &evt); err != nil {
		t.Fatal(err)
	}
	return evt, keep
}

// passChallenge — markaz, keyin buyruqdagi tomonga burilish.
func (f *identityFixture) passChallenge(t *testing.T, emb domain.Embedding) {
	t.Helper()
	f.send(t, emb, 0)
	for _, yaw := range []float64{-30, 30} {
		evt, _ := f.send(t, emb, yaw)
		if evt.ChallengeState == usecase.ChallengePassed {
			return
		}
		f.send(t, emb, 0) // markazga qaytish
	}
	t.Fatal("challenge o'tmadi")
}

func TestStreamRecognisesSamePersonAfterChallenge(t *testing.T) {
	f := newIdentityFixture(t)
	f.passChallenge(t, person(0))

	evt, _ := f.send(t, person(0), 0)
	if evt.Type != "recognized" {
		t.Fatalf("tanish kutilgan, keldi: %+v", evt)
	}
}

// Challenge'ni bir odam bajarib, tanish kadrida boshqasi bo'lsa — rad,
// bazaga murojaat qilinmaydi va oqim yopiladi.
func TestStreamRejectsPersonSwapAfterChallenge(t *testing.T) {
	f := newIdentityFixture(t)
	f.passChallenge(t, person(0))

	evt, keep := f.send(t, person(1), 0)
	if evt.Type != "spoof" || keep {
		t.Fatalf("almashish rad etilmadi: %+v keep=%v", evt, keep)
	}
	if f.repo.calls != 0 {
		t.Fatal("almashgan odam uchun bazaga murojaat qilindi")
	}
}

// Challenge davomida odam almashsa — darhol rad.
func TestStreamRejectsPersonSwapDuringChallenge(t *testing.T) {
	f := newIdentityFixture(t)
	f.send(t, person(0), 0)

	evt, keep := f.send(t, person(1), -30)
	if evt.Type != "spoof" || keep {
		t.Fatalf("challenge davomida almashish rad etilmadi: %+v", evt)
	}
}

// Vektorsiz kadr harakat sifatida hisoblanmaydi.
func TestStreamIgnoresActionOnFrameWithoutEmbedding(t *testing.T) {
	f := newIdentityFixture(t)
	f.send(t, person(0), 0)
	for _, yaw := range []float64{-30, 30} {
		evt, _ := f.send(t, nil, yaw)
		if evt.ChallengeState == usecase.ChallengePassed {
			t.Fatal("vektorsiz kadrda harakat qabul qilindi")
		}
	}
	if f.s.challengePassed {
		t.Fatal("challenge vektorsiz o'tdi")
	}
}
