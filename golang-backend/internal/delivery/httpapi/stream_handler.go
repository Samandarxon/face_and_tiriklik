package httpapi

import (
	"context"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// Real-time oqim: brauzer ⟷ WebSocket ⟷ Golang ⟷ gRPC bidi ⟷ Python.
//
// Nima uchun WebSocket, gRPC emas? Brauzer HTTP/2 trailer'larga kira olmaydi,
// shuning uchun sof gRPC brauzerda ishlamaydi (gRPC-Web + Envoy proxy kerak
// bo'lardi). WebSocket esa native qo'llab-quvvatlanadi. Golang ikkalasini
// bir-biriga ulaydi va BUTUN ULANISH DAVOMIDA bitta gRPC oqimini ushlab turadi
// — har kadrda yangi TCP/HTTP2 handshake bo'lmaydi.
//
// Avtomatik oqim mantiqi SERVERDA turadi: brauzer shunchaki kadr yuboradi va
// hodisa oladi. Shu tufayli mijoz (web, mobil, kiosk) sodda bo'ladi va
// biznes qoidasi bitta joyda saqlanadi.

const (
	// Ketma-ket shuncha "tirik" kadrdan keyin to'liq tanish ishga tushadi.
	defaultLiveStreak = 2
	// Ikki to'liq tekshiruv orasidagi minimal tanaffus.
	verifyCooldown = 3 * time.Second
	// Kadr qabul qilish tezligining YUQORI chegarasi — faqat DoS himoyasi.
	//
	// Haqiqiy tezlikni mijoz o'zi belgilaydi: u keyingi kadrni faqat oldingisiga
	// javob olgach yuboradi (pipelining), shuning uchun tezlik AI servisning
	// haqiqiy imkoniyatiga tabiiy ravishda moslashadi. Bu chegara faqat
	// buzuq/zararli mijoz oqimni bosib ketmasligi uchun kerak, shuning uchun
	// past qo'yilgan — aks holda u ramka harakatini sun'iy sekinlashtiradi.
	minFrameInterval = 50 * time.Millisecond

	wsReadLimit   = 8 << 20
	wsPongWait    = 60 * time.Second
	wsPingPeriod  = 25 * time.Second
	wsWriteWait   = 10 * time.Second
	streamMaxLife = 15 * time.Minute
)

// StreamHandler — WebSocket ↔ gRPC ko'prigi.
type StreamHandler struct {
	analyzer domain.FaceAnalyzer
	verifier *usecase.Verifier
	tickets  *TicketStore
	// challenges is nil when active liveness is disabled. Nil means the stream
	// behaves exactly as before — passive anti-spoofing only.
	challenges *usecase.ChallengeStore
	audit      domain.AuditSink
	metrics    *Metrics
	log        *slog.Logger
	upgrader   websocket.Upgrader
	liveStreak int
}

func NewStreamHandler(
	analyzer domain.FaceAnalyzer,
	verifier *usecase.Verifier,
	tickets *TicketStore,
	audit domain.AuditSink,
	metrics *Metrics,
	allowedOrigin string,
	log *slog.Logger,
	challenges *usecase.ChallengeStore,
) *StreamHandler {
	return &StreamHandler{
		analyzer:   analyzer,
		verifier:   verifier,
		tickets:    tickets,
		challenges: challenges,
		audit:      audit,
		metrics:    metrics,
		log:        log,
		liveStreak: defaultLiveStreak,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  4096,
			WriteBufferSize: 4096,
			CheckOrigin: func(r *http.Request) bool {
				if allowedOrigin == "" || allowedOrigin == "*" {
					return true
				}
				return r.Header.Get("Origin") == allowedOrigin
			},
		},
	}
}

// event — serverdan mijozga boradigan hodisa.
type event struct {
	Type    string `json:"type"` // liveness | spoof | recognized | not_found | error | ready
	FrameID uint64 `json:"frame_id,omitempty"`

	Liveness *domain.Liveness    `json:"liveness,omitempty"`
	BBox     *domain.BoundingBox `json:"bbox,omitempty"`

	// --- Active liveness ---
	// Challenge is sent once, right after the socket opens. ChallengeState
	// then rides along on every frame so the UI can show progress.
	Challenge      *usecase.Challenge     `json:"challenge,omitempty"`
	ChallengeState usecase.ChallengeState `json:"challenge_state,omitempty"`
	Pose           *domain.HeadPose       `json:"pose,omitempty"`

	PersonID string  `json:"person_id,omitempty"`
	Name     string  `json:"name,omitempty"`
	Score    float32 `json:"score,omitempty"`

	Code         domain.Code    `json:"code,omitempty"`
	Message      string         `json:"message,omitempty"`
	Details      map[string]any `json:"details,omitempty"`
	ProcessingMs int            `json:"processing_ms,omitempty"`
}

// Handle — `GET /api/v1/stream?ticket=...` ni WebSocket'ga ko'taradi.
//
// Autentifikatsiya oddiy middleware bilan emas, TICKET orqali: brauzer WS
// ulanishida `X-API-Key` sarlavhasini yubora olmaydi. Ticket bir martalik va
// 60 soniya yashaydi — batafsil izoh `ticket.go` da.
func (h *StreamHandler) Handle(c *gin.Context) {
	key, err := h.tickets.Redeem(c.Query("ticket"))
	if err != nil {
		// Hali WebSocket'ga ko'tarilmaganmiz — oddiy HTTP xatosi qaytadi.
		respondError(c, err)
		return
	}
	if !key.HasScope(domain.ScopeVerify) {
		respondError(c, domain.NewError(domain.CodeForbidden,
			"Oqim uchun `face:verify` huquqi kerak."))
		return
	}

	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		h.log.Warn("websocket upgrade muvaffaqiyatsiz", "error", err)
		return
	}
	defer conn.Close()

	h.metrics.StreamOpened()
	defer h.metrics.StreamClosed()

	requestID := requestIDOf(c)
	h.audit.Record(domain.AuditEntry{
		ClientID:  key.ClientID,
		KeyID:     key.ID,
		Action:    "stream",
		Result:    "ok",
		RequestID: requestID,
		IP:        c.ClientIP(),
	})
	ctx, cancel := context.WithTimeout(c.Request.Context(), streamMaxLife)
	defer cancel()

	stream, err := h.analyzer.OpenStream(ctx)
	if err != nil {
		h.writeEvent(conn, event{Type: "error", Code: domain.CodeAIUnavailable,
			Message: "AI servis oqimini ochib bo'lmadi."})
		return
	}
	defer stream.Close()

	h.log.Info("stream ochildi",
		"request_id", requestID, "client", c.ClientIP(), "api_client", key.ClientName)
	defer h.log.Info("stream yopildi", "request_id", requestID)

	session := &streamSession{handler: h, conn: conn, stream: stream, requestID: requestID}
	session.run(ctx, cancel)
}

// streamSession — bitta WebSocket ulanishining holati.
type streamSession struct {
	handler   *StreamHandler
	conn      *websocket.Conn
	stream    domain.FrameStream
	requestID string

	// wantEmbedding — keyingi kadr uchun to'liq rejim so'ralganini bildiradi.
	// Reader goroutine yozadi, u ham o'zi o'qiydi — atomik yetarli.
	wantEmbedding atomic.Bool
	liveStreak    int
	pausedUntil   time.Time

	// challengeNonce is empty when active liveness is off. While it is set,
	// recognition is GATED: a live frame alone is not enough, the requested
	// movement has to be seen first.
	challengeNonce  string
	challengePassed bool
}

func (s *streamSession) run(ctx context.Context, cancel context.CancelFunc) {
	s.conn.SetReadLimit(wsReadLimit)
	_ = s.conn.SetReadDeadline(time.Now().Add(wsPongWait))
	s.conn.SetPongHandler(func(string) error {
		return s.conn.SetReadDeadline(time.Now().Add(wsPongWait))
	})

	done := make(chan struct{})

	// --- Goroutine A: brauzerdan kadrlarni o'qib, gRPC oqimiga uzatadi ---
	go func() {
		defer close(done)
		defer func() { _ = s.stream.CloseSend() }()

		var frameID uint64
		lastForwarded := time.Time{}

		for {
			messageType, payload, err := s.conn.ReadMessage()
			if err != nil {
				return // mijoz uzildi — odatiy holat
			}
			if messageType != websocket.BinaryMessage || len(payload) == 0 {
				continue // matnli boshqaruv xabarlari hozircha ishlatilmaydi
			}

			// Tezlikni cheklaymiz — mijoz sekundiga 100 kadr yuborsa ham
			// AI servis ortiqcha yuklanmaydi.
			if time.Since(lastForwarded) < minFrameInterval {
				continue
			}
			lastForwarded = time.Now()

			frameID++
			if err := s.stream.Send(domain.FrameRequest{
				Image:         payload,
				FrameID:       frameID,
				WantEmbedding: s.wantEmbedding.Load(),
			}); err != nil {
				return
			}
		}
	}()

	// --- Goroutine B: gRPC javoblarini qayta ishlab, brauzerga yuboradi ---
	// WebSocket'ga YOZISH faqat shu goroutine'da bo'ladi (gorilla parallel
	// yozishga ruxsat bermaydi).
	ticker := time.NewTicker(wsPingPeriod)
	defer ticker.Stop()

	results := make(chan *domain.FrameResult, 4)
	streamErr := make(chan error, 1)
	go func() {
		for {
			result, err := s.stream.Recv()
			if err != nil {
				streamErr <- err
				return
			}
			results <- result
		}
	}()

	// The challenge is issued once, with the ready event, and only after the
	// socket is open. Issuing it earlier — on the HTTP request that fetches
	// the ticket, say — would give an attacker the extra round trip they need
	// to look up or generate matching footage.
	ready := event{Type: "ready"}
	if s.handler.challenges != nil {
		ch, err := s.handler.challenges.Issue()
		if err != nil {
			// Fail closed. Active liveness is configured on, so running
			// without it would silently downgrade the check the operator
			// believes is running.
			s.handler.log.Error("challenge berib bo'lmadi", "error", err)
			s.handler.writeEvent(s.conn, event{
				Type: "error", Code: domain.CodeInternal,
				Message: "Tekshiruvni boshlab bo'lmadi. Qaytadan urinib ko'ring.",
			})
			return
		}
		s.challengeNonce = ch.Nonce
		ready.Challenge = ch
		ready.ChallengeState = usecase.ChallengeAwaitingCentre
	}
	s.handler.writeEvent(s.conn, ready)

	for {
		select {
		case <-ctx.Done():
			return
		case <-done: // mijoz uzildi
			cancel()
			return
		case <-streamErr:
			s.handler.writeEvent(s.conn, event{Type: "error", Code: domain.CodeAIUnavailable,
				Message: "AI servis oqimi uzildi."})
			return
		case <-ticker.C:
			_ = s.conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := s.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		case result := <-results:
			if !s.handleResult(ctx, result) {
				return
			}
		}
	}
}

// handleResult — bitta kadr natijasini qayta ishlaydi.
// `false` qaytsa ulanish yopiladi.
func (s *streamSession) handleResult(ctx context.Context, result *domain.FrameResult) bool {
	// --- Kadr darajasidagi xato: oqim UZILMAYDI ---
	if !result.OK() {
		s.liveStreak = 0
		s.wantEmbedding.Store(false)

		evt := event{
			Type:         "error",
			FrameID:      result.FrameID,
			Code:         result.Err.Code,
			Message:      result.Err.Message,
			Details:      result.Err.Details,
			ProcessingMs: result.ProcessingMs,
		}
		if result.Err.Code == domain.CodeSpoofDetected {
			evt.Type = "spoof"
			evt.Liveness = result.Err.Liveness
		}
		return s.handler.writeEvent(s.conn, evt)
	}

	// --- Tirik emas ---
	if result.Liveness == nil || !result.Liveness.IsReal {
		s.liveStreak = 0
		s.wantEmbedding.Store(false)
		return s.handler.writeEvent(s.conn, event{
			Type: "spoof", FrameID: result.FrameID,
			Liveness: result.Liveness, BBox: &result.Face.BBox,
			ProcessingMs: result.ProcessingMs,
		})
	}

	// --- Tirik. Endi aktiv tekshiruv. ---
	//
	// Bu yerda joylashuvi muhim: kadr TIRIK deb topilgandan KEYIN, lekin
	// tanishdan OLDIN. Passiv model soxta materialni allaqachon to'sgan;
	// aktiv tekshiruv esa yozib olingan materialni to'sadi. Ikkalasi o'tmasa
	// bazaga umuman murojaat qilinmaydi.
	if s.challengeNonce != "" && !s.challengePassed {
		state := s.handler.challenges.Observe(s.challengeNonce, result.Face.Pose)

		switch state {
		case usecase.ChallengePassed:
			s.challengePassed = true
		case usecase.ChallengeExpired, usecase.ChallengeUnknown:
			// Muddat tugadi: yangi buyruq berish o'rniga oqim to'xtatiladi.
			// Cheksiz qayta urinish hujumchiga kerakli buyruq chiqquncha
			// kutish imkonini berardi.
			s.handler.writeEvent(s.conn, event{
				Type: "error", FrameID: result.FrameID,
				Code:           domain.CodeSpoofDetected,
				Message:        "Tekshiruv muddati tugadi. Sahifani yangilab, qaytadan urinib ko'ring.",
				ChallengeState: state,
			})
			return false
		default:
			// Hali bajarilmadi — foydalanuvchiga holatni ko'rsatamiz va
			// vektorni SO'RAMAYMIZ.
			s.wantEmbedding.Store(false)
			return s.handler.writeEvent(s.conn, event{
				Type: "liveness", FrameID: result.FrameID,
				Liveness: result.Liveness, BBox: &result.Face.BBox,
				Pose: result.Face.Pose, ChallengeState: state,
				ProcessingMs: result.ProcessingMs,
			})
		}
	}

	// --- Tirik va tekshiruvdan o'tgan. Vektor keldimi? ---
	if result.Face != nil && result.Face.Embedding.Dim() > 0 {
		s.wantEmbedding.Store(false)
		s.liveStreak = 0
		s.pausedUntil = time.Now().Add(verifyCooldown)
		return s.matchAndReport(ctx, result)
	}

	// --- Tirik, lekin hali faqat liveness. Ketma-ketlikni sanaymiz. ---
	s.liveStreak++
	if s.liveStreak >= s.handler.liveStreak && time.Now().After(s.pausedUntil) {
		// Keyingi kadrni to'liq rejimda so'raymiz.
		s.wantEmbedding.Store(true)
	}

	return s.handler.writeEvent(s.conn, event{
		Type: "liveness", FrameID: result.FrameID,
		Liveness: result.Liveness, BBox: &result.Face.BBox,
		ProcessingMs: result.ProcessingMs,
	})
}

// matchAndReport — STEP 3: vektorni bazadan qidiradi.
//
// Bu chaqiruv oqimni qisqa muddat bloklaydi (~5 ms) — bu ataylab: natija
// kelmaguncha yangi to'liq tekshiruv boshlanmaydi.
func (s *streamSession) matchAndReport(ctx context.Context, result *domain.FrameResult) bool {
	// Yakuniy tasdiq va buyruqni ISHLATIB YUBORISH. Observe() faqat holatni
	// kuzatadi; aynan shu chaqiruv buyruqni bir martalik qiladi, ya'ni
	// muvaffaqiyatli seansni qayta o'ynatib bo'lmaydi.
	if s.challengeNonce != "" {
		if err := s.handler.challenges.Consume(s.challengeNonce); err != nil {
			s.challengeNonce = ""
			domainErr, _ := domain.AsError(err)
			evt := event{Type: "error", FrameID: result.FrameID}
			if domainErr != nil {
				evt.Code = domainErr.Code
				evt.Message = domainErr.Message
			}
			s.handler.writeEvent(s.conn, evt)
			return false
		}
		// Ishlatildi — ikkinchi marta tekshirilmaydi.
		s.challengeNonce = ""
	}

	verification, err := s.handler.verifier.MatchEmbedding(ctx, result.Face, result.Liveness, s.requestID)
	if err != nil {
		domainErr, _ := domain.AsError(err)
		evt := event{Type: "error", FrameID: result.FrameID, BBox: &result.Face.BBox}
		if domainErr != nil {
			evt.Code = domainErr.Code
			evt.Message = domainErr.Message
			evt.Details = domainErr.Details
			evt.Liveness = domainErr.Liveness
			if domainErr.Code == domain.CodePersonNotFound {
				evt.Type = "not_found"
			}
		} else {
			evt.Code = domain.CodeInternal
			evt.Message = "Ichki xatolik."
		}
		return s.handler.writeEvent(s.conn, evt)
	}

	return s.handler.writeEvent(s.conn, event{
		Type:         "recognized",
		FrameID:      result.FrameID,
		PersonID:     verification.Person.ID,
		Name:         verification.Person.Name,
		Score:        verification.Score,
		Liveness:     verification.Liveness,
		BBox:         &result.Face.BBox,
		ProcessingMs: result.ProcessingMs,
	})
}

func (h *StreamHandler) writeEvent(conn *websocket.Conn, evt event) bool {
	_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
	if err := conn.WriteJSON(evt); err != nil {
		return false
	}
	return true
}
