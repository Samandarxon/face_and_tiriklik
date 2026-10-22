// Package faceai — `domain.FaceAnalyzer` portining gRPC implementatsiyasi.
//
// Bu paket protokolga oid BARCHA tafsilotlarni o'z ichida ushlab turadi:
// proto tuzilmalari, gRPC status kodlari, ulanish sozlamalari. Use-case
// qatlami faqat `domain` tiplarini ko'radi.
package faceai

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	facev1 "github.com/face_detecting_microservice/golang-backend/genproto/face/v1"
	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Client — AI servisga gRPC mijoz.
type Client struct {
	conn    *grpc.ClientConn
	rpc     facev1.FaceServiceClient
	timeout time.Duration
}

// Config — ulanish sozlamalari.
type Config struct {
	Target         string        // "ai-service:50051"
	Timeout        time.Duration // unary chaqiruvlar uchun
	MaxMessageSize int
	// TLS — nil bo'lmasa ulanish TLS bilan (AI servis sertifikati shu CA
	// bilan tekshiriladi, mTLS uchun mijoz sertifikati ham shu yerda).
	// Nil: avvalgidek ochiq kanal, faqat yopiq ichki tarmoqda.
	TLS *tls.Config
}

// New — ulanishni yaratadi. `grpc.NewClient` lazy ishlaydi: haqiqiy ulanish
// birinchi chaqiruvda o'rnatiladi, shuning uchun servislar navbat bilan
// ko'tarilishi muammo tug'dirmaydi.
func New(cfg Config) (*Client, error) {
	if cfg.MaxMessageSize <= 0 {
		cfg.MaxMessageSize = 16 << 20
	}

	transport := insecure.NewCredentials()
	if cfg.TLS != nil {
		transport = credentials.NewTLS(cfg.TLS)
	}

	conn, err := grpc.NewClient(cfg.Target,
		grpc.WithTransportCredentials(transport),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxMessageSize),
			grpc.MaxCallSendMsgSize(cfg.MaxMessageSize),
		),
		// Streaming ulanishlari NAT/proxy tomonidan uzilib qolmasligi uchun.
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("ai-service ulanishini yaratib bo'lmadi: %w", err)
	}

	return &Client{conn: conn, rpc: facev1.NewFaceServiceClient(conn), timeout: cfg.Timeout}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// LoadTLS — AI_GRPC_TLS_* fayllaridan mijoz TLS sozlamasini yasaydi.
//
// caFile bo'sh bo'lsa nil qaytadi, ya'ni TLS o'chiq (avvalgi xatti-harakat).
// CA berilsa AI servis sertifikati shu CA bilan tekshiriladi; certFile va
// keyFile ham berilsa — mTLS: AI servis faqat shu sertifikat egasini
// qabul qiladi. Bittasi berilib ikkinchisi berilmasa — xato, jimgina
// sertifikatsiz ulanib ketilmaydi.
func LoadTLS(caFile, certFile, keyFile, serverName string) (*tls.Config, error) {
	if caFile == "" {
		if certFile != "" || keyFile != "" {
			return nil, fmt.Errorf("AI_GRPC_TLS_CA_FILE ko'rsatilmagan, lekin mijoz sertifikati berilgan")
		}
		return nil, nil
	}
	pem, err := os.ReadFile(caFile) // #nosec G304 — operator bergan yo'l
	if err != nil {
		return nil, fmt.Errorf("AI gRPC CA o'qilmadi: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("AI gRPC CA faylida sertifikat topilmadi: %s", caFile)
	}
	out := &tls.Config{RootCAs: pool, ServerName: serverName, MinVersion: tls.VersionTLS12}

	if (certFile == "") != (keyFile == "") {
		return nil, fmt.Errorf("AI_GRPC_TLS_CERT_FILE va AI_GRPC_TLS_KEY_FILE birga berilishi kerak")
	}
	if certFile != "" {
		pair, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return nil, fmt.Errorf("AI gRPC mijoz sertifikati yuklanmadi: %w", err)
		}
		out.Certificates = []tls.Certificate{pair}
	}
	return out, nil
}

// Interfeys bajarilganini kompilyatsiya vaqtida tekshiramiz.
var _ domain.FaceAnalyzer = (*Client)(nil)

// ------------------------------------------------------------ unary metodlar

func (c *Client) Embed(ctx context.Context, image []byte, requestID string) (*domain.Face, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	resp, err := c.rpc.Embed(ctx, &facev1.ImageRequest{Image: image, RequestId: requestID})
	if err != nil {
		return nil, translateError(err)
	}
	return faceFromProto(resp.GetFace(), resp.GetEmbedding()), nil
}

func (c *Client) VerifyFrame(ctx context.Context, image []byte, requestID string) (*domain.Face, *domain.Liveness, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	resp, err := c.rpc.VerifyFrame(ctx, &facev1.ImageRequest{Image: image, RequestId: requestID})
	if err != nil {
		return nil, nil, translateError(err)
	}
	return faceFromProto(resp.GetFace(), resp.GetEmbedding()), livenessFromProto(resp.GetLiveness()), nil
}

func (c *Client) CheckLiveness(ctx context.Context, image []byte, requestID string) (*domain.Face, *domain.Liveness, error) {
	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	resp, err := c.rpc.CheckLiveness(ctx, &facev1.ImageRequest{Image: image, RequestId: requestID})
	if err != nil {
		return nil, nil, translateError(err)
	}
	return faceFromProto(resp.GetFace(), nil), livenessFromProto(resp.GetLiveness()), nil
}

func (c *Client) Health(ctx context.Context) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	resp, err := c.rpc.Health(ctx, &facev1.HealthRequest{})
	if err != nil {
		return nil, translateError(err)
	}

	return map[string]any{
		"ready":                      resp.GetReady(),
		"device":                     resp.GetDevice(),
		"insightface_model":          resp.GetInsightfaceModel(),
		"embedding_dim":              resp.GetEmbeddingDim(),
		"liveness_enabled":           resp.GetLivenessEnabled(),
		"liveness_loaded":            resp.GetLivenessLoaded(),
		"liveness_models":            resp.GetLivenessModels(),
		"liveness_threshold":         resp.GetLivenessThreshold(),
		"liveness_require_unanimous": resp.GetLivenessRequireUnanimous(),
		"liveness_error":             resp.GetLivenessError(),
	}, nil
}

func (c *Client) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, c.timeout)
}

// --------------------------------------------------------------- streaming

// OpenStream — ikki tomonlama oqim ochadi. Kontekst bekor qilinishi oqimni
// yopadi, shuning uchun chaqiruvchi `defer stream.Close()` qilishi shart.
func (c *Client) OpenStream(ctx context.Context) (domain.FrameStream, error) {
	// Streamda `c.timeout` QO'LLANMAYDI: oqim daqiqalar davomida ochiq turadi.
	// Uni chaqiruvchi kontekst boshqaradi.
	stream, err := c.rpc.StreamFrames(ctx)
	if err != nil {
		return nil, translateError(err)
	}
	return &frameStream{stream: stream}, nil
}

type frameStream struct {
	stream grpc.BidiStreamingClient[facev1.FrameRequest, facev1.FrameResponse]
}

func (s *frameStream) Send(frame domain.FrameRequest) error {
	mode := facev1.StreamMode_STREAM_MODE_LIVENESS_ONLY
	if frame.WantEmbedding {
		mode = facev1.StreamMode_STREAM_MODE_FULL
	}
	err := s.stream.Send(&facev1.FrameRequest{
		Image:   frame.Image,
		FrameId: frame.FrameID,
		Mode:    mode,
	})
	if err != nil {
		return translateError(err)
	}
	return nil
}

func (s *frameStream) Recv() (*domain.FrameResult, error) {
	resp, err := s.stream.Recv()
	if err != nil {
		// Bu OQIMNING tugashi (yoki tarmoq xatosi), bitta kadrning emas.
		return nil, translateError(err)
	}

	result := &domain.FrameResult{
		FrameID:      resp.GetFrameId(),
		ProcessingMs: int(resp.GetProcessingMs()),
	}

	if !resp.GetOk() {
		// Kadr darajasidagi xato — oqim davom etadi.
		result.Err = errorFromProto(resp.GetError())
		return result, nil
	}

	result.Face = faceFromProto(resp.GetFace(), resp.GetEmbedding())
	result.Liveness = livenessFromProto(resp.GetLiveness())
	return result, nil
}

func (s *frameStream) CloseSend() error { return s.stream.CloseSend() }

func (s *frameStream) Close() error {
	// Bidi oqimda mijoz tomonidan yopish — CloseSend + kontekst bekor qilish.
	return s.stream.CloseSend()
}

// ------------------------------------------------------- proto <-> domain

func faceFromProto(info *facev1.FaceInfo, embedding []float32) *domain.Face {
	face := &domain.Face{}
	if info != nil {
		face.DetScore = float64(info.GetDetScore())
		// Bosh burilishi — aktiv liveness uchun. Eski AI servis versiyasi bu
		// maydonni yubormaydi; u holda Pose nil qoladi va aktiv tekshiruv
		// "ma'lumot yo'q" deb hisoblab, kadrni o'tkazmaydi.
		if pose := info.GetPose(); pose != nil {
			face.Pose = &domain.HeadPose{
				Yaw:       float64(pose.GetYaw()),
				Pitch:     float64(pose.GetPitch()),
				Roll:      float64(pose.GetRoll()),
				Available: pose.GetAvailable(),
			}
		}
		if bbox := info.GetBbox(); bbox != nil {
			face.BBox = domain.BoundingBox{
				X1: int(bbox.GetX1()), Y1: int(bbox.GetY1()),
				X2: int(bbox.GetX2()), Y2: int(bbox.GetY2()),
			}
		}
	}
	if len(embedding) > 0 {
		face.Embedding = domain.Embedding(embedding)
	}
	return face
}

func livenessFromProto(l *facev1.Liveness) *domain.Liveness {
	if l == nil {
		return nil
	}
	perModel := make(map[string]float64, len(l.GetPerModel()))
	for name, score := range l.GetPerModel() {
		perModel[name] = float64(score)
	}
	return &domain.Liveness{
		IsReal:           l.GetIsReal(),
		RealProbability:  float64(l.GetRealProbability()),
		SpoofProbability: float64(l.GetSpoofProbability()),
		Threshold:        float64(l.GetThreshold()),
		PerModel:         perModel,
	}
}

// codeMap — proto xato kodlarini domain kodlariga o'giradi.
var codeMap = map[facev1.ErrorCode]domain.Code{
	facev1.ErrorCode_ERROR_CODE_INVALID_IMAGE:        domain.CodeInvalidInput,
	facev1.ErrorCode_ERROR_CODE_NO_FACE_DETECTED:     domain.CodeNoFaceDetected,
	facev1.ErrorCode_ERROR_CODE_MULTIPLE_FACES:       domain.CodeMultipleFaces,
	facev1.ErrorCode_ERROR_CODE_LOW_QUALITY_FACE:     domain.CodeLowQualityFace,
	facev1.ErrorCode_ERROR_CODE_SPOOF_DETECTED:       domain.CodeSpoofDetected,
	facev1.ErrorCode_ERROR_CODE_LIVENESS_UNAVAILABLE: domain.CodeAIUnavailable,
	facev1.ErrorCode_ERROR_CODE_MODEL_NOT_READY:      domain.CodeAIUnavailable,
	facev1.ErrorCode_ERROR_CODE_INTERNAL:             domain.CodeInternal,
}

func errorFromProto(fe *facev1.FaceError) *domain.Error {
	if fe == nil {
		return domain.NewError(domain.CodeInternal, "AI servis noma'lum xato qaytardi.")
	}

	code, ok := codeMap[fe.GetCode()]
	if !ok {
		code = domain.CodeInternal
	}

	err := domain.NewError(code, fe.GetMessage())
	if details := fe.GetDetails(); len(details) > 0 {
		converted := make(map[string]any, len(details))
		for k, v := range details {
			converted[k] = v
		}
		err = err.WithDetails(converted)
	}
	if fe.GetLiveness() != nil {
		err = err.WithLiveness(livenessFromProto(fe.GetLiveness()))
	}
	return err
}

// translateError — gRPC xatosini domain xatosiga o'giradi.
//
// Biznes xatolari `FaceError` detali sifatida keladi; detal bo'lmasa —
// bu transport darajasidagi muammo (servis o'chiq, timeout, ...).
func translateError(err error) error {
	if err == nil {
		return nil
	}

	st, ok := status.FromError(err)
	if !ok {
		return domain.WrapError(domain.CodeAIUnavailable, "AI servis bilan aloqa uzildi.", err)
	}

	for _, detail := range st.Details() {
		if faceErr, ok := detail.(*facev1.FaceError); ok {
			return errorFromProto(faceErr)
		}
	}

	switch st.Code() {
	case codes.Unavailable, codes.DeadlineExceeded, codes.Canceled:
		return domain.WrapError(domain.CodeAIUnavailable,
			"AI servis javob bermayapti. Keyinroq urinib ko'ring.", err)
	case codes.ResourceExhausted:
		return domain.WrapError(domain.CodeInvalidInput,
			"Rasm hajmi juda katta.", err)
	default:
		return domain.WrapError(domain.CodeInternal, "AI servisda xatolik yuz berdi.", err)
	}
}
