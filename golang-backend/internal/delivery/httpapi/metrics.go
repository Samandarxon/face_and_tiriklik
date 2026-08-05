package httpapi

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Metrics — Prometheus ko'rsatkichlari.
//
// Nima uchun aynan shular: yuz tanish tizimida eng muhim savollar — "qancha
// so'rov keldi", "qanchasi soxta deb rad etildi", "javob qancha vaqt oldi" va
// "AI/vektor bazasi ishlayaptimi". Ko'rsatkichlar mijoz (`client`) bo'yicha
// ajratilgan, shunda qaysi platforma qancha yuk berayotgani ko'rinadi.
type Metrics struct {
	requests   *prometheus.CounterVec
	duration   *prometheus.HistogramVec
	spoofs     *prometheus.CounterVec
	streams    prometheus.Gauge
	registry   *prometheus.Registry
	authFailed *prometheus.CounterVec
}

// NewMetrics — o'z registrini yasaydi (global holatga tegmaydi, shu tufayli
// testda ham xotirjam ishlatiladi).
func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()

	m := &Metrics{
		registry: reg,
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "face_requests_total",
			Help: "Amal va natija kodi bo'yicha so'rovlar soni.",
		}, []string{"action", "code", "client"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "face_request_duration_seconds",
			Help: "So'rovni qayta ishlash davomiyligi.",
			// Yuz tanish odatda 50 ms – 2 s oralig'ida; chegaralar shunga mos.
			Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10},
		}, []string{"action"}),
		spoofs: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "face_spoof_detected_total",
			Help: "Anti-spoofing rad etgan kadrlar soni.",
		}, []string{"client"}),
		streams: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "face_active_streams",
			Help: "Ayni damdagi ochiq WebSocket oqimlari soni.",
		}),
		authFailed: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "face_auth_failures_total",
			Help: "Rad etilgan autentifikatsiya urinishlari.",
		}, []string{"reason"}),
	}

	reg.MustRegister(m.requests, m.duration, m.spoofs, m.streams, m.authFailed)
	reg.MustRegister(collectors.NewGoCollector())
	reg.MustRegister(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	return m
}

// Observe — bitta amal yakunlanganini qayd etadi.
func (m *Metrics) Observe(action, client string, code domain.Code, elapsed time.Duration) {
	if m == nil {
		return
	}
	codeLabel := string(code)
	if codeLabel == "" {
		codeLabel = "ok"
	}
	if client == "" {
		client = "unknown"
	}

	m.requests.WithLabelValues(action, codeLabel, client).Inc()
	m.duration.WithLabelValues(action).Observe(elapsed.Seconds())
	if code == domain.CodeSpoofDetected {
		m.spoofs.WithLabelValues(client).Inc()
	}
}

// AuthFailure — autentifikatsiya rad etildi.
func (m *Metrics) AuthFailure(reason string) {
	if m == nil {
		return
	}
	m.authFailed.WithLabelValues(reason).Inc()
}

// StreamOpened / StreamClosed — ochiq oqimlar hisobi.
func (m *Metrics) StreamOpened() {
	if m != nil {
		m.streams.Inc()
	}
}

func (m *Metrics) StreamClosed() {
	if m != nil {
		m.streams.Dec()
	}
}

// HTTPHandler — `/metrics` endpointi uchun.
func (m *Metrics) HTTPHandler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}
