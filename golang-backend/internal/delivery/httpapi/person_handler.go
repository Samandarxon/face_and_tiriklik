package httpapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// PersonHandler — inson bilan bog'liq endpointlar.
type PersonHandler struct {
	registrar      *usecase.Registrar
	verifier       *usecase.Verifier
	confirmer      *usecase.Confirmer
	livenessProbe  *usecase.LivenessProbe
	directory      *usecase.Directory
	resolver       *usecase.Resolver
	audit          domain.AuditSink
	metrics        *Metrics
	maxUploadBytes int64
}

func NewPersonHandler(
	registrar *usecase.Registrar,
	verifier *usecase.Verifier,
	confirmer *usecase.Confirmer,
	probe *usecase.LivenessProbe,
	directory *usecase.Directory,
	resolver *usecase.Resolver,
	audit domain.AuditSink,
	metrics *Metrics,
	maxUploadBytes int64,
) *PersonHandler {
	return &PersonHandler{
		registrar:      registrar,
		verifier:       verifier,
		confirmer:      confirmer,
		livenessProbe:  probe,
		directory:      directory,
		resolver:       resolver,
		audit:          audit,
		metrics:        metrics,
		maxUploadBytes: maxUploadBytes,
	}
}

// finish — amal yakunini metrikaga va audit jurnaliga yozadi.
//
// Handler'lar bu chaqiruvni takrorlamasligi uchun bitta joyda: kim so'radi,
// nima bo'ldi, qancha vaqt ketdi — hammasi shu yerda qayd etiladi.
func (h *PersonHandler) finish(
	ac auditContext, action, personID string, score float32, err error,
) {
	code := domain.Code("")
	result := "ok"
	if err != nil {
		code = domain.CodeOf(err)
		result = "error"
	}

	elapsed := time.Since(ac.Started)
	h.metrics.Observe(action, ac.ClientName, code, elapsed)
	h.audit.Record(domain.AuditEntry{
		ClientID:  ac.ClientID,
		KeyID:     ac.KeyID,
		Action:    action,
		PersonID:  personID,
		Result:    result,
		Code:      code,
		Score:     score,
		LatencyMs: int(elapsed.Milliseconds()),
		RequestID: ac.RequestID,
		IP:        ac.IP,
	})
}

// ============================================================== POST /register

// Register godoc
//
//	@Summary		Yuzni bazaga qo'shish
//	@Description	`subject_id` (PNFL) yuborilsa, servis uni normallashtirib AYNAN o'zini `person_id` qilib oladi. Muqobil: tayyor `person_id` yuborish yoki ikkalasini ham qoldirmaslik (tasodifiy ID beriladi).
//	@Tags			persons
//	@Security		ApiKeyAuth
//	@Accept			multipart/form-data
//	@Produce		json
//	@Param			image		formData	file	true	"Yuz tasviri"
//	@Param			subject_id	formData	string	false	"PNFL (14 raqam)"
//	@Param			person_id	formData	string	false	"Tayyor person_id"
//	@Param			name				formData	string	false	"Ixtiyoriy yorliq"
//	@Param			require_liveness	formData	bool	false	"Jonli kameradan kelgan kadr uchun anti-spoofing (standart: false)"
//	@Success		201			{object}	map[string]any
//	@Failure		400			{object}	errorBody
//	@Failure		401			{object}	errorBody
//	@Failure		403			{object}	errorBody
//	@Failure		422			{object}	errorBody
//	@Router			/persons/register [post]
//
// AddSample — mavjud shaxsning galereyasiga qo'shimcha yuz namunasi qo'shadi.
//
// `Register` asosiy vektorni ALMASHTIRADI, bu esa QO'SHADI. 10-20 yillik
// arxivda bu farq hal qiluvchi: eski rasm yagona bo'lishi mumkin va uni
// yangisi bilan almashtirish qaytarib bo'lmaydigan yo'qotish.
func (h *PersonHandler) AddSample(c *gin.Context) {
	ac := auditContextOf(c)

	image, filename, ok := h.readImage(c)
	if !ok {
		return
	}
	requireLiveness, _ := strconv.ParseBool(c.PostForm("require_liveness"))

	personID := c.Param("person_id")
	result, err := h.registrar.AddSample(c.Request.Context(),
		personID, c.PostForm("sample"), usecase.RegisterCommand{
			Image:           image,
			FileName:        filename,
			RequestID:       requestIDOf(c),
			RequireLiveness: requireLiveness,
		})
	if err != nil {
		h.finish(ac, "sample.add", personID, 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "sample.add", result.Person.ID, 0, nil)

	respondOK(c, http.StatusCreated, gin.H{
		"message":   "Namuna galereyaga qo'shildi.",
		"person_id": result.Person.ID,
		"sample":    c.PostForm("sample"),
		"point_id":  result.PointID,
	})
}

func (h *PersonHandler) Register(c *gin.Context) {
	ac := auditContextOf(c)

	image, filename, ok := h.readImage(c)
	if !ok {
		return
	}

	// require_liveness — kadr JONLI kameradan kelganda yoqiladi.
	//
	// Bu maydon RegisterCommand da bor edi, lekin HTTP orqali unga qiymat
	// berib bo'lmasdi: handler uni o'qimasdi. Ya'ni anti-spoofing bilan
	// ro'yxatga olish imkoniyati kodda mavjud bo'lsa ham amalda o'chiq edi.
	//
	// Standart FALSE bo'lib qolishi shart: hodimlar odatda passport skani yoki
	// arxiv fotosidan ro'yxatga olinadi, ular esa ta'rifga ko'ra "spoof" —
	// standart true bo'lsa oddiy ro'yxatga olish butunlay ishlamay qolardi.
	requireLiveness, _ := strconv.ParseBool(c.PostForm("require_liveness"))

	result, err := h.registrar.Execute(c.Request.Context(), usecase.RegisterCommand{
		SubjectID:       c.PostForm("subject_id"),
		PersonID:        c.PostForm("person_id"),
		Name:            c.PostForm("name"),
		Image:           image,
		FileName:        filename,
		RequestID:       requestIDOf(c),
		RequireLiveness: requireLiveness,
	})
	if err != nil {
		h.finish(ac, "register", "", 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "register", result.Person.ID, 0, nil)

	status := http.StatusCreated
	message := "Inson muvaffaqiyatli ro'yxatdan o'tkazildi."
	if result.Updated {
		status = http.StatusOK
		message = "Mavjud insonning yuz vektori yangilandi."
	}

	body := gin.H{
		"message":      message,
		"person_id":    result.Person.ID,
		"generated_id": result.GeneratedID,
		"name":         result.Person.Name,
		"point_id":     result.PointID,
		"updated":      result.Updated,
		"det_score":    result.Face.DetScore,
		"bbox":         result.Face.BBox,
		"vector_dim":   result.Face.Embedding.Dim(),
	}
	// Anti-spoofing faqat `RequireLiveness` bilan ishlaydi. Ishlamagan holatda
	// maydonni umuman qo'shmaymiz — `"liveness": null` mijozga tekshiruv o'tgan
	// bo'lib ko'rinishi mumkin.
	if result.Liveness != nil {
		body["liveness"] = result.Liveness
	}
	respondOK(c, status, body)
}

// ================================================================ POST /verify

// Verify godoc
//
//	@Summary		Yuz orqali tanish
//	@Description	Avval anti-spoofing, keyin vektor qidiruvi. Kadr soxta bo'lsa bazaga so'rov UMUMAN yuborilmaydi va `400 spoof_detected` qaytadi.
//	@Tags			persons
//	@Security		ApiKeyAuth
//	@Accept			multipart/form-data
//	@Produce		json
//	@Param			image	formData	file	true	"Kamera kadri"
//	@Success		200		{object}	map[string]any
//	@Failure		400		{object}	errorBody	"spoof_detected"
//	@Failure		404		{object}	errorBody	"person_not_found"
//	@Failure		422		{object}	errorBody	"no_face_detected / low_quality_face"
//	@Router			/persons/verify [post]
//
// Search — 1:N nomzodlar ro'yxati. QAROR QABUL QILMAYDI.
//
// `Verify` bitta javob beradi va chegara qo'llaydi — turniket uchun to'g'ri.
// Bu esa ranjirlangan ro'yxatni qaytaradi va chegarani qo'llamaydi: tergov
// qidiruvida maqsad hech kimni o'tkazib yubormaslik, aniqlikni esa odam
// ta'minlaydi. Javobda `decided` maydoni ataylab YO'Q.
func (h *PersonHandler) Search(c *gin.Context) {
	ac := auditContextOf(c)

	image, _, ok := h.readImage(c)
	if !ok {
		return
	}
	limit, _ := strconv.Atoi(c.DefaultPostForm("limit", "10"))
	requireLiveness, _ := strconv.ParseBool(c.PostForm("require_liveness"))

	result, err := h.verifier.Search(
		c.Request.Context(), image, limit, requireLiveness, requestIDOf(c))
	if err != nil {
		h.finish(ac, "search", "", 0, err)
		respondError(c, err)
		return
	}

	best := float32(0)
	if len(result.Candidates) > 0 {
		best = result.Candidates[0].Score
	}
	h.finish(ac, "search", "", best, nil)

	candidates := make([]gin.H, 0, len(result.Candidates))
	for _, m := range result.Candidates {
		candidates = append(candidates, gin.H{
			"person_id": m.Person.ID,
			"name":      m.Person.Name,
			"score":     m.Score,
			// Nechta namuna mos keldi — ko'p namunali moslik bitta
			// namunalidan ishonchliroq, va bu re-ranking uchun signal.
			"samples":         m.Samples,
			"above_threshold": m.Score >= result.Threshold,
		})
	}

	respondOK(c, http.StatusOK, gin.H{
		"candidates": candidates,
		"count":      len(candidates),
		// Ma'lumot uchun: qaysi chegara QO'LLANMAGANINI bilish uchun.
		"threshold": result.Threshold,
		"det_score": result.Face.DetScore,
		"note":      "Bu nomzodlar ro'yxati, qaror emas. Yakuniy tasdiqni odam beradi.",
	})
}

func (h *PersonHandler) Verify(c *gin.Context) {
	ac := auditContextOf(c)

	image, _, ok := h.readImage(c)
	if !ok {
		return
	}

	result, err := h.verifier.Execute(c.Request.Context(), image, requestIDOf(c))
	if err != nil {
		h.finish(ac, "verify", "", 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "verify", result.Person.ID, result.Score, nil)

	respondOK(c, http.StatusOK, gin.H{
		"message":       "Inson muvaffaqiyatli tanildi.",
		"person_id":     result.Person.ID,
		"name":          result.Person.Name,
		"score":         result.Score,
		"threshold":     result.Threshold,
		"registered_at": result.Person.CreatedAt.Format(time.RFC3339),
		"liveness":      result.Liveness,
		"det_score":     result.Face.DetScore,
		"bbox":          result.Face.BBox,
	})
}

// ============================================ POST /persons/:person_id/verify

// Confirm godoc
//
//	@Summary		Yuzni tasdiqlash (1:1 verification)
//	@Description	"Bu odam haqiqatan shu odammi?" — butun baza bo'ylab qidirmaydi, faqat ko'rsatilgan `person_id` (PNFL) bilan solishtiradi. Avval anti-spoofing. `matched: false` ham 200 bilan qaytadi: bu xato emas, javob.
//	@Tags			persons
//	@Security		ApiKeyAuth
//	@Accept			multipart/form-data
//	@Produce		json
//	@Param			person_id	path		string	true	"PNFL"
//	@Param			image		formData	file	true	"Kamera kadri"
//	@Success		200			{object}	map[string]any
//	@Failure		400			{object}	errorBody	"spoof_detected"
//	@Failure		404			{object}	errorBody	"person_not_found"
//	@Failure		422			{object}	errorBody	"no_face_detected / low_quality_face"
//	@Router			/persons/{person_id}/verify [post]
func (h *PersonHandler) Confirm(c *gin.Context) {
	ac := auditContextOf(c)
	personID := strings.TrimSpace(c.Param("person_id"))

	image, _, ok := h.readImage(c)
	if !ok {
		return
	}

	result, err := h.confirmer.Execute(c.Request.Context(), personID, image, requestIDOf(c))
	if err != nil {
		h.finish(ac, "confirm", personID, 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "confirm", personID, result.Score, nil)

	message := "Yuz tasdiqlandi."
	if !result.Matched {
		message = "Yuz mos kelmadi."
	}
	respondOK(c, http.StatusOK, gin.H{
		"message":   message,
		"person_id": result.PersonID,
		"matched":   result.Matched,
		"score":     result.Score,
		"threshold": result.Threshold,
		"name":      result.Person.Name,
		"liveness":  result.Liveness,
		"det_score": result.Face.DetScore,
		"bbox":      result.Face.BBox,
	})
}

// ====================================================== POST /liveness/check

// CheckLiveness — PROBE endpoint: soxta kadrda ham `200` qaytadi.
// Mijoz uni har kadrda chaqirib jonli ko'rsatkich chizadi; avtorizatsiya
// qarori esa faqat `/persons/verify` da chiqariladi.
func (h *PersonHandler) CheckLiveness(c *gin.Context) {
	image, _, ok := h.readImage(c)
	if !ok {
		return
	}

	result, err := h.livenessProbe.Execute(c.Request.Context(), image, requestIDOf(c))
	if err != nil {
		respondError(c, err)
		return
	}

	respondOK(c, http.StatusOK, gin.H{
		"liveness": result.Liveness,
		"bbox":     result.Face.BBox,
	})
}

// ============================================================= POST /resolve

// resolveRequest — rasmsiz identifikatsiya so'rovi.
type resolveRequest struct {
	SubjectID string `json:"subject_id"`
}

// Resolve godoc
//
//	@Summary		subject_id dan person_id ni hisoblash (rasmsiz)
//	@Description	Yuz orqali tanish ishlamagan holatlar uchun zaxira yo'l: operator PNFL kiritadi, mijoz shu yerdan `person_id` ni va yozuv holatini oladi. `person_id` — normallashtirilgan PNFL ning o'zi.
//	@Tags			persons
//	@Security		ApiKeyAuth
//	@Accept			json
//	@Produce		json
//	@Param			body	body		resolveRequest	true	"Tashqi identifikator"
//	@Success		200		{object}	map[string]any
//	@Failure		400		{object}	errorBody
//	@Router			/persons/resolve [post]
func (h *PersonHandler) Resolve(c *gin.Context) {
	ac := auditContextOf(c)

	var req resolveRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		// Formadan ham qabul qilamiz — mijozlar uchun qulaylik.
		req.SubjectID = c.PostForm("subject_id")
	}

	result, err := h.resolver.Execute(c.Request.Context(), req.SubjectID)
	if err != nil {
		h.finish(ac, "resolve", "", 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "resolve", result.PersonID, 0, nil)

	respondOK(c, http.StatusOK, gin.H{
		"person_id":  result.PersonID,
		"registered": result.Registered,
		"person":     result.Person,
	})
}

// ============================================================ CRUD endpointlar

// Get godoc
//
//	@Summary	person_id bo'yicha yozuvni olish
//	@Tags		persons
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		person_id	path		string	true	"person_id"
//	@Success	200			{object}	map[string]any
//	@Failure	404			{object}	errorBody
//	@Router		/persons/{person_id} [get]
func (h *PersonHandler) Get(c *gin.Context) {
	person, err := h.directory.Get(c.Request.Context(), strings.TrimSpace(c.Param("person_id")))
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{"person": person})
}

// Delete godoc
//
//	@Summary	Yozuvni bazadan o'chirish
//	@Tags		persons
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		person_id	path		string	true	"person_id"
//	@Success	200			{object}	map[string]any
//	@Failure	404			{object}	errorBody
//	@Router		/persons/{person_id} [delete]
func (h *PersonHandler) Delete(c *gin.Context) {
	ac := auditContextOf(c)

	personID := strings.TrimSpace(c.Param("person_id"))
	if err := h.directory.Delete(c.Request.Context(), personID); err != nil {
		h.finish(ac, "delete", personID, 0, err)
		respondError(c, err)
		return
	}
	h.finish(ac, "delete", personID, 0, nil)
	respondOK(c, http.StatusOK, gin.H{
		"message":   "Inson bazadan o'chirildi.",
		"person_id": personID,
	})
}

func (h *PersonHandler) List(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))

	page, err := h.directory.List(c.Request.Context(), limit, c.Query("offset"))
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{
		"persons":     page.People,
		"count":       len(page.People),
		"next_offset": page.NextOffset,
	})
}

// ================================================================= yordamchi

// readImage — multipart'dan rasmni o'qiydi va validatsiya qiladi.
func (h *PersonHandler) readImage(c *gin.Context) (data []byte, filename string, ok bool) {
	fileHeader, err := firstImageFile(c)
	if err != nil {
		respondErrorCode(c, domain.CodeInvalidInput,
			"Rasm fayli topilmadi. `image` (yoki `photo`/`file`) maydonini yuboring.")
		return nil, "", false
	}
	if fileHeader.Size <= 0 {
		respondErrorCode(c, domain.CodeInvalidInput, "Rasm fayli bo'sh.")
		return nil, "", false
	}
	if fileHeader.Size > h.maxUploadBytes {
		respondErrorCode(c, domain.CodeInvalidInput,
			fmt.Sprintf("Rasm hajmi juda katta (maksimum %d MB).", h.maxUploadBytes/(1024*1024)))
		return nil, "", false
	}

	file, err := fileHeader.Open()
	if err != nil {
		respondErrorCode(c, domain.CodeInvalidInput, "Rasm faylini ochib bo'lmadi.")
		return nil, "", false
	}
	defer file.Close()

	data, err = io.ReadAll(io.LimitReader(file, h.maxUploadBytes))
	if err != nil {
		respondErrorCode(c, domain.CodeInvalidInput, "Rasm faylini o'qib bo'lmadi.")
		return nil, "", false
	}
	if !strings.HasPrefix(http.DetectContentType(data), "image/") {
		respondErrorCode(c, domain.CodeInvalidInput, "Fayl rasm emas. JPEG/PNG/WEBP yuboring.")
		return nil, "", false
	}

	return data, sanitizeFilename(fileHeader.Filename), true
}

// firstImageFile — bir nechta odatiy maydon nomini sinab ko'radi.
func firstImageFile(c *gin.Context) (*multipart.FileHeader, error) {
	for _, field := range []string{"image", "photo", "file"} {
		if fh, err := c.FormFile(field); err == nil {
			return fh, nil
		}
	}
	return nil, errors.New("rasm maydoni topilmadi")
}

func sanitizeFilename(name string) string {
	name = strings.TrimSpace(name)
	if idx := strings.LastIndexAny(name, `/\`); idx >= 0 {
		name = name[idx+1:]
	}
	if name == "" {
		return "upload.jpg"
	}
	return name
}

func requestIDOf(c *gin.Context) string { return c.GetString("request_id") }

// ===================================================================== /health

// HealthChecker — bog'liq tizimning holatini tekshiruvchi (Postgres kabi).
type HealthChecker interface {
	Healthy(ctx context.Context) error
}

// HealthHandler — tizim holati.
type HealthHandler struct {
	analyzer   domain.FaceAnalyzer
	repo       domain.PersonRepository
	store      HealthChecker
	collection string
}

func NewHealthHandler(
	analyzer domain.FaceAnalyzer,
	repo domain.PersonRepository,
	store HealthChecker,
	collection string,
) *HealthHandler {
	return &HealthHandler{analyzer: analyzer, repo: repo, store: store, collection: collection}
}

// Check godoc
//
//	@Summary		Yengil holat tekshiruvi
//	@Description	Orkestrator (Docker/Kubernetes) uchun. Kalit talab qilinmaydi.
//	@Tags			health
//	@Produce		json
//	@Success		200	{object}	map[string]any
//	@Router			/health [get]
//
// Bu endpoint "servis tirikmi" degan savolga javob beradi — bog'liq tizimlarga
// murojaat qilmaydi, shuning uchun tez va arzon.
func (h *HealthHandler) Check(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "face-api",
	})
}

// Ready godoc
//
//	@Summary		To'liq tayyorlik tekshiruvi
//	@Description	AI servis, Qdrant va Postgres holatini tekshiradi. Bittasi ishlamasa `503` qaytadi.
//	@Tags			health
//	@Produce		json
//	@Success		200	{object}	map[string]any
//	@Failure		503	{object}	map[string]any
//	@Router			/health/ready [get]
func (h *HealthHandler) Ready(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	body := gin.H{"service": "golang-backend", "collection": h.collection, "transport": "grpc"}
	healthy := true

	if status, err := h.analyzer.Health(ctx); err != nil {
		healthy = false
		body["ai_service"] = gin.H{"ok": false, "error": err.Error()}
	} else {
		body["ai_service"] = gin.H{"ok": true, "details": status}
	}

	if err := h.repo.Healthy(ctx); err != nil {
		healthy = false
		body["qdrant"] = gin.H{"ok": false, "error": err.Error()}
	} else {
		count, _ := h.repo.Count(ctx)
		body["qdrant"] = gin.H{"ok": true, "registered_persons": count}
	}

	if h.store != nil {
		if err := h.store.Healthy(ctx); err != nil {
			healthy = false
			body["postgres"] = gin.H{"ok": false, "error": err.Error()}
		} else {
			body["postgres"] = gin.H{"ok": true}
		}
	}

	body["status"] = "ok"
	status := http.StatusOK
	if !healthy {
		body["status"] = "degraded"
		status = http.StatusServiceUnavailable
	}
	c.JSON(status, body)
}
