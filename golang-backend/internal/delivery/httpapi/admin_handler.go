package httpapi

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
	"github.com/face_detecting_microservice/golang-backend/internal/usecase"
)

// AuditLister — jurnalni o'qish (admin uchun). Delivery qatlami Postgres
// adapterini bilmasligi uchun kichik interfeys sifatida e'lon qilingan.
type AuditLister interface {
	ListAudit(ctx context.Context, clientID string, limit, offset int) ([]domain.AuditEntry, error)
	CountAudit(ctx context.Context, clientID string) (int, error)
}

// AdminHandler — mijozlar va API kalitlarni boshqarish.
//
// Bu endpointlar `admin:keys` scope'i bilan himoyalangan. Aynan shu yerda
// boshqa platformalarga kalit beriladi.
type AdminHandler struct {
	keys  *usecase.KeyManager
	auth  *usecase.Authenticator
	audit AuditLister
	// sink yozuv uchun. Ilgari bu yerda faqat AuditLister bor edi, ya'ni
	// boshqaruv amallari — kalit berish, bekor qilish, o'chirish, mijoz
	// yaratish — HECH QAYERGA yozilmasdi. Aynan shu amallar servisdagi eng
	// nozik amallar: ular kimga biometrik bazaga kirish berilganini
	// belgilaydi. Jurnalda esa faqat yuz tanish so'rovlari ko'rinardi.
	sink domain.AuditSink
}

func NewAdminHandler(
	keys *usecase.KeyManager,
	auth *usecase.Authenticator,
	audit AuditLister,
	sink domain.AuditSink,
) *AdminHandler {
	return &AdminHandler{keys: keys, auth: auth, audit: audit, sink: sink}
}

// record — boshqaruv amalini jurnalga yozadi.
//
// `PersonID` o'rniga amal tegishli bo'lgan obyekt (mijoz yoki kalit) idsi
// yoziladi: jurnal ustuni "nimaga nisbatan qilindi" degan savolga javob
// beradi, va boshqaruv amallarida bu shaxs emas, yozuv bo'ladi.
func (h *AdminHandler) record(c *gin.Context, action, target string, err error) {
	if h.sink == nil {
		return
	}
	ac := auditContextOf(c)
	result := "ok"
	var code domain.Code
	if err != nil {
		result, code = "error", domain.CodeOf(err)
	}
	h.sink.Record(domain.AuditEntry{
		ClientID:  ac.ClientID,
		KeyID:     ac.KeyID,
		Action:    action,
		PersonID:  target,
		Result:    result,
		Code:      code,
		LatencyMs: int(time.Since(ac.Started).Milliseconds()),
		RequestID: ac.RequestID,
		IP:        ac.IP,
	})
}

// ============================================================== Mijozlar

type clientRequest struct {
	Name        string `json:"name"`
	Contact     string `json:"contact"`
	Description string `json:"description"`
	IsActive    *bool  `json:"is_active"`
}

// CreateClient godoc
//
//	@Summary		Yangi mijoz (platforma) qo'shish
//	@Description	Servisdan foydalanadigan tashqi platformani ro'yxatga oladi. Kalit keyin shu mijozga beriladi.
//	@Tags			admin
//	@Security		ApiKeyAuth
//	@Accept			json
//	@Produce		json
//	@Param			body	body		clientRequest	true	"Mijoz ma'lumotlari"
//	@Success		201		{object}	map[string]any
//	@Failure		400		{object}	errorBody
//	@Failure		409		{object}	errorBody
//	@Router			/admin/clients [post]
func (h *AdminHandler) CreateClient(c *gin.Context) {
	var req clientRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondErrorCode(c, domain.CodeInvalidInput, "So'rov tanasi noto'g'ri: "+err.Error())
		return
	}

	client, err := h.keys.CreateClient(c.Request.Context(), usecase.ClientCommand{
		Name:        req.Name,
		Contact:     req.Contact,
		Description: req.Description,
		IsActive:    req.IsActive,
	})
	h.record(c, "client.create", req.Name, err)
	if err != nil {
		respondError(c, err)
		return
	}

	respondOK(c, http.StatusCreated, gin.H{
		"message": "Mijoz qo'shildi.",
		"client":  client,
	})
}

// UpdateClient godoc
//
//	@Summary	Mijozni yangilash yoki faolsizlantirish
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Accept		json
//	@Produce	json
//	@Param		client_id	path		string			true	"Mijoz ID"
//	@Param		body		body		clientRequest	true	"Yangi qiymatlar"
//	@Success	200			{object}	map[string]any
//	@Router		/admin/clients/{client_id} [put]
func (h *AdminHandler) UpdateClient(c *gin.Context) {
	var req clientRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondErrorCode(c, domain.CodeInvalidInput, "So'rov tanasi noto'g'ri: "+err.Error())
		return
	}

	client, err := h.keys.UpdateClient(c.Request.Context(), c.Param("client_id"), usecase.ClientCommand{
		Name:        req.Name,
		Contact:     req.Contact,
		Description: req.Description,
		IsActive:    req.IsActive,
	})
	h.record(c, "client.update", c.Param("client_id"), err)
	if err != nil {
		respondError(c, err)
		return
	}

	// Mijoz faolsizlantirilgan bo'lishi mumkin — keshdagi kalitlar darhol
	// kuchini yo'qotishi kerak.
	h.auth.InvalidateAll()

	respondOK(c, http.StatusOK, gin.H{"message": "Mijoz yangilandi.", "client": client})
}

// ListClients godoc
//
//	@Summary	Mijozlar ro'yxati
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Success	200	{object}	map[string]any
//	@Router		/admin/clients [get]
func (h *AdminHandler) ListClients(c *gin.Context) {
	clients, err := h.keys.ListClients(c.Request.Context())
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{"clients": clients, "count": len(clients)})
}

// GetClient godoc
//
//	@Summary	Bitta mijoz
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		client_id	path		string	true	"Mijoz ID"
//	@Success	200			{object}	map[string]any
//	@Router		/admin/clients/{client_id} [get]
func (h *AdminHandler) GetClient(c *gin.Context) {
	client, err := h.keys.GetClient(c.Request.Context(), c.Param("client_id"))
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{"client": client})
}

// DeleteClient godoc
//
//	@Summary		Mijozni o'chirish
//	@Description	Mijoz bilan birga uning barcha kalitlari ham o'chadi. Audit jurnali saqlanib qoladi.
//	@Tags			admin
//	@Security		ApiKeyAuth
//	@Produce		json
//	@Param			client_id	path		string	true	"Mijoz ID"
//	@Success		200			{object}	map[string]any
//	@Router			/admin/clients/{client_id} [delete]
func (h *AdminHandler) DeleteClient(c *gin.Context) {
	err := h.keys.DeleteClient(c.Request.Context(), c.Param("client_id"))
	h.record(c, "client.delete", c.Param("client_id"), err)
	if err != nil {
		respondError(c, err)
		return
	}
	h.auth.InvalidateAll()
	respondOK(c, http.StatusOK, gin.H{"message": "Mijoz va uning kalitlari o'chirildi."})
}

// ============================================================== Kalitlar

type issueKeyRequest struct {
	ClientID        string   `json:"client_id"`
	Label           string   `json:"label"`
	Scopes          []string `json:"scopes"`
	RateLimitPerMin int      `json:"rate_limit_per_min"`
	ExpiresInDays   int      `json:"expires_in_days"`
}

// IssueKey godoc
//
//	@Summary		Yangi API kalit berish
//	@Description	Kalitning ochiq matni FAQAT shu javobda qaytariladi va boshqa hech qayerda saqlanmaydi. Uni yo'qotsangiz — rotatsiya qiling.
//	@Tags			admin
//	@Security		ApiKeyAuth
//	@Accept			json
//	@Produce		json
//	@Param			body	body		issueKeyRequest	true	"Kalit parametrlari"
//	@Success		201		{object}	map[string]any
//	@Failure		400		{object}	errorBody
//	@Router			/admin/keys [post]
func (h *AdminHandler) IssueKey(c *gin.Context) {
	var req issueKeyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondErrorCode(c, domain.CodeInvalidInput, "So'rov tanasi noto'g'ri: "+err.Error())
		return
	}

	issued, err := h.keys.IssueKey(c.Request.Context(), usecase.IssueKeyCommand{
		ClientID:        strings.TrimSpace(req.ClientID),
		Label:           req.Label,
		Scopes:          req.Scopes,
		RateLimitPerMin: req.RateLimitPerMin,
		ExpiresInDays:   req.ExpiresInDays,
	})
	// Nishon sifatida yorliq: kalit idsi xato holatda hali mavjud emas, va
	// "qaysi kalit berilmoqchi edi" degan savolga yorliq javob beradi.
	h.record(c, "key.issue", req.Label, err)
	if err != nil {
		respondError(c, err)
		return
	}

	respondOK(c, http.StatusCreated, gin.H{
		"message": "Kalit yaratildi. Uni saqlab qo'ying — bu qiymat boshqa ko'rsatilmaydi.",
		"key":     issued.Key,
		"secret":  issued.Secret,
	})
}

// ListKeys godoc
//
//	@Summary	Kalitlar ro'yxati (ochiq matnsiz)
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		client_id	query		string	false	"Faqat shu mijozning kalitlari"
//	@Success	200			{object}	map[string]any
//	@Router		/admin/keys [get]
func (h *AdminHandler) ListKeys(c *gin.Context) {
	keys, err := h.keys.ListKeys(c.Request.Context(), strings.TrimSpace(c.Query("client_id")))
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{"keys": keys, "count": len(keys)})
}

// RotateKey godoc
//
//	@Summary		Kalitni almashtirish
//	@Description	Eski kalit bekor qilinadi, aynan shu huquqlar bilan yangisi beriladi.
//	@Tags			admin
//	@Security		ApiKeyAuth
//	@Produce		json
//	@Param			key_id	path		string	true	"Kalit ID"
//	@Success		200		{object}	map[string]any
//	@Router			/admin/keys/{key_id}/rotate [post]
func (h *AdminHandler) RotateKey(c *gin.Context) {
	issued, oldPrefix, err := h.keys.RotateKey(c.Request.Context(), c.Param("key_id"))
	h.record(c, "key.rotate", c.Param("key_id"), err)
	if err != nil {
		respondError(c, err)
		return
	}
	// Eski kalit kesh muddati tugagunicha ishlab turmasligi uchun.
	h.auth.Invalidate(oldPrefix)

	respondOK(c, http.StatusOK, gin.H{
		"message": "Kalit almashtirildi. Eskisi darhol ishlamay qoldi.",
		"key":     issued.Key,
		"secret":  issued.Secret,
	})
}

// RevokeKey godoc
//
//	@Summary	Kalitni bekor qilish
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		key_id	path		string	true	"Kalit ID"
//	@Success	200		{object}	map[string]any
//	@Router		/admin/keys/{key_id} [delete]
//
// PurgeKey — bekor qilingan kalit yozuvini butunlay o'chiradi.
func (h *AdminHandler) PurgeKey(c *gin.Context) {
	err := h.keys.PurgeKey(c.Request.Context(), c.Param("key_id"))
	// Yozilgan yozuvning o'zi o'chirilgan kalit haqidagi yagona iz bo'lib
	// qoladi — shuning uchun bu yerda jurnalga yozish ayniqsa muhim.
	h.record(c, "key.purge", c.Param("key_id"), err)
	if err != nil {
		respondError(c, err)
		return
	}
	respondOK(c, http.StatusOK, gin.H{
		"message": "Kalit yozuvi o'chirildi.",
		"key_id":  c.Param("key_id"),
	})
}

func (h *AdminHandler) RevokeKey(c *gin.Context) {
	key, err := h.keys.RevokeKey(c.Request.Context(), c.Param("key_id"))
	h.record(c, "key.revoke", c.Param("key_id"), err)
	if err != nil {
		respondError(c, err)
		return
	}
	h.auth.Invalidate(key.Prefix)

	respondOK(c, http.StatusOK, gin.H{
		"message": "Kalit bekor qilindi.",
		"key_id":  key.ID,
	})
}

// Audit godoc
//
//	@Summary	So'rovlar jurnali
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Param		client_id	query		string	false	"Mijoz bo'yicha filtr"
//	@Param		limit		query		int		false	"Nechta yozuv (standart 100, maksimum 500)"
//	@Success	200			{object}	map[string]any
//	@Router		/admin/audit [get]
func (h *AdminHandler) Audit(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "100"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
	clientID := strings.TrimSpace(c.Query("client_id"))

	entries, err := h.audit.ListAudit(c.Request.Context(), clientID, limit, offset)
	if err != nil {
		respondError(c, err)
		return
	}

	// Jami son — sahifalash uchun. Xatosi jurnalni ko'rsatishga to'sqinlik
	// qilmaydi: yozuvlar allaqachon qo'lda, ularni yashirish foydasi yo'q.
	total, err := h.audit.CountAudit(c.Request.Context(), clientID)
	if err != nil {
		total = len(entries)
	}

	// domain.AuditEntry JSON teglarisiz (ichki tur) — mijozga aniq shakl beramiz.
	out := make([]gin.H, 0, len(entries))
	for _, e := range entries {
		out = append(out, gin.H{
			"client_id":   e.ClientID,
			"client_name": e.ClientName,
			"key_id":      e.KeyID,
			"action":      e.Action,
			"person_id":   e.PersonID,
			"result":      e.Result,
			"code":        e.Code,
			"score":       e.Score,
			"latency_ms":  e.LatencyMs,
			"request_id":  e.RequestID,
			"ip":          e.IP,
			"created_at":  e.CreatedAt,
		})
	}
	respondOK(c, http.StatusOK, gin.H{
		"entries": out,
		"count":   len(out),
		"total":   total,
	})
}

// Scopes godoc
//
//	@Summary	Mavjud scope'lar ro'yxati
//	@Tags		admin
//	@Security	ApiKeyAuth
//	@Produce	json
//	@Success	200	{object}	map[string]any
//	@Router		/admin/scopes [get]
func (h *AdminHandler) Scopes(c *gin.Context) {
	descriptions := map[domain.Scope]string{
		domain.ScopeVerify:   "Rasm bo'yicha tanish va real-time oqim.",
		domain.ScopeRegister: "Bazaga yuz qo'shish va yangilash.",
		domain.ScopeRead:     "Yozuvlarni o'qish va person_id ni hisoblash.",
		domain.ScopeDelete:   "Yozuvni bazadan o'chirish.",
		domain.ScopeAdmin:    "Mijoz va kalitlarni boshqarish.",
	}

	out := make([]gin.H, 0, len(domain.AllScopes))
	for _, s := range domain.AllScopes {
		out = append(out, gin.H{"scope": s, "description": descriptions[s]})
	}
	respondOK(c, http.StatusOK, gin.H{"scopes": out})
}
