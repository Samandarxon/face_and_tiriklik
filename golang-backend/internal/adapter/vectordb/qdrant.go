// Package vectordb — `domain.PersonRepository` portining Qdrant implementatsiyasi.
//
// Qdrant bilan REST (JSON) orqali ishlaydi. Kollektsiya 512 o'lchamli vektorlar
// va Cosine masofa bilan yaratiladi.
package vectordb

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// ErrUnauthorized — Qdrant so'rovni autentifikatsiya sababli rad etdi (401/403).
// Bu konfiguratsiya xatosi: qayta urinish yordam bermaydi.
var ErrUnauthorized = errors.New("qdrant autentifikatsiyani rad etdi")

// Config — Qdrant ulanish sozlamalari.
type Config struct {
	URL        string
	APIKey     string
	Collection string
	VectorSize int
	Timeout    time.Duration
}

// QdrantRepository — vektor ombori.
type QdrantRepository struct {
	baseURL    string
	apiKey     string
	collection string
	vectorSize int
	http       *http.Client
}

var _ domain.PersonRepository = (*QdrantRepository)(nil)

func NewQdrant(cfg Config) *QdrantRepository {
	return &QdrantRepository{
		baseURL:    strings.TrimRight(cfg.URL, "/"),
		apiKey:     cfg.APIKey,
		collection: cfg.Collection,
		vectorSize: cfg.VectorSize,
		http: &http.Client{
			Timeout: cfg.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

func (r *QdrantRepository) Collection() string { return r.collection }

// payload — Qdrant'da saqlanadigan metadata (tashqi ko'rinish, domain emas).
type payload struct {
	PersonID   string  `json:"person_id"`
	Name       string  `json:"name"`
	CreatedAt  string  `json:"created_at"`
	UpdatedAt  string  `json:"updated_at,omitempty"`
	DetScore   float64 `json:"det_score,omitempty"`
	SourceFile string  `json:"source_file,omitempty"`
	// Sample — bu nuqta galereyadagi QO'SHIMCHA namuna ekanini bildiradi.
	//
	// Asosiy yozuvda bo'sh. Aynan shu farq `List` ga bir shaxsni bir marta
	// ko'rsatish imkonini beradi: usiz galereyasi uchta namunali odam
	// ro'yxatda uch marta chiqardi.
	Sample string `json:"sample,omitempty"`
}

func toPayload(person domain.Person) payload {
	p := payload{
		PersonID:   person.ID,
		Name:       person.Name,
		CreatedAt:  person.CreatedAt.UTC().Format(time.RFC3339),
		DetScore:   person.DetScore,
		SourceFile: person.SourceFile,
	}
	if !person.UpdatedAt.IsZero() {
		p.UpdatedAt = person.UpdatedAt.UTC().Format(time.RFC3339)
	}
	return p
}

func (p payload) toDomain() domain.Person {
	person := domain.Person{
		ID:         p.PersonID,
		Name:       p.Name,
		DetScore:   p.DetScore,
		SourceFile: p.SourceFile,
	}
	if t, err := time.Parse(time.RFC3339, p.CreatedAt); err == nil {
		person.CreatedAt = t
	}
	if t, err := time.Parse(time.RFC3339, p.UpdatedAt); err == nil {
		person.UpdatedAt = t
	}
	return person
}

// ---------------------------------------------------------------- transport

func (r *QdrantRepository) do(ctx context.Context, method, path string, body, out any) (int, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return 0, fmt.Errorf("qdrant so'rovini kodlashda xato: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, r.baseURL+path, reader)
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if r.apiKey != "" {
		req.Header.Set("api-key", r.apiKey)
	}

	resp, err := r.http.Do(req)
	if err != nil {
		return 0, fmt.Errorf("qdrant bilan bog'lanib bo'lmadi: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return resp.StatusCode, fmt.Errorf("qdrant javobini o'qib bo'lmadi: %w", err)
	}

	// Auth xatosi "hali tayyor emas" degani emas — uni alohida ajratamiz,
	// shunda chaqiruvchi bekorga qayta urinib o'tirmaydi.
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return resp.StatusCode, fmt.Errorf("%w (HTTP %d): %s",
			ErrUnauthorized, resp.StatusCode, truncate(string(raw), 200))
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, fmt.Errorf("qdrant xatosi (HTTP %d): %s",
			resp.StatusCode, truncate(string(raw), 400))
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("qdrant javobi noto'g'ri formatda: %w", err)
		}
	}
	return resp.StatusCode, nil
}

// dbError — infratuzilma xatosini domain xatosiga o'raydi.
func dbError(message string, cause error) error {
	return domain.WrapError(domain.CodeVectorDBFailure, message, cause)
}

// ------------------------------------------------------------------ port

// EnsureReady — kollektsiya bo'lmasa yaratadi, bo'lsa sozlamalarini tekshiradi.
func (r *QdrantRepository) EnsureReady(ctx context.Context) error {
	var info struct {
		Result struct {
			Config struct {
				Params struct {
					Vectors struct {
						Size     int    `json:"size"`
						Distance string `json:"distance"`
					} `json:"vectors"`
				} `json:"params"`
			} `json:"config"`
		} `json:"result"`
	}

	status, err := r.do(ctx, http.MethodGet, "/collections/"+r.collection, nil, &info)
	switch {
	case err == nil:
		got := info.Result.Config.Params.Vectors
		if got.Size != 0 && got.Size != r.vectorSize {
			return fmt.Errorf(
				"mavjud '%s' kollektsiyasi %d o'lchamli, kutilgan %d — kollektsiyani o'chirib qayta yarating",
				r.collection, got.Size, r.vectorSize)
		}
		if got.Distance != "" && !strings.EqualFold(got.Distance, "Cosine") {
			return fmt.Errorf(
				"mavjud '%s' kollektsiyasi '%s' masofasidan foydalanadi, 'Cosine' kutilgan",
				r.collection, got.Distance)
		}
		return nil
	case status != http.StatusNotFound:
		return err
	}

	create := map[string]any{
		"vectors": map[string]any{
			"size": r.vectorSize,
			// InsightFace embeddinglari L2-normalizatsiyalangan, shuning uchun
			// Cosine masofa to'g'ridan-to'g'ri o'xshashlikni beradi.
			"distance": "Cosine",
		},
		"optimizers_config": map[string]any{"default_segment_number": 2},
		"hnsw_config":       map[string]any{"m": 16, "ef_construct": 128},
		"on_disk_payload":   true,
	}
	if _, err := r.do(ctx, http.MethodPut, "/collections/"+r.collection, create, nil); err != nil {
		return err
	}

	index := map[string]any{"field_name": "person_id", "field_schema": "keyword"}
	if _, err := r.do(ctx, http.MethodPut,
		"/collections/"+r.collection+"/index?wait=true", index, nil); err != nil {
		return fmt.Errorf("person_id indeksini yaratib bo'lmadi: %w", err)
	}
	return nil
}

func (r *QdrantRepository) Healthy(ctx context.Context) error {
	_, err := r.do(ctx, http.MethodGet, "/collections/"+r.collection, nil, nil)
	return err
}

func (r *QdrantRepository) Save(ctx context.Context, person domain.Person, embedding domain.Embedding) (string, error) {
	if embedding.Dim() != r.vectorSize {
		return "", domain.NewError(domain.CodeInternal,
			fmt.Sprintf("vektor o'lchami %d, kutilgan %d", embedding.Dim(), r.vectorSize))
	}

	return r.save(ctx, person, embedding, PointIDFor(person.ID), "")
}

// SaveSample — mavjud shaxsga QO'SHIMCHA namuna qo'shadi.
//
// Asosiy yozuvni almashtirmaydi: galereya o'sadi, eskisi joyida qoladi.
// Bir xil `sample` nomi bilan qayta yozilsa — o'sha namuna yangilanadi.
func (r *QdrantRepository) SaveSample(
	ctx context.Context, person domain.Person, embedding domain.Embedding, sample string,
) (string, error) {
	if embedding.Dim() != r.vectorSize {
		return "", domain.NewError(domain.CodeInternal,
			fmt.Sprintf("vektor o'lchami %d, kutilgan %d", embedding.Dim(), r.vectorSize))
	}
	return r.save(ctx, person, embedding, PointIDForSample(person.ID, sample), sample)
}

func (r *QdrantRepository) save(
	ctx context.Context, person domain.Person, embedding domain.Embedding,
	pointID, sample string,
) (string, error) {
	pl := toPayload(person)
	pl.Sample = sample
	body := map[string]any{
		"points": []map[string]any{{
			"id":      pointID,
			"vector":  []float32(embedding),
			"payload": pl,
		}},
	}
	// wait=true — javob qaytganda yozuv qidiruvda ko'rinadigan bo'ladi.
	if _, err := r.do(ctx, http.MethodPut,
		"/collections/"+r.collection+"/points?wait=true", body, nil); err != nil {
		return "", dbError("Vektor bazaga saqlab bo'lmadi.", err)
	}
	return pointID, nil
}

func (r *QdrantRepository) FindNearest(ctx context.Context, embedding domain.Embedding, limit int) ([]domain.Match, error) {
	if embedding.Dim() != r.vectorSize {
		return nil, domain.NewError(domain.CodeInternal,
			fmt.Sprintf("vektor o'lchami %d, kutilgan %d", embedding.Dim(), r.vectorSize))
	}
	if limit <= 0 {
		limit = 1
	}

	body := map[string]any{
		"vector":       []float32(embedding),
		"limit":        limit,
		"with_payload": true,
		"params":       map[string]any{"hnsw_ef": 128},
	}

	var resp struct {
		Result []struct {
			Score   float32 `json:"score"`
			Payload payload `json:"payload"`
		} `json:"result"`
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/search", body, &resp); err != nil {
		return nil, dbError("Vektor bazada qidirib bo'lmadi.", err)
	}

	// Bir shaxsning bir nechta namunasi qaytishi mumkin — ular bitta javobga
	// yig'iladi va ENG YUQORI ball olinadi.
	//
	// Nima uchun max, o'rtacha emas: 20 yillik galereyada bugungi kadr faqat
	// eng so'nggi namunaga mos kelishi normal. O'rtacha ball eski
	// namunalar hisobiga pasayadi va aynan biz izlayotgan mosllikni yashiradi.
	//
	// Tartib saqlanadi: Qdrant allaqachon ball bo'yicha saralab qaytaradi,
	// shuning uchun birinchi uchragan namuna — o'sha shaxsning eng yaxshisi.
	seen := make(map[string]int, len(resp.Result))
	matches := make([]domain.Match, 0, len(resp.Result))
	for _, item := range resp.Result {
		p := item.Payload.toDomain()
		if i, ok := seen[p.ID]; ok {
			if item.Score > matches[i].Score {
				matches[i].Score = item.Score
			}
			matches[i].Samples++
			continue
		}
		seen[p.ID] = len(matches)
		matches = append(matches, domain.Match{Person: p, Score: item.Score, Samples: 1})
	}
	return matches, nil
}

// ScoreAgainst — 1:1 solishtirish.
//
// Vektorni qo'lda o'qib, cosine'ni Go'da hisoblash o'rniga o'sha qidiruvni
// bitta nuqta bilan cheklaymiz (`has_id` filtri). Sabab: ball AYNAN 1:N
// yo'lidagi kabi Qdrant tomonidan, bir xil metrika va bir xil normalizatsiya
// bilan hisoblanadi. Ikki xil yo'lda ikki xil ball chiqishi chegarani
// (`MATCH_THRESHOLD`) ma'nosiz qilib qo'yardi.
func (r *QdrantRepository) ScoreAgainst(
	ctx context.Context,
	embedding domain.Embedding,
	personID string,
) (*domain.Match, bool, error) {
	if embedding.Dim() != r.vectorSize {
		return nil, false, domain.NewError(domain.CodeInternal,
			fmt.Sprintf("vektor o'lchami %d, kutilgan %d", embedding.Dim(), r.vectorSize))
	}

	body := map[string]any{
		"vector":       []float32(embedding),
		"limit":        1,
		"with_payload": true,
		// `has_id` EMAS, `person_id` bo'yicha: 1:1 tekshiruv ham shu shaxsning
		// BUTUN galereyasi ustidan qidirishi kerak.
		//
		// Ilgari bu yerda faqat asosiy nuqta bilan solishtirilardi, ya'ni
		// galereyaga yozilgan jonli kadrlar 1:1 yo'lida umuman ishlatilmasdi.
		// Aynan o'sha kadrlar eng qimmatli: ular bugungi ko'rinishga eng
		// yaqin, arxiv fotosi esa bir necha yillik bo'lishi mumkin.
		//
		// `limit: 1` saqlanadi — Qdrant eng yuqori ballni qaytaradi, ya'ni
		// galereyadagi eng mos keladigan namuna.
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "person_id", "match": map[string]any{"value": personID}},
			},
		},
	}

	var resp struct {
		Result []struct {
			Score   float32 `json:"score"`
			Payload payload `json:"payload"`
		} `json:"result"`
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/search", body, &resp); err != nil {
		return nil, false, dbError("Vektor bazada solishtirib bo'lmadi.", err)
	}
	if len(resp.Result) == 0 {
		return nil, false, nil
	}

	item := resp.Result[0]
	return &domain.Match{Person: item.Payload.toDomain(), Score: item.Score}, true, nil
}

// Get — shaxs bazada bormi.
//
// # Nima uchun nuqta idsi bo'yicha emas
//
// Ilgari bu yerda `points/<PointIDFor(personID)>` o'qilardi, ya'ni faqat
// ASOSIY nuqta. Galereyada qo'shimcha namunalar paydo bo'lgach bu noto'g'ri
// javob bera boshladi: asosiy nuqta o'chirilgan, namunalar esa qolgan
// holatda `Get` "yo'q" derdi, garchi shaxs bazada bo'lsa ham.
//
// Oqibati og'ir edi va aynan shu tarzda namoyon bo'ldi:
//
//	`Directory.Delete` avval `Get` ni chaqiradi → "topilmadi" → o'chirish
//	umuman bajarilmaydi → yetim namuna abadiy qoladi va odam yuz orqali
//	tanilishda davom etadi. Operator esa o'chirish tugmasini bosaveradi.
//
// `scroll` payload filtri bilan shaxsning har qanday nuqtasini topadi.
func (r *QdrantRepository) Get(ctx context.Context, personID string) (*domain.Person, error) {
	body := map[string]any{
		"limit":        1,
		"with_payload": true,
		"with_vector":  false,
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "person_id", "match": map[string]any{"value": personID}},
			},
		},
	}

	var resp struct {
		Result struct {
			Points []struct {
				Payload payload `json:"payload"`
			} `json:"points"`
		} `json:"result"`
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/scroll", body, &resp); err != nil {
		return nil, dbError("Vektor bazaga murojaat qilib bo'lmadi.", err)
	}
	if len(resp.Result.Points) == 0 {
		return nil, nil
	}
	person := resp.Result.Points[0].Payload.toDomain()
	return &person, nil
}

// Delete shaxsning BARCHA vektorlarini o'chiradi — asosiysini ham,
// galereyadagi qo'shimcha namunalarni ham.
//
// # Nima uchun nuqta idsi bo'yicha emas, filtr bo'yicha
//
// Ilgari bu yerda faqat `PointIDFor(personID)` — ya'ni ASOSIY nuqta —
// o'chirilardi. Bir shaxsning bitta vektori bo'lganda bu to'g'ri ishlardi,
// lekin `SaveSample` galereyaga qo'shimcha vektorlar qo'shadi va ularning
// nuqta idsi boshqa (`PointIDForSample`).
//
// Oqibati ko'rinish muammosidan ancha jiddiy edi:
//
//   - O'chirilgan shaxs ro'yxatda QOLAVERARDI, chunki `List` barcha
//     nuqtalarni o'qiydi.
//   - Va u yuz orqali TANILAVERARDI, chunki `FindNearest` ham barcha
//     nuqtalar ustidan qidiradi. Ya'ni ishdan ketgan hodim binoga kirishda
//     davom etardi, garchi operator uni o'chirgan bo'lsa ham.
//
// Filtr `person_id` payload maydoni bo'yicha ishlaydi, ya'ni namunalar
// qanday nomlangani ahamiyatsiz — hammasi ketadi.
func (r *QdrantRepository) Delete(ctx context.Context, personID string) error {
	body := map[string]any{
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "person_id", "match": map[string]any{"value": personID}},
			},
		},
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/delete?wait=true", body, nil); err != nil {
		return dbError("Yozuvni o'chirib bo'lmadi.", err)
	}
	return nil
}

func (r *QdrantRepository) List(ctx context.Context, limit int, offset string) ([]domain.Person, string, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	body := map[string]any{
		"limit":        limit,
		"with_payload": true,
		"with_vector":  false,
		// Faqat ASOSIY yozuvlar. Galereyadagi qo'shimcha namunalar ham
		// alohida nuqta bo'lib turadi, va ularsiz filtr uchta namunali
		// odamni ro'yxatda uch marta ko'rsatardi — bir xil PNFL bilan, ya'ni
		// operator uni dublikat deb o'ylardi.
		//
		// `is_empty` maydon umuman yo'q bo'lganini ham qamraydi, shuning
		// uchun bu o'zgarishdan OLDIN yozilgan yozuvlar ham to'g'ri chiqadi.
		"filter": map[string]any{
			"must": []map[string]any{
				{"is_empty": map[string]any{"key": "sample"}},
			},
		},
	}
	if offset != "" {
		body["offset"] = offset
	}

	var resp struct {
		Result struct {
			Points []struct {
				Payload payload `json:"payload"`
			} `json:"points"`
			NextPageOffset any `json:"next_page_offset"`
		} `json:"result"`
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/scroll", body, &resp); err != nil {
		return nil, "", dbError("Ro'yxatni olib bo'lmadi.", err)
	}

	people := make([]domain.Person, 0, len(resp.Result.Points))
	for _, point := range resp.Result.Points {
		people = append(people, point.Payload.toDomain())
	}

	next := ""
	if resp.Result.NextPageOffset != nil {
		next = fmt.Sprint(resp.Result.NextPageOffset)
	}
	return people, next, nil
}

func (r *QdrantRepository) Count(ctx context.Context) (int64, error) {
	var resp struct {
		Result struct {
			Count int64 `json:"count"`
		} `json:"result"`
	}
	if _, err := r.do(ctx, http.MethodPost,
		"/collections/"+r.collection+"/points/count", map[string]any{"exact": true}, &resp); err != nil {
		return 0, dbError("Yozuvlar sonini olib bo'lmadi.", err)
	}
	return resp.Result.Count, nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
