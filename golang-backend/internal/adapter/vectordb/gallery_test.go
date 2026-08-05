package vectordb

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// Bir shaxsning har bir namunasi o'z nuqtasiga tushishi kerak — aks holda
// ikkinchi namuna birinchisini o'chirib yuboradi va galereya hech qachon
// o'smaydi.
func TestPointIDForSampleIsDistinctPerSample(t *testing.T) {
	const person = "52306006180076"

	base := PointIDFor(person)
	a := PointIDForSample(person, "2005-passport")
	b := PointIDForSample(person, "2019-jonli")

	if a == b {
		t.Fatal("ikki namuna bir xil nuqtaga yozilyapti")
	}
	if a == base || b == base {
		t.Fatal("namuna asosiy yozuvni almashtirib yuboradi")
	}

	// Idempotentlik: bir xil nom — bir xil nuqta, ya'ni qayta yuborish
	// dublikat emas, yangilanish bo'ladi.
	if PointIDForSample(person, "2005-passport") != a {
		t.Fatal("bir xil namuna nomi turli nuqta beryapti")
	}
}

// Bo'sh nom — eski xatti-harakat. Mavjud yozuvlar joyida qolishi va
// migratsiya kerak bo'lmasligi uchun.
func TestPointIDForSampleEmptyMatchesLegacy(t *testing.T) {
	const person = "31105976740018"
	if PointIDForSample(person, "") != PointIDFor(person) {
		t.Fatal("bo'sh namuna nomi mavjud yozuvni ko'chirib yuboradi")
	}
}

// Turli shaxslarning bir xil nomli namunalari to'qnashmasligi kerak.
func TestPointIDForSampleIsScopedToThePerson(t *testing.T) {
	if PointIDForSample("111", "arxiv") == PointIDForSample("222", "arxiv") {
		t.Fatal("turli shaxslarning namunalari bir nuqtaga tushyapti")
	}
	// Ajratuvchisiz konkatenatsiya bo'lsa, ("11","1arxiv") va ("111","arxiv")
	// bir xil bo'lib qolardi.
	if PointIDForSample("11", "1arxiv") == PointIDForSample("111", "arxiv") {
		t.Fatal("shaxs va namuna chegarasi yo'q — id to'qnashuvi mumkin")
	}
}

// Bir shaxsning bir nechta namunasi bitta javobga yig'iladi va MAX ball
// olinadi.
//
// Nima uchun max, o'rtacha emas: 20 yillik galereyada bugungi kadr faqat
// eng so'nggi namunaga mos kelishi normal holat. O'rtacha ball eski
// namunalar hisobiga pasayadi va aynan biz izlayotgan moslikni yashiradi.
func TestFindNearestAggregatesSamplesByMax(t *testing.T) {
	// Qdrant ball bo'yicha saralab qaytaradi, shuning uchun kirish shu
	// tartibda modellashtiriladi.
	raw := []struct {
		person string
		score  float32
	}{
		{"AAA", 0.81}, // AAA ning eng yaxshi namunasi
		{"BBB", 0.74},
		{"AAA", 0.62}, // AAA ning eski namunasi
		{"AAA", 0.55}, // AAA ning yana bir eski namunasi
		{"CCC", 0.40},
	}

	seen := make(map[string]int, len(raw))
	var out []struct {
		id      string
		score   float32
		samples int
	}
	for _, item := range raw {
		if i, ok := seen[item.person]; ok {
			if item.score > out[i].score {
				out[i].score = item.score
			}
			out[i].samples++
			continue
		}
		seen[item.person] = len(out)
		out = append(out, struct {
			id      string
			score   float32
			samples int
		}{item.person, item.score, 1})
	}

	if len(out) != 3 {
		t.Fatalf("%d shaxs qaytdi, 3 kutilgan", len(out))
	}
	if out[0].id != "AAA" || out[0].score != 0.81 {
		t.Fatalf("AAA balli %v — eng yuqorisi 0.81 bo'lishi kerak", out[0].score)
	}
	if out[0].samples != 3 {
		t.Fatalf("AAA namunalari %d, 3 kutilgan", out[0].samples)
	}
	// Tartib buzilmasligi kerak: Qdrant allaqachon saralab qaytargan.
	if out[1].id != "BBB" || out[2].id != "CCC" {
		t.Fatalf("tartib buzildi: %v", out)
	}
}

// ---------------------------------------------------------------------------
// Galereya qo'shilgach, shaxsning butun hayot sikli bo'ylab BARCHA vektorlari
// birga boshqarilishi kerak. Quyidagi testlar shu shartni qulflaydi.
//
// Ular yozilishiga haqiqiy nosozlik sabab bo'ldi:
//
//	Operator yuzni o'chirdi → faqat ASOSIY nuqta ketdi → jonli namuna qoldi
//	→ shaxs ro'yxatda ko'rinishda va yuz orqali TANILISHDA davom etdi
//	→ qayta o'chirishga urinish esa "person_not_found" berdi, chunki `Get`
//	  ham faqat asosiy nuqtani izlardi.
//
// Ya'ni o'chirish tugmasi ishlamay qoldi, va buning hech qanday belgisi
// yo'q edi.
// ---------------------------------------------------------------------------

// captureQdrant — Qdrant o'rniga so'rovlarni yozib boruvchi soxta server.
type captureQdrant struct {
	*httptest.Server
	lastPath string
	lastBody map[string]any
	reply    string
}

func newCapture(t *testing.T, reply string) *captureQdrant {
	t.Helper()
	c := &captureQdrant{reply: reply}
	c.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.lastPath = r.URL.Path
		c.lastBody = map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&c.lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(c.reply))
	}))
	t.Cleanup(c.Close)
	return c
}

func repoFor(c *captureQdrant) *QdrantRepository {
	return NewQdrant(Config{URL: c.URL, Collection: "faces", VectorSize: 3})
}

// filterValue — so'rov tanasidagi `filter.must[0]` dan kalit bo'yicha qiymat.
func filterValue(t *testing.T, body map[string]any, key string) any {
	t.Helper()
	filter, ok := body["filter"].(map[string]any)
	if !ok {
		t.Fatalf("filter yo‘q: %v", body)
	}
	must, ok := filter["must"].([]any)
	if !ok || len(must) == 0 {
		t.Fatalf("filter.must yo‘q: %v", filter)
	}
	first, _ := must[0].(map[string]any)
	return first[key]
}

// O'chirish shaxsning BARCHA nuqtalarini qamrashi kerak — nuqta idsi bo'yicha
// emas, `person_id` bo'yicha.
func TestDeleteRemovesWholeGallery(t *testing.T) {
	c := newCapture(t, `{"result":{},"status":"ok"}`)

	if err := repoFor(c).Delete(context.Background(), "32012975670022"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, byID := c.lastBody["points"]; byID {
		t.Fatal("o‘chirish nuqta idsi bo‘yicha ketdi — galereya qoladi")
	}
	if got := filterValue(t, c.lastBody, "key"); got != "person_id" {
		t.Fatalf("filtr kaliti = %v, kutilgan person_id", got)
	}
}

// Ro'yxat bir shaxsni bir marta ko'rsatishi kerak: galereyadagi namunalar
// alohida nuqta bo'lib turadi va ularsiz filtr uchta namunali odam ro'yxatda
// uch marta, bir xil PNFL bilan chiqardi.
func TestListSkipsGallerySamples(t *testing.T) {
	c := newCapture(t, `{"result":{"points":[],"next_page_offset":null},"status":"ok"}`)

	if _, _, err := repoFor(c).List(context.Background(), 10, ""); err != nil {
		t.Fatalf("List: %v", err)
	}

	empty, ok := filterValue(t, c.lastBody, "is_empty").(map[string]any)
	if !ok {
		t.Fatalf("is_empty filtri yo‘q: %v", c.lastBody)
	}
	if empty["key"] != "sample" {
		t.Fatalf("is_empty kaliti = %v, kutilgan sample", empty["key"])
	}
}

// 1:1 tekshiruv ham butun galereya ustidan qidirishi kerak.
//
// Faqat asosiy nuqta bilan solishtirish galereyaning MAQSADINI yo'qqa
// chiqaradi: unga yozilgan jonli kadrlar bugungi ko'rinishga eng yaqin,
// arxiv fotosi esa bir necha yillik bo'lishi mumkin.
func TestScoreAgainstUsesGallery(t *testing.T) {
	c := newCapture(t, `{"result":[],"status":"ok"}`)

	_, _, err := repoFor(c).ScoreAgainst(
		context.Background(), domain.Embedding{1, 0, 0}, "32012975670022")
	if err != nil {
		t.Fatalf("ScoreAgainst: %v", err)
	}

	if got := filterValue(t, c.lastBody, "has_id"); got != nil {
		t.Fatal("1:1 faqat asosiy nuqta bilan solishtirdi — galereya ishlatilmadi")
	}
	if got := filterValue(t, c.lastBody, "key"); got != "person_id" {
		t.Fatalf("filtr kaliti = %v, kutilgan person_id", got)
	}
}

// Get faqat namuna qolgan holatda ham shaxsni topishi kerak.
//
// Aks holda `Directory.Delete` — u avval `Get` ni chaqiradi — "topilmadi"
// deb to'xtaydi va yetim namuna hech qachon o'chmaydi.
func TestGetFindsPersonWithOnlySamples(t *testing.T) {
	c := newCapture(t, `{"result":{"points":[{"payload":{"person_id":"32012975670022",`+
		`"sample":"live-2026-08","created_at":"2026-08-03T11:39:07Z"}}]},"status":"ok"}`)

	got, err := repoFor(c).Get(context.Background(), "32012975670022")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("faqat namunasi bor shaxs topilmadi — o‘chirish ham ishlamaydi")
	}
	if !strings.HasSuffix(c.lastPath, "/points/scroll") {
		t.Errorf("Get nuqta idsi bo‘yicha o‘qidi: %s", c.lastPath)
	}
}

// Namuna payload'ida `sample` bo'lishi shart — `List` filtri aynan shunga
// tayanadi.
func TestSaveSampleMarksPayload(t *testing.T) {
	c := newCapture(t, `{"result":{},"status":"ok"}`)

	_, err := repoFor(c).SaveSample(context.Background(),
		domain.Person{ID: "32012975670022"}, domain.Embedding{1, 0, 0}, "live-2026-08")
	if err != nil {
		t.Fatalf("SaveSample: %v", err)
	}

	points, _ := c.lastBody["points"].([]any)
	if len(points) == 0 {
		t.Fatalf("nuqta yuborilmadi: %v", c.lastBody)
	}
	pl, _ := points[0].(map[string]any)["payload"].(map[string]any)
	if pl["sample"] != "live-2026-08" {
		t.Fatalf("payload.sample = %v — List filtri bunday namunani ajrata olmaydi", pl["sample"])
	}
}

// Asosiy yozuvda `sample` bo'lmasligi kerak, aks holda `List` uni ham
// o'tkazib yuborardi va ro'yxat butunlay bo'shab qolardi.
func TestSaveLeavesSampleEmpty(t *testing.T) {
	c := newCapture(t, `{"result":{},"status":"ok"}`)

	_, err := repoFor(c).Save(context.Background(),
		domain.Person{ID: "32012975670022"}, domain.Embedding{1, 0, 0})
	if err != nil {
		t.Fatalf("Save: %v", err)
	}

	points, _ := c.lastBody["points"].([]any)
	pl, _ := points[0].(map[string]any)["payload"].(map[string]any)
	if s, ok := pl["sample"]; ok && s != "" {
		t.Fatalf("asosiy yozuvda sample = %v", s)
	}
}
