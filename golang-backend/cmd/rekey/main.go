// Command rekey — mavjud yuz yozuvlarini eski (HMAC hash) `person_id` dan
// yangi (PNFL) `person_id` ga o'tkazadi.
//
// # Muammo
//
// Avval `person_id` = HMAC-SHA256(pepper, PNFL) edi. Endi `person_id` = PNFL.
// Ya'ni Qdrant'dagi mavjud yozuvlar eski kalit ostida yotibdi va yangi kod
// ularni topa olmaydi. Rasmlarni qayta yig'ish — minglab odam uchun amalda
// imkonsiz.
//
// # Yechim
//
// Rasm KERAK EMAS. Vektorning o'zi Qdrant'da turibdi; faqat uni boshqa nuqta
// identifikatori ostiga ko'chirish kifoya. Eski hash bilan yangi PNFL
// o'rtasidagi bog'lanishni esa kirdi_chiqdi bazasi biladi: u har bir
// foydalanuvchi/hodim uchun ham `pnfl`, ham `face_person_id` ni saqlaydi.
//
// # Ishlatish
//
//  1. kirdi_chiqdi bazasidan moslik jadvalini eksport qiling.
//     (Izohda `\x27\x27` yozib bo'lmaydi — gofmt uni tirnoqqa aylantiradi,
//     shuning uchun bo'sh qatorni length() bilan tekshiramiz.)
//
//     psql -d kirdi_chiqdi -At -F, -c "
//     SELECT face_person_id, pnfl FROM users
//     WHERE length(face_person_id) > 0 AND length(pnfl) > 0
//     UNION
//     SELECT face_person_id, pnfl FROM employees
//     WHERE length(face_person_id) > 0 AND length(pnfl) > 0
//     UNION
//     SELECT face_person_id, pnfl FROM visitors
//     WHERE length(face_person_id) > 0 AND length(pnfl) > 0" > mapping.csv
//
//  2. Avval QURUQ ishga tushiring (hech nima o'zgarmaydi):
//
//     rekey -mapping mapping.csv -qdrant http://127.0.0.1:6333
//
//  3. Natija to'g'ri bo'lsa, haqiqiy ko'chirishni bajaring:
//
//     rekey -mapping mapping.csv -qdrant http://127.0.0.1:6333 -apply
//
// # Xavfsizlik
//
//   - ZAXIRA NUSXA OLING: `POST /collections/faces/snapshots`.
//   - Eski nuqta faqat yangisi muvaffaqiyatli yozilgandan KEYIN o'chiriladi.
//   - Idempotent: qayta ishga tushirsangiz allaqachon ko'chirilganlar
//     o'tkazib yuboriladi.
package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/adapter/vectordb"
)

var pnflRe = regexp.MustCompile(`^\d{14}$`)

type config struct {
	qdrantURL  string
	apiKey     string
	collection string
	mapping    string
	apply      bool
}

func main() {
	var cfg config
	flag.StringVar(&cfg.qdrantURL, "qdrant", "http://127.0.0.1:6333", "Qdrant manzili")
	flag.StringVar(&cfg.apiKey, "api-key", os.Getenv("QDRANT_API_KEY"), "Qdrant API kaliti")
	flag.StringVar(&cfg.collection, "collection", "faces", "Kolleksiya nomi")
	flag.StringVar(&cfg.mapping, "mapping", "", "CSV: eski_person_id,pnfl (sarlavhasiz)")
	flag.BoolVar(&cfg.apply, "apply", false, "Haqiqatan o'zgartirish (aks holda quruq ishga tushirish)")
	flag.Parse()

	if cfg.mapping == "" {
		fmt.Fprintln(os.Stderr, "XATO: -mapping majburiy. Yordam uchun: rekey -h")
		os.Exit(2)
	}

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "\nXATO: %v\n", err)
		os.Exit(1)
	}
}

type pair struct{ oldID, pnfl string }

func run(cfg config) error {
	pairs, err := readMapping(cfg.mapping)
	if err != nil {
		return err
	}
	fmt.Printf("Moslik jadvali: %d ta yozuv\n", len(pairs))

	if !cfg.apply {
		fmt.Println("\n*** QURUQ ISHGA TUSHIRISH — hech nima o'zgarmaydi. ***")
		fmt.Println("*** Haqiqiy ko'chirish uchun -apply qo'shing.         ***")
	} else {
		fmt.Println("\n!!! HAQIQIY KO'CHIRISH. Zaxira nusxa oldingizmi? !!!")
		fmt.Println("!!! 5 soniya ichida Ctrl+C bilan to'xtatishingiz mumkin. !!!")
		time.Sleep(5 * time.Second)
	}
	fmt.Println()

	c := &client{
		base:       strings.TrimRight(cfg.qdrantURL, "/"),
		apiKey:     cfg.apiKey,
		collection: cfg.collection,
		http:       &http.Client{Timeout: 30 * time.Second},
	}

	var moved, skipped, missing, failed int
	for _, p := range pairs {
		// Allaqachon ko'chirilganmi?
		if exists, err := c.pointExists(p.pnfl); err != nil {
			fmt.Printf("  [xato] %s: yangi nuqtani tekshirib bo'lmadi: %v\n", p.pnfl, err)
			failed++
			continue
		} else if exists {
			skipped++
			continue
		}

		point, err := c.fetchPoint(p.oldID)
		if err != nil {
			fmt.Printf("  [xato] %s: eski nuqtani o'qib bo'lmadi: %v\n", p.oldID, err)
			failed++
			continue
		}
		if point == nil {
			// Eski yozuv yo'q: odam yuzsiz ro'yxatga olingan bo'lishi mumkin.
			missing++
			continue
		}

		fmt.Printf("  %s...%s  ->  %s\n", p.oldID[:8], p.oldID[len(p.oldID)-4:], p.pnfl)
		if !cfg.apply {
			moved++
			continue
		}

		if err := c.writePoint(p.pnfl, point); err != nil {
			fmt.Printf("  [xato] %s: yangi nuqta yozilmadi: %v\n", p.pnfl, err)
			failed++
			continue
		}
		// Faqat yangisi joyiga tushgandan KEYIN eskisini o'chiramiz.
		if err := c.deletePoint(p.oldID); err != nil {
			fmt.Printf("  [ogoh] %s: yangi nuqta yozildi, lekin eskisi o'chmadi: %v\n", p.pnfl, err)
		}
		moved++
	}

	fmt.Printf("\n----------------------------------------\n")
	fmt.Printf("  ko'chirildi   : %d\n", moved)
	fmt.Printf("  o'tkazib yub. : %d  (allaqachon PNFL ostida)\n", skipped)
	fmt.Printf("  topilmadi     : %d  (eski nuqta yo'q)\n", missing)
	fmt.Printf("  xato          : %d\n", failed)
	fmt.Printf("----------------------------------------\n")

	if failed > 0 {
		return fmt.Errorf("%d ta yozuvda xato — yuqoridagi ro'yxatni tekshiring", failed)
	}
	if !cfg.apply {
		fmt.Println("\nQuruq ishga tushirish tugadi. Haqiqiy ko'chirish: -apply")
	}
	return nil
}

// readMapping — `eski_person_id,pnfl` ko'rinishidagi CSV.
func readMapping(path string) ([]pair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("moslik faylini ochib bo'lmadi: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	var out []pair
	seen := make(map[string]bool)
	line := 0
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%d-qator: %w", line+1, err)
		}
		line++
		if len(rec) < 2 {
			continue
		}

		oldID := strings.TrimSpace(rec[0])
		pnfl := normalize(rec[1])
		if oldID == "" || pnfl == "" {
			continue
		}
		if !pnflRe.MatchString(pnfl) {
			return nil, fmt.Errorf("%d-qator: %q PNFL emas (14 raqam kutilgan)", line, pnfl)
		}
		if oldID == pnfl {
			// Allaqachon PNFL — ko'chirish shart emas.
			continue
		}
		// Bir xil PNFL ikki xil eski ID ga bog'langan bo'lsa — bu ma'lumot
		// muammosi, jimgina bittasini tanlash noto'g'ri bo'lardi.
		if seen[pnfl] {
			return nil, fmt.Errorf("%d-qator: %s PNFL faylda bir necha marta uchraydi", line, pnfl)
		}
		seen[pnfl] = true
		out = append(out, pair{oldID: oldID, pnfl: pnfl})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("moslik faylida ko'chiriladigan yozuv yo'q")
	}
	return out, nil
}

func normalize(raw string) string {
	var b strings.Builder
	for _, r := range raw {
		switch r {
		case ' ', '\t', '\n', '\r', '-', '.', '_':
			continue
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ---------------------------------------------------------------- Qdrant

type client struct {
	base       string
	apiKey     string
	collection string
	http       *http.Client
}

type storedPoint struct {
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload"`
}

func (c *client) do(method, path string, body, out any) (int, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequest(method, c.base+path, reader)
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return resp.StatusCode, err
	}
	if resp.StatusCode >= 400 && resp.StatusCode != http.StatusNotFound {
		return resp.StatusCode, fmt.Errorf("qdrant HTTP %d: %s", resp.StatusCode, trunc(string(raw)))
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return resp.StatusCode, err
		}
	}
	return resp.StatusCode, nil
}

// fetchPoint — vektorni VA metadatani birga o'qiydi. `with_vector=true`
// bo'lmasa ko'chiriladigan narsaning o'zi qo'lga tushmaydi.
func (c *client) fetchPoint(personID string) (*storedPoint, error) {
	var resp struct {
		Result *storedPoint `json:"result"`
	}
	status, err := c.do(http.MethodGet,
		"/collections/"+c.collection+"/points/"+vectordb.PointIDFor(personID)+
			"?with_payload=true&with_vector=true", nil, &resp)
	if status == http.StatusNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if resp.Result == nil || len(resp.Result.Vector) == 0 {
		return nil, nil
	}
	return resp.Result, nil
}

func (c *client) pointExists(personID string) (bool, error) {
	var resp struct {
		Result *struct{} `json:"result"`
	}
	status, err := c.do(http.MethodGet,
		"/collections/"+c.collection+"/points/"+vectordb.PointIDFor(personID), nil, &resp)
	if status == http.StatusNotFound {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return resp.Result != nil, nil
}

func (c *client) writePoint(pnfl string, point *storedPoint) error {
	payload := make(map[string]any, len(point.Payload)+1)
	maps.Copy(payload, point.Payload)
	// Metadata ichidagi identifikator ham yangilanishi shart — aks holda
	// qidiruv natijasida eski hash qaytaverardi.
	payload["person_id"] = pnfl

	body := map[string]any{
		"points": []map[string]any{{
			"id":      vectordb.PointIDFor(pnfl),
			"vector":  point.Vector,
			"payload": payload,
		}},
	}
	_, err := c.do(http.MethodPut,
		"/collections/"+c.collection+"/points?wait=true", body, nil)
	return err
}

func (c *client) deletePoint(personID string) error {
	body := map[string]any{"points": []string{vectordb.PointIDFor(personID)}}
	_, err := c.do(http.MethodPost,
		"/collections/"+c.collection+"/points/delete?wait=true", body, nil)
	return err
}

func trunc(s string) string {
	if len(s) > 300 {
		return s[:300] + "..."
	}
	return s
}
