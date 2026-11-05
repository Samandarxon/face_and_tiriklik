package usecase

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

var defaultPlanarity = DefaultPassiveConfig().Planarity

// syntheticHead — yuzga o'xshash 3D nuqtalar: chekkalari orqaga ketadi, burun oldinga.
func syntheticHead(rng *rand.Rand, n int) [][3]float64 {
	points := make([][3]float64, n)
	for i := range points {
		x, y := rng.Float64()*2-1, rng.Float64()*2-1
		nose := 35 * math.Exp(-(x*x+y*y)/0.08)
		points[i] = [3]float64{70 * x, 90 * y, nose - 55*x*x - 25*y*y}
	}
	return points
}

// view — boshni burib, ortografik proyeksiya va kamera shovqini bilan kadr beradi.
func view(rng *rand.Rand, head [][3]float64, yawDeg, pitchDeg, jitter float64) []domain.Point {
	a, b := yawDeg*math.Pi/180, pitchDeg*math.Pi/180
	out := make([]domain.Point, len(head))
	for i, p := range head {
		x := p[0]*math.Cos(a) + p[2]*math.Sin(a)
		z := -p[0]*math.Sin(a) + p[2]*math.Cos(a)
		y := p[1]*math.Cos(b) - z*math.Sin(b)
		out[i] = domain.Point{X: 320 + x + rng.NormFloat64()*jitter, Y: 240 + y + rng.NormFloat64()*jitter}
	}
	return out
}

// liveSway — tabiiy qimirlash: bir necha gradusli sekin tebranish.
func liveSway(rng *rand.Rand, frames int, amplitude, jitter float64) [][]domain.Point {
	head := syntheticHead(rng, 106)
	out := make([][]domain.Point, frames)
	for i := range out {
		t := float64(i) / float64(frames-1)
		out[i] = view(rng, head, amplitude*math.Sin(2*math.Pi*t), 0.6*amplitude*math.Cos(2*math.Pi*t), jitter)
	}
	return out
}

// flatSequence — bitta suratning nuqtalari; har kadr tasodifiy gomografiya bilan
// o'tkazilgan: qo'lda qanday qimirlatilsa ham, tekislik shunday ko'rinadi.
func flatSequence(rng *rand.Rand, frames int, strength, jitter float64) [][]domain.Point {
	head := syntheticHead(rng, 106)
	base := view(rng, head, 0, 0, 0)
	out := make([][]domain.Point, frames)
	for i := range out {
		h := [8]float64{
			1 + strength*rng.NormFloat64()*0.1, strength * rng.NormFloat64() * 0.1, strength * rng.NormFloat64() * 10,
			strength * rng.NormFloat64() * 0.1, 1 + strength*rng.NormFloat64()*0.1, strength * rng.NormFloat64() * 10,
			strength * rng.NormFloat64() * 2e-4, strength * rng.NormFloat64() * 2e-4,
		}
		out[i] = make([]domain.Point, len(base))
		for k, p := range base {
			w := h[6]*p.X + h[7]*p.Y + 1
			out[i][k] = domain.Point{
				X: (h[0]*p.X+h[1]*p.Y+h[2])/w + rng.NormFloat64()*jitter,
				Y: (h[3]*p.X+h[4]*p.Y+h[5])/w + rng.NormFloat64()*jitter,
			}
		}
	}
	return out
}

func TestHomographyResidualIsZeroOnAPlane(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	seq := flatSequence(rng, 2, 1, 0)
	r, ok := homographyResidual(seq[0], seq[1])
	if !ok || r > 1e-6 {
		t.Fatalf("tekislikda qoldiq ~0 bo'lishi kerak: %v ok=%v", r, ok)
	}
}

func TestEpipolarResidualIsSmallOnARigidHead(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	head := syntheticHead(rng, 106)
	e, ok := epipolarResidual(view(rng, head, -4, 1, 0), view(rng, head, 4, -1, 0))
	if !ok || e > 1e-6 {
		t.Fatalf("qattiq jismda epipolar qoldiq ~0 bo'lishi kerak: %v ok=%v", e, ok)
	}
}

func TestPlanarityAcceptsParallax(t *testing.T) {
	passed := 0
	for seed := int64(0); seed < 20; seed++ {
		rng := rand.New(rand.NewSource(seed))
		if EvaluatePlanarity(liveSway(rng, 8, 4, 0.2), defaultPlanarity).NonPlanar {
			passed++
		}
	}
	if passed < 18 {
		t.Fatalf("parallaksli kadrlar o'tmadi: %d/20", passed)
	}
}

// ASOSIY TEST: tekislik har qanday qimirlash va titrashda ham o'tmaydi.
func TestPlanarityNeverAcceptsAPlane(t *testing.T) {
	for seed := int64(0); seed < 40; seed++ {
		for _, strength := range []float64{0.2, 0.5, 1, 2} {
			for _, jitter := range []float64{0, 0.3, 0.8, 2} {
				rng := rand.New(rand.NewSource(seed))
				for n := 6; n <= passiveWindow; n += 3 {
					r := EvaluatePlanarity(flatSequence(rng, n, strength, jitter), defaultPlanarity)
					if r.NonPlanar {
						t.Fatalf("tekislik o'tdi: seed=%d kuch=%v titrash=%v kadr=%d: %+v", seed, strength, jitter, n, r)
					}
				}
			}
		}
	}
}

// Modelning HAQIQIY chiqishi: namunaviy surat tekislik sifatida aylantirilgan va
// 2d106det bilan baholangan (scripts/passive_flat_fixture.py). Oyna kengayib
// borgani kabi har bir prefiks tekshiriladi — Observe ham shunday qiladi.
func TestPlanarityRejectsRealRegressorOnFlatPicture(t *testing.T) {
	file, err := os.Open("testdata/flat_regressor_landmarks.jsonl.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(gz)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)

	sequences := 0
	for scanner.Scan() {
		var seq struct {
			Motion string      `json:"motion"`
			Frames [][]float64 `json:"frames"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &seq); err != nil {
			t.Fatal(err)
		}
		sequences++
		var frames [][]domain.Point
		for _, raw := range seq.Frames {
			points := make([]domain.Point, len(raw)/2)
			for i := range points {
				points[i] = domain.Point{X: raw[2*i], Y: raw[2*i+1]}
			}
			frames = append(frames, points)
			if len(frames) < 6 {
				continue
			}
			if r := EvaluatePlanarity(frames, defaultPlanarity); r.NonPlanar {
				t.Errorf("%s: surat o'tdi (%d kadr): %+v", seq.Motion, len(frames), r)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if sequences < 10 {
		t.Fatalf("fixture juda kichik: %d", sequences)
	}
}

func TestPlanarityNeedsEnoughMotion(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	r := EvaluatePlanarity(liveSway(rng, 8, 0.05, 0), defaultPlanarity)
	if r.NonPlanar {
		t.Fatalf("harakatsiz yuz dalil emas: %+v", r)
	}
	if r.Pairs != 0 {
		t.Fatalf("kam harakatli juftlar hisobga olinmasligi kerak: %+v", r)
	}
}

// Bitta buzilgan kadr "harakat" yasab bera olmaydi.
func TestPlanarityIgnoresASingleBrokenFrame(t *testing.T) {
	for seed := int64(0); seed < 10; seed++ {
		rng := rand.New(rand.NewSource(seed))
		live := liveSway(rng, 8, 4, 0.1)
		seq := make([][]domain.Point, len(live))
		for i := range seq {
			seq[i] = live[0] // harakatsiz yuz
		}
		seq[4] = live[2] // bitta boshqa ko'rinishdagi kadr
		if r := EvaluatePlanarity(seq, defaultPlanarity); r.NonPlanar {
			t.Fatalf("bitta kadr o'tkazib yubordi: seed=%d %+v", seed, r)
		}
	}
}

// CHEGARA HUJJATI: ekrandagi video yozuvida parallaks kontentning o'zida bor.
// Ekran tekis bo'lsa ham test buni ko'radi va O'TKAZADI — geometriya replay'ni
// to'smaydi, uni faqat MiniFASNet to'sishi mumkin. Bu test noto'g'ri ishonchga
// qarshi: kimdir "ekran tekislik, demak yiqiladi" deb o'ylamasligi uchun.
func TestPlanarityCannotSeeReplayedVideo(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	video := liveSway(rng, 8, 4, 0.1)
	const tilt = 3e-4 // telefon biroz qiya ushlangan: butun kontentga bitta gomografiya
	for _, frame := range video {
		for k, p := range frame {
			w := tilt*p.X + 1
			frame[k] = domain.Point{X: (0.8*p.X + 30) / w, Y: (0.8*p.Y + 10) / w}
		}
	}
	if !EvaluatePlanarity(video, defaultPlanarity).NonPlanar {
		t.Fatal("test yiqildi — hujjatdagi chegarani qayta ko'rib chiqing, lekin bu replay'dan himoya kafolati EMAS")
	}
}

func BenchmarkEvaluatePlanarityFullWindow(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	frames := liveSway(rng, passiveWindow, 4, 0.3)
	for i := 0; i < b.N; i++ {
		EvaluatePlanarity(frames, defaultPlanarity)
	}
}
