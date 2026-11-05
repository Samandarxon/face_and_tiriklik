package usecase

import (
	"math"
	"sort"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// TEKISLIK TESTI — kadrlar orasidagi nuqtalar tekis yuzaga tegishlimi?
//
// # G'oya
//
// Qo'lda ushlangan surat qanday qimirlamasin, ikki kadr orasidagi nuqtalarni
// bitta gomografiya bog'laydi: u tekislik. Haqiqiy yuzda esa burun, yonoq va
// jag' turli chuqurlikda, shuning uchun bosh sal burilsa parallaks paydo
// bo'ladi — gomografiya uni tushuntira olmaydi.
//
// # Nega uchta o'lchov (bitta qoldiq yetmaydi)
//
// Nuqtalar qattiq geometriya emas, neyron tarmoq chiqishi. Surat qiya
// tutilganda ham, nuqtalar titrashidan ham gomografiya qoldig'i o'sadi. Shu
// sababli (surat buriladigan sintetik kadrlarda 2d106det bilan o'lchangan):
//
//	motion     — o'xshashlikdan (siljish, masshtab, tekis burilish) keyingi
//	             harakat. Kichik bo'lsa test hech narsani isbotlamaydi.
//	nonplanar  — qoldiq / motion. Tekis yuzada ~0.5–0.65: qiyalikning asosiy
//	             qismini gomografiya tushuntiradi. Haqiqiy yuzda ~0.9+.
//	epipolar   — fundamental matritsa qoldig'i / gomografiya qoldig'i. Qattiq
//	             3D jismda epipolar shart bajariladi, tasodifiy titrashda esa
//	             yo'q: tekis+titrash ~0.3–0.47, parallaks ~0.1–0.25.
//
// Oxirgisi muhim: yomon yorug'likda titrash ham motion, ham nonplanar'ni
// ko'taradi va bu ikkisi yolg'iz o'zi suratni o'tkazib yuborardi.
//
// # Nimani to'smaydi — ochiq aytiladi
//
// Test tekis JISMNI emas, nuqtalar HARAKATINI baholaydi. Ekranda qo'yilgan
// video yozuvida yuz o'zi burilgan bo'lsa, parallaks videoning ichida bor va
// test uni "uch o'lchamli" deb qabul qiladi. Ekranni faqat MiniFASNet to'sadi.
// Egilgan qog'oz va 3D niqob ham chegaradan tashqarida.

// PlanarityThresholds — qaror chegaralari.
type PlanarityThresholds struct {
	MinMotion   float64
	MinResidual float64
	// MinNonplanar — qoldiq / motion ning quyi chegarasi.
	MinNonplanar float64
	// MaxEpipolar — fundamental qoldiq / gomografiya qoldig'ining yuqori chegarasi.
	MaxEpipolar float64
}

// PlanarityResult — bitta oyna natijasi. Birliklar nuqtalar bulutining RMS
// radiusiga nisbatan: yuz o'lchami va kamera aniqligiga bog'liq emas.
type PlanarityResult struct {
	Frames    int     `json:"frames"`
	Pairs     int     `json:"pairs"`
	MaxMotion float64 `json:"max_motion"`
	Residual  float64 `json:"residual"`
	Nonplanar float64 `json:"nonplanar"`
	Epipolar  float64 `json:"epipolar"`
	NonPlanar bool    `json:"non_planar"`
}

// minPlanarPairs — hisobga olinadigan juftlarning eng kam soni.
const minPlanarPairs = 3

// pairEvidence — ikki kadr orasidagi o'lchovlar.
type pairEvidence struct {
	i, j                       int
	motion, residual, epipolar float64
}

// EvaluatePlanarity — kadrlar oynasini baholaydi.
//
// Dalil har qanday BITTA kadr olib tashlanganda ham saqlanishi shart: aks holda
// bitta buzuq kadr (xira, nuqtalar sakragan) suratga "harakat" yasab berardi.
func EvaluatePlanarity(frames [][]domain.Point, th PlanarityThresholds) PlanarityResult {
	result := PlanarityResult{Frames: len(frames)}
	pairs := make([]pairEvidence, 0, len(frames)*(len(frames)-1)/2)
	for i := 0; i < len(frames); i++ {
		for j := i + 1; j < len(frames); j++ {
			p, ok := measurePair(frames[i], frames[j])
			if !ok {
				continue
			}
			p.i, p.j = i, j
			pairs = append(pairs, p)
			result.MaxMotion = math.Max(result.MaxMotion, p.motion)
		}
	}

	full, ok := summarise(pairs, -1, th)
	result.Pairs = full.pairs
	result.Residual = full.residual
	result.Nonplanar = full.nonplanar
	result.Epipolar = full.epipolar
	if !ok {
		return result
	}
	for skip := range frames {
		if _, ok := summarise(pairs, skip, th); !ok {
			return result
		}
	}
	result.NonPlanar = true
	return result
}

type pairSummary struct {
	pairs                         int
	residual, nonplanar, epipolar float64
}

// summarise — yetarli harakatli juftlar medianasi. skip kadri hisobga olinmaydi.
func summarise(pairs []pairEvidence, skip int, th PlanarityThresholds) (pairSummary, bool) {
	var residuals, nonplanar, epipolar []float64
	for _, p := range pairs {
		if p.i == skip || p.j == skip || p.motion < th.MinMotion {
			continue
		}
		residuals = append(residuals, p.residual)
		nonplanar = append(nonplanar, p.residual/p.motion)
		if p.residual > 0 {
			epipolar = append(epipolar, p.epipolar/p.residual)
		} else {
			// Mutlaq tekislik — epipolar dalil yo'q.
			epipolar = append(epipolar, 1)
		}
	}
	if len(residuals) == 0 {
		return pairSummary{}, false
	}
	s := pairSummary{
		pairs:     len(residuals),
		residual:  median(residuals),
		nonplanar: median(nonplanar),
		epipolar:  median(epipolar),
	}
	return s, s.pairs >= minPlanarPairs &&
		s.residual >= th.MinResidual &&
		s.nonplanar >= th.MinNonplanar &&
		s.epipolar <= th.MaxEpipolar
}

// measurePair — juftning harakati va ikki model qoldig'i (a ning RMS radiusiga
// nisbatan). Nuqta soni mos kelmasa yoki hisob buzilsa ok=false.
func measurePair(a, b []domain.Point) (pairEvidence, bool) {
	if len(a) != len(b) || len(a) < 9 {
		return pairEvidence{}, false
	}
	radius := rmsRadius(a)
	if radius <= 0 {
		return pairEvidence{}, false
	}
	residual, ok := homographyResidual(a, b)
	if !ok {
		return pairEvidence{}, false
	}
	epipolar, ok := epipolarResidual(a, b)
	if !ok {
		return pairEvidence{}, false
	}
	return pairEvidence{
		motion:   similarityResidual(a, b) / radius,
		residual: residual / radius,
		epipolar: epipolar / radius,
	}, true
}

// similarityResidual — eng yaxshi o'xshashlik (siljish, masshtab, tekis
// burilish) olib tashlangandan keyingi RMS siljish. Yuz tekisligi ichidagi
// harakat 3D haqida hech narsa aytmaydi, shuning uchun hisobga olinmaydi.
func similarityResidual(a, b []domain.Point) float64 {
	ca, cb := centroid(a), centroid(b)
	var dot, cross, norm float64
	for i := range a {
		ax, ay := a[i].X-ca.X, a[i].Y-ca.Y
		bx, by := b[i].X-cb.X, b[i].Y-cb.Y
		dot += ax*bx + ay*by
		cross += ax*by - ay*bx
		norm += ax*ax + ay*ay
	}
	if norm == 0 {
		return 0
	}
	kr, ki := dot/norm, cross/norm
	var sum float64
	for i := range a {
		ax, ay := a[i].X-ca.X, a[i].Y-ca.Y
		ex := b[i].X - cb.X - (kr*ax - ki*ay)
		ey := b[i].Y - cb.Y - (kr*ay + ki*ax)
		sum += ex*ex + ey*ey
	}
	return math.Sqrt(sum / float64(len(a)))
}

// homographyResidual — DLT gomografiyaning RMS o'tkazish xatosi (piksel).
func homographyResidual(a, b []domain.Point) (float64, bool) {
	na, sa := hartley(a)
	nb, sb := hartley(b)
	if sa == 0 || sb == 0 {
		return 0, false
	}

	var ata [9][9]float64
	for i := range na {
		x, y, u, v := na[i].X, na[i].Y, nb[i].X, nb[i].Y
		accumulate(&ata, [9]float64{x, y, 1, 0, 0, 0, -u * x, -u * y, -u})
		accumulate(&ata, [9]float64{0, 0, 0, x, y, 1, -v * x, -v * y, -v})
	}
	h := smallestEigenvector(ata)

	var sum float64
	for i := range na {
		x, y := na[i].X, na[i].Y
		w := h[6]*x + h[7]*y + h[8]
		if math.Abs(w) < 1e-12 {
			return 0, false
		}
		// Normallangan fazodagi xato pikselga b masshtabi bilan qaytariladi.
		ex := ((h[0]*x+h[1]*y+h[2])/w - nb[i].X) / sb
		ey := ((h[3]*x+h[4]*y+h[5])/w - nb[i].Y) / sb
		sum += ex*ex + ey*ey
	}
	return math.Sqrt(sum / float64(len(na))), true
}

// epipolarResidual — 8 nuqtali fundamental matritsaning RMS Sampson masofasi
// (piksel, a masshtabida).
func epipolarResidual(a, b []domain.Point) (float64, bool) {
	na, sa := hartley(a)
	nb, sb := hartley(b)
	if sa == 0 || sb == 0 {
		return 0, false
	}

	var ata [9][9]float64
	for i := range na {
		x, y, u, v := na[i].X, na[i].Y, nb[i].X, nb[i].Y
		accumulate(&ata, [9]float64{u * x, u * y, u, v * x, v * y, v, x, y, 1})
	}
	f := rankTwo(smallestEigenvector(ata))

	var sum float64
	for i := range na {
		x, y, u, v := na[i].X, na[i].Y, nb[i].X, nb[i].Y
		// Fx va F^T x'
		fx0 := f[0]*x + f[1]*y + f[2]
		fx1 := f[3]*x + f[4]*y + f[5]
		fx2 := f[6]*x + f[7]*y + f[8]
		ft0 := f[0]*u + f[3]*v + f[6]
		ft1 := f[1]*u + f[4]*v + f[7]
		den := fx0*fx0 + fx1*fx1 + ft0*ft0 + ft1*ft1
		if den < 1e-18 {
			return 0, false
		}
		num := u*fx0 + v*fx1 + fx2
		sum += num * num / den
	}
	return math.Sqrt(sum/float64(len(na))) / sa, true
}

// rankTwo — F ni eng yaqin 2-darajali matritsaga olib keladi: F(I − vvᵀ), bu
// yerda v — F ning eng kichik singulyar qiymatiga mos o'ng vektori.
func rankTwo(f [9]float64) [9]float64 {
	var ftf [3][3]float64
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			ftf[r][c] = f[r]*f[c] + f[3+r]*f[3+c] + f[6+r]*f[6+c]
		}
	}
	values, vectors := jacobi(3, func(r, c int) float64 { return ftf[r][c] })
	best := 0
	for i := 1; i < 3; i++ {
		if values[i] < values[best] {
			best = i
		}
	}
	v := [3]float64{vectors[0][best], vectors[1][best], vectors[2][best]}

	var out [9]float64
	for r := 0; r < 3; r++ {
		row := f[3*r : 3*r+3]
		proj := row[0]*v[0] + row[1]*v[1] + row[2]*v[2]
		for c := 0; c < 3; c++ {
			out[3*r+c] = row[c] - proj*v[c]
		}
	}
	return out
}

// accumulate — AᵀA ga bitta qator qo'shadi.
func accumulate(m *[9][9]float64, row [9]float64) {
	for r := 0; r < 9; r++ {
		for c := r; c < 9; c++ {
			m[r][c] += row[r] * row[c]
		}
	}
}

// smallestEigenvector — simmetrik matritsaning (yuqori uchburchagi to'ldirilgan)
// eng kichik xos qiymatiga mos vektor: ‖Ah‖ ning birlik vektordagi minimumi.
func smallestEigenvector(m [9][9]float64) [9]float64 {
	for r := 0; r < 9; r++ {
		for c := 0; c < r; c++ {
			m[r][c] = m[c][r]
		}
	}
	values, vectors := jacobi(9, func(r, c int) float64 { return m[r][c] })
	best := 0
	for i := 1; i < 9; i++ {
		if values[i] < values[best] {
			best = i
		}
	}
	var out [9]float64
	for i := 0; i < 9; i++ {
		out[i] = vectors[i][best]
	}
	return out
}

// jacobi — simmetrik matritsaning xos qiymatlari va vektorlari (ustunlar).
//
// Nega o'zimizniki: kichik, deterministik va bitta shu hisob uchun og'ir
// chiziqli algebra kutubxonasini qo'shish arzimaydi. Tartib 9 dan oshmaydi.
func jacobi(n int, at func(r, c int) float64) ([]float64, [][]float64) {
	a := make([][]float64, n)
	v := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		v[i] = make([]float64, n)
		v[i][i] = 1
		for j := 0; j < n; j++ {
			a[i][j] = at(i, j)
		}
	}
	for sweep := 0; sweep < 32; sweep++ {
		var off float64
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += a[p][q] * a[p][q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				if a[p][q] == 0 {
					continue
				}
				theta := (a[q][q] - a[p][p]) / (2 * a[p][q])
				t := 1 / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta < 0 {
					t = -t
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := 0; k < n; k++ {
					akp, akq := a[k][p], a[k][q]
					a[k][p], a[k][q] = c*akp-s*akq, s*akp+c*akq
				}
				for k := 0; k < n; k++ {
					apk, aqk := a[p][k], a[q][k]
					a[p][k], a[q][k] = c*apk-s*aqk, s*apk+c*aqk
				}
				for k := 0; k < n; k++ {
					vkp, vkq := v[k][p], v[k][q]
					v[k][p], v[k][q] = c*vkp-s*vkq, s*vkp+c*vkq
				}
			}
		}
	}
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = a[i][i]
	}
	return values, v
}

// hartley — nuqtalarni markazga ko'chirib, o'rtacha masofani √2 ga keltiradi.
// Bu DLT'ni sonli barqaror qiladi. Masshtab 0 — nuqtalar bitta joyga tushgan.
func hartley(points []domain.Point) ([]domain.Point, float64) {
	c := centroid(points)
	var mean float64
	for _, p := range points {
		mean += math.Hypot(p.X-c.X, p.Y-c.Y)
	}
	mean /= float64(len(points))
	if mean == 0 {
		return nil, 0
	}
	scale := math.Sqrt2 / mean
	out := make([]domain.Point, len(points))
	for i, p := range points {
		out[i] = domain.Point{X: (p.X - c.X) * scale, Y: (p.Y - c.Y) * scale}
	}
	return out, scale
}

func centroid(points []domain.Point) domain.Point {
	var c domain.Point
	for _, p := range points {
		c.X += p.X
		c.Y += p.Y
	}
	n := float64(len(points))
	return domain.Point{X: c.X / n, Y: c.Y / n}
}

func rmsRadius(points []domain.Point) float64 {
	c := centroid(points)
	var sum float64
	for _, p := range points {
		dx, dy := p.X-c.X, p.Y-c.Y
		sum += dx*dx + dy*dy
	}
	return math.Sqrt(sum / float64(len(points)))
}

func median(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}
