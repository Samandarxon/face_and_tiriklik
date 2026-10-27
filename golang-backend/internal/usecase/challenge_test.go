package usecase

import (
	"sync"
	"testing"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// ============================================================================
//  Aktiv liveness oldindan YOZILGAN videoga qarshi ishlaydi. Uning kuchi
//  uchta shartga tayanadi — buyruq tasodifiy, bir martalik va javob tez
//  kelishi kerak. Quyidagi testlar aynan shu uchtasini qulflaydi: bittasi
//  buzilsa qolgan ikkitasi ham ma'nosiz bo'lib qoladi.
// ============================================================================

// testStore — vaqti boshqariladigan ombor, kutmasdan muddat tekshirish uchun.
func testStore() (*ChallengeStore, func(time.Duration)) {
	now := time.Now()
	var mu sync.Mutex

	s := &ChallengeStore{
		items: make(map[string]*Challenge),
		stop:  make(chan struct{}),
	}
	s.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	}
	return s, func(d time.Duration) {
		mu.Lock()
		defer mu.Unlock()
		now = now.Add(d)
	}
}

func pose(yaw float64) *domain.HeadPose {
	return &domain.HeadPose{Yaw: yaw, Available: true}
}

// ------------------------------------------------------------- to'g'ri oqim

func TestHappyPathCentreThenTurn(t *testing.T) {
	s, _ := testStore()

	ch, err := s.Issue()
	if err != nil {
		t.Fatalf("Issue: %v", err)
	}

	// Buyruqni bajaramiz — qaysi tomon so'ralgan bo'lsa, o'sha tomonga.
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}

	if got := s.Observe(ch.Nonce, pose(0)); got != ChallengeAwaitingAction {
		t.Fatalf("markazdan keyin awaiting_action kutilgan, olindi: %s", got)
	}
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengePassed {
		t.Fatalf("harakatdan keyin passed kutilgan, olindi: %s", got)
	}
	if err := s.Consume(ch.Nonce); err != nil {
		t.Fatalf("Consume: %v", err)
	}
}

// ------------------------------------------------- markazsiz o'tib bo'lmaydi

// Eng muhim test. Agar faqat "yon tomonga qaradi" tekshirilsa, DOIM yon
// tomonga qarab turgan surat hech narsa qilmasdan o'tib ketardi. Harakatning
// O'ZI ko'rinishi shart: avval markaz, keyin burilish.
func TestStaticSideFaceNeverPasses(t *testing.T) {
	s, _ := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}

	// Yigirma kadr — hammasi allaqachon burilgan holatda, markaz hech qachon
	// ko'rinmaydi.
	for range 20 {
		if got := s.Observe(ch.Nonce, pose(turn)); got == ChallengePassed {
			t.Fatal("markazga qaramasdan o'tib ketdi — statik yon surat aldab o'tadi")
		}
	}
	if err := s.Consume(ch.Nonce); err == nil {
		t.Fatal("Consume xato qaytarishi kerak edi")
	}
}

// -------------------------------------------------- noto'g'ri tomon o'tmaydi

func TestWrongDirectionDoesNotPass(t *testing.T) {
	s, _ := testStore()

	ch, _ := s.Issue()
	// Ataylab TESKARI tomonga buramiz.
	wrong := 30.0
	if ch.Action == ChallengeTurnRight {
		wrong = -30.0
	}

	s.Observe(ch.Nonce, pose(0))
	for range 10 {
		if got := s.Observe(ch.Nonce, pose(wrong)); got == ChallengePassed {
			t.Fatalf("%s so'ralgan edi, teskari tomon qabul qilindi", ch.Action)
		}
	}
}

// Chegaradan past burilish — tabiiy tebranishni harakat deb qabul qilmaslik.
func TestSmallMovementDoesNotPass(t *testing.T) {
	s, _ := testStore()

	ch, _ := s.Issue()
	small := -(challengeYawThreshold - 3)
	if ch.Action == ChallengeTurnRight {
		small = challengeYawThreshold - 3
	}

	s.Observe(ch.Nonce, pose(0))
	if got := s.Observe(ch.Nonce, pose(small)); got == ChallengePassed {
		t.Fatalf("chegaradan past burilish (%v) qabul qilindi", small)
	}
}

// ------------------------------------------------------------ bir martalik

// Ikkinchi shart: bitta muvaffaqiyatli seansni qayta o'ynatib bo'lmasligi
// kerak, aks holda tasodifiylikning ma'nosi qolmaydi.
func TestChallengeIsSingleUse(t *testing.T) {
	s, _ := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}
	s.Observe(ch.Nonce, pose(0))
	s.Observe(ch.Nonce, pose(turn))

	if err := s.Consume(ch.Nonce); err != nil {
		t.Fatalf("birinchi Consume ishlashi kerak edi: %v", err)
	}
	if err := s.Consume(ch.Nonce); err == nil {
		t.Fatal("ikkinchi Consume o'tdi — buyruq bir martalik EMAS")
	}
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengeUnknown {
		t.Fatalf("ishlatilgandan keyin unknown kutilgan, olindi: %s", got)
	}
}

// ----------------------------------------------------------------- muddat

// Uchinchi shart: javob tez kelishi kerak, aks holda hujumchi buyruqni ko'rib
// kerakli videoni tanlab yoki generatsiya qilib ulguradi.
func TestExpiredChallengeIsRejected(t *testing.T) {
	s, advance := testStore()

	ch, _ := s.Issue()
	advance(ChallengeTTL + time.Second)

	if got := s.Observe(ch.Nonce, pose(0)); got != ChallengeExpired {
		t.Fatalf("expired kutilgan, olindi: %s", got)
	}
	if err := s.Consume(ch.Nonce); err == nil {
		t.Fatal("muddati o'tgan buyruq qabul qilindi")
	}
}

// Muvaffaqiyatli bajarilgan, lekin muddati o'tgandan keyin tasdiqlangan
// urinish ham rad etilishi kerak — aks holda oyna amalda cheksiz bo'lardi.
func TestPassedButExpiredIsStillRejected(t *testing.T) {
	s, advance := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}
	s.Observe(ch.Nonce, pose(0))
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengePassed {
		t.Fatal("harakat qayd etilmadi")
	}

	advance(ChallengeTTL + time.Second)
	if err := s.Consume(ch.Nonce); err == nil {
		t.Fatal("muddati o'tgandan keyin ham qabul qilindi")
	}
}

func TestTTLIsShort(t *testing.T) {
	// Uzun harakat oynasi hujumchiga video tayyorlash uchun vaqt beradi.
	if ChallengeActionTTL > 10*time.Second {
		t.Fatalf("ChallengeActionTTL juda uzun: %v", ChallengeActionTTL)
	}
	if ChallengeTTL > 30*time.Second {
		t.Fatalf("ChallengeTTL juda uzun: %v", ChallengeTTL)
	}
}

// Sekin CPU: markazga qarash kech ko'rinsa ham urinish tirik qolishi kerak —
// harakat oynasi markaz ko'ringan paytdan boshlanadi, oqim ochilganidan emas.
func TestSlowCentreStillHasFullActionWindow(t *testing.T) {
	s, advance := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}

	advance(ChallengeActionTTL + 2*time.Second) // kamera/model sekin ochildi
	if got := s.Observe(ch.Nonce, pose(0)); got != ChallengeAwaitingAction {
		t.Fatalf("awaiting_action kutilgan, olindi: %s", got)
	}
	advance(ChallengeActionTTL - time.Second)
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengePassed {
		t.Fatalf("passed kutilgan, olindi: %s", got)
	}
	if err := s.Consume(ch.Nonce); err != nil {
		t.Fatalf("Consume: %v", err)
	}
}

// Umumiy oynaning oxirida bajarilgan harakat: tanish kadri ChallengeTTL dan
// keyin kelsa ham qabul qilinishi kerak — aks holda to'g'ri bajargan odamning
// oqimi yopilib, ekranda "ishlamayapti" chiqardi.
func TestPassedAtEndOfWindowIsStillConsumable(t *testing.T) {
	s, advance := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}

	advance(ChallengeTTL - 3*time.Second)
	s.Observe(ch.Nonce, pose(0))
	advance(2 * time.Second)
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengePassed {
		t.Fatalf("passed kutilgan, olindi: %s", got)
	}
	advance(ChallengeConsumeGrace - time.Second) // umumiy oyna o'tdi
	if err := s.Consume(ch.Nonce); err != nil {
		t.Fatalf("bajarilgan buyruq rad etildi: %v", err)
	}
}

// Markazdan keyin harakat oynasi o'tib ketsa — umumiy muddat qolgan bo'lsa ham
// rad etiladi.
func TestActionWindowExpiresAfterCentre(t *testing.T) {
	s, advance := testStore()

	ch, _ := s.Issue()
	turn := -30.0
	if ch.Action == ChallengeTurnRight {
		turn = 30.0
	}

	s.Observe(ch.Nonce, pose(0))
	advance(ChallengeActionTTL + time.Second)
	if got := s.Observe(ch.Nonce, pose(turn)); got != ChallengeExpired {
		t.Fatalf("expired kutilgan, olindi: %s", got)
	}
	if err := s.Consume(ch.Nonce); err == nil {
		t.Fatal("harakat oynasi o'tgandan keyin qabul qilindi")
	}
}

// --------------------------------------------------- ma'lumotsiz kadrlar

// Xira yoki burchakli kadr holatni o'zgartirmasligi kerak: uni "bajarildi"
// deb hisoblash xavfli, "bajarilmadi" deb hisoblash esa bitta yomon kadr
// tufayli butun urinishni bekor qilardi.
func TestUnavailablePoseDoesNotAdvanceOrReset(t *testing.T) {
	s, _ := testStore()

	ch, _ := s.Issue()

	if got := s.Observe(ch.Nonce, nil); got != ChallengeAwaitingCentre {
		t.Fatalf("nil pose holatni o'zgartirdi: %s", got)
	}
	unavailable := &domain.HeadPose{Yaw: 45, Available: false}
	if got := s.Observe(ch.Nonce, unavailable); got != ChallengeAwaitingCentre {
		t.Fatal("available=false bo'lgan kadr hisobga olindi")
	}

	// Markazni ko'rgandan keyin yomon kadr uni QAYTARIB OLMASLIGI kerak.
	s.Observe(ch.Nonce, pose(0))
	if got := s.Observe(ch.Nonce, nil); got != ChallengeAwaitingAction {
		t.Fatalf("yomon kadr markaz holatini bekor qildi: %s", got)
	}
}

// -------------------------------------------------------------- tasodif

// Birinchi shart: buyruq bashorat qilinadigan bo'lsa, bitta video har safar
// ishlardi. Ikkala variant ham chiqishi kerak.
func TestIssuedActionsVary(t *testing.T) {
	s, _ := testStore()

	seen := make(map[ChallengeAction]int)
	for range 200 {
		ch, err := s.Issue()
		if err != nil {
			t.Fatalf("Issue: %v", err)
		}
		seen[ch.Action]++
	}

	if len(seen) < len(challengeActions) {
		t.Fatalf("barcha harakatlar chiqmadi: %v", seen)
	}
	// 200 tanlovda har biri kamida 50 marta chiqishi kerak (kutilgani 100).
	for action, count := range seen {
		if count < 50 {
			t.Errorf("%s juda kam chiqdi (%d/200) — taqsimot qiyshiq", action, count)
		}
	}
}

func TestNoncesAreUnique(t *testing.T) {
	s, _ := testStore()

	seen := make(map[string]bool)
	for range 500 {
		ch, _ := s.Issue()
		if seen[ch.Nonce] {
			t.Fatal("nonce takrorlandi — buyruqlar bir-birini bosib ketadi")
		}
		seen[ch.Nonce] = true
	}
}

// -------------------------------------------------------------- boshqalar

func TestUnknownNonce(t *testing.T) {
	s, _ := testStore()

	if got := s.Observe("aslo-mavjud-bo'lmagan", pose(0)); got != ChallengeUnknown {
		t.Fatalf("unknown kutilgan, olindi: %s", got)
	}
	if err := s.Consume("aslo-mavjud-bo'lmagan"); err == nil {
		t.Fatal("noma'lum nonce qabul qilindi")
	}
}

// Yangi harakat qo'shilib, actionSatisfied da qo'llab-quvvatlanmasa, tekshiruv
// jimgina hammani o'tkazib yubormasligi kerak.
func TestUnknownActionNeverSatisfied(t *testing.T) {
	for _, yaw := range []float64{-90, -30, 0, 30, 90} {
		if actionSatisfied(ChallengeAction("nod_head"), yaw) {
			t.Fatalf("noma'lum harakat yaw=%v da bajarilgan deb topildi", yaw)
		}
	}
}

func TestConcurrentObserveIsRaceFree(t *testing.T) {
	s, _ := testStore()
	ch, _ := s.Issue()

	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				s.Observe(ch.Nonce, pose(0))
			}
		}()
	}
	wg.Wait()
	// Tasdiq — race detector; bu yergacha yetib kelish o'zi natija.
}
