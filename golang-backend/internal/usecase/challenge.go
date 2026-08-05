package usecase

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// AKTIV LIVENESS — server buyruq beradi, foydalanuvchi bajaradi.
//
// # Qanday hujumni to'sadi
//
// Passiv anti-spoofing (MiniFASNet) bosma suratni va ekrandagi tasvirni
// to'sadi: ularda tekstura, aks etish va chuqurlik boshqacha. Lekin u
// OLDINDAN YOZILGAN VIDEONI to'sa olmaydi — video haqiqiy odamning haqiqiy
// yuzidan olingan, ya'ni har qanday tekstura testidan o'tadi.
//
// Aktiv tekshiruv shu bo'shliqni yopadi. Server har safar TASODIFIY buyruq
// beradi ("chapga qarang"), va javob shu buyruqqa mos kelishi kerak. Yozib
// qo'yilgan videoda buyruq oldindan ma'lum emas, shuning uchun mos javob
// bo'lishi mumkin emas.
//
// # Nima uchun uch shart birga
//
//	1. Buyruq TASODIFIY   — aks holda bitta video har safar ishlardi.
//	2. Buyruq BIR MARTALIK — aks holda o'sha seansni qayta o'ynatib bo'lardi.
//	3. Javob TEZ kelishi   — aks holda hujumchi buyruqni ko'rib, kerakli
//	                         videoni tanlab yoki generatsiya qilib ulgurardi.
//
// Uchtasidan bittasi tushib qolsa qolgan ikkitasi ham ma'nosiz bo'ladi.
//
// # Nimani to'smaydi — ochiq aytiladi
//
// Real vaqtdagi deepfake buyruqqa javob bera oladi, ya'ni bu tekshiruvdan
// o'tadi. Aktiv liveness passiv modelning O'RNINI BOSMAYDI, uni to'ldiradi:
// passiv qatlam soxta materialni, aktiv qatlam esa yozib olingan materialni
// to'sadi. Deepfake ikkalasi uchun ham alohida muammo bo'lib qoladi.

// ChallengeAction — foydalanuvchidan so'raladigan harakat.
type ChallengeAction string

const (
	ChallengeTurnLeft  ChallengeAction = "turn_left"
	ChallengeTurnRight ChallengeAction = "turn_right"
)

// challengeActions — tasodifiy tanlanadigan to'plam.
//
// Faqat chap/o'ng: ular yaw orqali ishonchli o'lchanadi. "Boshni ko'taring"
// pitch talab qiladi, u esa 5 ta kalit nuqtadan noaniq chiqadi — noaniq
// o'lchov bilan qaror chiqarish tekshiruvni kuchsizlantirardi.
var challengeActions = []ChallengeAction{ChallengeTurnLeft, ChallengeTurnRight}

// Yaw chegaralari (gradus).
const (
	// challengeYawThreshold — harakat "bajarildi" deb hisoblanishi uchun
	// kerakli minimal burilish. Past qo'ysak tabiiy tebranish ham o'tib
	// ketardi; balandda esa haqiqiy odamlar qiynaladi.
	challengeYawThreshold = 15.0

	// challengeCentreTolerance — "markazda" deb hisoblanadigan oraliq.
	// Foydalanuvchi avval markazga qarashi shart: aks holda doim yon tomonga
	// qarab turgan odam hech narsa qilmasdan tekshiruvdan o'tardi.
	challengeCentreTolerance = 10.0
)

// Challenge — berilgan va hali bajarilmagan buyruq.
type Challenge struct {
	Nonce     string          `json:"nonce"`
	Action    ChallengeAction `json:"action"`
	IssuedAt  time.Time       `json:"issued_at"`
	ExpiresAt time.Time       `json:"expires_at"`

	// centred — foydalanuvchi markazga qaraganini ko'rdikmi.
	centred bool
	// done — harakat bajarildimi.
	done bool
}

// ChallengeTTL — buyruq berilgandan javob kelgunicha bo'lgan oyna.
//
// Qisqa bo'lishi ATAYLAB: hujumchi buyruqni ko'rgach kerakli videoni topishi
// yoki generatsiya qilishi uchun vaqt qolmasligi kerak. Odam uchun boshini
// burish bir soniyalik ish, shuning uchun bu cheklov real foydalanuvchiga
// sezilmaydi.
const ChallengeTTL = 5 * time.Second

// ChallengeStore — berilgan buyruqlar ombori.
//
// Jarayon xotirasida: buyruq umri besh soniya va u faqat bitta ochiq oqim
// doirasida ishlatiladi, ya'ni uni instansiyalar orasida bo'lishishning
// hojati yo'q. Redis bu yerda faqat qo'shimcha nosozlik nuqtasi bo'lardi.
type ChallengeStore struct {
	mu    sync.Mutex
	items map[string]*Challenge
	now   func() time.Time
	stop  chan struct{}
}

func NewChallengeStore() *ChallengeStore {
	s := &ChallengeStore{
		items: make(map[string]*Challenge),
		now:   time.Now,
		stop:  make(chan struct{}),
	}
	go s.collectGarbage()
	return s
}

func (s *ChallengeStore) Close() { close(s.stop) }

// Issue — yangi tasodifiy buyruq beradi.
//
// Tasodif crypto/rand dan: math/rand bashorat qilinadigan bo'lardi va
// hujumchi keyingi buyruqni oldindan bilib, mos videoni tayyorlab qo'yardi.
func (s *ChallengeStore) Issue() (*Challenge, error) {
	nonce, err := randomNonce()
	if err != nil {
		return nil, err
	}
	action, err := randomAction()
	if err != nil {
		return nil, err
	}

	now := s.now()
	ch := &Challenge{
		Nonce:     nonce,
		Action:    action,
		IssuedAt:  now,
		ExpiresAt: now.Add(ChallengeTTL),
	}

	s.mu.Lock()
	s.items[nonce] = ch
	s.mu.Unlock()
	return ch, nil
}

// ChallengeState — bitta kadrdan keyingi holat.
type ChallengeState string

const (
	// ChallengeAwaitingCentre — avval markazga qarash kerak.
	ChallengeAwaitingCentre ChallengeState = "awaiting_centre"
	// ChallengeAwaitingAction — markaz ko'rildi, endi buyruqni bajarish kerak.
	ChallengeAwaitingAction ChallengeState = "awaiting_action"
	// ChallengePassed — bajarildi.
	ChallengePassed ChallengeState = "passed"
	// ChallengeExpired — vaqt tugadi.
	ChallengeExpired ChallengeState = "expired"
	// ChallengeUnknown — bunday nonce yo'q yoki allaqachon ishlatilgan.
	ChallengeUnknown ChallengeState = "unknown"
)

// Observe — bitta kadrni buyruq bo'yicha baholaydi.
//
// Ketma-ketlik MAJBURIY: markaz -> harakat. Faqat "yon tomonga qaradi" ni
// tekshirish yetarli emas, chunki doim yon tomonga qarab turgan surat ham
// shu shartni qanoatlantirardi. Harakatning O'ZI ko'rinishi kerak.
func (s *ChallengeStore) Observe(nonce string, pose *domain.HeadPose) ChallengeState {
	s.mu.Lock()
	defer s.mu.Unlock()

	ch, ok := s.items[nonce]
	if !ok {
		return ChallengeUnknown
	}
	if s.now().After(ch.ExpiresAt) {
		delete(s.items, nonce)
		return ChallengeExpired
	}
	if ch.done {
		return ChallengePassed
	}

	// Ma'lumotsiz kadr holatni o'zgartirmaydi. Uni "bajarilmadi" deb hisoblash
	// ham noto'g'ri bo'lardi (bitta xira kadr butun urinishni bekor qilardi),
	// "bajarildi" deb hisoblash esa xavfli.
	if pose == nil || !pose.Available {
		if ch.centred {
			return ChallengeAwaitingAction
		}
		return ChallengeAwaitingCentre
	}

	if !ch.centred {
		if abs(pose.Yaw) <= challengeCentreTolerance {
			ch.centred = true
			return ChallengeAwaitingAction
		}
		return ChallengeAwaitingCentre
	}

	if actionSatisfied(ch.Action, pose.Yaw) {
		ch.done = true
		return ChallengePassed
	}
	return ChallengeAwaitingAction
}

// Consume — bajarilgan buyruqni yakuniy tekshiradi va o'chiradi.
//
// O'chirish MAJBURIY: aks holda bitta muvaffaqiyatli seansni qayta-qayta
// ishlatib bo'lardi va tasodifiylikning ma'nosi qolmasdi.
func (s *ChallengeStore) Consume(nonce string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	ch, ok := s.items[nonce]
	if !ok {
		return domain.NewError(domain.CodeInvalidInput,
			"Tekshiruv topilmadi yoki allaqachon ishlatilgan.")
	}
	delete(s.items, nonce)

	if s.now().After(ch.ExpiresAt) {
		return domain.NewError(domain.CodeInvalidInput,
			"Tekshiruv muddati tugadi. Qaytadan urinib ko'ring.")
	}
	if !ch.done {
		return domain.NewError(domain.CodeSpoofDetected,
			"So'ralgan harakat bajarilmadi.")
	}
	return nil
}

// actionSatisfied — yaw buyruqqa mos keladimi.
func actionSatisfied(action ChallengeAction, yaw float64) bool {
	switch action {
	case ChallengeTurnLeft:
		// Chapga burilish manfiy yaw beradi — headpose.py dagi kelishuv.
		return yaw <= -challengeYawThreshold
	case ChallengeTurnRight:
		return yaw >= challengeYawThreshold
	default:
		// Noma'lum buyruq hech qachon bajarilgan deb hisoblanmaydi: yangi
		// harakat qo'shilganda uni bu yerda ham qo'llab-quvvatlash unutilsa,
		// tekshiruv jimgina hammani o'tkazib yuborardi.
		return false
	}
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func randomNonce() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("nonce hosil qilib bo'lmadi: %w", err)
	}
	return fmt.Sprintf("%x", buf), nil
}

func randomAction() (ChallengeAction, error) {
	n, err := rand.Int(rand.Reader, big.NewInt(int64(len(challengeActions))))
	if err != nil {
		return "", fmt.Errorf("tasodifiy harakat tanlab bo'lmadi: %w", err)
	}
	return challengeActions[n.Int64()], nil
}

// collectGarbage — muddati o'tgan buyruqlarni tozalaydi.
//
// Har biri besh soniyada o'ladi, lekin bajarilmagan urinishlar o'z-o'zidan
// o'chmaydi: ularni tozalamasa, xotira sekin o'sib boradi.
func (s *ChallengeStore) collectGarbage() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.stop:
			return
		case <-ticker.C:
			now := s.now()
			s.mu.Lock()
			for nonce, ch := range s.items {
				if now.After(ch.ExpiresAt) {
					delete(s.items, nonce)
				}
			}
			s.mu.Unlock()
		}
	}
}
