// Package usecase — ilova stsenariylari (application layer).
//
// Bu qatlam domain portlarini birlashtiradi, lekin HTTP, gRPC yoki Qdrant
// haqida hech narsa bilmaydi. Barcha bog'liqliklar interfeys orqali kiritiladi.
package usecase

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// RegisterCommand — ro'yxatga olish so'rovi.
//
// Odamni identifikatsiya qilishning uchta yo'li bor va ular bir-birini
// istisno qiladi:
//
//	SubjectID  — tashqi identifikator, ya'ni PNFL. TAVSIYA ETILADI: servis uni
//	             normallashtiradi va AYNAN o'zini `person_id` qilib oladi.
//	PersonID   — tayyor `person_id` (u ham PNFL bo'ladi).
//	ikkalasi ham bo'sh — servis tasodifiy ID beradi (anonim yozuv).
type RegisterCommand struct {
	SubjectID string
	PersonID  string
	Name      string
	Image     []byte
	FileName  string
	RequestID string
	// RequireLiveness — kadr JONLI kameradan kelganda yoqiladi: avval
	// anti-spoofing, keyin vektor. Arxiv fotosi yoki passport skani uchun
	// o'chiq qoldiriladi (ular ta'rifga ko'ra "spoof").
	RequireLiveness bool
	// Replace — mavjud yozuvning vektorini almashtirishga ANIQ ruxsat.
	// Standart false: yozuv bor bo'lsa `already_exists` qaytadi. Aks holda
	// PNFL'ni bilgan har qanday mijoz boshqa odamning yuzini o'zinikiga
	// almashtirib, keyin uning nomidan yuz orqali kira olardi.
	Replace bool
}

// Registrar — insonni bazaga qo'shish stsenariysi.
type Registrar struct {
	analyzer   domain.FaceAnalyzer
	repo       domain.PersonRepository
	ids        domain.IDGenerator
	subjects   domain.SubjectResolver
	vectorSize int
	// gate — bazaga yozish uchun qattiqroq sifat talabi.
	gate EnrollGate
	log  *slog.Logger
	now  func() time.Time
	// writeMu — "bormi?" tekshiruvi va yozuvni bitta amalga aylantiradi:
	// bir PNFL uchun bir vaqtda kelgan ikki register ikkalasi ham "yo'q"
	// ko'rib, biri ikkinchisini jimgina almashtirib yubormasin. Faqat shu
	// jarayon ichida ishlaydi — bir nechta nusxada ishga tushirilsa, bu
	// kafolat yo'qoladi.
	writeMu sync.Mutex
}

// EnrollGate — ro'yxatga olish uchun sifat chegarasi.
//
// # Nima uchun qidiruvdagidan qattiqroq
//
// Qidiruvda tanlov yo'q: kamerada nima bo'lsa shu, va rad etish "odam
// o'tolmadi" degani. Yozishda esa tanlov bor — yaxshiroq kadr kutish
// mumkin. Va farq assimetrik: sifatsiz YOZUV bazada umrbod qoladi va
// har bir keyingi qidiruvga zarar yetkazadi, sifatsiz SO'ROV esa faqat
// bitta urinishni buzadi.
//
// Nol qiymatlar tekshiruvni o'chiradi.
type EnrollGate struct {
	MinDetScore float32
	MinFaceSize int
}

// check — yuz yozishga yaroqlimi.
func (g EnrollGate) check(face *domain.Face) error {
	if g.MinDetScore > 0 && face.DetScore < float64(g.MinDetScore) {
		return domain.NewError(domain.CodeLowQualityFace,
			"Rasm bazaga yozish uchun yetarli aniq emas. Yaxshiroq kadr oling.").
			WithDetails(map[string]any{
				"det_score":    face.DetScore,
				"min_required": g.MinDetScore,
				"reason":       "det_score",
			})
	}
	if g.MinFaceSize > 0 {
		if side := face.BBox.MinSide(); side > 0 && side < g.MinFaceSize {
			return domain.NewError(domain.CodeLowQualityFace,
				"Yuz bazaga yozish uchun juda kichik. Kameraga yaqinroq turing.").
				WithDetails(map[string]any{
					"face_size":    side,
					"min_required": g.MinFaceSize,
					"reason":       "face_size",
				})
		}
	}
	return nil
}

func NewRegistrar(
	analyzer domain.FaceAnalyzer,
	repo domain.PersonRepository,
	ids domain.IDGenerator,
	subjects domain.SubjectResolver,
	vectorSize int,
	gate EnrollGate,
	log *slog.Logger,
) *Registrar {
	return &Registrar{
		analyzer:   analyzer,
		repo:       repo,
		ids:        ids,
		subjects:   subjects,
		vectorSize: vectorSize,
		gate:       gate,
		log:        log,
		now:        func() time.Time { return time.Now().UTC() },
	}
}

// Execute stsenariysi:
//
//	a) rasm AI servisga yuboriladi
//	b) AI servis kadrda aynan 1 ta yuz borligini tekshirib, vektor qaytaradi
//	c) vektor `person_id` + `name` metadata bilan omborga saqlanadi
//
// Liveness sukut bo'yicha tekshirilmaydi: odamlar odatda passport skani yoki
// HR arxividagi fotodan ro'yxatga olinadi, ular esa ta'rifga ko'ra "spoof".
//
// `RequireLiveness` yoqilganda esa oqim `VerifyFrame` ga o'tadi: avval
// anti-spoofing, keyin vektor. Bu foydalanuvchi o'z yuzini KAMERA ORQALI
// biriktirayotgan holat uchun — u yerda "surat ko'tarib turgan odam" aynan
// oldini olinishi kerak bo'lgan hujum. Tekshiruv va vektor BIR XIL kadrdan
// olinadi, ya'ni bazaga tushgan vektor aynan anti-spoofingdan o'tgan kadrniki.
func (r *Registrar) Execute(ctx context.Context, cmd RegisterCommand) (*domain.RegistrationResult, error) {
	// `name` ixtiyoriy va ataylab shunday: bu servis bir nechta platformaga
	// xizmat qiladi, shuning uchun unga F.I.SH kabi shaxsiy ma'lumot
	// yubormaslik afzal — mijoz o'z bazasida `person_id` bo'yicha topadi.
	name := strings.TrimSpace(cmd.Name)
	if len(name) > 256 {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`name` 256 belgidan oshmasligi kerak.")
	}

	personID, generated, err := r.resolveID(cmd)
	if err != nil {
		return nil, err
	}

	// (a) + (b)
	var (
		face     *domain.Face
		liveness *domain.Liveness
	)
	if cmd.RequireLiveness {
		face, liveness, err = r.analyzer.VerifyFrame(ctx, cmd.Image, cmd.RequestID)
	} else {
		face, err = r.analyzer.Embed(ctx, cmd.Image, cmd.RequestID)
	}
	if err != nil {
		return nil, err
	}
	if face.Embedding.Dim() != r.vectorSize {
		return nil, domain.NewError(domain.CodeInternal,
			"AI servis kutilmagan o'lchamdagi vektor qaytardi.").
			WithDetails(map[string]any{
				"expected": r.vectorSize,
				"got":      face.Embedding.Dim(),
			})
	}
	// Yozishdan OLDIN. Keyin tekshirish ma'nosiz bo'lardi: yozuv allaqachon
	// bazada bo'lardi va uni qaytarib olish alohida amal talab qilardi.
	if err := r.gate.check(face); err != nil {
		return nil, err
	}

	// (c)
	now := r.now()
	person := domain.Person{
		ID:         personID,
		Name:       name,
		CreatedAt:  now,
		DetScore:   face.DetScore,
		SourceFile: cmd.FileName,
	}

	// Generatsiya qilingan ID uchun mavjud yozuv bo'lishi mumkin emas —
	// ortiqcha so'rov yubormaymiz.
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	updated := false
	if !generated {
		existing, err := r.repo.Get(ctx, personID)
		if err != nil {
			// Tekshirib bo'lmasa yozmaymiz: aks holda Replace'siz so'rov
			// ham mavjud yuzni almashtirib yuborishi mumkin edi.
			return nil, err
		}
		if existing != nil {
			if !cmd.Replace {
				return nil, domain.NewError(domain.CodeConflict,
					"Bu shaxs bazada allaqachon bor. Yuzni almashtirish uchun `replace=true` yuboring.")
			}
			updated = true
			person.CreatedAt = existing.CreatedAt
			person.UpdatedAt = now
		}
	}

	pointID, err := r.repo.Save(ctx, person, face.Embedding)
	if err != nil {
		return nil, err
	}

	r.log.Info("person registered",
		"request_id", cmd.RequestID, "person_id", personID,
		"generated_id", generated, "updated", updated, "det_score", face.DetScore)

	return &domain.RegistrationResult{
		Person:      person,
		PointID:     pointID,
		Updated:     updated,
		GeneratedID: generated,
		Face:        *face,
		// `RequireLiveness` bo'lmaganda nil — anti-spoofing umuman ishlamagan.
		// To'ldirilgan holda mijoz vektor aynan qaysi ballar bilan o'tgan
		// kadrdan olinganini ko'ra oladi.
		Liveness: liveness,
	}, nil
}

// resolveID — buyruqdagi identifikatorlardan `person_id` ni aniqlaydi.
//
// `subject_id` va `person_id` bir vaqtda kelishi mumkin emas: bu deyarli doim
// mijoz xatosi bo'ladi va jimgina birontasini tanlash keyinchalik "odam
// bazada ikki marta" degan muammoga olib keladi.
// AddSample — mavjud shaxsning galereyasiga yangi namuna qo'shadi.
//
// # Nima uchun bu alohida amal
//
// `Execute` shaxsning ASOSIY vektorini almashtiradi — ya'ni eskisini
// yo'qotadi. Bu 10-20 yillik arxiv uchun aynan noto'g'ri: eski rasm
// yagona bo'lishi mumkin va uni yangisi bilan almashtirish qaytarib
// bo'lmaydigan yo'qotish.
//
// AddSample esa qo'shadi. Galereya vaqt o'tishi bilan o'sadi va shaxsni
// turli yosh, rakurs va yoritishda qamrab oladi — qidiruvda esa eng mos
// keladigan namuna ishlaydi (`FindNearest` max bo'yicha yig'adi).
//
// Namuna nomi (`sample`) — kaliti. Bir xil nom bilan qayta yuborilsa
// o'sha namuna yangilanadi, ya'ni amal idempotent.
func (r *Registrar) AddSample(
	ctx context.Context, personID, sample string, cmd RegisterCommand,
) (*domain.RegistrationResult, error) {
	personID = strings.TrimSpace(personID)
	sample = strings.TrimSpace(sample)
	if personID == "" {
		return nil, domain.NewError(domain.CodeInvalidInput, "`person_id` majburiy.")
	}
	if sample == "" {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`sample` majburiy — namuna nomisiz uni keyin yangilab ham, o'chirib ham bo'lmaydi.")
	}
	if len(sample) > 64 {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`sample` 64 belgidan oshmasligi kerak.")
	}

	// Shaxs mavjudligini tekshiramiz: galereya egasiz bo'lmasligi kerak,
	// aks holda hech kimga tegishli bo'lmagan vektorlar to'planib qoladi.
	existing, err := r.repo.Get(ctx, personID)
	if err != nil {
		return nil, err
	}
	if existing == nil {
		return nil, domain.NewError(domain.CodeNotFound,
			"Bunday shaxs bazada yo'q — avval uni ro'yxatga oling.")
	}

	var face *domain.Face
	if cmd.RequireLiveness {
		face, _, err = r.analyzer.VerifyFrame(ctx, cmd.Image, cmd.RequestID)
	} else {
		face, err = r.analyzer.Embed(ctx, cmd.Image, cmd.RequestID)
	}
	if err != nil {
		return nil, err
	}
	if face.Embedding.Dim() != r.vectorSize {
		return nil, domain.NewError(domain.CodeInternal,
			"AI servis kutilmagan o'lchamdagi vektor qaytardi.")
	}
	// Galereyaga ham xuddi shunday qattiq talab. Sifatsiz namuna galereyani
	// yaxshilamaydi — u shovqin qo'shadi va max-agregatsiyada noto'g'ri
	// mosllikni kuchaytirishi mumkin.
	if err := r.gate.check(face); err != nil {
		return nil, err
	}

	person := *existing
	person.UpdatedAt = r.now()
	person.DetScore = face.DetScore
	if cmd.FileName != "" {
		person.SourceFile = cmd.FileName
	}

	pointID, err := r.repo.SaveSample(ctx, person, face.Embedding, sample)
	if err != nil {
		return nil, err
	}
	r.log.Info("galereyaga namuna qo'shildi",
		"person_id", personID, "sample", sample, "det_score", face.DetScore)

	return &domain.RegistrationResult{
		Person:  person,
		PointID: pointID,
		Updated: true,
	}, nil
}

func (r *Registrar) resolveID(cmd RegisterCommand) (personID string, generated bool, err error) {
	subjectID := strings.TrimSpace(cmd.SubjectID)
	explicitID := strings.TrimSpace(cmd.PersonID)

	switch {
	case subjectID != "" && explicitID != "":
		return "", false, domain.NewError(domain.CodeInvalidInput,
			"`subject_id` va `person_id` birgalikda yuborilmaydi — bittasini tanlang.")

	case subjectID != "":
		// Uzunlik va format tekshiruvi resolver ichida: qat'iy PNFL rejimida
		// u 14 xonalik shartni ham qo'llaydi.
		resolved, err := r.subjects.PersonID(subjectID)
		if err != nil {
			return "", false, err
		}
		return resolved, false, nil

	case explicitID != "":
		if len(explicitID) > 128 {
			return "", false, domain.NewError(domain.CodeInvalidInput,
				"`person_id` 128 belgidan oshmasligi kerak.")
		}
		return explicitID, false, nil

	default:
		newID, err := r.ids.NewID()
		if err != nil {
			return "", false, domain.WrapError(domain.CodeInternal,
				"Identifikator hosil qilib bo'lmadi.", err)
		}
		return newID, true, nil
	}
}
