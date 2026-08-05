package usecase

import (
	"context"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// KeyManager — mijozlar va API kalitlarni boshqarish stsenariylari.
//
// Bu use-case boshqa platformalarni servisga ulash uchun yagona yo'l:
// har bir platforma o'z mijoz yozuvini va o'z kalitini oladi, kalitning
// huquqlari esa scope'lar bilan cheklanadi.
type KeyManager struct {
	store  domain.APIKeyStore
	cipher domain.KeyCipher
	now    func() time.Time
}

func NewKeyManager(store domain.APIKeyStore, cipher domain.KeyCipher) *KeyManager {
	return &KeyManager{
		store:  store,
		cipher: cipher,
		now:    func() time.Time { return time.Now().UTC() },
	}
}

// ============================================================== Mijozlar

// ClientCommand — mijoz yaratish/yangilash uchun kirish.
type ClientCommand struct {
	Name        string
	Contact     string
	Description string
	IsActive    *bool // nil — o'zgarishsiz qoldirish (yangilashda)
}

func (m *KeyManager) CreateClient(ctx context.Context, cmd ClientCommand) (*domain.APIClient, error) {
	name := strings.TrimSpace(cmd.Name)
	if name == "" {
		return nil, domain.NewError(domain.CodeInvalidInput, "`name` maydoni majburiy.")
	}
	if len(name) > 128 {
		return nil, domain.NewError(domain.CodeInvalidInput, "`name` 128 belgidan oshmasligi kerak.")
	}

	client := &domain.APIClient{
		ID:          uuid.NewString(),
		Name:        name,
		Contact:     strings.TrimSpace(cmd.Contact),
		Description: strings.TrimSpace(cmd.Description),
		IsActive:    true,
	}
	if cmd.IsActive != nil {
		client.IsActive = *cmd.IsActive
	}

	if err := m.store.CreateClient(ctx, client); err != nil {
		return nil, err
	}
	return client, nil
}

func (m *KeyManager) UpdateClient(ctx context.Context, id string, cmd ClientCommand) (*domain.APIClient, error) {
	client, err := m.store.GetClient(ctx, id)
	if err != nil {
		return nil, err
	}

	if name := strings.TrimSpace(cmd.Name); name != "" {
		client.Name = name
	}
	client.Contact = strings.TrimSpace(cmd.Contact)
	client.Description = strings.TrimSpace(cmd.Description)
	if cmd.IsActive != nil {
		client.IsActive = *cmd.IsActive
	}

	if err := m.store.UpdateClient(ctx, client); err != nil {
		return nil, err
	}
	return client, nil
}

func (m *KeyManager) GetClient(ctx context.Context, id string) (*domain.APIClient, error) {
	return m.store.GetClient(ctx, id)
}

func (m *KeyManager) ListClients(ctx context.Context) ([]domain.APIClient, error) {
	return m.store.ListClients(ctx)
}

// DeleteClient — mijoz bilan birga uning kalitlari ham o'chadi.
func (m *KeyManager) DeleteClient(ctx context.Context, id string) error {
	return m.store.DeleteClient(ctx, id)
}

// ============================================================== Kalitlar

// IssueKeyCommand — yangi kalit berish so'rovi.
type IssueKeyCommand struct {
	ClientID string
	Label    string
	Scopes   []string
	// RateLimitPerMin — 0 bo'lsa servis standarti qo'llanadi.
	RateLimitPerMin int
	// ExpiresInDays — 0 bo'lsa kalit muddatsiz.
	ExpiresInDays int
}

// IssueKey — kalit yasaydi va uni faqat SHU javobda ochiq matnda qaytaradi.
// Bazaga hash yoziladi, shuning uchun kalitni keyin tiklab bo'lmaydi.
func (m *KeyManager) IssueKey(ctx context.Context, cmd IssueKeyCommand) (*domain.IssuedKey, error) {
	client, err := m.store.GetClient(ctx, cmd.ClientID)
	if err != nil {
		return nil, err
	}
	if !client.IsActive {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"Mijoz faol emas — avval uni faollashtiring.")
	}

	scopes, err := domain.ParseScopes(cmd.Scopes)
	if err != nil {
		return nil, err
	}
	if cmd.RateLimitPerMin < 0 {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`rate_limit_per_min` manfiy bo'lishi mumkin emas.")
	}
	if cmd.ExpiresInDays < 0 {
		return nil, domain.NewError(domain.CodeInvalidInput,
			"`expires_in_days` manfiy bo'lishi mumkin emas.")
	}

	generated, err := m.cipher.Generate()
	if err != nil {
		return nil, domain.WrapError(domain.CodeInternal, "Kalit yasab bo'lmadi.", err)
	}

	key := &domain.APIKey{
		ID:              uuid.NewString(),
		ClientID:        client.ID,
		ClientName:      client.Name,
		Label:           strings.TrimSpace(cmd.Label),
		Prefix:          generated.Prefix,
		Scopes:          scopes,
		RateLimitPerMin: cmd.RateLimitPerMin,
		ClientActive:    client.IsActive,
	}
	if cmd.ExpiresInDays > 0 {
		expires := m.now().AddDate(0, 0, cmd.ExpiresInDays)
		key.ExpiresAt = &expires
	}

	if err := m.store.CreateKey(ctx, key, generated.Hash); err != nil {
		return nil, err
	}
	return &domain.IssuedKey{Key: *key, Secret: generated.Secret}, nil
}

// RotateKey — eskisini bekor qilib, aynan shu huquqlar bilan yangisini beradi.
//
// Rotatsiya "bekor qilish + yangi berish" dan ataylab alohida: mijoz scope'lar
// ro'yxatini qayta yozib chiqishi shart emas, xato qilish ehtimoli kamayadi.
// Eski kalitning prefiksi ham qaytariladi — chaqiruvchi uni keshdan
// chiqarishi kerak, aks holda bekor qilingan kalit kesh muddati tugagunicha
// ishlab turaveradi.
func (m *KeyManager) RotateKey(ctx context.Context, keyID string) (*domain.IssuedKey, string, error) {
	old, err := m.store.GetKey(ctx, keyID)
	if err != nil {
		return nil, "", err
	}

	scopes := make([]string, len(old.Scopes))
	for i, s := range old.Scopes {
		scopes[i] = string(s)
	}

	var expiresInDays int
	if old.ExpiresAt != nil {
		// Eski kalitning qolgan amal muddatini saqlab qolamiz.
		if remaining := int(time.Until(*old.ExpiresAt).Hours() / 24); remaining > 0 {
			expiresInDays = remaining
		} else {
			expiresInDays = 1
		}
	}

	issued, err := m.IssueKey(ctx, IssueKeyCommand{
		ClientID:        old.ClientID,
		Label:           old.Label,
		Scopes:          scopes,
		RateLimitPerMin: old.RateLimitPerMin,
		ExpiresInDays:   expiresInDays,
	})
	if err != nil {
		return nil, "", err
	}

	// Yangisi muvaffaqiyatli yaratilgandan KEYIN eskisi bekor qilinadi —
	// aks holda xatolik yuz bersa mijoz kalitsiz qolib ketardi.
	if err := m.store.RevokeKey(ctx, keyID, m.now()); err != nil {
		return nil, "", err
	}
	return issued, old.Prefix, nil
}

func (m *KeyManager) RevokeKey(ctx context.Context, keyID string) (*domain.APIKey, error) {
	key, err := m.store.GetKey(ctx, keyID)
	if err != nil {
		return nil, err
	}
	if err := m.store.RevokeKey(ctx, keyID, m.now()); err != nil {
		return nil, err
	}
	return key, nil
}

// PurgeKey — bekor qilingan kalit yozuvini butunlay o'chiradi.
//
// # Nega avval bekor qilish shart
//
// Ishlab turgan kalitni bevosita o'chirish uni jimgina ishlamas holga
// keltiradi va ayni paytda undan qolgan barcha izni ham yo'q qiladi: kim
// foydalanayotgani, qachon berilgani, oxirgi murojaati. Buzilgan integratsiya
// sababini keyin aniqlab bo'lmaydi.
//
// Shuning uchun tartib ikki qadamli: avval bekor qilinadi (tizim to'xtaydi,
// yozuv qoladi), keyin — agar ro'yxatni tozalash kerak bo'lsa — o'chiriladi.
func (m *KeyManager) PurgeKey(ctx context.Context, keyID string) error {
	key, err := m.store.GetKey(ctx, keyID)
	if err != nil {
		return err
	}
	if key.RevokedAt == nil {
		return domain.NewError(domain.CodeConflict,
			"Avval kalitni bekor qiling — ishlab turgan kalitni o'chirish undan qolgan izni ham yo'q qiladi.")
	}
	return m.store.PurgeKey(ctx, keyID)
}

func (m *KeyManager) ListKeys(ctx context.Context, clientID string) ([]domain.APIKey, error) {
	return m.store.ListKeys(ctx, clientID)
}
