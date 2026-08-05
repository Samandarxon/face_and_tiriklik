package usecase

import (
	"context"
	"testing"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// setupKeys — mijoz + kalit yaratib, ochiq matnini qaytaradi.
func setupKeys(t *testing.T, scopes ...domain.Scope) (*fakeKeyStore, *KeyManager, *domain.IssuedKey) {
	t.Helper()

	store := newFakeKeyStore()
	manager := NewKeyManager(store, &fakeCipher{})

	client, err := manager.CreateClient(context.Background(), ClientCommand{Name: "kirdi_chiqdi"})
	if err != nil {
		t.Fatalf("mijoz yaratilmadi: %v", err)
	}

	raw := make([]string, len(scopes))
	for i, s := range scopes {
		raw[i] = string(s)
	}
	issued, err := manager.IssueKey(context.Background(), IssueKeyCommand{
		ClientID: client.ID,
		Label:    "test",
		Scopes:   raw,
	})
	if err != nil {
		t.Fatalf("kalit berilmadi: %v", err)
	}
	return store, manager, issued
}

func newTestAuthenticator(store domain.APIKeyStore, ttl time.Duration, bootstrap string) *Authenticator {
	return NewAuthenticator(store, &fakeCipher{}, ttl, bootstrap, testLogger())
}

func TestAuthenticateAcceptsValidKey(t *testing.T) {
	store, _, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, 0, "")

	key, err := auth.Authenticate(context.Background(), issued.Secret)
	if err != nil {
		t.Fatalf("haqiqiy kalit rad etildi: %v", err)
	}
	if !key.HasScope(domain.ScopeVerify) {
		t.Error("scope yo'qolgan")
	}
	if key.HasScope(domain.ScopeAdmin) {
		t.Error("berilmagan scope paydo bo'ldi")
	}
}

func TestAuthenticateRejectsBadKeys(t *testing.T) {
	store, _, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, 0, "")

	cases := map[string]string{
		"bo'sh":            "",
		"format xato":      "abc",
		"noma'lum prefiks": "test_yoq_s1",
		"noto'g'ri sir":    "test_p1_boshqa",
		"ortiqcha belgi":   issued.Secret + "x",
	}

	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := auth.Authenticate(context.Background(), raw); domain.CodeOf(err) != domain.CodeUnauthenticated {
				t.Fatalf("unauthenticated kutilgan, olindi: %v", err)
			}
		})
	}
}

// Bekor qilingan kalit darhol ishlamay qolishi kerak.
func TestAuthenticateRejectsRevokedKey(t *testing.T) {
	store, manager, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, 0, "")

	if _, err := manager.RevokeKey(context.Background(), issued.Key.ID); err != nil {
		t.Fatalf("bekor qilinmadi: %v", err)
	}

	if _, err := auth.Authenticate(context.Background(), issued.Secret); domain.CodeOf(err) != domain.CodeUnauthenticated {
		t.Fatalf("bekor qilingan kalit qabul qilindi: %v", err)
	}
}

func TestAuthenticateRejectsExpiredKey(t *testing.T) {
	store := newFakeKeyStore()
	manager := NewKeyManager(store, &fakeCipher{})

	client, _ := manager.CreateClient(context.Background(), ClientCommand{Name: "x"})
	issued, err := manager.IssueKey(context.Background(), IssueKeyCommand{
		ClientID:      client.ID,
		Scopes:        []string{string(domain.ScopeVerify)},
		ExpiresInDays: 1,
	})
	if err != nil {
		t.Fatalf("kalit berilmadi: %v", err)
	}

	auth := newTestAuthenticator(store, 0, "")
	// Vaqtni oldinga suramiz.
	auth.now = func() time.Time { return time.Now().UTC().AddDate(0, 0, 2) }

	if _, err := auth.Authenticate(context.Background(), issued.Secret); domain.CodeOf(err) != domain.CodeUnauthenticated {
		t.Fatalf("muddati o'tgan kalit qabul qilindi: %v", err)
	}
}

// Mijoz faolsizlantirilsa uning kalitlari ham ishlamasligi kerak.
func TestAuthenticateRejectsInactiveClient(t *testing.T) {
	store, manager, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, 0, "")

	inactive := false
	if _, err := manager.UpdateClient(context.Background(), issued.Key.ClientID,
		ClientCommand{Name: "kirdi_chiqdi", IsActive: &inactive}); err != nil {
		t.Fatalf("mijoz yangilanmadi: %v", err)
	}

	if _, err := auth.Authenticate(context.Background(), issued.Secret); domain.CodeOf(err) != domain.CodeUnauthenticated {
		t.Fatalf("faolsiz mijoz kaliti qabul qilindi: %v", err)
	}
}

// Kesh ishlayotganini tekshiramiz: ikkinchi so'rov bazaga bormasligi kerak.
func TestAuthenticateUsesCache(t *testing.T) {
	store, _, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, time.Minute, "")

	for range 3 {
		if _, err := auth.Authenticate(context.Background(), issued.Secret); err != nil {
			t.Fatalf("kutilmagan xato: %v", err)
		}
	}

	if calls := store.callCount(); calls != 1 {
		t.Fatalf("bazaga %d marta borildi — 1 kutilgan (kesh ishlamayapti)", calls)
	}
}

// Bekor qilingandan keyin kesh tozalansa, kalit darhol ishlamay qolishi kerak.
func TestInvalidateDropsCachedKey(t *testing.T) {
	store, manager, issued := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, time.Minute, "")

	if _, err := auth.Authenticate(context.Background(), issued.Secret); err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if _, err := manager.RevokeKey(context.Background(), issued.Key.ID); err != nil {
		t.Fatalf("bekor qilinmadi: %v", err)
	}
	auth.Invalidate(issued.Key.Prefix)

	if _, err := auth.Authenticate(context.Background(), issued.Secret); err == nil {
		t.Fatal("keshdan chiqarilgandan keyin ham kalit ishladi")
	}
}

// ============================================================== bootstrap

const bootstrapSecret = "bootstrap-key-at-least-32-characters-long"

// Baza bo'sh bo'lganda bootstrap kaliti admin huquqini berishi kerak.
func TestBootstrapWorksOnEmptyStore(t *testing.T) {
	store := newFakeKeyStore()
	auth := newTestAuthenticator(store, 0, bootstrapSecret)

	key, err := auth.Authenticate(context.Background(), bootstrapSecret)
	if err != nil {
		t.Fatalf("bootstrap kaliti rad etildi: %v", err)
	}
	if !key.HasScope(domain.ScopeAdmin) {
		t.Fatal("bootstrap kalitida admin huquqi yo'q")
	}
}

// Birinchi haqiqiy kalit berilishi bilan bootstrap yopilishi kerak — aks holda
// u doimiy orqa eshikka aylanadi.
func TestBootstrapClosesAfterFirstKey(t *testing.T) {
	store, _, _ := setupKeys(t, domain.ScopeVerify)
	auth := newTestAuthenticator(store, 0, bootstrapSecret)

	if _, err := auth.Authenticate(context.Background(), bootstrapSecret); err == nil {
		t.Fatal("faol kalit mavjud bo'lsa ham bootstrap ishladi")
	}
}

func TestBootstrapDisabledWhenNotConfigured(t *testing.T) {
	store := newFakeKeyStore()
	auth := newTestAuthenticator(store, 0, "")

	if _, err := auth.Authenticate(context.Background(), bootstrapSecret); err == nil {
		t.Fatal("sozlanmagan bootstrap kaliti qabul qilindi")
	}
}

// ============================================================== kalitlar

func TestIssueKeyValidatesScopes(t *testing.T) {
	store := newFakeKeyStore()
	manager := NewKeyManager(store, &fakeCipher{})
	client, _ := manager.CreateClient(context.Background(), ClientCommand{Name: "x"})

	cases := map[string][]string{
		"scope yo'q":     {},
		"noma'lum scope": {"face:everything"},
		"bo'sh qiymat":   {""},
	}

	for name, scopes := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := manager.IssueKey(context.Background(), IssueKeyCommand{
				ClientID: client.ID,
				Scopes:   scopes,
			})
			if domain.CodeOf(err) != domain.CodeInvalidInput {
				t.Fatalf("invalid_request kutilgan, olindi: %v", err)
			}
		})
	}
}

// Rotatsiya: eskisi ishlamaydi, yangisi aynan shu huquqlar bilan ishlaydi.
func TestRotateKeyReplacesOldOne(t *testing.T) {
	store, manager, old := setupKeys(t, domain.ScopeVerify, domain.ScopeRead)
	auth := newTestAuthenticator(store, 0, "")

	fresh, oldPrefix, err := manager.RotateKey(context.Background(), old.Key.ID)
	if err != nil {
		t.Fatalf("rotatsiya xatosi: %v", err)
	}
	if oldPrefix != old.Key.Prefix {
		t.Errorf("eski prefiks qaytmadi: %q", oldPrefix)
	}
	if fresh.Secret == old.Secret {
		t.Error("yangi kalit eskisi bilan bir xil")
	}

	if _, err := auth.Authenticate(context.Background(), old.Secret); err == nil {
		t.Error("eski kalit hali ham ishlayapti")
	}

	key, err := auth.Authenticate(context.Background(), fresh.Secret)
	if err != nil {
		t.Fatalf("yangi kalit ishlamadi: %v", err)
	}
	if !key.HasScope(domain.ScopeVerify) || !key.HasScope(domain.ScopeRead) {
		t.Error("rotatsiyada huquqlar yo'qoldi")
	}
}

func TestIssueKeyRejectsInactiveClient(t *testing.T) {
	store := newFakeKeyStore()
	manager := NewKeyManager(store, &fakeCipher{})

	inactive := false
	client, _ := manager.CreateClient(context.Background(), ClientCommand{
		Name: "x", IsActive: &inactive,
	})

	_, err := manager.IssueKey(context.Background(), IssueKeyCommand{
		ClientID: client.ID,
		Scopes:   []string{string(domain.ScopeVerify)},
	})
	if domain.CodeOf(err) != domain.CodeInvalidInput {
		t.Fatalf("faolsiz mijozga kalit berildi: %v", err)
	}
}

func TestCreateClientRejectsDuplicateName(t *testing.T) {
	store := newFakeKeyStore()
	manager := NewKeyManager(store, &fakeCipher{})

	if _, err := manager.CreateClient(context.Background(), ClientCommand{Name: "kirdi_chiqdi"}); err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	_, err := manager.CreateClient(context.Background(), ClientCommand{Name: "kirdi_chiqdi"})
	if domain.CodeOf(err) != domain.CodeConflict {
		t.Fatalf("konflikt kutilgan, olindi: %v", err)
	}
}
