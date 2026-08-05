package security

import (
	"strings"
	"testing"
)

// Bu testlar production'da kalitlarning taxminan YARMINI ishlamas holga
// keltirgan xato uchun.
//
// Sir base64.RawURLEncoding bilan kodlanadi — alifbosi "A-Za-z0-9-_", ya'ni
// sirning O'ZIDA pastki chiziq bo'lishi mumkin. ParseAPIKey esa strings.Split
// ishlatib "aynan 3 ta bo'lak" talab qilardi, natijada bunday kalit
// "formati noto'g'ri" deb rad etilardi. Kalit bazada turardi va faol edi,
// xato xabari esa "API kalit noto'g'ri" derdi — sababni topish deyarli
// imkonsiz edi.
//
// Mavjud TestParseRejectsMalformed faqat RAD ETISH holatlarini tekshirardi.
// Yaroqli kalit aylanib qaytishini hech kim sinamagandi — bo'shliq aynan shu
// yerda edi.

func TestParseKeepsUnderscoresInSecret(t *testing.T) {
	// Bir nechta pastki chiziqli sir — eng og'ir holat.
	raw := "fdsk_2a9e049b8daf_rbNU_xY_zQ12"

	prefix, secret, err := ParseAPIKey(raw)
	if err != nil {
		t.Fatalf("pastki chiziqli sir rad etildi: %v", err)
	}
	if prefix != "2a9e049b8daf" {
		t.Errorf("prefiks noto'g'ri: %q", prefix)
	}
	if secret != "rbNU_xY_zQ12" {
		t.Errorf("sir kesilib qoldi: %q, kutilgan %q", secret, "rbNU_xY_zQ12")
	}
}

// ENG MUHIM TEST: yasalgan kalit HAR DOIM o'zini parse qila olishi va hash
// bilan mos kelishi kerak. Bu generatsiya va tahlilni bir-biriga bog'laydi —
// kodlash o'zgarsa test darhol yiqiladi.
func TestGeneratedKeysAlwaysRoundTrip(t *testing.T) {
	// 200 ta kalit. Xato bo'lganda bu test taxminan yarmida yiqilardi.
	for i := range 200 {
		key, err := GenerateAPIKey()
		if err != nil {
			t.Fatalf("%d: GenerateAPIKey: %v", i, err)
		}

		prefix, secret, err := ParseAPIKey(key.Secret)
		if err != nil {
			t.Fatalf("%d: yasalgan kalit parse bo'lmadi (%q): %v", i, key.Secret, err)
		}
		if prefix != key.Prefix {
			t.Fatalf("%d: prefiks mos kelmadi: %q != %q", i, prefix, key.Prefix)
		}
		// Kesilgan sir hash bilan hech qachon mos kelmaydi — autentifikatsiya
		// aynan shu yerda yiqilardi.
		if !SecretMatches(secret, key.Hash) {
			t.Fatalf("%d: sir hash bilan mos kelmadi — kalit kesilgan (%q)", i, key.Secret)
		}
	}
}

// Yuqoridagi test haqiqatan biror narsa isbotlashi uchun pastki chiziqli
// sirlar amalda uchrashi kerak. Aks holda u bekorga yashil bo'lardi.
func TestGeneratedSecretsDoContainUnderscores(t *testing.T) {
	for range 300 {
		key, err := GenerateAPIKey()
		if err != nil {
			t.Fatal(err)
		}
		if _, secret, err := ParseAPIKey(key.Secret); err == nil && strings.Contains(secret, "_") {
			return // topildi — test o'z maqsadini bajardi
		}
	}
	t.Error("300 ta kalitda pastki chiziq chiqmadi — kodlash o'zgarganmi? " +
		"Agar shunday bo'lsa, TestGeneratedKeysAlwaysRoundTrip endi hech narsa isbotlamaydi.")
}

// Defis ham base64url alifbosida bor — u ham saqlanishi kerak.
func TestParseKeepsHyphensInSecret(t *testing.T) {
	_, secret, err := ParseAPIKey("fdsk_abc123_aB-cD-eF")
	if err != nil {
		t.Fatalf("defisli sir rad etildi: %v", err)
	}
	if secret != "aB-cD-eF" {
		t.Errorf("sir buzildi: %q", secret)
	}
}
