package security

import (
	"strings"
	"testing"
)

// Yasalgan kalit o'zining hash'iga mos kelishi, boshqasiga esa mos
// kelmasligi kerak.
func TestGenerateAndMatch(t *testing.T) {
	generated, err := GenerateAPIKey()
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}

	prefix, secret, err := ParseAPIKey(generated.Secret)
	if err != nil {
		t.Fatalf("yasalgan kalit o'qilmadi: %v", err)
	}
	if prefix != generated.Prefix {
		t.Fatalf("prefiks mos kelmadi: %q != %q", prefix, generated.Prefix)
	}
	if !SecretMatches(secret, generated.Hash) {
		t.Fatal("kalit o'z hash'iga mos kelmadi")
	}
	if SecretMatches(secret+"x", generated.Hash) {
		t.Fatal("o'zgartirilgan kalit qabul qilindi")
	}
}

// Ochiq matn hech qachon hash bilan bir xil bo'lmasligi kerak — ya'ni
// bazadagi qiymat kalitning o'zi emas.
func TestSecretIsNotStoredInPlain(t *testing.T) {
	generated, _ := GenerateAPIKey()
	if strings.Contains(generated.Hash, generated.Secret) {
		t.Fatal("hash ichida ochiq kalit bor")
	}
}

func TestGenerateIsUnique(t *testing.T) {
	seen := make(map[string]bool, 100)
	for range 100 {
		generated, err := GenerateAPIKey()
		if err != nil {
			t.Fatalf("kutilmagan xato: %v", err)
		}
		if seen[generated.Prefix] {
			t.Fatalf("prefiks takrorlandi: %s", generated.Prefix)
		}
		seen[generated.Prefix] = true
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	cases := []string{
		"",
		"fdsk",
		"fdsk_abc",
		"wrong_abc_def",
		"fdsk__def",
		"fdsk_abc_",
	}
	for _, raw := range cases {
		if _, _, err := ParseAPIKey(raw); err == nil {
			t.Errorf("%q qabul qilindi — rad etilishi kerak edi", raw)
		}
	}
}

// ================================================================ person_id

// Bir xil kirish doim bir xil natija berishi kerak — aks holda bitta odam
// bazada bir necha marta paydo bo'lardi.
func TestPersonIDIsDeterministic(t *testing.T) {
	subjects := NewSubjectResolver(true)

	first, err := subjects.PersonID("12345678901234")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	second, _ := subjects.PersonID("12345678901234")

	if first != second {
		t.Fatalf("bir xil PNFL uchun ikki xil ID: %s != %s", first, second)
	}
}

// person_id — bu PNFL ning O'ZI. Aynan shu xususiyat bazani ko'chma qiladi:
// yozuvni o'qigan istalgan tizim odamni darhol taniydi.
func TestPersonIDIsThePNFLItself(t *testing.T) {
	subjects := NewSubjectResolver(true)

	id, err := subjects.PersonID("12345678901234")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if id != "12345678901234" {
		t.Fatalf("person_id PNFL ga teng emas: %q", id)
	}
}

// Formatlash farqlari bitta odamni ikkiga bo'lib yubormasligi kerak.
func TestPersonIDNormalizesInput(t *testing.T) {
	subjects := NewSubjectResolver(true)

	expected, _ := subjects.PersonID("12345678901234")
	for _, variant := range []string{
		" 12345678901234 ",
		"1234-5678-901234",
		"12345678901234\n",
		"1234.5678.901234",
		"1234_5678_901234",
	} {
		got, err := subjects.PersonID(variant)
		if err != nil {
			t.Fatalf("%q: kutilmagan xato: %v", variant, err)
		}
		if got != expected {
			t.Errorf("%q boshqa ID berdi: %q", variant, got)
		}
	}
}

// Qat'iy rejim — standart. PNFL bo'lmagan hamma narsa rad etilishi kerak,
// aks holda bazaga aralash formatdagi ID lar tushib, "bir odam — bir ID"
// qoidasi buziladi.
func TestStrictModeRejectsNonPNFL(t *testing.T) {
	subjects := NewSubjectResolver(true)

	for _, raw := range []string{
		"1234567890123",   // 13 ta — kam
		"123456789012345", // 15 ta — ko'p
		"1234567890123a",  // harf bor
		"AB1234567",       // passport
		"tabel-42",        // ichki kod
	} {
		if _, err := subjects.PersonID(raw); err == nil {
			t.Errorf("%q qat'iy rejimda qabul qilindi", raw)
		}
	}
}

// Erkin rejim boshqa formatdagi identifikatorlarga ruxsat beradi, lekin
// normallashtirishni baribir qo'llaydi.
func TestFreeModeAcceptsOtherFormats(t *testing.T) {
	subjects := NewSubjectResolver(false)

	got, err := subjects.PersonID(" Tabel-0042 ")
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if got != "tabel0042" {
		t.Fatalf("normallashtirish kutilganday emas: %q", got)
	}
}

func TestFreeModeRejectsTooLong(t *testing.T) {
	subjects := NewSubjectResolver(false)

	if _, err := subjects.PersonID(strings.Repeat("9", MaxSubjectIDLen+1)); err == nil {
		t.Fatal("juda uzun subject_id qabul qilindi")
	}
}

func TestPersonIDRejectsEmptySubject(t *testing.T) {
	for _, strict := range []bool{true, false} {
		subjects := NewSubjectResolver(strict)
		for _, raw := range []string{"", "   ", "---", "..."} {
			if _, err := subjects.PersonID(raw); err == nil {
				t.Errorf("strict=%v: %q qabul qilindi", strict, raw)
			}
		}
	}
}

func TestIsPNFL(t *testing.T) {
	if !IsPNFL(" 1234-5678-901234 ") {
		t.Fatal("normallashtirilgandan keyin PNFL tanilmadi")
	}
	if IsPNFL("AB1234567") {
		t.Fatal("passport PNFL deb topildi")
	}
}
