package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeCSV(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mapping.csv")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadMappingNormalizesAndSkips(t *testing.T) {
	// 3-qator allaqachon PNFL ostida — ko'chirish shart emas.
	// 4-qator to'liq emas.
	path := writeCSV(t, `abc123,1234-5678-901234
def456, 22222222222222 
12345678901233,12345678901233
onlyonefield
`)

	pairs, err := readMapping(path)
	if err != nil {
		t.Fatalf("kutilmagan xato: %v", err)
	}
	if len(pairs) != 2 {
		t.Fatalf("2 ta juftlik kutilgan, olindi: %d (%+v)", len(pairs), pairs)
	}
	if pairs[0].pnfl != "12345678901234" {
		t.Fatalf("defis olib tashlanmadi: %q", pairs[0].pnfl)
	}
	if pairs[1].pnfl != "22222222222222" {
		t.Fatalf("bo'shliq olib tashlanmadi: %q", pairs[1].pnfl)
	}
}

// Noto'g'ri PNFL jimgina o'tib ketmasligi kerak: bunday yozuv bazaga
// tushsa uni keyin topib bo'lmaydi.
func TestReadMappingRejectsNonPNFL(t *testing.T) {
	path := writeCSV(t, "abc123,1234567890123\n")

	if _, err := readMapping(path); err == nil {
		t.Fatal("13 xonali raqam qabul qilindi")
	}
}

// Bir xil PNFL ikki xil eski ID ga bog'langan bo'lsa — bu ma'lumot muammosi.
// Jimgina bittasini tanlash boshqa odamning yuzini o'chirib yuborardi.
func TestReadMappingRejectsDuplicatePNFL(t *testing.T) {
	path := writeCSV(t, "abc123,12345678901234\ndef456,1234-5678-901234\n")

	if _, err := readMapping(path); err == nil {
		t.Fatal("takrorlangan PNFL qabul qilindi")
	}
}

func TestReadMappingRejectsEmptyResult(t *testing.T) {
	path := writeCSV(t, "\n\n")

	if _, err := readMapping(path); err == nil {
		t.Fatal("bo'sh fayl qabul qilindi")
	}
}
