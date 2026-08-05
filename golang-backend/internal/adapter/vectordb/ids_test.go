package vectordb

import (
	"regexp"
	"testing"
)

// PointIDFor RFC 4122 UUIDv5 ni to'g'ri hisoblashi kerak (namespace = DNS).
// Kutilgan qiymatlar Python'ning `uuid.uuid5(uuid.NAMESPACE_DNS, name)` natijasi.
func TestPointIDFor(t *testing.T) {
	cases := map[string]string{
		"test":     "4be0643f-1d98-573b-97cd-ca98a65347dd",
		"person-1": "b42d06b8-5fa7-5bd9-bbdb-7bebc3035da7",
		"":         "4ebd0208-8328-5d69-8c44-ec50939c0967",
	}

	for input, want := range cases {
		if got := PointIDFor(input); got != want {
			t.Errorf("PointIDFor(%q) = %q, kutilgan %q", input, got, want)
		}
	}
}

// Bir xil person_id har doim bir xil ID beradi — bu upsert semantikasi uchun shart.
func TestPointIDForIsDeterministic(t *testing.T) {
	first, second := PointIDFor("ali-001"), PointIDFor("ali-001")
	if first != second {
		t.Fatalf("deterministik emas: %q != %q", first, second)
	}
	if PointIDFor("ali-001") == PointIDFor("ali-002") {
		t.Fatal("turli person_id lar bir xil point ID berdi")
	}
}

var uuidV4Pattern = regexp.MustCompile(
	`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func TestUUIDGeneratorProducesValidV4(t *testing.T) {
	var generator UUIDGenerator
	seen := make(map[string]bool, 100)

	for i := 0; i < 100; i++ {
		id, err := generator.NewID()
		if err != nil {
			t.Fatalf("NewID: %v", err)
		}
		if !uuidV4Pattern.MatchString(id) {
			t.Fatalf("UUIDv4 formatiga mos emas: %q", id)
		}
		if seen[id] {
			t.Fatalf("takrorlangan ID: %q", id)
		}
		seen[id] = true
	}
}
