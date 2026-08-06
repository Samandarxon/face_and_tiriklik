package httpapi

import "testing"

// Eng muhim holat: `CORS_ALLOW_ORIGIN` ga aniq domen yozilganda ham
// server-server mijoz (Origin'siz) oqimga ulana olishi kerak. Aks holda
// chaqiruvchi backendning liveness proxysi jimgina 403 oladi va sabab
// hech qayerda ko'rinmaydi.
func TestOriginPolicyNonBrowserAlwaysAllowed(t *testing.T) {
	for _, spec := range []string{"*", "none", "https://10.181.1.57", ""} {
		if !newOriginPolicy(spec).allowsNonBrowser("") {
			t.Errorf("CORS_ALLOW_ORIGIN=%q: Origin'siz mijoz rad etildi", spec)
		}
	}
}

func TestOriginPolicyBrowser(t *testing.T) {
	tests := []struct {
		spec   string
		origin string
		want   bool
	}{
		{"*", "https://evil.example", true},
		{"", "https://evil.example", true},

		// `none` — production tanlovi: hech bir sahifa o'tmaydi.
		{"none", "https://10.181.1.57", false},
		{"none", "", false},

		{"https://10.181.1.57", "https://10.181.1.57", true},
		{"https://10.181.1.57", "https://10.181.1.58", false},
		{"https://10.181.1.57", "http://10.181.1.57", false},

		// Sozlamadagi oxirgi `/` va katta harf integratsiyani buzmasligi kerak.
		{"https://10.181.1.57/", "https://10.181.1.57", true},
		{"HTTPS://A.UZ", "https://a.uz", true},

		{"https://a.uz, https://b.uz", "https://b.uz", true},
		{"https://a.uz, https://b.uz", "https://c.uz", false},
	}

	for _, tt := range tests {
		if got := newOriginPolicy(tt.spec).allows(tt.origin); got != tt.want {
			t.Errorf("spec=%q origin=%q: %v, kutilgan %v",
				tt.spec, tt.origin, got, tt.want)
		}
	}
}

// Ro'yxat rejimida javob `*` emas, aynan so'ralgan origin bilan qaytishi
// kerak — `*` bir nechta domen bilan ishlamaydi.
func TestOriginPolicyHeader(t *testing.T) {
	p := newOriginPolicy("https://a.uz,https://b.uz")

	if got := p.header("https://b.uz"); got != "https://b.uz" {
		t.Errorf("ruxsat etilgan origin: %q", got)
	}
	if got := p.header("https://c.uz"); got != "" {
		t.Errorf("begona origin uchun sarlavha qaytdi: %q", got)
	}
	if got := newOriginPolicy("none").header("https://a.uz"); got != "" {
		t.Errorf("none rejimida sarlavha qaytdi: %q", got)
	}
	if got := newOriginPolicy("*").header("https://a.uz"); got != "*" {
		t.Errorf("* rejimi: %q", got)
	}
}
