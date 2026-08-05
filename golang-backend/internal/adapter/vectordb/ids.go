package vectordb

import (
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"fmt"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// namespace — RFC 4122 DNS namespace'i (6ba7b810-9dad-11d1-80b4-00c04fd430c8).
// DIQQAT: bu qiymat o'zgartirilsa barcha point ID'lar o'zgaradi va mavjud
// yozuvlar "yo'qolgandek" ko'rinadi.
var namespace = [16]byte{
	0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1,
	0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
}

// PointIDFor — `person_id` dan deterministik UUIDv5 yasaydi.
//
// Qdrant point ID sifatida faqat uint64 yoki UUID qabul qiladi. Deterministik
// o'girish tufayli bitta odamni qayta ro'yxatdan o'tkazish dublikat yaratmaydi,
// balki mavjud yozuvni almashtiradi. `person_id` ixtiyoriy satr bo'lishi mumkin
// ("emp-001" ham, generatsiya qilingan UUID ham).
func PointIDFor(personID string) string {
	h := sha1.New()
	h.Write(namespace[:])
	h.Write([]byte(personID))
	sum := h.Sum(nil)

	var uuid [16]byte
	copy(uuid[:], sum[:16])
	uuid[6] = (uuid[6] & 0x0f) | 0x50 // version 5
	uuid[8] = (uuid[8] & 0x3f) | 0x80 // RFC 4122 variant
	return formatUUID(uuid)
}

// PointIDForSample — bir shaxsning QO'SHIMCHA namunasi uchun nuqta idsi.
//
// # Nima uchun bir shaxsga bir nechta vektor
//
// 10-20 yillik farqda bitta vektor yetmaydi: 10 yoshdagi va 25 yoshdagi
// vektorlar bir-biridan uzoq va to'g'ridan-to'g'ri mos kelmaydi. Lekin
// galereyada 10, 14, 18, 22 yoshdagi namunalar bo'lsa, ular orasida
// "ko'prik" hosil bo'ladi: har bir qo'shni juftlik yaqin, va bugungi kadr
// eng so'nggisiga mos keladi.
//
// Xuddi shu narsa rakurs, yoritish, soqol va ko'zoynak uchun ham amal
// qiladi. Bu — hech qanday model almashtirish bermaydigan yaxshilanish.
//
// `sample` bo'sh bo'lsa PointIDFor bilan bir xil natija qaytadi, ya'ni
// mavjud yozuvlar joyida qoladi va migratsiya kerak emas.
func PointIDForSample(personID, sample string) string {
	if sample == "" {
		return PointIDFor(personID)
	}
	return PointIDFor(personID + "\x00" + sample)
}

// UUIDGenerator — `domain.IDGenerator` portining implementatsiyasi.
type UUIDGenerator struct{}

var _ domain.IDGenerator = UUIDGenerator{}

// NewID — tasodifiy UUIDv4.
func (UUIDGenerator) NewID() (string, error) {
	var uuid [16]byte
	if _, err := rand.Read(uuid[:]); err != nil {
		return "", fmt.Errorf("tasodifiy ID hosil qilib bo'lmadi: %w", err)
	}
	uuid[6] = (uuid[6] & 0x0f) | 0x40 // version 4
	uuid[8] = (uuid[8] & 0x3f) | 0x80 // RFC 4122 variant
	return formatUUID(uuid), nil
}

// formatUUID — 16 baytni kanonik 8-4-4-4-12 ko'rinishga o'giradi.
func formatUUID(uuid [16]byte) string {
	buf := make([]byte, 36)
	hex.Encode(buf[0:8], uuid[0:4])
	buf[8] = '-'
	hex.Encode(buf[9:13], uuid[4:6])
	buf[13] = '-'
	hex.Encode(buf[14:18], uuid[6:8])
	buf[18] = '-'
	hex.Encode(buf[19:23], uuid[8:10])
	buf[23] = '-'
	hex.Encode(buf[24:36], uuid[10:16])
	return string(buf)
}
