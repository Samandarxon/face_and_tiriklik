package usecase

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/face_detecting_microservice/golang-backend/internal/domain"
)

// fakeKeyStore — `domain.APIKeyStore` ning xotiradagi soxta implementatsiyasi.
type fakeKeyStore struct {
	mu      sync.Mutex
	clients map[string]*domain.APIClient
	keys    map[string]*domain.APIKey // id -> key
	hashes  map[string]string         // id -> hash
	seq     int

	// findCalls — keshning haqiqatan ishlayotganini tekshirish uchun.
	findCalls int
	touched   []string
	failWith  error
}

var _ domain.APIKeyStore = (*fakeKeyStore)(nil)

func newFakeKeyStore() *fakeKeyStore {
	return &fakeKeyStore{
		clients: make(map[string]*domain.APIClient),
		keys:    make(map[string]*domain.APIKey),
		hashes:  make(map[string]string),
	}
}

func (s *fakeKeyStore) nextID(prefix string) string {
	s.seq++
	return fmt.Sprintf("%s-%d", prefix, s.seq)
}

func (s *fakeKeyStore) CreateClient(_ context.Context, client *domain.APIClient) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, existing := range s.clients {
		if existing.Name == client.Name {
			return domain.NewError(domain.CodeConflict, "mavjud")
		}
	}
	if client.ID == "" {
		client.ID = s.nextID("client")
	}
	copied := *client
	s.clients[client.ID] = &copied
	return nil
}

func (s *fakeKeyStore) UpdateClient(_ context.Context, client *domain.APIClient) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.clients[client.ID]; !ok {
		return domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	copied := *client
	s.clients[client.ID] = &copied
	return nil
}

func (s *fakeKeyStore) GetClient(_ context.Context, id string) (*domain.APIClient, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	client, ok := s.clients[id]
	if !ok {
		return nil, domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	copied := *client
	return &copied, nil
}

func (s *fakeKeyStore) ListClients(context.Context) ([]domain.APIClient, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]domain.APIClient, 0, len(s.clients))
	for _, c := range s.clients {
		out = append(out, *c)
	}
	return out, nil
}

func (s *fakeKeyStore) DeleteClient(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.clients[id]; !ok {
		return domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	delete(s.clients, id)
	for keyID, key := range s.keys {
		if key.ClientID == id {
			delete(s.keys, keyID)
		}
	}
	return nil
}

func (s *fakeKeyStore) CreateKey(_ context.Context, key *domain.APIKey, hash string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if key.ID == "" {
		key.ID = s.nextID("key")
	}
	copied := *key
	s.keys[key.ID] = &copied
	s.hashes[key.ID] = hash
	return nil
}

func (s *fakeKeyStore) FindKeyByPrefix(_ context.Context, prefix string) (*domain.APIKey, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.findCalls++
	if s.failWith != nil {
		return nil, "", s.failWith
	}

	for id, key := range s.keys {
		if key.Prefix == prefix {
			copied := *key
			if client, ok := s.clients[key.ClientID]; ok {
				copied.ClientActive = client.IsActive
				copied.ClientName = client.Name
			}
			return &copied, s.hashes[id], nil
		}
	}
	return nil, "", domain.NewError(domain.CodeUnauthenticated, "kalit noto'g'ri")
}

func (s *fakeKeyStore) GetKey(_ context.Context, id string) (*domain.APIKey, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	key, ok := s.keys[id]
	if !ok {
		return nil, domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	copied := *key
	if client, ok := s.clients[key.ClientID]; ok {
		copied.ClientActive = client.IsActive
	}
	return &copied, nil
}

func (s *fakeKeyStore) ListKeys(_ context.Context, clientID string) ([]domain.APIKey, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]domain.APIKey, 0, len(s.keys))
	for _, key := range s.keys {
		if clientID == "" || key.ClientID == clientID {
			out = append(out, *key)
		}
	}
	return out, nil
}

func (s *fakeKeyStore) PurgeKey(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.keys[id]; !ok {
		return domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	delete(s.keys, id)
	return nil
}

func (s *fakeKeyStore) RevokeKey(_ context.Context, id string, at time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key, ok := s.keys[id]
	if !ok {
		return domain.NewError(domain.CodeNotFound, "topilmadi")
	}
	if key.RevokedAt == nil {
		key.RevokedAt = &at
	}
	return nil
}

func (s *fakeKeyStore) TouchKey(_ context.Context, id string, _ time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.touched = append(s.touched, id)
	return nil
}

func (s *fakeKeyStore) CountKeys(context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := 0
	for _, key := range s.keys {
		if key.RevokedAt == nil {
			n++
		}
	}
	return n, nil
}

func (s *fakeKeyStore) Healthy(context.Context) error { return nil }

func (s *fakeKeyStore) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.findCalls
}

// fakeCipher — kriptografiyasiz, bashorat qilinadigan kalit: "test_<prefix>_<sir>",
// hash esa "hash:" + sir.
type fakeCipher struct{ next int }

var _ domain.KeyCipher = (*fakeCipher)(nil)

func (c *fakeCipher) Generate() (domain.GeneratedKey, error) {
	c.next++
	prefix := fmt.Sprintf("p%d", c.next)
	secret := fmt.Sprintf("s%d", c.next)
	return domain.GeneratedKey{
		Prefix: prefix,
		Secret: fmt.Sprintf("test_%s_%s", prefix, secret),
		Hash:   "hash:" + secret,
	}, nil
}

func (c *fakeCipher) Parse(raw string) (string, string, error) {
	parts := strings.Split(raw, "_")
	if len(parts) != 3 || parts[0] != "test" {
		return "", "", fmt.Errorf("format xato")
	}
	return parts[1], parts[2], nil
}

func (c *fakeCipher) Matches(secret, hash string) bool { return "hash:"+secret == hash }
