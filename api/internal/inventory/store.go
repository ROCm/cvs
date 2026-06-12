package inventory

import (
	"errors"
	"sync"
	"time"

	"github.com/ROCm/cvs/api/internal/store"
)

// Store persists the single fleet inventory document.
type Store interface {
	// Get returns the inventory and whether one has been saved.
	Get() (Inventory, bool, error)
	// Save persists the inventory (stamping UpdatedAt) and returns it.
	Save(Inventory) (Inventory, error)
}

// FileStore is a JSON-file-backed Store with an in-memory cache.
type FileStore struct {
	file *store.JSONFile
	mu   sync.RWMutex
	cur  *Inventory
}

// NewFileStore opens (or initializes) the inventory document at path.
func NewFileStore(path string) (*FileStore, error) {
	fs := &FileStore{file: store.NewJSONFile(path)}
	var inv Inventory
	switch err := fs.file.Load(&inv); {
	case err == nil:
		fs.cur = &inv
	case errors.Is(err, store.ErrNotFound):
		// No inventory yet; tiles stay gated until the first Save.
	default:
		return nil, err
	}
	return fs, nil
}

// Get returns the current inventory and whether one exists.
func (s *FileStore) Get() (Inventory, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.cur == nil {
		return Inventory{}, false, nil
	}
	return *s.cur, true, nil
}

// Save persists inv atomically and updates the cache.
func (s *FileStore) Save(inv Inventory) (Inventory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	inv.UpdatedAt = time.Now().UTC()
	if err := s.file.Save(&inv); err != nil {
		return Inventory{}, err
	}
	cp := inv
	s.cur = &cp
	return inv, nil
}
