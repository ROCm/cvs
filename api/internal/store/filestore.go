// Package store provides small, dependency-free persistence primitives shared
// across tiles. The v1 FileStore writes JSON documents atomically (temp file +
// fsync + rename) and is safe for concurrent use. It targets low-write config
// documents (inventory) and small collections (clusters, executions) at
// single-instance scale; the interfaces are drop-in for a future DB.
package store

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

// ErrNotFound indicates the requested document does not exist yet.
var ErrNotFound = errors.New("store: not found")

// JSONFile is a single-document JSON store with atomic writes.
type JSONFile struct {
	path string
	mu   sync.RWMutex
}

// NewJSONFile returns a store backed by the file at path. The file and its
// parent directory are created lazily on the first Save.
func NewJSONFile(path string) *JSONFile {
	return &JSONFile{path: path}
}

// Path returns the backing file path.
func (f *JSONFile) Path() string { return f.path }

// Exists reports whether the backing file is present.
func (f *JSONFile) Exists() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	_, err := os.Stat(f.path)
	return err == nil
}

// Load decodes the document into v. Returns ErrNotFound if the file is absent.
func (f *JSONFile) Load(v any) error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	data, err := os.ReadFile(f.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ErrNotFound
		}
		return err
	}
	return json.Unmarshal(data, v)
}

// Save atomically encodes v to disk via a temp file, fsync, and rename so a
// crash mid-write can never leave a partially written document.
func (f *JSONFile) Save(v any) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(f.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename succeeds

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, f.path); err != nil {
		return err
	}

	// Best-effort durability for the rename itself.
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// Remove deletes the backing file if present. Missing file is not an error.
func (f *JSONFile) Remove() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	err := os.Remove(f.path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}
