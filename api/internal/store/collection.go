package store

import (
	"errors"
	"sync"
)

// Collection is a small, concurrency-safe, file-backed map of records keyed by
// string ID. The whole collection is persisted as one JSON document on every
// mutation (atomic temp+fsync+rename via JSONFile). It targets small
// collections (saved clusters, execution records) at single-instance scale; the
// interface is a drop-in for a future DB-backed store.
type Collection[T any] struct {
	file *JSONFile
	mu   sync.RWMutex
	// items is the in-memory cache, always consistent with the file.
	items map[string]T
}

// NewCollection opens (or initializes) a collection at path. A missing file is
// treated as an empty collection (created lazily on the first Put).
func NewCollection[T any](path string) (*Collection[T], error) {
	c := &Collection[T]{file: NewJSONFile(path), items: map[string]T{}}
	var loaded map[string]T
	switch err := c.file.Load(&loaded); {
	case err == nil:
		if loaded != nil {
			c.items = loaded
		}
	case errors.Is(err, ErrNotFound):
		// empty collection
	default:
		return nil, err
	}
	return c, nil
}

// Get returns the record for id and whether it exists.
func (c *Collection[T]) Get(id string) (T, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.items[id]
	return v, ok
}

// List returns a snapshot of all records (unordered).
func (c *Collection[T]) List() []T {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]T, 0, len(c.items))
	for _, v := range c.items {
		out = append(out, v)
	}
	return out
}

// Put inserts or replaces the record for id and persists the collection.
func (c *Collection[T]) Put(id string, v T) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[id] = v
	return c.file.Save(c.items)
}

// Delete removes the record for id and persists. Deleting a missing id is a
// no-op (no error, no write).
func (c *Collection[T]) Delete(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.items[id]; !ok {
		return nil
	}
	delete(c.items, id)
	return c.file.Save(c.items)
}

// Update applies fn to every record and persists once. It is used at startup to
// reconcile non-terminal records (e.g. mark interrupted executions).
func (c *Collection[T]) Update(fn func(id string, v T) (T, bool)) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	changed := false
	for id, v := range c.items {
		if nv, ok := fn(id, v); ok {
			c.items[id] = nv
			changed = true
		}
	}
	if !changed {
		return nil
	}
	return c.file.Save(c.items)
}
