package cluster

import "github.com/ROCm/cvs/api/internal/store"

// Store persists the saved-cluster catalog.
type Store interface {
	List() []Cluster
	Get(id string) (Cluster, bool)
	Put(c Cluster) error
	Delete(id string) error
}

// FileStore is a JSON-collection-backed Store.
type FileStore struct {
	c *store.Collection[Cluster]
}

// NewFileStore opens (or initializes) the cluster catalog at path.
func NewFileStore(path string) (*FileStore, error) {
	c, err := store.NewCollection[Cluster](path)
	if err != nil {
		return nil, err
	}
	return &FileStore{c: c}, nil
}

func (s *FileStore) List() []Cluster          { return s.c.List() }
func (s *FileStore) Get(id string) (Cluster, bool) { return s.c.Get(id) }
func (s *FileStore) Put(c Cluster) error      { return s.c.Put(c.ID, c) }
func (s *FileStore) Delete(id string) error   { return s.c.Delete(id) }

// FilePath resolves a cluster's generated cluster_json path by ID. It satisfies
// the resolver the Test Execution tile uses to turn a cluster_id into a
// --cluster_file path.
func (s *FileStore) FilePath(id string) (string, bool) {
	c, ok := s.c.Get(id)
	if !ok {
		return "", false
	}
	return c.FilePath, true
}
