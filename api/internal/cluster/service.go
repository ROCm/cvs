package cluster

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/ROCm/cvs/api/internal/store"
)

// Inventory is the subset of the shared inventory the cluster service needs to
// generate a cluster_json: the SSH user, the resolved key file, and the full
// node set (used to validate the requested subset).
type Inventory struct {
	Username string
	KeyFile  string
	Nodes    []string
}

// InventoryProvider supplies the current inventory (decoupling this package
// from the inventory implementation).
type InventoryProvider interface {
	Current() (Inventory, bool, error)
}

// Validation/precondition errors surfaced to handlers.
var (
	ErrNoInventory = errors.New("no inventory saved")
	ErrNoNodes     = errors.New("at least one node is required")
	ErrNotSubset   = errors.New("selected nodes are not part of the inventory")
	ErrNoName      = errors.New("cluster name is required")
)

// Service creates, lists, and deletes saved clusters.
type Service struct {
	store Store
	gen   Generator
	inv   InventoryProvider
	dir   string // directory for generated cluster_json files
}

// NewService wires the catalog store, the cluster_json generator, the inventory
// provider, and the directory generated files are written to.
func NewService(s Store, gen Generator, inv InventoryProvider, dir string) *Service {
	return &Service{store: s, gen: gen, inv: inv, dir: dir}
}

// List returns saved clusters sorted by creation time (newest first).
func (s *Service) List() []Cluster {
	cs := s.store.List()
	sort.Slice(cs, func(i, j int) bool { return cs[i].CreatedAt.After(cs[j].CreatedAt) })
	return cs
}

// Get returns a saved cluster by ID.
func (s *Service) Get(id string) (Cluster, bool) { return s.store.Get(id) }

// ErrNoFile indicates the generated cluster_json file is missing.
var ErrNoFile = errors.New("cluster file not found")

// Content returns the raw generated cluster_json file for a saved cluster.
func (s *Service) Content(id string) ([]byte, bool, error) {
	c, ok := s.store.Get(id)
	if !ok {
		return nil, false, nil
	}
	if c.FilePath == "" {
		return nil, true, ErrNoFile
	}
	b, err := os.ReadFile(c.FilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, true, ErrNoFile
		}
		return nil, true, err
	}
	return b, true, nil
}

// CreateParams is a request to generate and save a new cluster.
type CreateParams struct {
	Name     string
	Nodes    []string
	HeadNode string
}

// Create validates the node subset, generates the cluster_json via the CLI, and
// persists the record.
func (s *Service) Create(ctx context.Context, p CreateParams) (Cluster, error) {
	if p.Name == "" {
		return Cluster{}, ErrNoName
	}
	if len(p.Nodes) == 0 {
		return Cluster{}, ErrNoNodes
	}
	inv, ok, err := s.inv.Current()
	if err != nil {
		return Cluster{}, err
	}
	if !ok {
		return Cluster{}, ErrNoInventory
	}
	if !subset(p.Nodes, inv.Nodes) {
		return Cluster{}, ErrNotSubset
	}

	id := store.NewID()
	out := filepath.Join(s.dir, id+".json")
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return Cluster{}, fmt.Errorf("create clusters dir: %w", err)
	}

	if err := s.gen.Generate(ctx, GenerateParams{
		Hosts:    p.Nodes,
		Username: inv.Username,
		KeyFile:  inv.KeyFile,
		Output:   out,
		HeadNode: p.HeadNode,
	}); err != nil {
		return Cluster{}, err
	}

	now := time.Now().UTC()
	c := Cluster{
		ID:        id,
		Name:      p.Name,
		Nodes:     p.Nodes,
		HeadNode:  p.HeadNode,
		FilePath:  out,
		Source:    "generated",
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := s.store.Put(c); err != nil {
		_ = os.Remove(out) // best-effort cleanup so we don't orphan a file without a record
		return Cluster{}, err
	}
	return c, nil
}

// UpdateParams is a request to edit a saved cluster. Nodes/HeadNode changes
// trigger regeneration of the cluster_json file; a name-only change does not.
type UpdateParams struct {
	Name     string
	Nodes    []string
	HeadNode string
}

// Update edits a saved cluster's name and/or node set. When the node set or head
// node changes, the cluster_json file is regenerated in place (re-validating the
// subset against the current inventory). Returns (_, false, nil) if not found.
func (s *Service) Update(ctx context.Context, id string, p UpdateParams) (Cluster, bool, error) {
	existing, ok := s.store.Get(id)
	if !ok {
		return Cluster{}, false, nil
	}
	if p.Name == "" {
		return Cluster{}, true, ErrNoName
	}
	if len(p.Nodes) == 0 {
		return Cluster{}, true, ErrNoNodes
	}

	updated := existing
	updated.Name = p.Name

	if !sameNodes(existing.Nodes, p.Nodes) || existing.HeadNode != p.HeadNode {
		inv, hasInv, err := s.inv.Current()
		if err != nil {
			return Cluster{}, true, err
		}
		if !hasInv {
			return Cluster{}, true, ErrNoInventory
		}
		if !subset(p.Nodes, inv.Nodes) {
			return Cluster{}, true, ErrNotSubset
		}
		if existing.FilePath == "" {
			return Cluster{}, true, ErrNoFile
		}
		if err := s.gen.Generate(ctx, GenerateParams{
			Hosts:    p.Nodes,
			Username: inv.Username,
			KeyFile:  inv.KeyFile,
			Output:   existing.FilePath,
			HeadNode: p.HeadNode,
		}); err != nil {
			return Cluster{}, true, err
		}
		updated.Nodes = p.Nodes
		updated.HeadNode = p.HeadNode
	}

	updated.UpdatedAt = time.Now().UTC()
	if err := s.store.Put(updated); err != nil {
		return Cluster{}, true, err
	}
	return updated, true, nil
}

// Delete removes the record and its generated cluster_json file.
func (s *Service) Delete(id string) (bool, error) {
	c, ok := s.store.Get(id)
	if !ok {
		return false, nil
	}
	if err := s.store.Delete(id); err != nil {
		return false, err
	}
	if c.FilePath != "" {
		_ = os.Remove(c.FilePath) // best-effort; the record is gone regardless
	}
	return true, nil
}

// sameNodes reports whether two node lists contain the same set (order-insensitive).
func sameNodes(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	set := make(map[string]struct{}, len(a))
	for _, x := range a {
		set[x] = struct{}{}
	}
	for _, y := range b {
		if _, ok := set[y]; !ok {
			return false
		}
	}
	return true
}

// subset reports whether every element of want is present in have.
func subset(want, have []string) bool {
	set := make(map[string]struct{}, len(have))
	for _, h := range have {
		set[h] = struct{}{}
	}
	for _, w := range want {
		if _, ok := set[w]; !ok {
			return false
		}
	}
	return true
}
