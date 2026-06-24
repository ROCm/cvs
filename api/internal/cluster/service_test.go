package cluster

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// fakeGen records params and writes a stub cluster_json instead of shelling out.
type fakeGen struct {
	last GenerateParams
	err  error
}

func (g *fakeGen) Generate(_ context.Context, p GenerateParams) error {
	g.last = p
	if g.err != nil {
		return g.err
	}
	return os.WriteFile(p.Output, []byte(`{"stub":true}`), 0o644)
}

type fakeInv struct {
	inv Inventory
	ok  bool
}

func (f fakeInv) Current() (Inventory, bool, error) { return f.inv, f.ok, nil }

func newSvc(t *testing.T, gen Generator, inv InventoryProvider) (*Service, string) {
	t.Helper()
	dir := t.TempDir()
	st, err := NewFileStore(filepath.Join(dir, "clusters.json"))
	if err != nil {
		t.Fatal(err)
	}
	return NewService(st, gen, inv, filepath.Join(dir, "clusters")), dir
}

func TestCreateGeneratesAndPersists(t *testing.T) {
	gen := &fakeGen{}
	inv := fakeInv{ok: true, inv: Inventory{
		Username: "amd", KeyFile: "/keys/id_rsa",
		Nodes: []string{"n1", "n2", "n3"},
	}}
	svc, _ := newSvc(t, gen, inv)

	c, err := svc.Create(context.Background(), CreateParams{
		Name: "two-node", Nodes: []string{"n1", "n2"}, HeadNode: "n1",
	})
	if err != nil {
		t.Fatal(err)
	}
	// Generator received inventory creds + selected subset only.
	if gen.last.Username != "amd" || gen.last.KeyFile != "/keys/id_rsa" {
		t.Fatalf("generator creds wrong: %+v", gen.last)
	}
	if len(gen.last.Hosts) != 2 || gen.last.HeadNode != "n1" {
		t.Fatalf("generator hosts wrong: %+v", gen.last)
	}
	if _, err := os.Stat(c.FilePath); err != nil {
		t.Fatalf("cluster file not written: %v", err)
	}

	// Persisted + resolvable.
	if fp, ok := svc.store.(*FileStore).FilePath(c.ID); !ok || fp != c.FilePath {
		t.Fatalf("FilePath resolve failed: %q ok=%v", fp, ok)
	}
	if got := svc.List(); len(got) != 1 {
		t.Fatalf("expected 1 saved cluster, got %d", len(got))
	}
}

func TestCreateRejectsNonSubset(t *testing.T) {
	svc, _ := newSvc(t, &fakeGen{}, fakeInv{ok: true, inv: Inventory{Username: "amd", Nodes: []string{"n1"}}})
	_, err := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1", "n9"}})
	if err != ErrNotSubset {
		t.Fatalf("want ErrNotSubset, got %v", err)
	}
}

func TestCreateRequiresInventory(t *testing.T) {
	svc, _ := newSvc(t, &fakeGen{}, fakeInv{ok: false})
	_, err := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1"}})
	if err != ErrNoInventory {
		t.Fatalf("want ErrNoInventory, got %v", err)
	}
}

func TestContentReturnsGeneratedFile(t *testing.T) {
	svc, _ := newSvc(t, &fakeGen{}, fakeInv{ok: true, inv: Inventory{Username: "amd", Nodes: []string{"n1"}}})
	c, err := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1"}})
	if err != nil {
		t.Fatal(err)
	}

	b, ok, err := svc.Content(c.ID)
	if err != nil || !ok {
		t.Fatalf("content failed ok=%v err=%v", ok, err)
	}
	if string(b) != `{"stub":true}` {
		t.Fatalf("content = %q", string(b))
	}

	if _, ok, _ := svc.Content("missing"); ok {
		t.Fatal("missing cluster should report not found")
	}

	// File removed out-of-band -> ErrNoFile but cluster still known.
	_ = os.Remove(c.FilePath)
	if _, ok, err := svc.Content(c.ID); !ok || err != ErrNoFile {
		t.Fatalf("want ok=true ErrNoFile, got ok=%v err=%v", ok, err)
	}
}

func TestUpdateRenameDoesNotRegenerate(t *testing.T) {
	gen := &fakeGen{}
	svc, _ := newSvc(t, gen, fakeInv{ok: true, inv: Inventory{Username: "amd", Nodes: []string{"n1", "n2"}}})
	c, err := svc.Create(context.Background(), CreateParams{Name: "old", Nodes: []string{"n1"}})
	if err != nil {
		t.Fatal(err)
	}
	gen.last = GenerateParams{} // reset to detect a regen

	up, ok, err := svc.Update(context.Background(), c.ID, UpdateParams{Name: "new", Nodes: []string{"n1"}})
	if err != nil || !ok {
		t.Fatalf("update failed ok=%v err=%v", ok, err)
	}
	if up.Name != "new" {
		t.Fatalf("name = %q, want new", up.Name)
	}
	if len(gen.last.Hosts) != 0 {
		t.Fatalf("rename should not regenerate, got %+v", gen.last)
	}
}

func TestUpdateNodesRegenerates(t *testing.T) {
	gen := &fakeGen{}
	svc, _ := newSvc(t, gen, fakeInv{ok: true, inv: Inventory{Username: "amd", KeyFile: "/k", Nodes: []string{"n1", "n2", "n3"}}})
	c, err := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1"}})
	if err != nil {
		t.Fatal(err)
	}
	gen.last = GenerateParams{}

	up, ok, err := svc.Update(context.Background(), c.ID, UpdateParams{Name: "x", Nodes: []string{"n1", "n2"}})
	if err != nil || !ok {
		t.Fatalf("update failed ok=%v err=%v", ok, err)
	}
	if len(up.Nodes) != 2 {
		t.Fatalf("nodes = %v, want 2", up.Nodes)
	}
	// Regenerated in place at the same file path.
	if gen.last.Output != c.FilePath || len(gen.last.Hosts) != 2 {
		t.Fatalf("expected regen to %s with 2 hosts, got %+v", c.FilePath, gen.last)
	}
}

func TestUpdateRejectsNonSubsetAndMissing(t *testing.T) {
	svc, _ := newSvc(t, &fakeGen{}, fakeInv{ok: true, inv: Inventory{Username: "amd", Nodes: []string{"n1"}}})
	c, _ := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1"}})

	if _, _, err := svc.Update(context.Background(), c.ID, UpdateParams{Name: "x", Nodes: []string{"n1", "n9"}}); err != ErrNotSubset {
		t.Fatalf("want ErrNotSubset, got %v", err)
	}
	if _, ok, err := svc.Update(context.Background(), "missing", UpdateParams{Name: "x", Nodes: []string{"n1"}}); ok || err != nil {
		t.Fatalf("missing should report ok=false err=nil, got ok=%v err=%v", ok, err)
	}
}

func TestDeleteRemovesFileAndRecord(t *testing.T) {
	svc, _ := newSvc(t, &fakeGen{}, fakeInv{ok: true, inv: Inventory{Username: "amd", Nodes: []string{"n1"}}})
	c, err := svc.Create(context.Background(), CreateParams{Name: "x", Nodes: []string{"n1"}})
	if err != nil {
		t.Fatal(err)
	}
	ok, err := svc.Delete(c.ID)
	if err != nil || !ok {
		t.Fatalf("delete failed ok=%v err=%v", ok, err)
	}
	if _, err := os.Stat(c.FilePath); !os.IsNotExist(err) {
		t.Fatalf("cluster file should be removed, stat err=%v", err)
	}
	if _, ok := svc.Get(c.ID); ok {
		t.Fatal("record should be gone")
	}
	if ok, _ := svc.Delete("missing"); ok {
		t.Fatal("deleting missing should report not found")
	}
}
