package inventory

import (
	"path/filepath"
	"testing"
)

func TestFileStore_GetEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inventory.json")
	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	if _, ok, err := s.Get(); err != nil || ok {
		t.Fatalf("expected empty inventory, got ok=%v err=%v", ok, err)
	}
}

func TestFileStore_SaveAndPersistAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inventory.json")
	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}

	saved, err := s.Save(Inventory{
		Nodes:      []string{"10.0.0.1", "10.0.0.2"},
		Username:   "amd",
		AuthMethod: AuthKey,
		KeyName:    "id_rsa",
	})
	if err != nil {
		t.Fatalf("Save: %v", err)
	}
	if saved.UpdatedAt.IsZero() {
		t.Fatal("expected UpdatedAt to be stamped")
	}

	// Simulate a restart: a fresh store reading the same file.
	s2, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	got, ok, err := s2.Get()
	if err != nil || !ok {
		t.Fatalf("expected inventory after restart, ok=%v err=%v", ok, err)
	}
	if len(got.Nodes) != 2 || got.Username != "amd" || got.KeyName != "id_rsa" {
		t.Fatalf("unexpected inventory after restart: %+v", got)
	}
}
