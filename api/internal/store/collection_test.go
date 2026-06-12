package store

import (
	"path/filepath"
	"testing"
)

type rec struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

func TestCollectionRoundTripAndPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "coll.json")

	c, err := NewCollection[rec](path)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.List()) != 0 {
		t.Fatal("new collection should be empty")
	}
	if err := c.Put("a", rec{ID: "a", Name: "alpha"}); err != nil {
		t.Fatal(err)
	}
	if err := c.Put("b", rec{ID: "b", Name: "beta"}); err != nil {
		t.Fatal(err)
	}
	if v, ok := c.Get("a"); !ok || v.Name != "alpha" {
		t.Fatalf("get a = %+v ok=%v", v, ok)
	}

	// Reopen: data persisted.
	c2, err := NewCollection[rec](path)
	if err != nil {
		t.Fatal(err)
	}
	if len(c2.List()) != 2 {
		t.Fatalf("expected 2 persisted records, got %d", len(c2.List()))
	}

	if err := c2.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if _, ok := c2.Get("a"); ok {
		t.Fatal("a should be deleted")
	}
	if err := c2.Delete("missing"); err != nil {
		t.Fatalf("deleting missing id should be a no-op, got %v", err)
	}
}

func TestCollectionUpdate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "coll.json")
	c, _ := NewCollection[rec](path)
	_ = c.Put("a", rec{ID: "a", Name: "x"})
	_ = c.Put("b", rec{ID: "b", Name: "y"})

	err := c.Update(func(_ string, v rec) (rec, bool) {
		if v.Name == "x" {
			v.Name = "X"
			return v, true
		}
		return v, false
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := c.Get("a"); v.Name != "X" {
		t.Fatalf("update not applied: %+v", v)
	}
	if v, _ := c.Get("b"); v.Name != "y" {
		t.Fatalf("unrelated record changed: %+v", v)
	}
}
