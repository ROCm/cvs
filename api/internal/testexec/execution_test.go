package testexec

import (
	"path/filepath"
	"testing"
	"time"
)

func TestExecutionStoreReconcilesInterrupted(t *testing.T) {
	path := filepath.Join(t.TempDir(), "exec.json")
	s, err := NewFileExecutionStore(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	_ = s.Save(Execution{ID: "run1", Status: StatusRunning, CreatedAt: now})
	_ = s.Save(Execution{ID: "run2", Status: StatusQueued, CreatedAt: now})
	_ = s.Save(Execution{ID: "run3", Status: StatusPassed, CreatedAt: now})

	// Reopen -> non-terminal records become interrupted, terminal untouched.
	s2, err := NewFileExecutionStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if e, _ := s2.Get("run1"); e.Status != StatusInterrupted {
		t.Fatalf("run1 = %s, want interrupted", e.Status)
	}
	if e, _ := s2.Get("run2"); e.Status != StatusInterrupted {
		t.Fatalf("run2 = %s, want interrupted", e.Status)
	}
	if e, _ := s2.Get("run3"); e.Status != StatusPassed {
		t.Fatalf("run3 = %s, want passed (unchanged)", e.Status)
	}
}

func TestExecutionListNewestFirst(t *testing.T) {
	s, _ := NewFileExecutionStore(filepath.Join(t.TempDir(), "exec.json"))
	t0 := time.Now().UTC()
	_ = s.Save(Execution{ID: "old", Status: StatusPassed, CreatedAt: t0.Add(-time.Hour)})
	_ = s.Save(Execution{ID: "new", Status: StatusPassed, CreatedAt: t0})
	list := s.List()
	if len(list) != 2 || list[0].ID != "new" {
		t.Fatalf("expected newest-first, got %+v", list)
	}
}
