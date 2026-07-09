package testexec

import (
	"sort"
	"time"

	"github.com/ROCm/cvs/api/internal/store"
)

// timeNow is overridable in tests.
var timeNow = func() time.Time { return time.Now().UTC() }

// Status is an execution's lifecycle state.
type Status string

const (
	StatusQueued      Status = "queued"
	StatusRunning     Status = "running"
	StatusPassed      Status = "passed"
	StatusFailed      Status = "failed"
	StatusError       Status = "error"       // could not run (spawn/setup failure)
	StatusInterrupted Status = "interrupted" // process did not survive a daemon restart
)

// Terminal reports whether the status is final.
func (s Status) Terminal() bool {
	switch s {
	case StatusPassed, StatusFailed, StatusError, StatusInterrupted:
		return true
	default:
		return false
	}
}

// Execution is one test run record. It is the one-shot, non-re-derivable artifact
// the ExecutionStore must persist (vs. live monitor state, which is re-derivable).
type Execution struct {
	ID          string     `json:"id"`
	Suite       string     `json:"suite"`
	ClusterID   string     `json:"cluster_id"`
	ClusterFile string     `json:"cluster_file"`
	ConfigPath  string     `json:"config_path,omitempty"`
	Status      Status     `json:"status"`
	ExitCode    *int       `json:"exit_code,omitempty"`
	Error       string     `json:"error,omitempty"`
	LogPath     string     `json:"log_path,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
	StartedAt   *time.Time `json:"started_at,omitempty"`
	FinishedAt  *time.Time `json:"finished_at,omitempty"`
}

// ExecutionStore persists execution records.
type ExecutionStore interface {
	Save(Execution) error
	Get(id string) (Execution, bool)
	List() []Execution
}

// FileExecutionStore is a JSON-collection-backed ExecutionStore. On open it
// reconciles any non-terminal record to interrupted, because child CLI
// processes do not survive a daemon restart.
type FileExecutionStore struct {
	c *store.Collection[Execution]
}

// NewFileExecutionStore opens (or initializes) the execution store at path and
// reconciles interrupted runs.
func NewFileExecutionStore(path string) (*FileExecutionStore, error) {
	c, err := store.NewCollection[Execution](path)
	if err != nil {
		return nil, err
	}
	s := &FileExecutionStore{c: c}
	if err := s.reconcile(); err != nil {
		return nil, err
	}
	return s, nil
}

// reconcile marks any queued/running record as interrupted at startup.
func (s *FileExecutionStore) reconcile() error {
	now := time.Now().UTC()
	return s.c.Update(func(_ string, e Execution) (Execution, bool) {
		if e.Status.Terminal() {
			return e, false
		}
		e.Status = StatusInterrupted
		e.Error = "daemon restarted while execution was " + string(e.Status)
		e.FinishedAt = &now
		return e, true
	})
}

func (s *FileExecutionStore) Save(e Execution) error { return s.c.Put(e.ID, e) }

func (s *FileExecutionStore) Get(id string) (Execution, bool) { return s.c.Get(id) }

// List returns executions sorted newest-first.
func (s *FileExecutionStore) List() []Execution {
	es := s.c.List()
	sort.Slice(es, func(i, j int) bool { return es[i].CreatedAt.After(es[j].CreatedAt) })
	return es
}
