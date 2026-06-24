package testexec

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
)

// waitTerminal polls the store until the execution reaches a terminal state or
// the deadline passes.
func waitTerminal(t *testing.T, s ExecutionStore, id string) Execution {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if e, ok := s.Get(id); ok && e.Status.Terminal() {
			return e
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("execution %s did not reach terminal state", id)
	return Execution{}
}

func TestExecutorRunsToPassed(t *testing.T) {
	dir := t.TempDir()
	s, _ := NewFileExecutionStore(filepath.Join(dir, "exec.json"))
	ex := NewExecutor(s, FakeRunner{}, nil, 2, 16, nil)
	defer ex.Shutdown()

	logPath := filepath.Join(dir, "run1", "output.log")
	_ = s.Save(Execution{ID: "run1", Suite: "rccl_perf", Status: StatusQueued, LogPath: logPath, CreatedAt: timeNow()})
	if err := ex.Submit(Job{ID: "run1", Suite: "rccl_perf", ClusterFile: "/c.json", LogPath: logPath}); err != nil {
		t.Fatal(err)
	}

	got := waitTerminal(t, s, "run1")
	if got.Status != StatusPassed {
		t.Fatalf("status = %s, want passed", got.Status)
	}
	if got.ExitCode == nil || *got.ExitCode != 0 {
		t.Fatalf("exit code = %v, want 0", got.ExitCode)
	}
	if got.StartedAt == nil || got.FinishedAt == nil {
		t.Fatal("timestamps not set")
	}
	data, err := os.ReadFile(logPath)
	if err != nil || !strings.Contains(string(data), "fake runner") {
		t.Fatalf("log not written: %v / %q", err, string(data))
	}
}

// exitRunner returns a fixed exit code, to exercise failed/error mapping.
type exitRunner struct {
	code int
	err  error
}

func (r exitRunner) Run(_ context.Context, _ Job, out io.Writer) (int, error) {
	_, _ = io.WriteString(out, "ran\n")
	return r.code, r.err
}

// captureEvents records executor lifecycle events for assertions.
type captureEvents struct {
	mu       sync.Mutex
	logs     []string
	statuses []Status
	complete *Execution
}

func (c *captureEvents) Log(_, line string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.logs = append(c.logs, line)
}
func (c *captureEvents) Status(_ string, ex Execution) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.statuses = append(c.statuses, ex.Status)
}
func (c *captureEvents) Complete(ex Execution) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e := ex
	c.complete = &e
}

func TestExecutorEmitsEvents(t *testing.T) {
	dir := t.TempDir()
	s, _ := NewFileExecutionStore(filepath.Join(dir, "exec.json"))
	ev := &captureEvents{}
	ex := NewExecutor(s, FakeRunner{}, ev, 1, 4, nil)
	defer ex.Shutdown()

	logPath := filepath.Join(dir, "run1", "output.log")
	_ = s.Save(Execution{ID: "run1", Suite: "rccl_perf", Status: StatusQueued, LogPath: logPath, CreatedAt: timeNow()})
	_ = ex.Submit(Job{ID: "run1", Suite: "rccl_perf", LogPath: logPath})
	waitTerminal(t, s, "run1")

	ev.mu.Lock()
	defer ev.mu.Unlock()
	if len(ev.logs) == 0 {
		t.Fatal("expected streamed log lines")
	}
	if ev.complete == nil || ev.complete.Status != StatusPassed {
		t.Fatalf("complete event = %#v", ev.complete)
	}
	if len(ev.statuses) == 0 || ev.statuses[0] != StatusRunning {
		t.Fatalf("status events = %v, want running first", ev.statuses)
	}
}

func TestExecutorMapsFailed(t *testing.T) {
	s, _ := NewFileExecutionStore(filepath.Join(t.TempDir(), "exec.json"))
	ex := NewExecutor(s, exitRunner{code: 1}, nil, 1, 4, nil)
	defer ex.Shutdown()
	_ = s.Save(Execution{ID: "r", Status: StatusQueued, CreatedAt: timeNow()})
	_ = ex.Submit(Job{ID: "r"})
	if got := waitTerminal(t, s, "r"); got.Status != StatusFailed {
		t.Fatalf("status = %s, want failed", got.Status)
	}
}

// fakeResolver implements ClusterResolver.
type fakeResolver map[string]string

func (f fakeResolver) FilePath(id string) (string, bool) { v, ok := f[id]; return v, ok }

func newExecServer(t *testing.T) (*httptest.Server, ExecutionStore, string) {
	t.Helper()
	dir := t.TempDir()
	s, _ := NewFileExecutionStore(filepath.Join(dir, "exec.json"))
	ex := NewExecutor(s, FakeRunner{}, nil, 2, 16, nil)
	t.Cleanup(ex.Shutdown)

	lister := fakeLister{suites: []Suite{{Name: "rccl_perf"}}}
	deps := &ExecutionDeps{
		Store:    s,
		Executor: ex,
		Clusters: fakeResolver{"cl1": filepath.Join(dir, "cl1.json")},
		Dir:      filepath.Join(dir, "executions"),
	}
	r := chi.NewRouter()
	NewHandler(lister, NewConfigService(""), nil, deps).Routes(r)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv, s, dir
}

func TestArtifactsServingAndHasReport(t *testing.T) {
	srv, s, _ := newExecServer(t)

	// Persist an execution and stage a report.html + a sidecar per-test log.
	dir := t.TempDir()
	logPath := filepath.Join(dir, "output.log")
	_ = os.WriteFile(logPath, []byte("out"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "report.html"), []byte("<html>report</html>"), 0o644)
	_ = os.MkdirAll(filepath.Join(dir, "rccl_perf_html"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "rccl_perf_html", "t1.log"), []byte("per-test"), 0o644)
	_ = s.Save(Execution{ID: "rpt", Suite: "rccl_perf", Status: StatusPassed, LogPath: logPath, CreatedAt: timeNow()})

	// list reports has_report=true.
	lr, _ := http.Get(srv.URL + "/executions")
	defer lr.Body.Close()
	var list struct {
		Executions []struct {
			ID        string `json:"id"`
			HasReport bool   `json:"has_report"`
		} `json:"executions"`
	}
	_ = json.NewDecoder(lr.Body).Decode(&list)
	found := false
	for _, e := range list.Executions {
		if e.ID == "rpt" {
			found = true
			if !e.HasReport {
				t.Fatal("has_report = false, want true")
			}
		}
	}
	if !found {
		t.Fatal("execution not in list")
	}

	// Artifacts: report.html and sidecar resolve.
	for path, want := range map[string]string{
		"/executions/rpt/artifacts/report.html":           "<html>report</html>",
		"/executions/rpt/artifacts/rccl_perf_html/t1.log": "per-test",
	} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || string(b) != want {
			t.Fatalf("GET %s = %d %q, want 200 %q", path, resp.StatusCode, string(b), want)
		}
	}

	// Path traversal is rejected.
	resp, _ := http.Get(srv.URL + "/executions/rpt/artifacts/../../exec.json")
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode == 200 {
		t.Fatal("traversal should not return 200")
	}
}

func TestExecuteHandlerLifecycle(t *testing.T) {
	srv, s, _ := newExecServer(t)

	body := `{"suite":"rccl_perf","cluster_id":"cl1","config":{"rccl":{"mpi_pml":"auto"}}}`
	resp, err := http.Post(srv.URL+"/tests/execute", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("status = %d, want 202", resp.StatusCode)
	}
	var created Execution
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	if created.ID == "" || created.ConfigPath == "" {
		t.Fatalf("unexpected execution: %+v", created)
	}

	got := waitTerminal(t, s, created.ID)
	if got.Status != StatusPassed {
		t.Fatalf("status = %s, want passed", got.Status)
	}

	// Logs endpoint returns captured output.
	lr, err := http.Get(srv.URL + "/executions/" + created.ID + "/logs")
	if err != nil {
		t.Fatal(err)
	}
	defer lr.Body.Close()
	var logs struct {
		Logs   string `json:"logs"`
		Status string `json:"status"`
	}
	_ = json.NewDecoder(lr.Body).Decode(&logs)
	if !strings.Contains(logs.Logs, "fake runner") {
		t.Fatalf("logs missing transcript: %q", logs.Logs)
	}

	// Config staged to disk.
	if _, err := os.Stat(created.ConfigPath); err != nil {
		t.Fatalf("config not staged: %v", err)
	}
}

func TestExecuteHandlerValidation(t *testing.T) {
	srv, _, _ := newExecServer(t)

	cases := []struct {
		body string
		want int
	}{
		{`{"suite":"nope","cluster_id":"cl1"}`, http.StatusNotFound},
		{`{"suite":"rccl_perf"}`, http.StatusBadRequest},                      // missing cluster_id
		{`{"suite":"rccl_perf","cluster_id":"ghost"}`, http.StatusBadRequest}, // unknown cluster
	}
	for _, tc := range cases {
		resp, err := http.Post(srv.URL+"/tests/execute", "application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != tc.want {
			t.Errorf("body %s -> status %d, want %d", tc.body, resp.StatusCode, tc.want)
		}
	}
}
