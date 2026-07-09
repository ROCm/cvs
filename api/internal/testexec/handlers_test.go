package testexec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
)

type fakeLister struct {
	suites []Suite
	err    error
}

func (f fakeLister) List(context.Context) ([]Suite, error) { return f.suites, f.err }

type fakeClusters struct{ file string }

func (f fakeClusters) FilePath(id string) (string, bool) {
	if f.file == "" {
		return "", false
	}
	return f.file, true
}

// TestRerunHandler verifies that POST /executions/{id}/rerun creates a new
// queued execution reusing the source's suite, cluster, and staged config.
func TestRerunHandler(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileExecutionStore(filepath.Join(dir, "exec.json"))
	if err != nil {
		t.Fatal(err)
	}
	exec := NewExecutor(store, FakeRunner{Delay: time.Millisecond}, nil, 1, 4, nil)
	t.Cleanup(exec.Shutdown)

	clusterFile := filepath.Join(dir, "cluster.json")
	if err := os.WriteFile(clusterFile, []byte(`{"nodes":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	deps := &ExecutionDeps{
		Store:    store,
		Executor: exec,
		Clusters: fakeClusters{file: clusterFile},
		Dir:      dir,
	}
	lister := fakeLister{suites: []Suite{{Name: "rccl_perf"}}}

	r := chi.NewRouter()
	NewHandler(lister, NewConfigService(""), nil, deps).Routes(r)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	// Seed a finished source execution with a staged config.
	const srcID = "src-1"
	srcDir := filepath.Join(dir, srcID)
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfgPath := filepath.Join(srcDir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{"rccl":{"mpi_pml":"ucx"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(Execution{
		ID: srcID, Suite: "rccl_perf", ClusterID: "c1", ClusterFile: clusterFile,
		ConfigPath: cfgPath, Status: StatusFailed, CreatedAt: time.Now().UTC(),
	}); err != nil {
		t.Fatal(err)
	}

	resp, err := http.Post(srv.URL+"/executions/"+srcID+"/rerun", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("status = %d, want 202", resp.StatusCode)
	}
	var got Execution
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.ID == srcID {
		t.Fatal("rerun must mint a new execution id")
	}
	if got.Suite != "rccl_perf" || got.ClusterID != "c1" {
		t.Fatalf("rerun did not carry over suite/cluster: %+v", got)
	}
	// The new execution's config must match the source's staged bytes.
	newCfg, err := os.ReadFile(got.ConfigPath)
	if err != nil {
		t.Fatalf("read new config: %v", err)
	}
	var orig, copied map[string]any
	_ = json.Unmarshal([]byte(`{"rccl":{"mpi_pml":"ucx"}}`), &orig)
	if err := json.Unmarshal(newCfg, &copied); err != nil {
		t.Fatalf("new config not valid JSON: %v", err)
	}
	if a, _ := json.Marshal(orig); !bytes.Equal(a, mustMarshal(t, copied)) {
		t.Fatalf("rerun config mismatch: got %s", newCfg)
	}

	// Let the async rerun job finish writing before TempDir cleanup runs.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if ex, ok := store.Get(got.ID); ok && ex.Status.Terminal() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Unknown source -> 404.
	resp2, err := http.Post(srv.URL+"/executions/nope/rerun", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp2.StatusCode)
	}
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newTestServer(t *testing.T, l SuiteLister) *httptest.Server {
	t.Helper()
	return newTestServerWithConfig(t, l, NewConfigService(""))
}

func newTestServerWithConfig(t *testing.T, l SuiteLister, cfg *ConfigService) *httptest.Server {
	t.Helper()
	r := chi.NewRouter()
	NewHandler(l, cfg, nil, nil).Routes(r)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv
}

func TestListSuitesHandler_OK(t *testing.T) {
	lister := fakeLister{suites: []Suite{
		{Name: "rccl_perf", ModulePath: "cvs.tests.rccl.rccl_perf", Module: "cvs.tests.rccl", Package: "cvs", Category: "rccl"},
		{Name: "rccl_regression", ModulePath: "cvs.tests.rccl.rccl_regression", Module: "cvs.tests.rccl", Package: "cvs", Category: "rccl"},
	}}
	srv := newTestServer(t, lister)

	resp, err := http.Get(srv.URL + "/suites")
	if err != nil {
		t.Fatalf("GET /suites: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var body struct {
		Suites []Suite `json:"suites"`
		Total  int     `json:"total"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Total != 2 || len(body.Suites) != 2 {
		t.Fatalf("unexpected total/len: %d / %d", body.Total, len(body.Suites))
	}
	if body.Suites[0].Name != "rccl_perf" || body.Suites[0].Category != "rccl" {
		t.Fatalf("unexpected first suite: %+v", body.Suites[0])
	}
}

func TestListSuitesHandler_ListerError(t *testing.T) {
	srv := newTestServer(t, fakeLister{err: errors.New("boom")})

	resp, err := http.Get(srv.URL + "/suites")
	if err != nil {
		t.Fatalf("GET /suites: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
}

func TestSuiteSchemaHandler(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root+"/rccl/rccl_config.json", `{"rccl":{"_comment_mpi_pml":"MPI layer","mpi_pml":"auto"}}`)

	lister := fakeLister{suites: []Suite{
		{Name: "rccl_perf", Module: "cvs.tests.rccl", Category: "rccl", Package: "cvs", ModulePath: "cvs.tests.rccl.rccl_perf"},
		{Name: "conftest", Module: "cvs.tests", Category: "tests", Package: "cvs", ModulePath: "cvs.tests.conftest"},
	}}
	srv := newTestServerWithConfig(t, lister, NewConfigService(root))

	// Suite with a config.
	resp, err := http.Get(srv.URL + "/suites/rccl_perf/schema")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var body struct {
		HasConfig  bool    `json:"has_config"`
		ConfigPath string  `json:"config_path"`
		Schema     []Field `json:"schema"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if !body.HasConfig || body.ConfigPath != "rccl/rccl_config.json" {
		t.Fatalf("unexpected schema response: %+v", body)
	}
	if len(body.Schema) != 1 || body.Schema[0].Name != "rccl" || len(body.Schema[0].Fields) != 1 {
		t.Fatalf("unexpected schema tree: %+v", body.Schema)
	}
	if body.Schema[0].Fields[0].Description != "MPI layer" {
		t.Fatalf("comment not folded into description: %+v", body.Schema[0].Fields[0])
	}

	// Suite without a config -> has_config false, 200.
	resp2, err := http.Get(srv.URL + "/suites/conftest/schema")
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var body2 struct {
		HasConfig bool `json:"has_config"`
	}
	if err := json.NewDecoder(resp2.Body).Decode(&body2); err != nil {
		t.Fatal(err)
	}
	if body2.HasConfig {
		t.Fatal("conftest should have no config")
	}

	// Unknown suite -> 404.
	resp3, err := http.Get(srv.URL + "/suites/does_not_exist/schema")
	if err != nil {
		t.Fatal(err)
	}
	defer resp3.Body.Close()
	if resp3.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp3.StatusCode)
	}
}
