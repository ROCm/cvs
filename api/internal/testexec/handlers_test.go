package testexec

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
)

type fakeLister struct {
	suites []Suite
	err    error
}

func (f fakeLister) List(context.Context) ([]Suite, error) { return f.suites, f.err }

func newTestServer(t *testing.T, l SuiteLister) *httptest.Server {
	t.Helper()
	return newTestServerWithConfig(t, l, NewConfigService(""))
}

func newTestServerWithConfig(t *testing.T, l SuiteLister, cfg *ConfigService) *httptest.Server {
	t.Helper()
	r := chi.NewRouter()
	NewHandler(l, cfg, nil).Routes(r)
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
