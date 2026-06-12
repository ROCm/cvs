package inventory

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
)

func newTestHandler(t *testing.T) (*Handler, *KeyStore) {
	t.Helper()
	dir := t.TempDir()
	st, err := NewFileStore(filepath.Join(dir, "inventory.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	keys := NewKeyStore(filepath.Join(dir, "keys"))
	return NewHandler(st, keys, nil, nil), keys
}

// fakeProber returns canned statuses without touching the network.
type fakeProber struct {
	statuses []NodeStatus
	called   bool
}

func (f *fakeProber) Probe(_ context.Context, inv Inventory) ([]NodeStatus, error) {
	f.called = true
	if f.statuses != nil {
		return f.statuses, nil
	}
	out := make([]NodeStatus, len(inv.Nodes))
	for i, n := range inv.Nodes {
		out[i] = NodeStatus{Host: n, Reachable: true, GPUType: "MI300X", GPUCount: 8, ROCmVersion: "7.0.2"}
	}
	return out, nil
}

func mountedRouter(h *Handler) http.Handler {
	r := chi.NewRouter()
	r.Route("/api/v1", func(api chi.Router) { h.Routes(api) })
	return r
}

func TestHandler_GetBeforeSave(t *testing.T) {
	h, _ := newTestHandler(t)
	srv := mountedRouter(h)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/inventory", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp["configured"] != false {
		t.Fatalf("expected configured=false, got %v", resp["configured"])
	}
}

func TestHandler_SaveThenGet(t *testing.T) {
	h, _ := newTestHandler(t)
	srv := mountedRouter(h)

	body := `{"nodes":["10.0.0.1","# comment","10.0.0.1","10.0.0.2"],"username":"amd","auth_method":"key","key_name":"id_rsa"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/inventory", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("save status = %d (%s), want 200", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodGet, "/api/v1/inventory", nil)
	rec = httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	var resp struct {
		Configured bool      `json:"configured"`
		Inventory  Inventory `json:"inventory"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !resp.Configured {
		t.Fatal("expected configured=true after save")
	}
	// Comment dropped + duplicate de-duped -> 2 nodes.
	if len(resp.Inventory.Nodes) != 2 {
		t.Fatalf("nodes = %v, want 2 deduped entries", resp.Inventory.Nodes)
	}
}

func TestHandler_SaveValidation(t *testing.T) {
	h, _ := newTestHandler(t)
	srv := mountedRouter(h)

	cases := map[string]string{
		"no nodes":    `{"nodes":["# only a comment"],"username":"amd"}`,
		"no username": `{"nodes":["10.0.0.1"],"username":"  "}`,
		"bad json":    `{`,
	}
	for name, body := range cases {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/inventory", strings.NewReader(body))
		rec := httptest.NewRecorder()
		srv.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", name, rec.Code)
		}
	}
}

func TestHandler_UploadKey(t *testing.T) {
	h, keys := newTestHandler(t)
	srv := mountedRouter(h)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, err := mw.CreateFormFile("key", "cluster_id_rsa")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := fw.Write([]byte("-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/v1/inventory/keys", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("upload status = %d (%s), want 201", rec.Code, rec.Body.String())
	}
	var resp struct {
		KeyName string `json:"key_name"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.KeyName != "cluster_id_rsa" {
		t.Fatalf("key_name = %q, want cluster_id_rsa", resp.KeyName)
	}

	got, err := keys.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 || got[0] != "cluster_id_rsa" {
		t.Fatalf("stored keys = %v, want [cluster_id_rsa]", got)
	}
}

func TestKeyStore_RejectsTraversal(t *testing.T) {
	keys := NewKeyStore(t.TempDir())
	if _, err := keys.Save("../escape", strings.NewReader("x")); err == nil {
		t.Fatal("expected traversal filename to be rejected")
	}
}

func TestHandler_ProbeBeforeSave(t *testing.T) {
	dir := t.TempDir()
	st, err := NewFileStore(filepath.Join(dir, "inventory.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	h := NewHandler(st, nil, &fakeProber{}, nil)
	srv := mountedRouter(h)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/inventory/probe", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("probe before save status = %d, want 409", rec.Code)
	}
}

func TestHandler_ProbePersistsStatuses(t *testing.T) {
	dir := t.TempDir()
	st, err := NewFileStore(filepath.Join(dir, "inventory.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	fp := &fakeProber{}
	h := NewHandler(st, nil, fp, nil)
	srv := mountedRouter(h)

	body := `{"nodes":["10.0.0.1","10.0.0.2"],"username":"amd","auth_method":"key","key_name":"id_rsa"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/inventory", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("save status = %d (%s)", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/inventory/probe", nil)
	rec = httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("probe status = %d (%s), want 200", rec.Code, rec.Body.String())
	}
	if !fp.called {
		t.Fatal("prober was not invoked")
	}

	var resp struct {
		Inventory Inventory `json:"inventory"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Inventory.Statuses) != 2 {
		t.Fatalf("statuses = %v, want 2", resp.Inventory.Statuses)
	}
	if resp.Inventory.Statuses[0].GPUType != "MI300X" || resp.Inventory.Statuses[0].GPUCount != 8 {
		t.Fatalf("unexpected status[0] = %+v", resp.Inventory.Statuses[0])
	}

	// Statuses must survive a reload from disk.
	st2, err := NewFileStore(filepath.Join(dir, "inventory.json"))
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	got, ok, err := st2.Get()
	if err != nil || !ok {
		t.Fatalf("reload: ok=%v err=%v", ok, err)
	}
	if len(got.Statuses) != 2 {
		t.Fatalf("persisted statuses = %v, want 2", got.Statuses)
	}
}
