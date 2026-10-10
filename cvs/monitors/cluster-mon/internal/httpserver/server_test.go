package httpserver

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/app"
)

func TestHealth(t *testing.T) {
	a := app.New(slog.Default(), t.TempDir())
	s := &Server{App: a, Logger: slog.Default()}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status %d body %s", rr.Code, rr.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["status"] != "healthy" {
		t.Fatalf("%v", body)
	}
}

func TestConfigCurrentEmpty(t *testing.T) {
	dir := t.TempDir()
	a := app.New(slog.Default(), dir)
	s := &Server{App: a, Logger: slog.Default()}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/config/current", nil)
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status %d body %s", rr.Code, rr.Body.String())
	}
}

func TestSSHKeyUploadRequiresToken(t *testing.T) {
	t.Setenv("CLUSTER_MON_API_TOKEN", "secret-token")
	a := app.New(slog.Default(), t.TempDir())
	s := &Server{App: a, Logger: slog.Default()}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/ssh-keys/list", nil)
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 401 {
		t.Fatalf("status %d", rr.Code)
	}
	rr = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/api/ssh-keys/list", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("authed status %d body %s", rr.Code, rr.Body.String())
	}
}

func TestBackendRestartUsesCallback(t *testing.T) {
	called := make(chan error, 1)
	a := app.New(slog.Default(), t.TempDir())
	s := &Server{
		App:    a,
		Logger: slog.Default(),
		Restart: func() error {
			called <- errors.New("exec failed")
			return errors.New("exec failed")
		},
	}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/backend/restart", nil)
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	select {
	case <-called:
	case <-time.After(3 * time.Second):
		t.Fatal("restart callback was not invoked")
	}
}

func TestMetricsLatestEmpty(t *testing.T) {
	a := app.New(slog.Default(), t.TempDir())
	s := &Server{App: a}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/metrics/latest", nil)
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != 503 {
		t.Fatalf("status %d", rr.Code)
	}
}
