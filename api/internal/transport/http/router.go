// Package http builds the shared HTTP router for the CVS unified daemon.
//
// At S0 this wires only the health/version endpoints and the embedded SPA.
// Subsequent slices mount per-tile routers under /api/v1 and the WebSocket hub.
package http

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"github.com/ROCm/cvs/api/internal/inventory"
	"github.com/ROCm/cvs/api/internal/testexec"
	"github.com/ROCm/cvs/api/internal/version"
	"github.com/ROCm/cvs/api/internal/webui"
)

// Options configures the router.
type Options struct {
	Logger *slog.Logger
	// CvsBin is the CVS CLI executable used to discover suites (Test Execution tile).
	CvsBin string
	// ConfigDir is the CVS config_file directory used for suite config forms.
	ConfigDir string
	// InventoryStore backs the inventory-first gate. When nil, inventory routes
	// are not mounted (e.g. narrow unit tests).
	InventoryStore inventory.Store
	// KeyStore handles SSH private-key uploads. May be nil to disable uploads.
	KeyStore *inventory.KeyStore
}

// NewRouter constructs the top-level chi router.
func NewRouter(opts Options) http.Handler {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	r := chi.NewRouter()

	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(requestLogger(logger))
	r.Use(middleware.Recoverer)

	// Liveness probe (kept outside /api/v1 for simple container healthchecks).
	r.Get("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	testexecHandler := testexec.NewHandler(
		testexec.NewCLISuiteLister(opts.CvsBin),
		testexec.NewConfigService(opts.ConfigDir),
		logger,
	)

	r.Route("/api/v1", func(api chi.Router) {
		api.Get("/health", func(w http.ResponseWriter, _ *http.Request) {
			writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		})
		api.Get("/version", func(w http.ResponseWriter, _ *http.Request) {
			writeJSON(w, http.StatusOK, map[string]string{
				"version":    version.Version,
				"commit":     version.Commit,
				"build_time": version.BuildTime,
			})
		})

		// Inventory-first gate (shared by all tiles) + F2 connectivity probe.
		if opts.InventoryStore != nil {
			prober := inventory.NewSSHProber(opts.KeyStore)
			inventory.NewHandler(opts.InventoryStore, opts.KeyStore, prober, logger).Routes(api)
		}

		// Test Execution tile.
		testexecHandler.Routes(api)
	})

	// Embedded single-page app + client-side routing fallback.
	r.Handle("/*", webui.Handler())

	return r
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// requestLogger emits one structured log line per request.
func requestLogger(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)
			logger.Info("http_request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", ww.Status(),
				"bytes", ww.BytesWritten(),
				"duration_ms", time.Since(start).Milliseconds(),
				"request_id", middleware.GetReqID(r.Context()),
			)
		})
	}
}
