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

	"github.com/ROCm/cvs/api/internal/cluster"
	"github.com/ROCm/cvs/api/internal/clustermon"
	"github.com/ROCm/cvs/api/internal/inventory"
	"github.com/ROCm/cvs/api/internal/testexec"
	"github.com/ROCm/cvs/api/internal/transport/ws"
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
	// Prober backs the /inventory/probe endpoint. When nil the probe endpoint
	// returns 503; constructed in main.go with the fleet singleton pool.
	Prober inventory.Prober
	// ClusterService backs the saved-cluster catalog (S3b). May be nil.
	ClusterService *cluster.Service
	// Execution wires the Test Execution run capability (S3). May be nil.
	Execution *testexec.ExecutionDeps
	// WSHub + WSSnapshots wire the live log/status streams (S4). May be nil.
	WSHub       *ws.Hub
	WSSnapshots ws.SnapshotProvider
	// MetricsService backs the Cluster Monitor live GPU metrics (S7/S8). May be nil.
	MetricsService *clustermon.MetricsService
	// SoftwareService backs the Cluster Monitor on-demand software collectors
	// (S9b: GPU software/firmware, NIC devlink). May be nil.
	SoftwareService *clustermon.SoftwareService
	// LogsService backs the Cluster Monitor dmesg log collectors + grep search
	// (S9c). May be nil.
	LogsService *clustermon.LogsService
	// TopologyService backs the LLDP network topology endpoint (S9g). May be nil.
	TopologyService *clustermon.TopologyService
	// WSMetrics supplies the latest snapshot for the /ws/clustermon stream (S8).
	// May be nil (the stream still works, just no immediate first frame).
	WSMetrics ws.MetricsProvider

	// OnInventorySave, when non-nil, is wired to the inventory handler's OnSave
	// callback and called (in a goroutine) after a successful POST /inventory.
	OnInventorySave func(inventory.Inventory)
	// OnInventoryClear, when non-nil, is wired to the inventory handler's
	// OnClear callback and called (in a goroutine) after DELETE /inventory.
	OnInventoryClear func()
	// PoolStatus, when non-nil, is called by GET /api/v1/inventory/probe-status
	// to return the current SSH pool probe state. When nil the endpoint returns
	// a zeroed (not-ready) status so the UI can still poll safely.
	PoolStatus func() ProbeStatusResponse
}

// ProbeStatusResponse is the JSON shape served by GET /api/v1/inventory/probe-status.
// It is defined here (transport layer) because the SSH pool does not need to
// know about it — the ready flag is app-level state owned by main.go.
type ProbeStatusResponse struct {
	Ready       bool `json:"ready"`
	Reachable   int  `json:"reachable"`
	Unreachable int  `json:"unreachable"`
	Total       int  `json:"total"`
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
		opts.Execution,
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
			invHandler := inventory.NewHandler(opts.InventoryStore, opts.KeyStore, opts.Prober, logger)
			invHandler.OnSave = opts.OnInventorySave
			invHandler.OnClear = opts.OnInventoryClear
			invHandler.Routes(api)

			// Probe-status endpoint: UI polls this until ready:true before
			// allowing access to Cluster Monitor / Fleet Metrics tiles.
			poolStatusFn := opts.PoolStatus
			api.Get("/inventory/probe-status", func(w http.ResponseWriter, _ *http.Request) {
				var st ProbeStatusResponse
				if poolStatusFn != nil {
					st = poolStatusFn()
				}
				writeJSON(w, http.StatusOK, st)
			})

			// Cluster Monitor tile (S6–S9): all services use the fleet singleton
			// pool injected from main.go; nil services are skipped at runtime.
			clustermon.NewHandler(
				opts.InventoryStore,
				opts.MetricsService,
				opts.SoftwareService,
				opts.LogsService,
				opts.TopologyService,
				logger,
			).Routes(api)
		}

		// Saved-cluster catalog (S3b), shared by Test Execution run config.
		if opts.ClusterService != nil {
			cluster.NewHandler(opts.ClusterService, logger).Routes(api)
		}

		// Test Execution tile.
		testexecHandler.Routes(api)
	})

	// WebSocket streams (live logs/status + notifications). Outside /api/v1 to
	// match the ws-scheme convention and keep upgrade handling separate.
	if opts.WSHub != nil {
		wsHandler := ws.NewHandler(opts.WSHub, opts.WSSnapshots, logger)
		if opts.WSMetrics != nil {
			wsHandler.SetMetricsProvider(opts.WSMetrics)
		}
		r.Route("/ws", func(wr chi.Router) { wsHandler.Routes(wr) })
	}

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
