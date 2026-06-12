// Command server is the CVS unified platform daemon.
//
// It serves three tiles (Test Execution, Cluster Monitor, Fleet Metrics) behind
// one binary. At S0 it serves the embedded React shell plus health/version.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/ROCm/cvs/api/internal/cluster"
	"github.com/ROCm/cvs/api/internal/clustermon"
	"github.com/ROCm/cvs/api/internal/inventory"
	"github.com/ROCm/cvs/api/internal/testexec"
	httptransport "github.com/ROCm/cvs/api/internal/transport/http"
	"github.com/ROCm/cvs/api/internal/transport/ws"
	"github.com/ROCm/cvs/api/internal/version"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	addr := envOr("CVS_LISTEN_ADDR", ":8080")
	cvsBin := envOr("CVS_BIN", "cvs")
	configDir := testexec.LocateConfigDir(os.Getenv("CVS_CONFIG_DIR"), cvsBin)
	logger.Info("config_dir_resolved", "dir", configDir)

	// Persistent data volume backing every FileStore plus uploaded SSH keys.
	dataDir := envOr("CVS_DATA_DIR", "/app/data")
	keyDir := envOr("CVS_SSH_KEY_DIR", filepath.Join(dataDir, "keys"))
	logger.Info("data_dir_resolved", "data_dir", dataDir, "key_dir", keyDir)

	invStore, err := inventory.NewFileStore(filepath.Join(dataDir, "inventory.json"))
	if err != nil {
		logger.Error("inventory_store_init_failed", "err", err)
		os.Exit(1)
	}
	keyStore := inventory.NewKeyStore(keyDir)

	reprobeInterval := time.Duration(
		envOrInt("PSSH_REPROBE_INTERVAL", defaultPsshReprobeSeconds),
	) * time.Second

	// FleetManager owns the pssh pool lifecycle and the probeReady gate.
	fm := NewFleetManager(invStore, keyStore, reprobeInterval, logger)

	// Saved-cluster catalog (S3b): generated cluster_json files live on the data
	// volume; the catalog index is a JSON collection beside them.
	clusterStore, err := cluster.NewFileStore(filepath.Join(dataDir, "clusters.json"))
	if err != nil {
		logger.Error("cluster_store_init_failed", "err", err)
		os.Exit(1)
	}
	clusterSvc := cluster.NewService(
		clusterStore,
		cluster.NewCLIGenerator(cvsBin),
		invAdapter{store: invStore, keys: keyStore},
		filepath.Join(dataDir, "clusters"),
	)

	// Test Execution runs (S3): persisted execution records + a bounded worker
	// pool. S3 uses a fake runner; S5 swaps in testexec.NewCLIRunner(cvsBin).
	execStore, err := testexec.NewFileExecutionStore(filepath.Join(dataDir, "executions.json"))
	if err != nil {
		logger.Error("execution_store_init_failed", "err", err)
		os.Exit(1)
	}
	// Test runner (S5): the real CLI runner shells out to `cvs run`. The fake
	// runner (used to build/test the lifecycle without real nodes) is still
	// available via CVS_RUNNER=fake for dev/demo.
	var runner testexec.Runner
	if envOr("CVS_RUNNER", "cli") == "fake" {
		runner = testexec.FakeRunner{Delay: 400 * time.Millisecond}
		logger.Info("runner_selected", "runner", "fake")
	} else {
		runner = testexec.NewCLIRunner(cvsBin)
		logger.Info("runner_selected", "runner", "cli", "bin", cvsBin)
	}

	// Live log/status streaming (S4): the executor publishes lifecycle events to
	// the WS hub, which fans them out to connected UI clients. The hub also
	// carries the S8 Cluster Monitor metrics broadcast.
	hub := ws.NewHub()

	// Cluster Monitor services (S7/S8/S9): all use the fleet singleton pool.
	metricsSvc := clustermon.NewMetricsService(fm.Get, logger)
	softwareSvc := clustermon.NewSoftwareService(fm.Get, logger)
	logsSvc := clustermon.NewLogsService(fm.Get, logger)
	topologySvc := clustermon.NewTopologyService(fm.Get, logger)

	// InventoryProber uses the fleet singleton pool to collect and persist
	// per-node basic info (GPU type, ROCm version) to the inventory store.
	// Assigned to fm.Prober so fm.Set can call it during auto-enrichment.
	fm.Prober = inventory.NewInventoryProber(fm.Get)

	// Cluster Monitor poll loop (S8): refreshes + broadcasts GPU metrics and
	// debounced reachability until shutdown.
	poller := clustermon.NewPoller(
		metricsSvc,
		invStore,
		fm.Prober,
		hub,
		clustermon.PollerConfig{
			PollInterval:     time.Duration(envOrInt("POLLING__INTERVAL", 60)) * time.Second,
			FailureThreshold: envOrInt("POLLING__FAILURE_THRESHOLD", 5),
		},
		logger,
	)
	// Demand-driven polling (S8b): the GPU+NIC sweep runs only when at least one
	// browser subscriber has "Live updates on". The WS hub calls poller.Subscribe
	// on the first metrics client and poller.Unsubscribe when the last one leaves.
	hub.SetMetricsLifecycle(poller.Subscribe, poller.Unsubscribe)

	executor := testexec.NewExecutor(
		execStore,
		runner,
		wsEvents{hub: hub},
		envOrInt("CVS_MAX_CONCURRENT", 2),
		1024,
		logger,
	)
	defer executor.Shutdown()

	// Auto-initialize the fleet pool at startup if a saved inventory already
	// exists on the data volume. Without this, a container restart would leave
	// the fleet nil until the user re-saves the inventory through the UI.
	//
	// Clear stale checked_at timestamps synchronously before the server starts
	// accepting requests so every page load during the initial probe sees
	// "probing…" rather than results from a previous container run whose SSH
	// connections are gone.
	if savedInv, ok, err := invStore.Get(); err == nil && ok {
		for i := range savedInv.Statuses {
			savedInv.Statuses[i].CheckedAt = nil
		}
		if _, err := invStore.Save(savedInv); err != nil {
			logger.Warn("startup_stale_clear_failed", "err", err)
		}
		logger.Info("inventory_found_on_startup", "nodes", len(savedInv.Nodes))
		go fm.Set(savedInv)
	}

	handler := httptransport.NewRouter(httptransport.Options{
		Logger:         logger,
		CvsBin:         cvsBin,
		ConfigDir:      configDir,
		InventoryStore: invStore,
		KeyStore:       keyStore,
		ClusterService: clusterSvc,
		Execution: &testexec.ExecutionDeps{
			Store:    execStore,
			Executor: executor,
			Clusters: clusterStore,
			Dir:      filepath.Join(dataDir, "executions"),
		},
		WSHub:            hub,
		WSSnapshots:      execSnapshots{store: execStore},
		Prober:           fm.Prober,
		MetricsService:   metricsSvc,
		SoftwareService:  softwareSvc,
		LogsService:      logsSvc,
		TopologyService:  topologySvc,
		WSMetrics:        metricsProvider{svc: metricsSvc},
		OnInventorySave:  fm.Set,
		OnInventoryClear: fm.Clear,
		PoolStatus:       fm.ProbeStatus,
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		logger.Info("server_starting",
			"addr", addr,
			"version", version.Version,
			"commit", version.Commit,
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server_error", "err", err)
			os.Exit(1)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go poller.Run(ctx)

	<-ctx.Done()

	logger.Info("server_shutting_down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("server_shutdown_error", "err", err)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// defaultPsshReprobeSeconds is the fallback for PSSH_REPROBE_INTERVAL.
const defaultPsshReprobeSeconds = 300

func envOrInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}
