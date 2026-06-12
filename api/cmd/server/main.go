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
	"syscall"
	"time"

	"github.com/ROCm/cvs/api/internal/inventory"
	"github.com/ROCm/cvs/api/internal/testexec"
	httptransport "github.com/ROCm/cvs/api/internal/transport/http"
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

	handler := httptransport.NewRouter(httptransport.Options{
		Logger:         logger,
		CvsBin:         cvsBin,
		ConfigDir:      configDir,
		InventoryStore: invStore,
		KeyStore:       keyStore,
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
