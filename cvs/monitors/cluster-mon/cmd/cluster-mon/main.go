package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/app"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/httpserver"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/sshkeys"
)

func main() {
	listen := flag.String("listen", envOr("LISTEN_ADDR", ":8001"), "HTTP listen address")
	configDir := flag.String("config-dir", "", "cluster.yaml / nodes.txt directory")
	staticDir := flag.String("static", envOr("STATIC_DIR", ""), "built React dist directory")
	flag.Parse()

	level := slog.LevelInfo
	d := os.Getenv("DEBUG")
	if d == "true" || d == "1" || d == "yes" {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	dir := *configDir
	if dir == "" {
		if h := os.Getenv("CLUSTER_MONITOR_HOME"); h != "" {
			dir = filepath.Join(h, "config")
		} else {
			dir = config.ResolveConfigDir()
		}
	}

	static := *staticDir
	if static == "" {
		for _, p := range []string{"/app/static", "static", "frontend/dist"} {
			if st, err := os.Stat(p); err == nil && st.IsDir() {
				static = p
				break
			}
		}
	}

	application := app.New(logger, dir)
	if err := sshkeys.EnsureDir(); err != nil {
		logger.Warn("ssh_dir", "err", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	application.Start(ctx)

	srv := &httpserver.Server{App: application, StaticDir: static, Logger: logger, Restart: reexec}
	httpSrv := &http.Server{Addr: *listen, Handler: srv.Handler(), ReadHeaderTimeout: 15 * time.Second}

	go func() {
		logger.Info("cluster_mon_listen", "addr", *listen, "config_dir", dir, "static", static)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("http_server", "err", err)
			os.Exit(1)
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	shutdown, scancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer scancel()
	_ = httpSrv.Shutdown(shutdown)
	application.Stop()
}

func reexec() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	return syscall.Exec(exe, os.Args, os.Environ())
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
