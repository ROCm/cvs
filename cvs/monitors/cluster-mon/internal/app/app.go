package app

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/collectors"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/fleet"
	insp "github.com/ROCm/cvs/monitors/cluster-mon/internal/inspector"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/rccl"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/ws"
)

type CollectorResult struct {
	State     string `json:"state"`
	Timestamp string `json:"timestamp"`
	Error     string `json:"error,omitempty"`
	Critical  bool   `json:"-"`
}

type cacheEntry struct {
	data any
	at   time.Time
}

type App struct {
	Logger    *slog.Logger
	ConfigDir string
	Fleet     *fleet.Fleet
	Logs      *collectors.LogsService
	Soft      *collectors.SoftwareService

	MetricsHub *ws.Hub
	RCCLHub    *ws.Hub

	mu           sync.RWMutex
	cfg          *config.Config
	sshPassword  string
	jumpPassword string
	latest       map[string]any
	latestRCCL   map[string]any
	collecting   bool
	failCount    map[string]int
	health       map[string]string
	results      map[string]CollectorResult
	nicSoftCache cacheEntry
	nicAdvCache  cacheEntry
	softwareTTL  time.Duration

	Store   *rccl.Store
	rcclCol *rccl.Collector

	appliedSSH  string
	appliedJump string

	cancel context.CancelFunc
}

func New(logger *slog.Logger, configDir string) *App {
	if logger == nil {
		logger = slog.Default()
	}
	a := &App{
		Logger:      logger,
		ConfigDir:   configDir,
		Fleet:       fleet.New(logger),
		MetricsHub:  ws.NewHub(),
		RCCLHub:     ws.NewHub(),
		failCount:   map[string]int{},
		health:      map[string]string{},
		results:     map[string]CollectorResult{},
		softwareTTL: 180 * time.Second,
		Store:       rccl.NewStore(),
	}
	a.Logs = collectors.NewLogsService(a.Pool, logger)
	a.Soft = collectors.NewSoftwareService(a.Pool, logger)
	return a
}

func (a *App) Pool() *pssh.Pool {
	if a.Fleet == nil {
		return nil
	}
	return a.Fleet.Pool()
}

func (a *App) Cfg() *config.Config {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.cfg
}

func (a *App) SetPasswords(ssh, jump string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if ssh != "" {
		a.sshPassword = ssh
	}
	if jump != "" {
		a.jumpPassword = jump
	}
}

func (a *App) Passwords() (string, string) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.sshPassword, a.jumpPassword
}

func (a *App) LatestMetrics() map[string]any {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.latest
}

func (a *App) LatestRCCL() map[string]any {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.latestRCCL
}

func (a *App) Collecting() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.collecting
}

func (a *App) NodeHealth() (map[string]string, map[string]int) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	h := map[string]string{}
	f := map[string]int{}
	for k, v := range a.health {
		h[k] = v
	}
	for k, v := range a.failCount {
		f[k] = v
	}
	return h, f
}

func (a *App) CollectorResults() map[string]CollectorResult {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := map[string]CollectorResult{}
	for k, v := range a.results {
		out[k] = v
	}
	return out
}

func (a *App) Start(ctx context.Context) {
	ctx, a.cancel = context.WithCancel(ctx)
	cfg, err := config.Load(a.ConfigDir)
	if err != nil {
		a.Logger.Warn("config_load_failed", "err", err)
		a.Store.ConnectRedis(config.RedisConfig{}, a.Logger)
	} else {
		a.mu.Lock()
		a.cfg = cfg
		a.mu.Unlock()
		a.Store.ConnectRedis(cfg.Storage.Redis, a.Logger)
		if len(cfg.Nodes) > 0 {
			if err := a.Fleet.Rebuild(cfg, a.sshPassword, a.jumpPassword); err != nil {
				a.Logger.Warn("fleet_rebuild_failed", "err", err)
			} else {
				a.appliedSSH, a.appliedJump = a.sshPassword, a.jumpPassword
			}
		}
	}
	go a.pollMetrics(ctx)
	go a.pollRCCL(ctx)
	go a.pollInspector(ctx)
}

func (a *App) Stop() {
	if a.cancel != nil {
		a.cancel()
	}
	a.Fleet.Close()
	if a.Store != nil {
		a.Store.Close()
	}
}

func (a *App) pollInterval() time.Duration {
	cfg := a.Cfg()
	n := 60
	if cfg != nil && cfg.Polling.Interval > 0 {
		n = cfg.Polling.Interval
	}
	if v := os.Getenv("POLLING__INTERVAL"); v != "" {
		if i, err := strconv.Atoi(v); err == nil && i > 0 {
			n = i
		}
	}
	return time.Duration(n) * time.Second
}

func (a *App) failureThreshold() int {
	cfg := a.Cfg()
	n := 5
	if cfg != nil && cfg.Polling.FailureThreshold > 0 {
		n = cfg.Polling.FailureThreshold
	}
	return n
}

func (a *App) pollMetrics(ctx context.Context) {
	a.waitInitialProbe(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		a.collectOnce(ctx)
		t := time.NewTimer(a.pollInterval())
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
}

// waitInitialProbe blocks until the SSH pool has classified the fleet (or the
// app is shutting down). Exec() is empty until then because New() starts every
// node unreachable; collecting immediately produces gpu_hosts=0 and a blank UI.
func (a *App) waitInitialProbe(ctx context.Context) {
	t := time.NewTicker(200 * time.Millisecond)
	defer t.Stop()
	for {
		pool := a.Pool()
		if pool != nil && pool.InitialProbeDone() {
			a.Logger.Info("initial_probe_done", "reachable", len(pool.Reachable()), "unreachable", len(pool.Unreachable()))
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (a *App) collectOnce(ctx context.Context) {
	pool := a.Pool()
	if pool == nil {
		a.mu.Lock()
		a.collecting = false
		a.mu.Unlock()
		return
	}
	if !pool.InitialProbeDone() || len(pool.Reachable()) == 0 {
		a.updateHealth(pool)
		a.Logger.Info("metrics_skipped", "probe_done", pool.InitialProbeDone(), "reachable", len(pool.Reachable()))
		return
	}
	a.mu.Lock()
	a.collecting = true
	a.mu.Unlock()

	cctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	var gpu map[string]collectors.NodeGPUMetrics
	var nic map[string]collectors.NodeNICMetrics
	var lldp map[string]any
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); gpu = collectors.CollectGPU(cctx, pool) }()
	go func() { defer wg.Done(); nic = collectors.CollectNIC(cctx, pool) }()
	go func() { defer wg.Done(); lldp = collectors.CollectLLDPRaw(cctx, pool) }()
	wg.Wait()

	gpuErr, nicErr := "", ""
	if len(gpu) == 0 {
		gpuErr = "no gpu data"
	}
	if len(nic) == 0 {
		nicErr = "no nic data"
	}

	metrics := map[string]any{
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"gpu":       collectors.PythonGPUPayload(gpu),
		"nic":       collectors.PythonNICPayload(nic, lldp),
	}

	a.updateHealth(pool)
	a.mu.Lock()
	a.latest = metrics
	a.results["gpu"] = CollectorResult{State: stateOK(gpuErr), Timestamp: metrics["timestamp"].(string), Error: gpuErr, Critical: true}
	a.results["nic"] = CollectorResult{State: stateOK(nicErr), Timestamp: metrics["timestamp"].(string), Error: nicErr, Critical: true}
	a.mu.Unlock()
	a.MetricsHub.Broadcast(map[string]any{"type": "metrics", "data": metrics})
	a.Logger.Info("metrics_collected", "gpu_hosts", len(gpu), "nic_hosts", len(nic))
}

func stateOK(err string) string {
	if err != "" {
		return "error"
	}
	return "ok"
}

func (a *App) updateHealth(pool *pssh.Pool) {
	th := a.failureThreshold()
	all := pool.All()
	unreach := map[string]struct{}{}
	for _, h := range pool.Unreachable() {
		unreach[h] = struct{}{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, h := range all {
		if _, bad := unreach[h]; bad {
			a.failCount[h]++
			if a.failCount[h] >= th {
				a.health[h] = "unreachable"
			} else if a.health[h] == "" {
				a.health[h] = "healthy"
			}
		} else {
			a.failCount[h] = 0
			if a.health[h] == "unreachable" || a.health[h] == "" {
				a.health[h] = "healthy"
			}
		}
	}
}

func (a *App) pollRCCL(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		cfg := a.Cfg()
		interval := 30 * time.Second
		if cfg != nil && cfg.RCCL.PollInterval > 0 {
			interval = time.Duration(cfg.RCCL.PollInterval) * time.Second
		}
		pool := a.Pool()
		if pool != nil {
			timeout := 8 * time.Second
			rasPort := 28028
			if cfg != nil {
				if cfg.RCCL.CollectiveTimeoutSecs > 0 {
					timeout = time.Duration(cfg.RCCL.CollectiveTimeoutSecs) * time.Second
				}
				if cfg.RCCL.RASPort > 0 {
					rasPort = cfg.RCCL.RASPort
				}
			}
			maxBytes := 0
			if cfg != nil {
				maxBytes = cfg.RCCL.RASMaxResponseBytes
			}
			col := a.rcclCollector(pool, rasPort, maxBytes)
			snap := col.Collect(ctx, timeout)
			m := snapToMap(snap)
			st := "ok"
			switch snap.State {
			case "no_job":
				st = "no_service"
			case "degraded", "error":
				st = "error"
			case "unreachable":
				st = "unreachable"
			}
			a.mu.Lock()
			a.latestRCCL = m
			a.results["rccl"] = CollectorResult{State: st, Timestamp: time.Now().UTC().Format(time.RFC3339), Critical: false}
			a.mu.Unlock()
			a.Store.PushSnapshot(m)
			a.RCCLHub.Broadcast(map[string]any{"type": "rccl_snapshot", "data": m})
		}
		t := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
}

func (a *App) pollInspector(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		cfg := a.Cfg()
		interval := 30 * time.Second
		enabled := false
		inspCfg := config.InspectorConfig{}
		if cfg != nil {
			inspCfg = cfg.RCCL.Inspector
			enabled = inspCfg.Enabled
			if inspCfg.PollInterval > 0 {
				interval = time.Duration(inspCfg.PollInterval) * time.Second
			}
		}
		if enabled {
			pids := a.activePIDs()
			snap := insp.Collect(ctx, a.Pool(), inspCfg, pids)
			if col := a.rcclCollector(a.Pool(), 0, 0); col != nil {
				col.MarkInspectorV5(insp.InspectorV5Hosts(snap.Records))
			}
			b, _ := json.Marshal(snap)
			var m map[string]any
			_ = json.Unmarshal(b, &m)
			a.Store.PushInspector(m)
			a.mu.Lock()
			a.results["inspector"] = CollectorResult{State: "ok", Timestamp: time.Now().UTC().Format(time.RFC3339), Critical: false}
			a.mu.Unlock()
		}
		t := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
}

func (a *App) activePIDs() map[int]struct{} {
	out := map[int]struct{}{}
	snap := a.LatestRCCL()
	if snap == nil {
		return out
	}
	comms, _ := snap["communicators"].([]any)
	for _, c := range comms {
		cm, _ := c.(map[string]any)
		ranks, _ := cm["ranks"].([]any)
		for _, r := range ranks {
			rm, _ := r.(map[string]any)
			switch t := rm["pid"].(type) {
			case float64:
				if t > 0 {
					out[int(t)] = struct{}{}
				}
			case int:
				if t > 0 {
					out[t] = struct{}{}
				}
			}
		}
	}
	return out
}

func (a *App) NICSoftware(ctx context.Context) map[string]any {
	a.mu.RLock()
	if a.nicSoftCache.data != nil && time.Since(a.nicSoftCache.at) < a.softwareTTL {
		d := a.nicSoftCache.data.(map[string]any)
		a.mu.RUnlock()
		return d
	}
	a.mu.RUnlock()
	d := collectors.CollectNICSoftware(ctx, a.Pool(), a.Logger)
	a.mu.Lock()
	a.nicSoftCache = cacheEntry{data: d, at: time.Now()}
	a.mu.Unlock()
	return d
}

func (a *App) NICAdvanced(ctx context.Context) map[string]any {
	a.mu.RLock()
	if a.nicAdvCache.data != nil && time.Since(a.nicAdvCache.at) < a.softwareTTL {
		d := a.nicAdvCache.data.(map[string]any)
		a.mu.RUnlock()
		return d
	}
	a.mu.RUnlock()
	d := collectors.CollectNICAdvanced(ctx, a.Pool(), a.Logger)
	a.mu.Lock()
	a.nicAdvCache = cacheEntry{data: d, at: time.Now()}
	a.mu.Unlock()
	return d
}

func snapToMap(s rccl.Snapshot) map[string]any {
	b, err := json.Marshal(s)
	if err != nil {
		return map[string]any{"state": "no_job"}
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

func (a *App) rcclCollector(pool *pssh.Pool, rasPort, maxBytes int) *rccl.Collector {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.rcclCol == nil {
		port := rasPort
		if port <= 0 && a.cfg != nil {
			port = a.cfg.RCCL.RASPort
		}
		a.rcclCol = rccl.NewCollector(pool, a.Store, port, a.Logger)
	} else {
		a.rcclCol.SetPool(pool)
		if rasPort > 0 {
			a.rcclCol.SetRASPort(rasPort)
		}
	}
	if maxBytes > 0 {
		a.rcclCol.SetRASMaxBytes(maxBytes)
	}
	return a.rcclCol
}
