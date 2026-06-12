package clustermon

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/ROCm/cvs/api/pkg/pssh"
)

// MetricsSnapshot is the result of one fleet-wide live metrics sweep. As of S9
// it carries both the GPU and NIC collectors (the two live/critical collectors,
// mirroring the cluster-mon Python poll loop); software/logs collectors are
// on-demand with TTL caches and live elsewhere.
type MetricsSnapshot struct {
	CollectedAt time.Time        `json:"collected_at"`
	GPU         []NodeGPUMetrics `json:"gpu"`
	NIC         []NodeNICMetrics `json:"nic"`
}

// MetricsService collects live GPU + NIC metrics over the fleet singleton SSH
// pool and caches the latest snapshot. Collection runs on-demand (REST) and on
// the S8 poll loop.
type MetricsService struct {
	getPool func() *pssh.Pool
	logger  *slog.Logger

	mu     sync.RWMutex
	latest *MetricsSnapshot
}

// NewMetricsService wires the collector to the fleet singleton pool getter.
func NewMetricsService(getPool func() *pssh.Pool, logger *slog.Logger) *MetricsService {
	if logger == nil {
		logger = slog.Default()
	}
	return &MetricsService{getPool: getPool, logger: logger}
}

// Latest returns the most recent cached snapshot, or nil if none yet.
func (s *MetricsService) Latest() *MetricsSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.latest
}

// Collect runs a GPU + NIC metrics sweep over the reachable nodes, caches it as
// the latest snapshot, and returns it.
func (s *MetricsService) Collect(ctx context.Context) (*MetricsSnapshot, error) {
	pool := s.getPool()
	if pool == nil {
		return nil, fmt.Errorf("no SSH pool: inventory not configured")
	}

	t0 := time.Now()
	s.logger.Info("metrics_collect_start", "reachable", len(pool.Reachable()))

	var (
		gpuResults map[string]NodeGPUMetrics
		nicResults map[string]NodeNICMetrics
		wg         sync.WaitGroup
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		t := time.Now()
		gpuResults = CollectGPU(ctx, pool)
		s.logger.Info("gpu_collect_done", "nodes", len(gpuResults), "elapsed_ms", time.Since(t).Milliseconds())
	}()
	go func() {
		defer wg.Done()
		t := time.Now()
		nicResults = CollectNIC(ctx, pool)
		s.logger.Info("nic_collect_done", "nodes", len(nicResults), "elapsed_ms", time.Since(t).Milliseconds())
	}()
	wg.Wait()
	s.logger.Info("metrics_collect_done", "elapsed_ms", time.Since(t0).Milliseconds())

	gpu := make([]NodeGPUMetrics, 0, len(gpuResults))
	for _, m := range gpuResults {
		gpu = append(gpu, m)
	}
	nic := make([]NodeNICMetrics, 0, len(nicResults))
	for _, m := range nicResults {
		nic = append(nic, m)
	}
	sort.Slice(gpu, func(i, j int) bool { return gpu[i].Host < gpu[j].Host })
	sort.Slice(nic, func(i, j int) bool { return nic[i].Host < nic[j].Host })

	snap := &MetricsSnapshot{CollectedAt: time.Now().UTC(), GPU: gpu, NIC: nic}
	s.mu.Lock()
	s.latest = snap
	s.mu.Unlock()
	s.logger.Info("live_metrics_collected", "gpu_nodes", len(gpu), "nic_nodes", len(nic))
	return snap, nil
}
