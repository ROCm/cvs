package clustermon

import (
	"context"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/ROCm/cvs/api/pkg/pssh"
)

const (
	lldpTTL            = 3 * time.Minute
	lldpCollectTimeout = 90 * time.Second
)

// TopologySnapshot is the result of one LLDP collection sweep.
type TopologySnapshot struct {
	CollectedAt time.Time      `json:"collected_at"`
	Nodes       []NodeLLDPData `json:"nodes"`
}

// TopologyService collects LLDP neighbor data with a 3-minute TTL cache.
// A stale-on-error fallback means a transient SSH failure doesn't blank the
// topology page.
type TopologyService struct {
	getPool func() *pssh.Pool
	logger  *slog.Logger

	mu      sync.RWMutex
	cached  *TopologySnapshot
	cacheAt time.Time
}

// NewTopologyService creates a topology service backed by the fleet singleton pool getter.
func NewTopologyService(getPool func() *pssh.Pool, logger *slog.Logger) *TopologyService {
	if logger == nil {
		logger = slog.Default()
	}
	return &TopologyService{getPool: getPool, logger: logger}
}

// LLDP returns the cached LLDP snapshot (refreshed if older than lldpTTL).
func (s *TopologyService) LLDP(ctx context.Context) (*TopologySnapshot, error) {
	s.mu.RLock()
	fresh := s.cached != nil && time.Since(s.cacheAt) < lldpTTL
	cached := s.cached
	s.mu.RUnlock()

	if fresh {
		return cached, nil
	}

	snap, err := s.collect(ctx)
	if err != nil {
		if cached != nil {
			s.logger.Warn("lldp_collect_failed_using_stale", "err", err)
			return cached, nil
		}
		return nil, err
	}
	s.mu.Lock()
	s.cached = snap
	s.cacheAt = time.Now()
	s.mu.Unlock()
	return snap, nil
}

func (s *TopologyService) collect(ctx context.Context) (*TopologySnapshot, error) {
	pool := s.getPool()
	if pool == nil {
		return &TopologySnapshot{CollectedAt: time.Now().UTC()}, nil
	}

	results := CollectLLDP(ctx, pool)

	nodes := make([]NodeLLDPData, 0, len(results))
	for _, nd := range results {
		nodes = append(nodes, nd)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Host < nodes[j].Host })

	s.logger.Info("lldp_collected", "nodes", len(nodes))
	return &TopologySnapshot{CollectedAt: time.Now().UTC(), Nodes: nodes}, nil
}

