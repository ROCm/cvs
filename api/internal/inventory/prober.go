package inventory

import (
	"context"
	"time"

	"github.com/ROCm/cvs/api/internal/probe"
	"github.com/ROCm/cvs/api/pkg/pssh"
)

// Prober runs connectivity + basic-info collection for an inventory and returns
// the per-node statuses. It is the F2 capability that populates reachability,
// GPU type/count, and ROCm version for the tiles.
type Prober interface {
	Probe(ctx context.Context, inv Inventory) ([]NodeStatus, error)
}

// InventoryProber implements Prober using the fleet singleton pssh.Pool as the
// authoritative reachability source. The pool tracks reachability via SSH
// keepalives, Probe sweeps, and per-Exec connection-error detection — all of
// which catch SSH-level failures (auth, key, sshd config) that a plain TCP
// sweep cannot detect. The prober reads pool.Reachable() directly and collects
// basic info (GPU type, ROCm version) on the reachable set, persisting it to
// the inventory NodeStatus records.
type InventoryProber struct {
	getPool func() *pssh.Pool
}

// NewInventoryProber returns a prober that uses the fleet singleton pool for
// both reachability state and SSH basic-info collection. The pool is obtained
// lazily on each Probe call so it picks up the latest configured pool
// automatically.
func NewInventoryProber(getPool func() *pssh.Pool) *InventoryProber {
	return &InventoryProber{getPool: getPool}
}

// Probe reads the fleet pool's SSH-verified reachability state, collects basic
// info (GPU type, ROCm version) on the reachable nodes, and returns one
// NodeStatus per inventory node.
//
// When the pool is nil (inventory not yet configured) all nodes are returned
// with a "ssh pool not configured" error and Reachable=false.
func (p *InventoryProber) Probe(ctx context.Context, inv Inventory) ([]NodeStatus, error) {
	now := time.Now().UTC()

	pool := p.getPool()
	if pool == nil {
		statuses := make([]NodeStatus, 0, len(inv.Nodes))
		for _, host := range inv.Nodes {
			statuses = append(statuses, NodeStatus{
				Host:      host,
				CheckedAt: &now,
				Error:     "ssh pool not configured",
			})
		}
		return statuses, nil
	}

	// pool.Reachable() reflects the latest SSH-verified sweep (keepalive /
	// Probe / pruneAfterExec). Use it directly — no redundant TCP sweep.
	reachable := pool.Reachable()
	reachSet := make(map[string]struct{}, len(reachable))
	for _, h := range reachable {
		reachSet[h] = struct{}{}
	}

	// Collect basic info (GPU type, ROCm version) fleet-wide via SSH.
	infos := probe.Collect(ctx, pool)

	statuses := make([]NodeStatus, 0, len(inv.Nodes))
	for _, host := range inv.Nodes {
		st := NodeStatus{Host: host, CheckedAt: &now}
		if _, ok := reachSet[host]; ok {
			st.Reachable = true
			if info, found := infos[host]; found {
				st.GPUType = info.GPUType
				st.GPUCount = info.GPUCount
				st.ROCmVersion = info.ROCmVersion
				st.Error = info.Err
			}
		} else {
			if reason := pool.NodeError(host); reason != "" {
				st.Error = reason
			} else {
				st.Error = "ssh: host unreachable"
			}
		}
		statuses = append(statuses, st)
	}
	return statuses, nil
}
