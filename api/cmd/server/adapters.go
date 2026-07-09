package main

import (
	"os"

	"github.com/ROCm/cvs/api/internal/cluster"
	"github.com/ROCm/cvs/api/internal/clustermon"
	"github.com/ROCm/cvs/api/internal/inventory"
	"github.com/ROCm/cvs/api/internal/testexec"
	"github.com/ROCm/cvs/api/internal/transport/ws"
)

// invAdapter bridges the inventory store + key store to the cluster package's
// InventoryProvider, resolving the uploaded key name to an absolute path for
// the `cvs generate cluster_json --key_file` flag.
type invAdapter struct {
	store inventory.Store
	keys  *inventory.KeyStore
}

func (a invAdapter) Current() (cluster.Inventory, bool, error) {
	inv, ok, err := a.store.Get()
	if err != nil || !ok {
		return cluster.Inventory{}, ok, err
	}
	keyFile := ""
	if a.keys != nil && inv.KeyName != "" {
		if p, e := a.keys.Path(inv.KeyName); e == nil {
			keyFile = p
		}
	}
	return cluster.Inventory{Username: inv.Username, KeyFile: keyFile, Nodes: inv.Nodes}, true, nil
}

// wsEvents adapts the WS hub to testexec.Events so the executor can stream live
// log lines and lifecycle transitions without importing the ws package.
type wsEvents struct{ hub *ws.Hub }

func (e wsEvents) Log(id, line string)                     { e.hub.PublishLog(id, line) }
func (e wsEvents) Status(id string, ex testexec.Execution) { e.hub.PublishStatus(id, ex) }
func (e wsEvents) Complete(ex testexec.Execution)          { e.hub.PublishCompletion(ex.ID, ex) }

// metricsProvider adapts the Cluster Monitor metrics service to
// ws.MetricsProvider so a newly connected /ws/clustermon client gets the cached
// latest snapshot as its first frame.
type metricsProvider struct{ svc *clustermon.MetricsService }

func (m metricsProvider) LatestMetrics() (any, bool) {
	snap := m.svc.Latest()
	if snap == nil {
		return nil, false
	}
	return snap, true
}

// execSnapshots adapts the execution store to ws.SnapshotProvider, supplying the
// terminal/late-joiner fallback (persisted log + final status).
type execSnapshots struct{ store testexec.ExecutionStore }

func (s execSnapshots) Snapshot(id string) (ws.ExecutionSnapshot, bool) {
	ex, ok := s.store.Get(id)
	if !ok {
		return ws.ExecutionSnapshot{}, false
	}
	logs := ""
	if ex.LogPath != "" {
		if b, err := os.ReadFile(ex.LogPath); err == nil {
			logs = string(b)
		}
	}
	return ws.ExecutionSnapshot{Terminal: ex.Status.Terminal(), Logs: logs, Status: ex}, true
}
