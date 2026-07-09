package main

// FleetManager owns the lifecycle of the pssh connection pool (the "fleet")
// and the app-level probeReady gate that the UI polls before unlocking cluster
// tiles.
//
// It is constructed once in main and wired into the HTTP router via its method
// values (Set, Clear, ProbeStatus). The Prober field must be set after
// construction because inventory.NewInventoryProber takes fm.Get as an
// argument, creating a forward reference that would otherwise require a second
// parameter to New.

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ROCm/cvs/api/internal/inventory"
	httptransport "github.com/ROCm/cvs/api/internal/transport/http"
	"github.com/ROCm/cvs/api/pkg/pssh"
)

// FleetManager manages the pssh.Pool singleton and the probe-ready UI gate.
type FleetManager struct {
	mu              sync.Mutex
	pool            *pssh.Pool
	probeReady      atomic.Bool
	reprobeInterval time.Duration

	invStore inventory.Store
	keyStore *inventory.KeyStore
	logger   *slog.Logger

	// Prober is set after construction (see main.go) because it depends on
	// fm.Get, which would otherwise create a circular dependency.
	Prober inventory.Prober
}

// NewFleetManager creates a FleetManager. Set fm.Prober before calling fm.Set.
func NewFleetManager(
	invStore inventory.Store,
	keyStore *inventory.KeyStore,
	reprobeInterval time.Duration,
	logger *slog.Logger,
) *FleetManager {
	return &FleetManager{
		invStore:        invStore,
		keyStore:        keyStore,
		reprobeInterval: reprobeInterval,
		logger:          logger,
	}
}

// Get returns the current pool, or nil if no inventory has been configured.
// Safe for concurrent use.
func (fm *FleetManager) Get() *pssh.Pool {
	fm.mu.Lock()
	defer fm.mu.Unlock()
	return fm.pool
}

// Clear tears down the current pool and resets probe state. Called when the
// inventory is deleted via DELETE /inventory.
func (fm *FleetManager) Clear() {
	fm.mu.Lock()
	if fm.pool != nil {
		fm.pool.Close()
		fm.pool = nil
	}
	fm.mu.Unlock()
	fm.probeReady.Store(false)
}

// Set creates or incrementally updates the fleet for the given inventory.
//
//   - First call (no pool yet): creates a full pssh.Pool, sets probeReady=false,
//     then probes all nodes in the background (probeReady flips true when done).
//   - Subsequent calls: diffs old vs new node list; only probes added nodes and
//     drops removed ones. probeReady is never reset — the existing fleet stays warm.
func (fm *FleetManager) Set(inv inventory.Inventory) {
	if inv.AuthMethod == inventory.AuthPassword || inv.KeyName == "" {
		fm.logger.Error("pssh_init_skipped", "reason", "password auth or no key configured")
		return
	}
	keyPath, err := fm.keyStore.Path(inv.KeyName)
	if err != nil {
		fm.logger.Error("pssh_init_failed", "err", err, "key_name", inv.KeyName)
		return
	}

	// ── Incremental path ────────────────────────────────────────────────
	// A pool already exists: compute the diff and only touch changed nodes.
	// Existing connections stay open; probeReady is never reset.
	fm.mu.Lock()
	existing := fm.pool
	fm.mu.Unlock()

	if existing != nil {
		added, removed := diffNodeLists(existing.All(), inv.Nodes)
		if len(added) == 0 && len(removed) == 0 {
			return // no-op save (e.g. only SSH settings changed)
		}
		if len(removed) > 0 {
			existing.RemoveNodes(removed)
		}
		if len(added) > 0 {
			existing.AddNodes(added)
			go fm.probeSubsetAndEnrich(existing, added, inv)
		}
		return // probeReady stays true — fleet remains warm
	}

	// ── First-time path ──────────────────────────────────────────────────
	// No pool yet: create one and do a full probe before unlocking tiles.
	p, err := pssh.New(inv.Username, keyPath, inv.Nodes, 0, 0, fm.reprobeInterval, 0, fm.logger)
	if err != nil {
		fm.logger.Error("pssh_init_failed", "err", err)
		return
	}
	fm.mu.Lock()
	fm.pool = p
	fm.mu.Unlock()
	fm.probeReady.Store(false) // tiles block until probeAllAndEnrich finishes

	go fm.probeAllAndEnrich(p, inv)
}

// ProbeStatus returns the current probe state for GET /api/v1/inventory/probe-status.
func (fm *FleetManager) ProbeStatus() httptransport.ProbeStatusResponse {
	p := fm.Get()
	if p == nil {
		return httptransport.ProbeStatusResponse{}
	}
	return httptransport.ProbeStatusResponse{
		Ready:       fm.probeReady.Load(),
		Reachable:   len(p.Reachable()),
		Unreachable: len(p.Unreachable()),
		Total:       len(p.All()),
	}
}

// probeAllAndEnrich runs a full fleet probe then enriches the inventory with
// per-node GPU/ROCm basic info. Called in a goroutine on first-time setup.
func (fm *FleetManager) probeAllAndEnrich(pool *pssh.Pool, savedInv inventory.Inventory) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	pool.Probe(ctx)
	fm.probeReady.Store(true)
	fm.logger.Info("pssh_pool_ready",
		"reachable", len(pool.Reachable()),
		"unreachable", len(pool.Unreachable()),
		"total", len(pool.All()),
	)
	if fm.Prober == nil {
		return
	}
	statuses, err := fm.Prober.Probe(ctx, savedInv)
	if err != nil {
		fm.logger.Warn("auto_probe_failed", "err", err)
		return
	}
	savedInv.Statuses = statuses
	if _, err := fm.invStore.Save(savedInv); err != nil {
		fm.logger.Warn("auto_probe_save_failed", "err", err)
	}
	fm.logger.Info("auto_probe_done", "nodes", len(statuses))
}

// probeSubsetAndEnrich runs a targeted probe on added nodes and merges the
// new results into the persisted inventory. Called in a goroutine on incremental
// saves.
func (fm *FleetManager) probeSubsetAndEnrich(pool *pssh.Pool, addedHosts []string, savedInv inventory.Inventory) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	pool.ProbeSubset(ctx, addedHosts)
	fm.logger.Info("pssh_subset_ready",
		"added", len(addedHosts),
		"reachable", len(pool.Reachable()),
		"total", len(pool.All()),
	)
	if fm.Prober == nil {
		return
	}
	// Probe only the newly added nodes for basic info.
	deltaInv := savedInv
	deltaInv.Nodes = addedHosts
	statuses, err := fm.Prober.Probe(ctx, deltaInv)
	if err != nil {
		fm.logger.Warn("auto_probe_delta_failed", "err", err)
		return
	}
	// Merge delta statuses into the full inventory document and persist.
	fullInv, ok, err := fm.invStore.Get()
	if err != nil || !ok {
		return
	}
	byHost := make(map[string]inventory.NodeStatus, len(fullInv.Statuses))
	for _, s := range fullInv.Statuses {
		byHost[s.Host] = s
	}
	for _, s := range statuses {
		byHost[s.Host] = s
	}
	merged := make([]inventory.NodeStatus, 0, len(byHost))
	for _, s := range byHost {
		merged = append(merged, s)
	}
	fullInv.Statuses = merged
	if _, err := fm.invStore.Save(fullInv); err != nil {
		fm.logger.Warn("auto_probe_delta_save_failed", "err", err)
	}
	fm.logger.Info("auto_probe_delta_done", "nodes", len(statuses))
}

// diffNodeLists returns nodes added to (in newList but not oldList) and
// removed from (in oldList but not newList) the fleet. Order in the added
// slice matches newList; removed order is unspecified.
func diffNodeLists(oldList, newList []string) (added, removed []string) {
	oldSet := make(map[string]struct{}, len(oldList))
	for _, h := range oldList {
		oldSet[h] = struct{}{}
	}
	newSet := make(map[string]struct{}, len(newList))
	for _, h := range newList {
		newSet[h] = struct{}{}
	}
	for _, h := range newList {
		if _, ok := oldSet[h]; !ok {
			added = append(added, h)
		}
	}
	for _, h := range oldList {
		if _, ok := newSet[h]; !ok {
			removed = append(removed, h)
		}
	}
	return
}
