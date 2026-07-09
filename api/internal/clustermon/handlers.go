// Package clustermon serves the Cluster Monitor tile. At S6 it presents the
// shared fleet inventory (F1) plus the latest connectivity/basic-info probe
// results (F2) as a node grid; later slices add live GPU/NIC collectors on the
// shared SSH pool.
package clustermon

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/ROCm/cvs/api/internal/inventory"
)

// collectTimeout bounds a single on-demand metrics sweep (GPU + NIC in parallel).
// With 165 nodes and maxParallelCollect=64, each collector needs ceil(165/64)*max_node_time.
// Both run concurrently, so 120s covers up to ~40s/node.
const collectTimeout = 120 * time.Second

// Handler serves the Cluster Monitor REST API.
type Handler struct {
	store    inventory.Store
	metrics  *MetricsService
	software *SoftwareService
	logs     *LogsService
	topology *TopologyService
	logger   *slog.Logger
}

// NewHandler constructs a Cluster Monitor handler backed by the shared
// inventory store. Any service field may be nil to disable its routes.
func NewHandler(s inventory.Store, metrics *MetricsService, software *SoftwareService, logs *LogsService, topology *TopologyService, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{store: s, metrics: metrics, software: software, logs: logs, topology: topology, logger: logger}
}

// Routes mounts the tile's routes onto the given router (expected under /api/v1).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/clustermon/nodes", h.nodes)
	r.Get("/clustermon/nodes/{nodeID}", h.nodeDetail)
	if h.metrics != nil {
		r.Get("/clustermon/metrics/latest", h.latestMetrics)
		r.Post("/clustermon/metrics/collect", h.collectMetrics)
		r.Get("/clustermon/cluster/status", h.clusterStatus)
	}
	if h.software != nil {
		r.Get("/clustermon/software/gpu", h.gpuSoftware)
		r.Get("/clustermon/software/nic/devlink", h.nicDevlink)
	}
	if h.logs != nil {
		r.Get("/clustermon/logs/dmesg", h.dmesgLogs)
		r.Get("/clustermon/logs/search", h.searchLogs)
	}
	if h.topology != nil {
		r.Get("/clustermon/topology/lldp", h.topologyLLDP)
	}
}

// dmesgLogs returns the cached log snapshot immediately, triggering a background
// collection if the cache is absent or stale. Returns 204 while the first sweep
// is still in progress so the UI can show a "collecting" placeholder.
func (h *Handler) dmesgLogs(w http.ResponseWriter, r *http.Request) {
	snap, err := h.logs.Logs(r.Context())
	if err != nil {
		h.logger.Error("dmesg_logs_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	if snap == nil {
		// Background collection triggered; no data yet.
		w.WriteHeader(http.StatusNoContent)
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// searchLogs runs a validated ad-hoc grep search against dmesg on each node.
func (h *Handler) searchLogs(w http.ResponseWriter, r *http.Request) {
	grep := r.URL.Query().Get("grep_command")
	ctx, cancel := context.WithTimeout(r.Context(), logsCollectTimeout)
	defer cancel()
	res, err := h.logs.Search(ctx, grep)
	if err != nil {
		var bad *invalidGrepError
		if errors.As(err, &bad) {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": bad.Error()})
			return
		}
		h.logger.Error("logs_search_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// gpuSoftware returns the cached GPU software snapshot immediately (never blocks).
// When the cache is cold a background collection is triggered; the response
// includes "collecting":true so the UI can poll and show a progress state.
func (h *Handler) gpuSoftware(w http.ResponseWriter, r *http.Request) {
	snap, err := h.software.GPUSoftware(r.Context())
	if err != nil {
		h.logger.Error("gpu_software_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	if snap == nil {
		writeJSON(w, http.StatusOK, map[string]any{"collecting": true, "nodes": []any{}})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// nicDevlink returns the cached NIC devlink snapshot immediately (never blocks).
func (h *Handler) nicDevlink(w http.ResponseWriter, r *http.Request) {
	snap, err := h.software.NICDevlink(r.Context())
	if err != nil {
		h.logger.Error("nic_devlink_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	if snap == nil {
		writeJSON(w, http.StatusOK, map[string]any{"collecting": true, "nodes": []any{}})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// latestMetrics returns the last cached GPU metrics snapshot (may be empty).
func (h *Handler) latestMetrics(w http.ResponseWriter, _ *http.Request) {
	snap := h.metrics.Latest()
	if snap == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"collected_at": nil,
			"gpu":          []NodeGPUMetrics{},
			"nic":          []NodeNICMetrics{},
		})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// collectMetrics runs an on-demand GPU metrics sweep over the reachable nodes,
// caches it, and returns the fresh snapshot.
func (h *Handler) collectMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), collectTimeout)
	defer cancel()
	snap, err := h.metrics.Collect(ctx)
	if err != nil {
		h.logger.Error("gpu_metrics_collect_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// NodeView is one node's row in the monitor grid: its identity plus the last
// known probe result. Probed is false for a node that has never been swept.
type NodeView struct {
	Host        string     `json:"host"`
	Probed      bool       `json:"probed"`
	Reachable   bool       `json:"reachable"`
	GPUType     string     `json:"gpu_type,omitempty"`
	GPUCount    int        `json:"gpu_count,omitempty"`
	ROCmVersion string     `json:"rocm_version,omitempty"`
	Error       string     `json:"error,omitempty"`
	CheckedAt   *time.Time `json:"checked_at,omitempty"`
}

func (h *Handler) nodes(w http.ResponseWriter, _ *http.Request) {
	inv, ok, err := h.store.Get()
	if err != nil {
		h.logger.Error("clustermon_inventory_get_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read inventory"})
		return
	}
	if !ok {
		// Tiles are gated on a saved inventory, but stay defensive.
		writeJSON(w, http.StatusOK, map[string]any{"nodes": []NodeView{}, "total": 0, "configured": false})
		return
	}

	// Index the latest probe statuses by host for an O(1) merge.
	byHost := make(map[string]inventory.NodeStatus, len(inv.Statuses))
	for _, s := range inv.Statuses {
		byHost[s.Host] = s
	}

	views := make([]NodeView, 0, len(inv.Nodes))
	for _, host := range inv.Nodes {
		v := NodeView{Host: host}
		if s, found := byHost[host]; found {
			v.Probed = true
			v.Reachable = s.Reachable
			v.GPUType = s.GPUType
			v.GPUCount = s.GPUCount
			v.ROCmVersion = s.ROCmVersion
			v.Error = s.Error
			v.CheckedAt = s.CheckedAt
		}
		views = append(views, v)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"nodes":      views,
		"total":      len(views),
		"configured": true,
		"updated_at": inv.UpdatedAt,
	})
}

// clusterStatus returns a lightweight fleet-level aggregate over the latest
// cached metrics snapshot. Fixed-size response (~250 bytes) at any fleet scale.
func (h *Handler) clusterStatus(w http.ResponseWriter, _ *http.Request) {
	inv, ok, _ := h.store.Get()
	totalNodes := 0
	if ok {
		totalNodes = len(inv.Nodes)
	}

	snap := h.metrics.Latest()
	if snap == nil || len(snap.GPU) == 0 {
		writeJSON(w, http.StatusOK, map[string]any{
			"total_nodes":        totalNodes,
			"healthy_nodes":      0,
			"unhealthy_nodes":    0,
			"unreachable_nodes":  0,
			"total_gpus":         0,
			"avg_gpu_util_pct":   0.0,
			"avg_gpu_mem_pct":    0.0,
			"avg_gpu_temp_c":     0.0,
			"collected_at":       nil,
			"metrics_available":  false,
		})
		return
	}

	var healthy, unhealthy, unreachable int
	var totalGPUs, gpuCount int
	var sumUtil, sumMem, sumTemp float64

	for _, node := range snap.GPU {
		if node.Error != "" {
			unreachable++
			continue
		}
		nodeHealthy := true
		for _, g := range node.GPUs {
			totalGPUs++
			sumUtil += g.UtilizationPct
			sumMem += g.MemUsedPct
			sumTemp += g.TempHotspotC
			gpuCount++
			if g.ECC != nil && g.ECC.Uncorrectable > 0 {
				nodeHealthy = false
			}
			if g.TempHotspotC >= 85 {
				nodeHealthy = false
			}
		}
		if nodeHealthy {
			healthy++
		} else {
			unhealthy++
		}
	}
	// Nodes present in the inventory but absent from the snapshot are unreachable.
	if notCollected := totalNodes - len(snap.GPU); notCollected > 0 {
		unreachable += notCollected
	}

	avg := func(sum float64, n int) float64 {
		if n == 0 {
			return 0
		}
		return sum / float64(n)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"total_nodes":        totalNodes,
		"healthy_nodes":      healthy,
		"unhealthy_nodes":    unhealthy,
		"unreachable_nodes":  unreachable,
		"total_gpus":         totalGPUs,
		"avg_gpu_util_pct":   avg(sumUtil, gpuCount),
		"avg_gpu_mem_pct":    avg(sumMem, gpuCount),
		"avg_gpu_temp_c":     avg(sumTemp, gpuCount),
		"collected_at":       snap.CollectedAt,
		"metrics_available":  true,
	})
}

// nodeDetail returns the GPU + NIC details for a single node from the latest
// cached snapshot, keyed by host (URL param :nodeID).
func (h *Handler) nodeDetail(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "nodeID")
	snap := h.metrics.Latest()
	if snap == nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no metrics snapshot available yet"})
		return
	}
	for _, n := range snap.GPU {
		if n.Host != nodeID {
			continue
		}
		var links []RDMALink
		for _, nic := range snap.NIC {
			if nic.Host == nodeID {
				links = nic.RDMALinks
				break
			}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"host":       n.Host,
			"gpus":       n.GPUs,
			"rdma_links": links,
			"error":      n.Error,
		})
		return
	}
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "node not found in latest metrics"})
}

// topologyLLDP returns LLDP neighbor data for all inventory nodes. The
// response is cached for lldpTTL (3 min) with a stale-on-error fallback.
// Returns 200 with an empty nodes list when lldpd is not installed.
func (h *Handler) topologyLLDP(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), lldpCollectTimeout)
	defer cancel()
	snap, err := h.topology.LLDP(ctx)
	if err != nil {
		h.logger.Error("topology_lldp_failed", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
