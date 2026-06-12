package inventory

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
)

// maxKeyUploadBytes caps an uploaded SSH key (keys are a few KB at most).
const maxKeyUploadBytes = 1 << 20 // 1 MiB

// probeTimeout bounds a full connectivity + basic-info sweep.
const probeTimeout = 120 * time.Second

// Handler serves the inventory + SSH key REST API that gates every tile.
type Handler struct {
	store  Store
	keys   *KeyStore
	prober Prober
	logger *slog.Logger

	// OnSave is called asynchronously after a successful inventory save.
	// The callback receives the persisted Inventory. May be nil.
	OnSave func(Inventory)
	// OnClear is called asynchronously after a successful inventory delete.
	// May be nil.
	OnClear func()
}

// NewHandler constructs an inventory handler. keys may be nil to disable uploads;
// prober may be nil to disable the connectivity probe endpoint.
func NewHandler(s Store, keys *KeyStore, prober Prober, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{store: s, keys: keys, prober: prober, logger: logger}
}

// Routes mounts the inventory routes onto the given router (under /api/v1).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/inventory", h.get)
	r.Post("/inventory", h.save)
	r.Delete("/inventory", h.deleteInventory)
	r.Post("/inventory/probe", h.probe)
	r.Get("/inventory/keys", h.listKeys)
	r.Post("/inventory/keys", h.uploadKey)
}

func (h *Handler) get(w http.ResponseWriter, _ *http.Request) {
	inv, ok, err := h.store.Get()
	if err != nil {
		h.logger.Error("inventory_get_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read inventory"})
		return
	}
	resp := map[string]any{"configured": ok}
	if ok {
		resp["inventory"] = inv
	}
	writeJSON(w, http.StatusOK, resp)
}

// deleteInventory removes the saved fleet inventory. Uploaded SSH keys are
// unchanged (see /inventory/keys to manage those).
func (h *Handler) deleteInventory(w http.ResponseWriter, _ *http.Request) {
	if err := h.store.Clear(); err != nil {
		h.logger.Error("inventory_clear_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to clear inventory"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"configured": false})
	if h.OnClear != nil {
		go h.OnClear()
	}
}

type saveRequest struct {
	Nodes      []string   `json:"nodes"`
	Username   string     `json:"username"`
	AuthMethod AuthMethod `json:"auth_method"`
	KeyName    string     `json:"key_name"`
	JumpHost   *JumpHost  `json:"jump_host"`
}

func (h *Handler) save(w http.ResponseWriter, r *http.Request) {
	var req saveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	nodes := NormalizeNodes(req.Nodes)
	if len(nodes) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "at least one node is required"})
		return
	}
	if strings.TrimSpace(req.Username) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "username is required"})
		return
	}

	method := req.AuthMethod
	if method == "" {
		method = AuthKey
	}

	// Carry forward probe results for nodes that remain in the new list so
	// that removing or re-ordering nodes doesn't reset every node to "probing…".
	// New nodes have no entry and will be probed incrementally by FleetManager.
	var keptStatuses []NodeStatus
	if existing, ok, err := h.store.Get(); err == nil && ok && len(existing.Statuses) > 0 {
		keep := make(map[string]struct{}, len(nodes))
		for _, n := range nodes {
			keep[n] = struct{}{}
		}
		for _, s := range existing.Statuses {
			if _, ok := keep[s.Host]; ok {
				keptStatuses = append(keptStatuses, s)
			}
		}
	}

	inv := Inventory{
		Nodes:      nodes,
		Username:   strings.TrimSpace(req.Username),
		AuthMethod: method,
		KeyName:    strings.TrimSpace(req.KeyName),
		JumpHost:   req.JumpHost,
		Statuses:   keptStatuses,
	}
	saved, err := h.store.Save(inv)
	if err != nil {
		h.logger.Error("inventory_save_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to save inventory"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"configured": true, "inventory": saved})
	if h.OnSave != nil {
		go h.OnSave(saved)
	}
}

// probe runs a connectivity + basic-info sweep against the saved inventory and
// persists the resulting per-node statuses.
func (h *Handler) probe(w http.ResponseWriter, r *http.Request) {
	if h.prober == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "probe not configured"})
		return
	}
	inv, ok, err := h.store.Get()
	if err != nil {
		h.logger.Error("inventory_get_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read inventory"})
		return
	}
	if !ok {
		writeJSON(w, http.StatusConflict, map[string]string{"error": "no inventory saved yet"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), probeTimeout)
	defer cancel()
	statuses, err := h.prober.Probe(ctx, inv)
	if err != nil {
		h.logger.Error("inventory_probe_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "probe failed"})
		return
	}

	inv.Statuses = statuses
	saved, err := h.store.Save(inv)
	if err != nil {
		h.logger.Error("inventory_save_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to persist probe results"})
		return
	}
	h.logger.Info("inventory_probed", "nodes", len(statuses))
	writeJSON(w, http.StatusOK, map[string]any{"configured": true, "inventory": saved})
}

func (h *Handler) listKeys(w http.ResponseWriter, _ *http.Request) {
	if h.keys == nil {
		writeJSON(w, http.StatusOK, map[string]any{"keys": []string{}})
		return
	}
	names, err := h.keys.List()
	if err != nil {
		h.logger.Error("key_list_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to list keys"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": names})
}

func (h *Handler) uploadKey(w http.ResponseWriter, r *http.Request) {
	if h.keys == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "key storage not configured"})
		return
	}
	if err := r.ParseMultipartForm(maxKeyUploadBytes); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid multipart form"})
		return
	}
	file, header, err := r.FormFile("key")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing 'key' file field"})
		return
	}
	defer file.Close()

	name, err := h.keys.Save(header.Filename, http.MaxBytesReader(w, file, maxKeyUploadBytes))
	if err != nil {
		h.logger.Error("key_upload_failed", "name", header.Filename, "err", err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "failed to store key"})
		return
	}
	path, _ := h.keys.Path(name)
	writeJSON(w, http.StatusCreated, map[string]any{"key_name": name, "path": path})
}

// NormalizeNodes splits, trims, drops comments/blanks, and de-duplicates node
// entries. It accepts both a list of hosts and multi-line textarea payloads.
func NormalizeNodes(in []string) []string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(in))
	for _, raw := range in {
		for _, line := range strings.Split(raw, "\n") {
			n := strings.TrimSpace(line)
			if n == "" || strings.HasPrefix(n, "#") {
				continue
			}
			if _, dup := seen[n]; dup {
				continue
			}
			seen[n] = struct{}{}
			out = append(out, n)
		}
	}
	return out
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
