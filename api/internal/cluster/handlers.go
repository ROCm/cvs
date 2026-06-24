package cluster

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
)

// Handler serves the saved-cluster catalog REST API.
type Handler struct {
	svc    *Service
	logger *slog.Logger
}

// NewHandler constructs a cluster handler.
func NewHandler(svc *Service, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{svc: svc, logger: logger}
}

// Routes mounts the cluster routes onto the given router (under /api/v1).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/clusters", h.list)
	r.Post("/clusters", h.create)
	r.Get("/clusters/{id}", h.get)
	r.Get("/clusters/{id}/content", h.content)
	r.Put("/clusters/{id}", h.update)
	r.Delete("/clusters/{id}", h.del)
}

func (h *Handler) list(w http.ResponseWriter, _ *http.Request) {
	cs := h.svc.List()
	writeJSON(w, http.StatusOK, map[string]any{"clusters": cs, "total": len(cs)})
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	c, ok := h.svc.Get(chi.URLParam(r, "id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "cluster not found"})
		return
	}
	writeJSON(w, http.StatusOK, c)
}

// content streams the generated cluster_json file as-is so the UI can display it.
func (h *Handler) content(w http.ResponseWriter, r *http.Request) {
	b, ok, err := h.svc.Content(chi.URLParam(r, "id"))
	switch {
	case !ok:
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "cluster not found"})
	case errors.Is(err, ErrNoFile):
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "cluster file not found"})
	case err != nil:
		h.logger.Error("cluster_content_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read cluster file"})
	default:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(b)
	}
}

type createRequest struct {
	Name     string   `json:"name"`
	Nodes    []string `json:"nodes"`
	HeadNode string   `json:"head_node"`
}

func (h *Handler) create(w http.ResponseWriter, r *http.Request) {
	var req createRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	c, err := h.svc.Create(r.Context(), CreateParams{
		Name:     strings.TrimSpace(req.Name),
		Nodes:    req.Nodes,
		HeadNode: strings.TrimSpace(req.HeadNode),
	})
	if err != nil {
		h.writeCreateErr(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, c)
}

// writeCreateErr maps validation errors to 400 and generation/persist failures
// to 5xx.
func (h *Handler) writeCreateErr(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrNoName), errors.Is(err, ErrNoNodes), errors.Is(err, ErrNotSubset):
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
	case errors.Is(err, ErrNoInventory):
		writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
	default:
		h.logger.Error("cluster_create_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create cluster"})
	}
}

func (h *Handler) update(w http.ResponseWriter, r *http.Request) {
	var req createRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	c, ok, err := h.svc.Update(r.Context(), chi.URLParam(r, "id"), UpdateParams{
		Name:     strings.TrimSpace(req.Name),
		Nodes:    req.Nodes,
		HeadNode: strings.TrimSpace(req.HeadNode),
	})
	if !ok && err == nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "cluster not found"})
		return
	}
	if err != nil {
		h.writeCreateErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, c)
}

func (h *Handler) del(w http.ResponseWriter, r *http.Request) {
	ok, err := h.svc.Delete(chi.URLParam(r, "id"))
	if err != nil {
		h.logger.Error("cluster_delete_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to delete cluster"})
		return
	}
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "cluster not found"})
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
