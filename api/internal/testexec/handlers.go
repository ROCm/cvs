package testexec

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"
)

// Handler serves the Test Execution REST API.
type Handler struct {
	lister  SuiteLister
	configs *ConfigService
	logger  *slog.Logger
}

// NewHandler constructs a Test Execution handler.
func NewHandler(lister SuiteLister, configs *ConfigService, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{lister: lister, configs: configs, logger: logger}
}

// Routes mounts the tile's routes onto the given router (expected under /api/v1).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/suites", h.listSuites)
	r.Get("/suites/{suite}/examples", h.listExamples)
	r.Get("/suites/{suite}/schema", h.suiteSchema)
}

func (h *Handler) listSuites(w http.ResponseWriter, r *http.Request) {
	suites, err := h.lister.List(r.Context())
	if err != nil {
		h.logger.Error("list_suites_failed", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to list suites"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"suites": suites, "total": len(suites)})
}

func (h *Handler) listExamples(w http.ResponseWriter, r *http.Request) {
	suite, ok := h.resolveSuite(w, r)
	if !ok {
		return
	}
	examples, err := h.configs.Examples(suite)
	if err != nil {
		h.logger.Error("list_examples_failed", "suite", suite.Name, "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to list examples"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"suite":    suite.Name,
		"examples": examples,
	})
}

func (h *Handler) suiteSchema(w http.ResponseWriter, r *http.Request) {
	suite, ok := h.resolveSuite(w, r)
	if !ok {
		return
	}
	exampleName := r.URL.Query().Get("example")
	value, ref, err := h.configs.Load(suite, exampleName)
	if err != nil {
		if IsNoConfig(err) {
			writeJSON(w, http.StatusOK, map[string]any{
				"suite":       suite.Name,
				"test_name":   suite.Name,
				"has_config":  false,
				"config_path": nil,
				"schema":      []Field{},
				"example":     nil,
			})
			return
		}
		h.logger.Error("suite_schema_failed", "suite", suite.Name, "err", err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"suite":       suite.Name,
		"test_name":   suite.Name,
		"has_config":  true,
		"config_path": ref.Path,
		"format":      ref.Format,
		"schema":      InferSchema(value),
		"example":     value,
	})
}

// resolveSuite looks up the {suite} path param against the suite list.
func (h *Handler) resolveSuite(w http.ResponseWriter, r *http.Request) (Suite, bool) {
	name := chi.URLParam(r, "suite")
	suites, err := h.lister.List(r.Context())
	if err != nil {
		h.logger.Error("list_suites_failed", "err", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to list suites"})
		return Suite{}, false
	}
	for _, s := range suites {
		if s.Name == name {
			return s, true
		}
	}
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown suite: " + name})
	return Suite{}, false
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
