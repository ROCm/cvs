package testexec

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/ROCm/cvs/api/internal/store"
)

// ClusterResolver turns a saved cluster_id into its generated --cluster_file
// path (decouples this tile from the cluster package).
type ClusterResolver interface {
	FilePath(id string) (string, bool)
}

// ExecutionDeps wires the (optional) execution capability into the handler.
// When nil, the execute/executions routes are not mounted.
type ExecutionDeps struct {
	Store    ExecutionStore
	Executor *Executor
	Clusters ClusterResolver
	// Dir is where per-execution artifacts (config + logs) are written.
	Dir string
}

// Handler serves the Test Execution REST API.
type Handler struct {
	lister  SuiteLister
	configs *ConfigService
	logger  *slog.Logger
	exec    *ExecutionDeps // optional
}

// NewHandler constructs a Test Execution handler. exec may be nil to mount only
// the read-only discovery routes (suites/schema), e.g. in narrow tests.
func NewHandler(lister SuiteLister, configs *ConfigService, logger *slog.Logger, exec *ExecutionDeps) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{lister: lister, configs: configs, logger: logger, exec: exec}
}

// Routes mounts the tile's routes onto the given router (expected under /api/v1).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/suites", h.listSuites)
	r.Get("/suites/catalog", h.catalog)
	r.Get("/suites/{suite}/examples", h.listExamples)
	r.Get("/suites/{suite}/schema", h.suiteSchema)

	if h.exec != nil {
		r.Post("/tests/execute", h.execute)
		r.Get("/executions", h.listExecutions)
		r.Get("/executions/{id}", h.getExecution)
		r.Post("/executions/{id}/rerun", h.rerun)
		r.Get("/executions/{id}/logs", h.executionLogs)
		// Serve the generated cluster_file used for the run (it lives in the
		// cluster catalog dir, outside the execution's artifact dir).
		r.Get("/executions/{id}/cluster", h.executionCluster)
		// Serve per-execution artifacts (pytest report.html + its <suite>_html
		// sidecar of per-test logs, pytest.log, the zip bundle, etc.) so the UI
		// can open the report and its links resolve.
		r.Get("/executions/{id}/artifacts/*", h.artifacts)
	}
}

// executionView augments a stored Execution with derived, non-persisted fields
// the UI needs (whether a pytest HTML report was produced).
type executionView struct {
	Execution
	HasReport bool `json:"has_report"`
}

// reportPath is the deterministic location of the pytest HTML report for an
// execution (written by the runner into the per-execution dir).
func reportPath(ex Execution) string {
	if ex.LogPath == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(ex.LogPath), "report.html")
}

func toView(ex Execution) executionView {
	v := executionView{Execution: ex}
	if p := reportPath(ex); p != "" {
		if _, err := os.Stat(p); err == nil {
			v.HasReport = true
		}
	}
	return v
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

func (h *Handler) catalog(w http.ResponseWriter, r *http.Request) {
	cat, err := h.configs.Catalog()
	if err != nil {
		h.logger.Error("catalog_failed", "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to build config catalog"})
		return
	}
	writeJSON(w, http.StatusOK, cat)
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

type executeRequest struct {
	Suite     string          `json:"suite"`
	ClusterID string          `json:"cluster_id"`
	Config    json.RawMessage `json:"config"` // optional edited config object
}

// execute validates the request, persists a queued Execution, and submits it to
// the worker pool, returning 202 with the new record.
func (h *Handler) execute(w http.ResponseWriter, r *http.Request) {
	var req executeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	// Suite must exist.
	suites, err := h.lister.List(r.Context())
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to list suites"})
		return
	}
	if !suiteExists(suites, req.Suite) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown suite: " + req.Suite})
		return
	}

	// Cluster must resolve to a generated cluster_file (cvs run requires it).
	if req.ClusterID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "cluster_id is required"})
		return
	}

	ex, status, err := h.startExecution(req.Suite, req.ClusterID, req.Config)
	if err != nil {
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, status, ex)
}

// rerun launches a fresh execution from a prior one, reusing its suite, cluster,
// and staged config. The cluster is re-resolved at rerun time (so a deleted
// cluster is rejected and an edited cluster's current file is used).
func (h *Handler) rerun(w http.ResponseWriter, r *http.Request) {
	src, ok := h.exec.Store.Get(chi.URLParam(r, "id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "execution not found"})
		return
	}

	// Suite must still exist.
	suites, err := h.lister.List(r.Context())
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to list suites"})
		return
	}
	if !suiteExists(suites, src.Suite) {
		writeJSON(w, http.StatusConflict, map[string]string{"error": "suite no longer exists: " + src.Suite})
		return
	}

	// Replay the original config bytes (if any were staged).
	var config json.RawMessage
	if src.ConfigPath != "" {
		data, err := os.ReadFile(src.ConfigPath)
		if err != nil {
			h.logger.Error("rerun_config_read_failed", "id", src.ID, "err", err)
			writeJSON(w, http.StatusConflict, map[string]string{"error": "original config is no longer available"})
			return
		}
		config = json.RawMessage(data)
	}

	ex, status, err := h.startExecution(src.Suite, src.ClusterID, config)
	if err != nil {
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, status, ex)
}

// startExecution re-resolves the cluster, stages the config, persists a queued
// Execution, and submits it to the worker pool. On failure it returns the HTTP
// status to surface alongside the error.
func (h *Handler) startExecution(suite, clusterID string, config json.RawMessage) (Execution, int, error) {
	clusterFile, ok := h.exec.Clusters.FilePath(clusterID)
	if !ok || clusterFile == "" {
		return Execution{}, http.StatusBadRequest, fmt.Errorf("unknown cluster_id: %s", clusterID)
	}

	id := store.NewID()
	dir := filepath.Join(h.exec.Dir, id)

	configPath, err := h.writeConfig(dir, config)
	if err != nil {
		h.logger.Error("execution_config_write_failed", "id", id, "err", err)
		return Execution{}, http.StatusInternalServerError, fmt.Errorf("failed to stage config")
	}

	logPath := filepath.Join(dir, "output.log")
	ex := Execution{
		ID:          id,
		Suite:       suite,
		ClusterID:   clusterID,
		ClusterFile: clusterFile,
		ConfigPath:  configPath,
		Status:      StatusQueued,
		LogPath:     logPath,
		CreatedAt:   timeNow(),
	}
	if err := h.exec.Store.Save(ex); err != nil {
		h.logger.Error("execution_save_failed", "id", id, "err", err)
		return Execution{}, http.StatusInternalServerError, fmt.Errorf("failed to record execution")
	}

	if err := h.exec.Executor.Submit(Job{
		ID:          id,
		Suite:       suite,
		ClusterFile: clusterFile,
		ConfigFile:  configPath,
		LogPath:     logPath,
	}); err != nil {
		ex.Status = StatusError
		ex.Error = err.Error()
		_ = h.exec.Store.Save(ex)
		return Execution{}, http.StatusServiceUnavailable, err
	}

	return ex, http.StatusAccepted, nil
}

// writeConfig persists a provided config object to dir/config.json, returning
// its path (empty when no config was provided).
func (h *Handler) writeConfig(dir string, raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "config.json")
	// Re-indent for readability; the bytes are already valid JSON.
	var pretty any
	if err := json.Unmarshal(raw, &pretty); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(pretty, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func (h *Handler) listExecutions(w http.ResponseWriter, _ *http.Request) {
	es := h.exec.Store.List()
	views := make([]executionView, len(es))
	for i, ex := range es {
		views[i] = toView(ex)
	}
	writeJSON(w, http.StatusOK, map[string]any{"executions": views, "total": len(views)})
}

func (h *Handler) getExecution(w http.ResponseWriter, r *http.Request) {
	ex, ok := h.exec.Store.Get(chi.URLParam(r, "id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "execution not found"})
		return
	}
	writeJSON(w, http.StatusOK, toView(ex))
}

// artifacts serves files from an execution's directory (report.html and its
// sidecar, pytest.log, zip bundle). Path traversal outside the dir is rejected.
func (h *Handler) artifacts(w http.ResponseWriter, r *http.Request) {
	ex, ok := h.exec.Store.Get(chi.URLParam(r, "id"))
	if !ok || ex.LogPath == "" {
		http.NotFound(w, r)
		return
	}
	base := filepath.Clean(filepath.Dir(ex.LogPath))
	// Clean the wildcard against root to neutralize any ".." segments.
	rel := filepath.Clean("/" + chi.URLParam(r, "*"))
	full := filepath.Join(base, rel)
	if full != base && !strings.HasPrefix(full, base+string(os.PathSeparator)) {
		http.NotFound(w, r)
		return
	}
	info, err := os.Stat(full)
	if err != nil || info.IsDir() {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, full)
}

// executionCluster serves the cluster_file (cluster_json) that was used for the
// run, read from the path recorded on the execution.
func (h *Handler) executionCluster(w http.ResponseWriter, r *http.Request) {
	ex, ok := h.exec.Store.Get(chi.URLParam(r, "id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "execution not found"})
		return
	}
	if ex.ClusterFile == "" {
		http.NotFound(w, r)
		return
	}
	if _, err := os.Stat(ex.ClusterFile); err != nil {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, ex.ClusterFile)
}

func (h *Handler) executionLogs(w http.ResponseWriter, r *http.Request) {
	ex, ok := h.exec.Store.Get(chi.URLParam(r, "id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "execution not found"})
		return
	}
	logs := ""
	if ex.LogPath != "" {
		if data, err := os.ReadFile(ex.LogPath); err == nil {
			logs = string(data)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":     ex.ID,
		"status": ex.Status,
		"logs":   logs,
	})
}

func suiteExists(suites []Suite, name string) bool {
	for _, s := range suites {
		if s.Name == name {
			return true
		}
	}
	return false
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
