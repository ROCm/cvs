package httpserver

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/app"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/collectors"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/installers"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/sshkeys"
)

type Server struct {
	App       *app.App
	StaticDir string
	Logger    *slog.Logger
	// Restart replaces the process. Tests inject a fake; main uses syscall.Exec.
	Restart func() error
}

func (s *Server) Handler() http.Handler {
	r := chi.NewRouter()
	r.Use(s.cors)
	r.Use(s.logRequest)

	r.Get("/health", s.health)
	r.Get("/ws/metrics", func(w http.ResponseWriter, r *http.Request) { s.App.MetricsHub.ServeWS(w, r) })
	r.Get("/ws/rccl", func(w http.ResponseWriter, r *http.Request) { s.App.RCCLHub.ServeWS(w, r) })

	r.Route("/api", func(r chi.Router) {
		r.Get("/cluster/status", s.clusterStatus)
		r.Get("/cluster/health", s.clusterHealth)
		r.Get("/nodes", s.listNodes)
		r.Get("/nodes/{nodeID}", s.nodeDetails)
		r.Get("/metrics/latest", s.metricsLatest)
		r.Get("/metrics/history", s.metricsHistory)
		r.Get("/software/gpu", s.softwareGPU)
		r.Get("/software/nic", s.softwareNIC)
		r.Get("/software/nic/advanced", s.softwareNICAdvanced)
		r.Get("/software/nic/devlink", s.softwareDevlink)
		r.Post("/backend/restart", s.backendRestart)
		r.Post("/packages/install", s.packageInstall)
		r.Get("/packages/status/{package}", s.packageStatus)
		r.Get("/packages/list", s.packageList)
		r.Get("/logs/dmesg", s.logsDmesg)
		r.Get("/logs/search", s.logsSearch)
		r.Get("/collectors/status", s.collectorsStatus)
		r.Get("/rccl/status", s.rcclStatus)
		r.Get("/rccl/communicators", s.rcclComms)
		r.Get("/rccl/communicators/{commHash}", s.rcclComm)
		r.Get("/rccl/events", s.rcclEvents)
		r.Get("/rccl/performance", s.rcclPerf)
		r.Get("/rccl/performance/history", s.rcclPerfHistory)
		r.Post("/rccl/markers", s.rcclMarker)

		r.Group(func(r chi.Router) {
			r.Use(s.requireAPIToken)
			r.Post("/config/update", s.configUpdate)
			r.Post("/config/reload", s.configReload)
			r.Get("/config/current", s.configCurrent)
			r.Post("/ssh-keys/upload", s.sshUpload)
			r.Get("/ssh-keys/list", s.sshList)
			r.Delete("/ssh-keys/{filename}", s.sshDelete)
		})
	})

	if s.StaticDir != "" {
		if st, err := os.Stat(s.StaticDir); err == nil && st.IsDir() {
			h := spa(s.StaticDir)
			r.NotFound(h.ServeHTTP)
		}
	}
	return r
}

func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origins := []string{}
		if s.App != nil {
			if cfg := s.App.Cfg(); cfg != nil {
				origins = cfg.HTTP.CORSOrigins
			}
		}
		reqOrig := r.Header.Get("Origin")
		if reqOrig != "" {
			for _, o := range origins {
				if o == "*" || o == reqOrig {
					allow := reqOrig
					if o == "*" {
						allow = "*"
					}
					w.Header().Set("Access-Control-Allow-Origin", allow)
					w.Header().Set("Vary", "Origin")
					break
				}
			}
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Token")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) requireAPIToken(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimSpace(os.Getenv("CLUSTER_MON_API_TOKEN"))
		if token == "" {
			next.ServeHTTP(w, r)
			return
		}
		got := r.Header.Get("X-API-Token")
		if got == "" {
			auth := r.Header.Get("Authorization")
			if len(auth) > 7 && strings.EqualFold(auth[:7], "Bearer ") {
				got = strings.TrimSpace(auth[7:])
			}
		}
		if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
			writeErr(w, 401, "unauthorized")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) logRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		if s.Logger != nil && strings.HasPrefix(r.URL.Path, "/api") {
			s.Logger.Debug("http", "method", r.Method, "path", r.URL.Path, "dur", time.Since(start).String())
		}
	})
}

func spa(dir string) http.Handler {
	fs := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := filepath.Join(dir, filepath.Clean(r.URL.Path))
		if _, err := os.Stat(p); os.IsNotExist(err) {
			http.ServeFile(w, r, filepath.Join(dir, "index.html"))
			return
		}
		fs.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"detail": msg})
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{
		"status":      "healthy",
		"ssh_manager": s.App.Pool() != nil,
		"collecting":  s.App.Collecting(),
		"clients":     s.App.MetricsHub.Count(),
	})
}

func (s *Server) metricsLatest(w http.ResponseWriter, _ *http.Request) {
	m := s.App.LatestMetrics()
	if m == nil {
		writeErr(w, 503, "No metrics available yet")
		return
	}
	writeJSON(w, 200, m)
}

func (s *Server) metricsHistory(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	writeJSON(w, 200, map[string]any{
		"message": "Historical metrics not yet implemented",
		"note":    "This will query InfluxDB in production",
		"params": map[string]any{
			"node":        q.Get("node"),
			"metric_type": q.Get("metric_type"),
			"duration":    q.Get("duration"),
		},
	})
}

func (s *Server) clusterStatus(w http.ResponseWriter, _ *http.Request) {
	m := s.App.LatestMetrics()
	pool := s.App.Pool()
	if m == nil {
		writeJSON(w, 200, map[string]any{
			"total_nodes": 0, "healthy_nodes": 0, "unhealthy_nodes": 0,
			"unreachable_nodes": 0, "total_gpus": 0, "status": "no_data",
		})
		return
	}
	if pool == nil {
		writeJSON(w, 200, map[string]any{
			"total_nodes": 0, "healthy_nodes": 0, "unhealthy_nodes": 0,
			"unreachable_nodes": 0, "total_gpus": 0, "status": "no_ssh_manager",
		})
		return
	}
	gpu, _ := m["gpu"].(map[string]any)
	util, _ := gpu["utilization"].(map[string]any)
	mem, _ := gpu["memory"].(map[string]any)
	temp, _ := gpu["temperature"].(map[string]any)

	healthy, unhealthy := 0, 0
	for _, h := range pool.Reachable() {
		st, _ := nodeHealth(h, gpu, s.gpuTempThreshold())
		if st == "healthy" {
			healthy++
		} else {
			unhealthy++
		}
	}
	totalGPUs, gpuCount, totalUtil := 0, 0, 0.0
	for _, nodeData := range util {
		nd, ok := nodeData.(map[string]any)
		if !ok || nd["error"] != nil {
			continue
		}
		totalGPUs += len(nd)
		for _, gm := range nd {
			g, ok := gm.(map[string]any)
			if !ok {
				continue
			}
			if u := toFloat(g["GPU use (%)"]); u != 0 || g["GPU use (%)"] != nil {
				totalUtil += u
				gpuCount++
			}
		}
	}
	memCount, totalMem := 0, 0.0
	for _, nodeData := range mem {
		nd, ok := nodeData.(map[string]any)
		if !ok || nd["error"] != nil {
			continue
		}
		for _, gm := range nd {
			g, ok := gm.(map[string]any)
			if !ok {
				continue
			}
			used := toFloat(g["VRAM Total Used Memory (B)"])
			tot := toFloat(g["VRAM Total Memory (B)"])
			if tot > 0 {
				totalMem += used / tot * 100
				memCount++
			}
		}
	}
	tempCount, totalTemp := 0, 0.0
	for _, nodeData := range temp {
		nd, ok := nodeData.(map[string]any)
		if !ok || nd["error"] != nil {
			continue
		}
		for _, gm := range nd {
			g, ok := gm.(map[string]any)
			if !ok {
				continue
			}
			t := toFloat(g["Temperature (Sensor junction) (C)"])
			if t == 0 {
				t = toFloat(g["Temperature (Sensor edge) (C)"])
			}
			if t != 0 {
				totalTemp += t
				tempCount++
			}
		}
	}
	avgUtil, avgMem, avgTemp := 0.0, 0.0, 0.0
	if gpuCount > 0 {
		avgUtil = totalUtil / float64(gpuCount)
	}
	if memCount > 0 {
		avgMem = totalMem / float64(memCount)
	}
	if tempCount > 0 {
		avgTemp = totalTemp / float64(tempCount)
	}
	unreach := len(pool.Unreachable())
	status := "healthy"
	if unreach > 0 {
		status = "critical"
	} else if unhealthy > 0 {
		status = "degraded"
	}
	writeJSON(w, 200, map[string]any{
		"total_nodes":                len(pool.All()),
		"healthy_nodes":              healthy,
		"unhealthy_nodes":            unhealthy,
		"unreachable_nodes":          unreach,
		"total_gpus":                 totalGPUs,
		"avg_gpu_utilization":        round2(avgUtil),
		"avg_gpu_memory_utilization": round2(avgMem),
		"avg_gpu_temperature":        round1(avgTemp),
		"status":                     status,
		"last_update":                m["timestamp"],
	})
}

func (s *Server) clusterHealth(w http.ResponseWriter, _ *http.Request) {
	m := s.App.LatestMetrics()
	pool := s.App.Pool()
	if m == nil {
		writeErr(w, 503, "No metrics available yet")
		return
	}
	gpu, _ := m["gpu"].(map[string]any)
	out := map[string]any{"timestamp": m["timestamp"], "nodes": map[string]any{}, "alerts": []any{}}
	nodes := map[string]any{}
	var alerts []any
	if pool != nil {
		for _, h := range pool.Reachable() {
			st, issues := nodeHealth(h, gpu, s.gpuTempThreshold())
			nodes[h] = map[string]any{"status": st, "issues": issues}
			if st == "unhealthy" {
				for _, issue := range issues {
					alerts = append(alerts, map[string]any{"severity": "warning", "node": h, "message": issue})
				}
			}
		}
		for _, h := range pool.Unreachable() {
			nodes[h] = map[string]any{"status": "unreachable", "issues": []string{"Node is unreachable via SSH"}}
			alerts = append(alerts, map[string]any{"severity": "critical", "node": h, "message": "Node " + h + " is unreachable"})
		}
	}
	out["nodes"] = nodes
	if alerts == nil {
		alerts = []any{}
	}
	out["alerts"] = alerts
	writeJSON(w, 200, out)
}

func (s *Server) listNodes(w http.ResponseWriter, _ *http.Request) {
	pool := s.App.Pool()
	if pool == nil {
		writeJSON(w, 200, []any{})
		return
	}
	m := s.App.LatestMetrics()
	gpu := map[string]any{}
	if m != nil {
		gpu, _ = m["gpu"].(map[string]any)
	}
	util, _ := gpu["utilization"].(map[string]any)
	temp, _ := gpu["temperature"].(map[string]any)
	health, fails := s.App.NodeHealth()
	var list []any
	for _, node := range pool.All() {
		st := health[node]
		if st == "" {
			st = "healthy"
		}
		issues := []string{}
		hs, hi := nodeHealth(node, gpu, s.gpuTempThreshold())
		if st == "unreachable" {
			hs = "unreachable"
			hi = []string{"Unreachable after " + strconv.Itoa(fails[node]) + " consecutive failures"}
		} else {
			st = hs
			issues = hi
		}
		info := map[string]any{
			"hostname": node, "status": st, "gpu_count": 0,
			"avg_gpu_util": 0.0, "avg_gpu_temp": 0.0, "health_issues": issues,
		}
		if nd, ok := util[node].(map[string]any); ok && nd["error"] == nil {
			info["gpu_count"] = len(nd)
			sum := 0.0
			n := 0
			for _, gm := range nd {
				g, ok := gm.(map[string]any)
				if !ok {
					continue
				}
				sum += toFloat(g["GPU use (%)"])
				n++
			}
			if n > 0 {
				info["avg_gpu_util"] = round2(sum / float64(n))
			}
		}
		if nd, ok := temp[node].(map[string]any); ok && nd["error"] == nil {
			sum := 0.0
			n := 0
			for _, gm := range nd {
				g, ok := gm.(map[string]any)
				if !ok {
					continue
				}
				t := toFloat(g["Temperature (Sensor junction) (C)"])
				if t == 0 {
					t = toFloat(g["Temperature (Sensor edge) (C)"])
				}
				if t != 0 {
					sum += t
					n++
				}
			}
			if n > 0 {
				info["avg_gpu_temp"] = round2(sum / float64(n))
			}
		}
		list = append(list, info)
	}
	if list == nil {
		list = []any{}
	}
	writeJSON(w, 200, list)
}

func (s *Server) nodeDetails(w http.ResponseWriter, r *http.Request) {
	pool := s.App.Pool()
	if pool == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	id := chi.URLParam(r, "nodeID")
	found := false
	for _, h := range pool.All() {
		if h == id {
			found = true
			break
		}
	}
	if !found {
		writeErr(w, 404, "Node "+id+" not found")
		return
	}
	reach := false
	for _, h := range pool.Reachable() {
		if h == id {
			reach = true
			break
		}
	}
	m := s.App.LatestMetrics()
	ts := any(nil)
	if m != nil {
		ts = m["timestamp"]
	}
	out := map[string]any{
		"hostname": id, "status": map[bool]string{true: "reachable", false: "unreachable"}[reach],
		"last_update": ts, "gpus": []any{}, "nics": []any{},
	}
	if !reach || m == nil {
		writeJSON(w, 200, out)
		return
	}
	gpu, _ := m["gpu"].(map[string]any)
	nic, _ := m["nic"].(map[string]any)
	util := nested(gpu, "utilization", id)
	memd := nested(gpu, "memory", id)
	temp := nested(gpu, "temperature", id)
	pwr := nested(gpu, "power", id)
	var gpus []any
	for gid, gv := range util {
		gm, ok := gv.(map[string]any)
		if !ok {
			continue
		}
		info := map[string]any{
			"id": gid, "utilization": toFloat(gm["GPU use (%)"]),
			"memory_used_mb": 0, "memory_total_mb": 0, "memory_util_percent": 0.0,
			"temperature_c": 0.0, "power_w": 0.0,
		}
		if mm, ok := memd[gid].(map[string]any); ok {
			used := int64(toFloat(mm["VRAM Total Used Memory (B)"]))
			tot := int64(toFloat(mm["VRAM Total Memory (B)"]))
			info["memory_used_mb"] = used / (1024 * 1024)
			info["memory_total_mb"] = tot / (1024 * 1024)
			if tot > 0 {
				info["memory_util_percent"] = round2(float64(used) / float64(tot) * 100)
			}
		}
		if tm, ok := temp[gid].(map[string]any); ok {
			t := toFloat(tm["Temperature (Sensor junction) (C)"])
			if t == 0 {
				t = toFloat(tm["Temperature (Sensor edge) (C)"])
			}
			info["temperature_c"] = t
		}
		if pm, ok := pwr[gid].(map[string]any); ok {
			info["power_w"] = firstFloatToken(stringify(pm["socket_power"]))
		}
		gpus = append(gpus, info)
	}
	out["gpus"] = gpus
	ip := nested(nic, "ip_addr", id)
	rdma := nested(nic, "rdma_links", id)
	var nics []any
	for name, nv := range ip {
		ni, ok := nv.(map[string]any)
		if !ok || ni["error"] != nil {
			continue
		}
		entry := map[string]any{
			"name": name, "state": ni["state"], "mtu": ni["mtu"],
			"mac_addr": ni["mac_addr"], "ipv4_addrs": ni["ipv4_addr_list"], "rdma": nil,
		}
		if entry["ipv4_addrs"] == nil {
			entry["ipv4_addrs"] = ni["ipv4"]
		}
		for dev, dv := range rdma {
			dm, ok := dv.(map[string]any)
			if !ok {
				continue
			}
			if stringify(dm["netdev"]) == name {
				entry["rdma"] = map[string]any{"device": dev, "state": dm["state"], "physical_state": dm["physical_state"]}
			}
		}
		nics = append(nics, entry)
	}
	out["nics"] = nics
	writeJSON(w, 200, out)
}

type jumpIn struct {
	Host         string `json:"host"`
	Username     string `json:"username"`
	AuthMethod   string `json:"auth_method"`
	Password     string `json:"password"`
	KeyFilePath  string `json:"key_file_path"`
	NodeUsername string `json:"node_username"`
	NodeKeyFile  string `json:"node_key_file"`
}

type configIn struct {
	Nodes       []string `json:"nodes"`
	Username    string   `json:"username"`
	AuthMethod  string   `json:"auth_method"`
	Password    string   `json:"password"`
	KeyFilePath string   `json:"key_file_path"`
	UseJumpHost bool     `json:"use_jump_host"`
	JumpHost    *jumpIn  `json:"jump_host"`
}

func (s *Server) configUpdate(w http.ResponseWriter, r *http.Request) {
	var in configIn
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	if len(in.Nodes) == 0 {
		writeErr(w, 400, "No nodes provided")
		return
	}
	if in.UseJumpHost && (in.JumpHost == nil || in.JumpHost.Host == "") {
		writeErr(w, 400, "Jump host IP/hostname is required")
		return
	}
	cfg, err := config.Load(s.App.ConfigDir)
	if err != nil || cfg == nil {
		cfg = &config.Config{}
		*cfg = *mustDefaults(s.App.ConfigDir)
	}
	cfg.SSH.Username = in.Username
	cfg.SSH.KeyFile = config.NormalizeSSHKeyPath(in.KeyFilePath)
	if cfg.SSH.KeyFile == "" {
		cfg.SSH.KeyFile = "/root/.ssh/id_rsa"
	}
	cfg.SSH.Timeout = 30
	if in.AuthMethod == "password" && in.Password != "" {
		s.App.SetPasswords(in.Password, "")
	}
	if in.UseJumpHost && in.JumpHost != nil {
		jh := in.JumpHost
		cfg.SSH.JumpHost.Enabled = true
		cfg.SSH.JumpHost.Host = jh.Host
		cfg.SSH.JumpHost.Username = jh.Username
		cfg.SSH.JumpHost.KeyFile = config.NormalizeSSHKeyPath(jh.KeyFilePath)
		if cfg.SSH.JumpHost.KeyFile == "" {
			cfg.SSH.JumpHost.KeyFile = "/root/.ssh/id_rsa"
		}
		cfg.SSH.JumpHost.NodeUsername = jh.NodeUsername
		if cfg.SSH.JumpHost.NodeUsername == "" {
			cfg.SSH.JumpHost.NodeUsername = in.Username
		}
		cfg.SSH.JumpHost.NodeKeyFile = jh.NodeKeyFile
		if cfg.SSH.JumpHost.NodeKeyFile == "" {
			cfg.SSH.JumpHost.NodeKeyFile = "~/.ssh/id_ed25519"
		}
		if jh.AuthMethod == "password" && jh.Password != "" {
			s.App.SetPasswords("", jh.Password)
		}
	} else {
		cfg.SSH.JumpHost.Enabled = false
	}
	if err := config.SaveNodes(s.App.ConfigDir, in.Nodes); err != nil {
		writeErr(w, 500, "Failed to save configuration: "+err.Error())
		return
	}
	if err := config.SaveYAML(s.App.ConfigDir, cfg); err != nil {
		writeErr(w, 500, "Failed to save configuration: "+err.Error())
		return
	}
	note := ""
	if in.AuthMethod == "password" || (in.UseJumpHost && in.JumpHost != nil && in.JumpHost.AuthMethod == "password") {
		note = " Passwords are stored in memory only."
	}
	sec := any(nil)
	if note != "" {
		sec = "Passwords are never persisted to disk - stored in memory only"
	}
	writeJSON(w, 200, map[string]any{
		"success":           true,
		"message":           "Configuration saved successfully! " + strconv.Itoa(len(in.Nodes)) + " node(s) configured." + note,
		"nodes_saved":       len(in.Nodes),
		"jump_host_enabled": in.UseJumpHost,
		"security_note":     sec,
	})
}

func mustDefaults(dir string) *config.Config {
	c, err := config.Load(dir)
	if err != nil {
		return &config.Config{ConfigDir: dir}
	}
	return c
}

func (s *Server) configReload(w http.ResponseWriter, _ *http.Request) {
	res, err := s.App.Reload()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, res)
}

func (s *Server) configCurrent(w http.ResponseWriter, _ *http.Request) {
	cfg := s.App.Cfg()
	if cfg == nil {
		c, err := config.Load(s.App.ConfigDir)
		if err != nil {
			writeJSON(w, 200, map[string]any{"nodes": []string{}})
			return
		}
		cfg = c
	}
	ssh, _ := s.App.Passwords()
	auth := "key"
	if ssh != "" {
		auth = "password"
	}
	out := map[string]any{
		"nodes": cfg.Nodes, "username": cfg.SSH.Username, "auth_method": auth,
		"key_file": cfg.SSH.KeyFile, "jump_host_enabled": cfg.SSH.JumpHost.Enabled,
	}
	if cfg.SSH.JumpHost.Enabled {
		out["jump_host"] = cfg.SSH.JumpHost.Host
		out["jump_host_username"] = cfg.SSH.JumpHost.Username
		out["jump_host_key_file"] = cfg.SSH.JumpHost.KeyFile
		out["node_username_via_jump"] = cfg.SSH.JumpHost.NodeUsername
		out["node_key_file_on_jumphost"] = cfg.SSH.JumpHost.NodeKeyFile
	} else {
		out["jump_host"] = nil
		out["jump_host_username"] = nil
		out["jump_host_key_file"] = nil
		out["node_username_via_jump"] = nil
		out["node_key_file_on_jumphost"] = nil
	}
	writeJSON(w, 200, out)
}

func (s *Server) softwareGPU(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 150*time.Second)
	defer cancel()
	snap, err := s.App.Soft.GPUSoftware(ctx)
	if err != nil && snap == nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, collectors.PythonGPUSoftwarePayload(snap))
}

func (s *Server) softwareNIC(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 150*time.Second)
	defer cancel()
	writeJSON(w, 200, s.App.NICSoftware(ctx))
}

func (s *Server) softwareNICAdvanced(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 150*time.Second)
	defer cancel()
	writeJSON(w, 200, s.App.NICAdvanced(ctx))
}

func (s *Server) softwareDevlink(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 150*time.Second)
	defer cancel()
	snap, err := s.App.Soft.NICDevlink(ctx)
	if err != nil && snap == nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, collectors.PythonDevlinkPayload(snap))
}

func (s *Server) backendRestart(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{
		"success": true,
		"message": "Backend is restarting... Please wait 10 seconds and refresh the page.",
	})
	if s.Restart == nil {
		if s.Logger != nil {
			s.Logger.Error("restart_not_configured")
		}
		return
	}
	go func() {
		time.Sleep(time.Second)
		if err := s.Restart(); err != nil && s.Logger != nil {
			s.Logger.Error("restart_failed", "err", err)
		}
	}()
}

func (s *Server) packageInstall(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Package string   `json:"package"`
		Nodes   []string `json:"nodes"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	if strings.ToLower(in.Package) != "lldp" {
		writeErr(w, 400, "Unsupported package: "+in.Package+". Supported: lldp")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	result := installers.LLDPInstall(ctx, s.App.Pool(), in.Nodes)
	ok, _ := result["successful"].(int)
	fail, _ := result["failed"].(int)
	result["success"] = true
	result["message"] = "Installation complete: " + strconv.Itoa(ok) + " successful, " + strconv.Itoa(fail) + " failed"
	writeJSON(w, 200, result)
}

func (s *Server) packageStatus(w http.ResponseWriter, r *http.Request) {
	pkg := chi.URLParam(r, "package")
	if s.App.Pool() == nil {
		writeJSON(w, 200, map[string]any{
			"package": pkg, "total_nodes": 0, "installed_count": 0, "not_installed_count": 0,
			"installed_nodes": []string{}, "not_installed_nodes": []string{}, "status_by_node": map[string]any{},
			"note": "SSH not configured yet. Save configuration first.",
		})
		return
	}
	if strings.ToLower(pkg) != "lldp" {
		writeErr(w, 400, "Unsupported package: "+pkg+". Supported: lldp")
		return
	}
	writeJSON(w, 200, installers.LLDPStatus(r.Context(), s.App.Pool()))
}

func (s *Server) packageList(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{
		"packages": []any{
			map[string]any{
				"id": "lldp", "name": "LLDP Daemon",
				"description":  "Link Layer Discovery Protocol daemon for network topology discovery",
				"package_name": "lldpd", "check_command": "lldpcli",
			},
		},
	})
}

func (s *Server) logsDmesg(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()
	snap, err := s.App.Logs.Logs(ctx)
	if err != nil {
		writeErr(w, 500, "Failed to collect logs: "+err.Error())
		return
	}
	writeJSON(w, 200, collectors.PythonLogsPayload(snap))
}

func (s *Server) logsSearch(w http.ResponseWriter, r *http.Request) {
	if s.App.Pool() == nil {
		writeErr(w, 503, "SSH manager not initialized")
		return
	}
	cmd := r.URL.Query().Get("grep_command")
	if cmd == "" {
		writeErr(w, 400, "Invalid grep command: Empty command")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	res, err := s.App.Logs.Search(ctx, cmd)
	if err != nil {
		if collectors.IsInvalidGrep(err) {
			writeErr(w, 400, "Invalid grep command: "+err.Error())
			return
		}
		writeErr(w, 500, "Failed to search logs: "+err.Error())
		return
	}
	writeJSON(w, 200, collectors.PythonSearchPayload(res))
}

func (s *Server) sshUpload(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(2 << 20); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	f, hdr, err := r.FormFile("file")
	if err != nil {
		writeErr(w, 400, "No filename provided")
		return
	}
	defer f.Close()
	content, err := io.ReadAll(io.LimitReader(f, 1<<20))
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if err := sshkeys.Save(hdr.Filename, content); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"success": true, "message": "SSH key '" + hdr.Filename + "' uploaded successfully",
		"filename": hdr.Filename, "size": len(content), "path": filepath.Join(sshkeys.Dir(), hdr.Filename),
	})
}

func (s *Server) sshList(w http.ResponseWriter, _ *http.Request) {
	keys, err := sshkeys.List()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"keys": keys, "count": len(keys)})
}

func (s *Server) sshDelete(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "filename")
	if err := sshkeys.Delete(name); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeErr(w, 404, "Key '"+name+"' not found")
			return
		}
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"success": true, "message": "SSH key '" + name + "' deleted successfully"})
}

func (s *Server) collectorsStatus(w http.ResponseWriter, _ *http.Request) {
	res := s.App.CollectorResults()
	out := map[string]any{}
	overall := "healthy"
	for name, cr := range res {
		out[name] = map[string]any{"state": cr.State, "timestamp": cr.Timestamp, "error": nilOr(cr.Error)}
		if (cr.State == "error" || cr.State == "unreachable") && cr.Critical {
			overall = "critical"
		} else if overall != "critical" && (cr.State == "error" || cr.State == "unreachable") {
			overall = "degraded"
		}
	}
	out["overall_status"] = overall
	writeJSON(w, 200, out)
}

func (s *Server) rcclStatus(w http.ResponseWriter, _ *http.Request) {
	snap := s.App.LatestRCCL()
	if snap == nil {
		writeJSON(w, 200, map[string]any{"state": "no_job", "message": "No RCCL snapshot collected yet"})
		return
	}
	writeJSON(w, 200, snap)
}

func (s *Server) rcclComms(w http.ResponseWriter, _ *http.Request) {
	snap := s.App.LatestRCCL()
	if snap == nil {
		writeJSON(w, 200, []any{})
		return
	}
	comms, _ := snap["communicators"]
	if comms == nil {
		comms = []any{}
	}
	writeJSON(w, 200, comms)
}

func (s *Server) rcclComm(w http.ResponseWriter, r *http.Request) {
	snap := s.App.LatestRCCL()
	if snap == nil {
		writeErr(w, 404, "No snapshot available")
		return
	}
	want := chi.URLParam(r, "commHash")
	comms, _ := snap["communicators"].([]any)
	for _, c := range comms {
		cm, _ := c.(map[string]any)
		if stringify(cm["comm_hash"]) == want {
			writeJSON(w, 200, cm)
			return
		}
	}
	writeErr(w, 404, "Communicator "+want+" not found")
}

func (s *Server) rcclEvents(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	since, _ := strconv.ParseFloat(q.Get("since"), 64)
	until, _ := strconv.ParseFloat(q.Get("until"), 64)
	if until == 0 {
		until = float64(time.Now().Unix())
	}
	if since == 0 {
		since = until - 3600
	}
	writeJSON(w, 200, map[string]any{
		"events":    s.App.Store.Events(since, until, q.Get("type")),
		"truncated": s.App.Store.IsCapped(),
	})
}

func (s *Server) rcclPerf(w http.ResponseWriter, _ *http.Request) {
	snap := s.App.Store.InspectorCurrent()
	if snap == nil {
		writeErr(w, 503, "No Inspector snapshot available. Check that rccl.inspector.enabled=true and a job is running.")
		return
	}
	writeJSON(w, 200, snap)
}

func (s *Server) rcclPerfHistory(w http.ResponseWriter, r *http.Request) {
	n, _ := strconv.Atoi(r.URL.Query().Get("count"))
	if n <= 0 {
		n = 50
	}
	snaps := s.App.Store.InspectorHistory(n)
	writeJSON(w, 200, map[string]any{"snapshots": snaps, "count": len(snaps)})
}

func (s *Server) rcclMarker(w http.ResponseWriter, r *http.Request) {
	var e map[string]any
	if err := json.NewDecoder(r.Body).Decode(&e); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	if _, ok := e["event_type"]; !ok {
		e["event_type"] = "training_marker"
	}
	if _, ok := e["timestamp"]; !ok {
		e["timestamp"] = float64(time.Now().UnixNano()) / 1e9
	}
	s.App.Store.PushEvent(e)
	writeJSON(w, 201, map[string]string{"status": "accepted"})
}

func (s *Server) gpuTempThreshold() float64 {
	if s != nil && s.App != nil {
		if cfg := s.App.Cfg(); cfg != nil && cfg.Alerts.GPUTempThreshold > 0 {
			return cfg.Alerts.GPUTempThreshold
		}
	}
	return 85
}

func nodeHealth(node string, gpu map[string]any, tempLimit float64) (string, []string) {
	if tempLimit <= 0 {
		tempLimit = 85
	}
	status := "healthy"
	var issues []string
	if gpu == nil {
		return status, issues
	}
	if ras, ok := gpu["ras_errors"].(map[string]any); ok {
		if nd, ok := ras[node].(map[string]any); ok {
			arr, _ := nd["gpu_data"].([]any)
			for _, g := range arr {
				gm, _ := g.(map[string]any)
				ecc, _ := gm["ecc"].(map[string]any)
				unc := int(toFloat(ecc["total_uncorrectable"]))
				cor := int(toFloat(ecc["total_correctable"]))
				gpuID := stringify(gm["gpu"])
				if unc > 0 {
					status = "unhealthy"
					issues = append(issues, "GPU "+gpuID+": "+strconv.Itoa(unc)+" uncorrectable ECC errors")
				} else if cor > 10 {
					status = "unhealthy"
					issues = append(issues, "GPU "+gpuID+": "+strconv.Itoa(cor)+" correctable ECC errors")
				}
			}
		}
	}
	if pcie, ok := gpu["pcie"].(map[string]any); ok {
		if nd, ok := pcie[node].(map[string]any); ok {
			arr, _ := nd["gpu_data"].([]any)
			for _, g := range arr {
				gm, _ := g.(map[string]any)
				pc, _ := gm["pcie"].(map[string]any)
				replay := int(toFloat(pc["replay_count"]))
				nak := int(toFloat(pc["nak_count"]))
				if replay > 100 || nak > 100 {
					status = "unhealthy"
					gpuID := stringify(gm["gpu"])
					issues = append(issues, "GPU "+gpuID+": PCIe errors (replay: "+strconv.Itoa(replay)+", nak: "+strconv.Itoa(nak)+")")
				}
			}
		}
	}
	if xgmi, ok := gpu["xgmi"].(map[string]any); ok {
		if nd, ok := xgmi[node].(map[string]any); ok {
			arr, _ := nd["gpu_data"].([]any)
			for _, g := range arr {
				gm, _ := g.(map[string]any)
				xg, _ := gm["xgmi"].(map[string]any)
				n := int(toFloat(xg["error_count"]))
				if n > 10 {
					status = "unhealthy"
					gpuID := stringify(gm["gpu"])
					issues = append(issues, "GPU "+gpuID+": "+strconv.Itoa(n)+" XGMI errors (threshold 10)")
				}
			}
		}
	}
	if temp, ok := gpu["temperature"].(map[string]any); ok {
		if nd, ok := temp[node].(map[string]any); ok {
			for id, raw := range nd {
				if id == "gpu_data" || id == "error" {
					continue
				}
				gm, ok := raw.(map[string]any)
				if !ok {
					continue
				}
				t := gpuTempC(gm)
				if t > tempLimit {
					status = "unhealthy"
					issues = append(issues, "GPU "+id+": temperature "+strconv.FormatFloat(t, 'f', 0, 64)+" C exceeds "+strconv.FormatFloat(tempLimit, 'f', 0, 64)+" C")
				}
			}
		}
	}
	return status, issues
}

func gpuTempC(gm map[string]any) float64 {
	for _, k := range []string{
		"Temperature (Sensor junction) (C)",
		"Temperature (Sensor edge) (C)",
		"Temperature (Sensor memory) (C)",
	} {
		v, ok := gm[k]
		if !ok {
			continue
		}
		t := toFloat(v)
		if t > 0 {
			return t
		}
	}
	return 0
}

func nested(root map[string]any, key, host string) map[string]any {
	if root == nil {
		return map[string]any{}
	}
	m, _ := root[key].(map[string]any)
	if m == nil {
		return map[string]any{}
	}
	h, _ := m[host].(map[string]any)
	if h == nil {
		return map[string]any{}
	}
	return h
}

func toFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	case string:
		return firstFloatToken(t)
	}
	return 0
}

func firstFloatToken(s string) float64 {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0
	}
	f, _ := strconv.ParseFloat(strings.TrimSpace(fields[0]), 64)
	return f
}

func stringify(v any) string {
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	default:
		return strings.Trim(string(mustMarshal(t)), `"`)
	}
}

func mustMarshal(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func round2(f float64) float64 { return float64(int(f*100+0.5)) / 100 }
func round1(f float64) float64 { return float64(int(f*10+0.5)) / 10 }

func nilOr(s string) any {
	if s == "" {
		return nil
	}
	return s
}
