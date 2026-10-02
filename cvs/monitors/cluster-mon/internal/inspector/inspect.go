package inspector

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

type GPU struct {
	Index            int            `json:"index"`
	Name             string         `json:"name"`
	Serial           string         `json:"serial"`
	UniqueID         string         `json:"unique_id"`
	VBIOS            string         `json:"vbios"`
	Busy             float64        `json:"busy"`
	MemoryUsedBytes  int64          `json:"memory_used_bytes"`
	MemoryTotalBytes int64          `json:"memory_total_bytes"`
	Processes        []Process      `json:"processes"`
	XGMI             map[string]any `json:"xgmi,omitempty"`
}

type Process struct {
	PID     int    `json:"pid"`
	Name    string `json:"name"`
	User    string `json:"user"`
	GPUMem  int64  `json:"gpu_memory_bytes"`
	Command string `json:"command"`
}

type NodeInspect struct {
	Host      string `json:"host"`
	Error     string `json:"error,omitempty"`
	GPUs      []GPU  `json:"gpus"`
	Timestamp string `json:"timestamp"`
}

func CollectGPUs(ctx context.Context, pool *pssh.Pool, hosts []string) map[string]NodeInspect {
	out := map[string]NodeInspect{}
	if pool == nil || len(hosts) == 0 {
		return out
	}
	cmd := `bash -c 'amd-smi --json 2>/dev/null || rocm-smi --json 2>/dev/null; echo "---PROCS---"; amd-smi process --json 2>/dev/null || true'`
	res := pool.ExecHosts(ctx, cmd, hosts)
	now := time.Now().UTC().Format(time.RFC3339)
	for _, h := range hosts {
		r := res[h]
		ni := NodeInspect{Host: h, Timestamp: now, GPUs: []GPU{}}
		if r.Err != nil {
			ni.Error = r.Err.Error()
			out[h] = ni
			continue
		}
		ni.GPUs = parseInspect(r.Output)
		out[h] = ni
	}
	return out
}

func parseInspect(raw string) []GPU {
	parts := strings.SplitN(raw, "---PROCS---", 2)
	smi := strings.TrimSpace(parts[0])
	gpus := parseSMI(smi)
	if len(parts) > 1 {
		attachProcesses(gpus, strings.TrimSpace(parts[1]))
	}
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	return gpus
}

func parseSMI(s string) []GPU {
	var gpus []GPU
	s = strings.TrimSpace(s)
	if s == "" {
		return gpus
	}
	var root any
	if err := json.Unmarshal([]byte(s), &root); err != nil {
		return gpus
	}
	cards := extractCards(root)
	for i, card := range cards {
		g := GPU{Index: i, Processes: []Process{}}
		g.Name = firstString(card, "Device Name", "device_name", "Card series", "card_series")
		g.Serial = firstString(card, "Serial Number", "serial_number", "Serial")
		g.UniqueID = firstString(card, "Unique ID", "unique_id", "GUID")
		g.VBIOS = firstString(card, "VBIOS version", "vbios_version", "VBIOS")
		g.Busy = firstFloat(card, "GPU use (%)", "gpu_use_percent", "Average Graphics Package Power (W)")
		used, total := memoryPair(card)
		g.MemoryUsedBytes = used
		g.MemoryTotalBytes = total
		if idx := firstInt(card, "GPU ID", "gpu_id", "Card"); idx >= 0 {
			g.Index = idx
		}
		gpus = append(gpus, g)
	}
	return gpus
}

func extractCards(root any) []map[string]any {
	switch t := root.(type) {
	case []any:
		var out []map[string]any
		for _, e := range t {
			if m, ok := e.(map[string]any); ok {
				out = append(out, m)
			}
		}
		if len(out) > 0 {
			return out
		}
	case map[string]any:
		if cards, ok := t["card"].([]any); ok {
			return mapsFromArr(cards)
		}
		if devices, ok := t["devices"].([]any); ok {
			return mapsFromArr(devices)
		}
		var out []map[string]any
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if strings.HasPrefix(strings.ToLower(k), "card") {
				if m, ok := t[k].(map[string]any); ok {
					out = append(out, m)
				}
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return nil
}

func mapsFromArr(a []any) []map[string]any {
	var out []map[string]any
	for _, e := range a {
		if m, ok := e.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func attachProcesses(gpus []GPU, raw string) {
	if raw == "" {
		return
	}
	var root any
	if err := json.Unmarshal([]byte(raw), &root); err != nil {
		return
	}
	arr := []any{}
	switch t := root.(type) {
	case []any:
		arr = t
	case map[string]any:
		if p, ok := t["process_list"].([]any); ok {
			arr = p
		} else if p, ok := t["processes"].([]any); ok {
			arr = p
		}
	}
	byGPU := map[int][]Process{}
	for _, e := range arr {
		m, _ := e.(map[string]any)
		if m == nil {
			continue
		}
		p := Process{
			PID:     firstInt(m, "pid", "process_id", "PID"),
			Name:    firstString(m, "name", "process_name", "Name"),
			User:    firstString(m, "user", "username"),
			GPUMem:  int64(firstInt(m, "gpu_memory_usage", "mem", "memory")),
			Command: firstString(m, "command", "cmd"),
		}
		gi := firstInt(m, "gpu_id", "GPU ID", "card")
		byGPU[gi] = append(byGPU[gi], p)
	}
	for i := range gpus {
		if ps, ok := byGPU[gpus[i].Index]; ok {
			gpus[i].Processes = ps
		}
	}
}

func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := lookup(m, k); ok {
			switch t := v.(type) {
			case string:
				if t != "" {
					return t
				}
			default:
				s := strings.TrimSpace(fmt.Sprintf("%v", t))
				if s != "" && s != "<nil>" {
					return s
				}
			}
		}
	}
	return ""
}

func firstFloat(m map[string]any, keys ...string) float64 {
	for _, k := range keys {
		if v, ok := lookup(m, k); ok {
			switch t := v.(type) {
			case float64:
				return t
			case string:
				f, _ := strconv.ParseFloat(strings.TrimSpace(t), 64)
				return f
			}
		}
	}
	return 0
}

func firstInt(m map[string]any, keys ...string) int {
	for _, k := range keys {
		if v, ok := lookup(m, k); ok {
			switch t := v.(type) {
			case float64:
				return int(t)
			case int:
				return t
			case string:
				n, _ := strconv.Atoi(strings.TrimSpace(t))
				return n
			}
		}
	}
	return 0
}

func memoryPair(m map[string]any) (used, total int64) {
	usedKeys := []string{"GPU Memory Used (B)", "vram_used_bytes", "used_memory"}
	totalKeys := []string{"GPU Memory Total (B)", "vram_total_bytes", "total_memory"}
	if hasField(m, usedKeys...) {
		used = int64(firstInt(m, usedKeys...))
	}
	if hasField(m, totalKeys...) {
		total = int64(firstInt(m, totalKeys...))
	}
	if !hasField(m, usedKeys...) && total > 0 {
		if pct, ok := parseVRAMPercent(firstString(m, "GPU Memory Allocated (VRAM%)")); ok {
			used = total * int64(pct) / 100
		}
	}
	return
}

func hasField(m map[string]any, keys ...string) bool {
	for _, k := range keys {
		if _, ok := lookup(m, k); ok {
			return true
		}
	}
	return false
}

func parseVRAMPercent(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if !strings.HasSuffix(s, "%") {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimSpace(strings.TrimSuffix(s, "%")))
	if err != nil || n < 0 || n > 100 {
		return 0, false
	}
	return n, true
}

func lookup(m map[string]any, key string) (any, bool) {
	if v, ok := m[key]; ok {
		return v, true
	}
	lk := strings.ToLower(key)
	for k, v := range m {
		if strings.ToLower(k) == lk {
			return v, true
		}
		if nested, ok := v.(map[string]any); ok {
			if nv, ok := lookup(nested, key); ok {
				return nv, true
			}
		}
	}
	return nil, false
}
