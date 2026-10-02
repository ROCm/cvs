package rccl

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type Snapshot struct {
	Timestamp     float64        `json:"timestamp"`
	State         string         `json:"state"`
	JobSummary    *JobSummary    `json:"job_summary,omitempty"`
	Communicators []Communicator `json:"communicators"`
	Peers         []any          `json:"peers"`
	DeadPeers     []string       `json:"dead_peers"`
	Errors        []string       `json:"errors"`
}

type JobSummary struct {
	TotalNodes             int    `json:"total_nodes"`
	TotalProcesses         int    `json:"total_processes"`
	TotalGPUs              int    `json:"total_gpus"`
	RCCLVersion            string `json:"rccl_version"`
	HIPRuntimeVersion      int    `json:"hip_runtime_version"`
	AMDGPUDriverVersion    int    `json:"amdgpu_driver_version"`
	InconsistentTopology   bool   `json:"inconsistent_topology"`
	HIPRuntimeVersionStr   string `json:"hip_runtime_version_str,omitempty"`
	AMDGPUDriverVersionStr string `json:"amdgpu_driver_version_str,omitempty"`
	DriverRuntimeMismatch  bool   `json:"driver_runtime_mismatch,omitempty"`
}

type Communicator struct {
	CommHash        string `json:"comm_hash"`
	TotalRanks      int    `json:"total_ranks"`
	RespondingRanks int    `json:"responding_ranks"`
	MissingRanks    int    `json:"missing_ranks"`
	Ranks           []Rank `json:"ranks"`
	Health          string `json:"health"`
}

type Rank struct {
	CommRank     int            `json:"comm_rank"`
	NodeAddr     string         `json:"node_addr"`
	PID          int            `json:"pid"`
	CUDADev      int            `json:"cuda_dev"`
	NVMLDev      int            `json:"nvml_dev"`
	CollOpCounts map[string]int `json:"coll_op_counts"`
	Status       RankStatus     `json:"status"`
}

type RankStatus struct {
	InitState      int  `json:"init_state"`
	AsyncError     int  `json:"async_error"`
	FinalizeCalled bool `json:"finalize_called"`
	DestroyFlag    bool `json:"destroy_flag"`
	AbortFlag      bool `json:"abort_flag"`
}

func emptySnap(state string) Snapshot {
	return Snapshot{
		Timestamp:     float64(time.Now().UnixNano()) / 1e9,
		State:         state,
		Communicators: []Communicator{},
		Peers:         []any{},
		DeadPeers:     []string{},
		Errors:        []string{},
	}
}

func hipVersionStr(v int) string {
	if v <= 0 {
		return "unknown"
	}
	major := v / 10_000_000
	minor := (v / 100_000) % 100
	patch := v % 100_000
	return fmt.Sprintf("%d.%d.%d", major, minor, patch)
}

func ParseJSON(raw string) Snapshot {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return emptySnap("no_job")
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return ParseText(raw)
	}
	commsAny, _ := data["communicators"].([]any)
	var comms []Communicator
	hosts := map[string]struct{}{}
	procs := map[string]struct{}{}
	totalGPUs := 0
	for _, c := range commsAny {
		cm, _ := c.(map[string]any)
		if cm == nil {
			continue
		}
		total := asInt(cm["size"])
		missing := asInt(cm["missing_ranks_count"])
		ranksCount := asInt(cm["ranks_count"])
		if ranksCount == 0 {
			ranksCount = total
		}
		totalGPUs += total
		var ranks []Rank
		anyErr := false
		for _, r := range asArr(cm["ranks"]) {
			rm, _ := r.(map[string]any)
			if rm == nil {
				continue
			}
			st, _ := rm["status"].(map[string]any)
			status := RankStatus{
				InitState:      asInt(st["init_state"]),
				AsyncError:     asInt(st["async_error"]),
				FinalizeCalled: asBool(st["finalize_called"]),
				DestroyFlag:    asBool(st["destroy_flag"]),
				AbortFlag:      asBool(st["abort_flag"]),
			}
			if status.AsyncError != 0 || status.AbortFlag || status.InitState != 0 {
				anyErr = true
			}
			counts := map[string]int{}
			if cc, ok := rm["collective_counts"].(map[string]any); ok {
				for k, v := range cc {
					counts[k] = asInt(v)
				}
			}
			host := asString(rm["host"])
			pid := asInt(rm["pid"])
			if host != "" {
				hosts[host] = struct{}{}
				procs[host+":"+fmt.Sprintf("%d", pid)] = struct{}{}
			}
			ranks = append(ranks, Rank{
				CommRank:     asInt(rm["rank"]),
				NodeAddr:     host,
				PID:          pid,
				CUDADev:      asInt(rm["cuda_dev"]),
				NVMLDev:      asInt(rm["nvml_dev"]),
				CollOpCounts: counts,
				Status:       status,
			})
		}
		health := "healthy"
		if missing > 0 || anyErr {
			health = "degraded"
		}
		comms = append(comms, Communicator{
			CommHash:        asString(cm["hash"]),
			TotalRanks:      total,
			RespondingRanks: ranksCount - missing,
			MissingRanks:    missing,
			Ranks:           ranks,
			Health:          health,
		})
	}
	hip := asInt(data["cuda_runtime_version"])
	drv := asInt(data["cuda_driver_version"])
	summary := &JobSummary{
		TotalNodes:             max(len(hosts), 1),
		TotalProcesses:         max(len(procs), 1),
		TotalGPUs:              totalGPUs,
		RCCLVersion:            asString(data["nccl_version"]),
		HIPRuntimeVersion:      hip,
		AMDGPUDriverVersion:    drv,
		HIPRuntimeVersionStr:   hipVersionStr(hip),
		AMDGPUDriverVersionStr: hipVersionStr(drv),
		DriverRuntimeMismatch:  hip != drv,
	}
	if summary.RCCLVersion == "" {
		summary.RCCLVersion = "unknown"
	}
	state := "healthy"
	if len(comms) == 0 {
		state = "no_job"
	}
	for _, c := range comms {
		if c.Health == "degraded" {
			state = "degraded"
			break
		}
	}
	dead := asStringSlice(data["dead_peers"])
	errs := asStringSlice(data["errors"])
	if len(dead) > 0 || len(errs) > 0 {
		state = "degraded"
	}
	return Snapshot{
		Timestamp:     float64(time.Now().UnixNano()) / 1e9,
		State:         state,
		JobSummary:    summary,
		Communicators: comms,
		Peers:         []any{},
		DeadPeers:     dead,
		Errors:        errs,
	}
}

func asStringSlice(v any) []string {
	arr, ok := v.([]any)
	if !ok || len(arr) == 0 {
		return []string{}
	}
	out := make([]string, 0, len(arr))
	for _, x := range arr {
		if s, ok := x.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	return out
}

func asInt(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case json.Number:
		n, _ := t.Int64()
		return int(n)
	case string:
		n := 0
		for _, c := range t {
			if c < '0' || c > '9' {
				break
			}
			n = n*10 + int(c-'0')
		}
		return n
	}
	return 0
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func asArr(v any) []any {
	a, _ := v.([]any)
	return a
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
