package inspector

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

type CollPerf struct {
	Timestamp     float64        `json:"timestamp"`
	CommHash      string         `json:"comm_hash"`
	Rank          int            `json:"rank"`
	NRanks        int            `json:"nranks"`
	NNodes        int            `json:"nnodes"`
	Hostname      string         `json:"hostname"`
	PID           int            `json:"pid"`
	Collective    string         `json:"collective"`
	SequenceNum   int            `json:"sequence_num"`
	MsgSizeBytes  int64          `json:"msg_size_bytes"`
	ExecTimeUS    int64          `json:"exec_time_us"`
	TimingSource  string         `json:"timing_source"`
	AlgoBWGbps    float64        `json:"algo_bw_gbps"`
	BusBWGbps     float64        `json:"bus_bw_gbps"`
	FormatVersion string         `json:"inspector_format_version"`
	GraphCaptured *bool          `json:"graph_captured,omitempty"`
	EventTrace    map[string]any `json:"event_trace,omitempty"`
}

type Snapshot struct {
	Timestamp           float64        `json:"timestamp"`
	Records             []CollPerf     `json:"records"`
	AvgBusBWGbps        *float64       `json:"avg_bus_bw_gbps"`
	MinBusBWGbps        *float64       `json:"min_bus_bw_gbps"`
	MaxBusBWGbps        *float64       `json:"max_bus_bw_gbps"`
	SlowestRank         *int           `json:"slowest_rank"`
	CollectiveBreakdown map[string]int `json:"collective_breakdown"`
}

func ParseFile(path string, tail int) []CollPerf {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLines(string(b), tail)
}

func ParseLines(text string, tail int) []CollPerf {
	lines := strings.Split(text, "\n")
	if tail > 0 && len(lines) > tail {
		lines = lines[len(lines)-tail:]
	}
	var out []CollPerf
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if rec, ok := parseLine(line); ok {
			out = append(out, rec)
		}
	}
	return out
}

func parseLine(line string) (CollPerf, bool) {
	var obj map[string]any
	if err := json.Unmarshal([]byte(line), &obj); err != nil {
		return CollPerf{}, false
	}
	header, _ := obj["header"].(map[string]any)
	meta, _ := obj["metadata"].(map[string]any)
	perf, _ := obj["coll_perf"].(map[string]any)
	if header == nil || meta == nil || perf == nil {
		return CollPerf{}, false
	}
	fmtVer, _ := meta["inspector_output_format_version"].(string)
	if fmtVer == "" {
		fmtVer = "v4.0"
	}
	var graph *bool
	if v, ok := perf["graphCaptured"].(bool); ok {
		graph = &v
	}
	rec := CollPerf{
		Timestamp:     asFloat(meta["dump_timestamp_us"]) / 1_000_000.0,
		CommHash:      asString(header["id"]),
		Rank:          asInt(header["rank"]),
		NRanks:        asInt(header["n_ranks"]),
		NNodes:        asInt(header["nnodes"]),
		Hostname:      asString(meta["hostname"]),
		PID:           asInt(meta["pid"]),
		Collective:    asString(perf["coll"]),
		SequenceNum:   asInt(perf["coll_sn"]),
		MsgSizeBytes:  asInt64(perf["coll_msg_size_bytes"]),
		ExecTimeUS:    asInt64(perf["coll_exec_time_us"]),
		TimingSource:  asString(perf["coll_timing_source"]),
		AlgoBWGbps:    asFloat(perf["coll_algobw_gbs"]),
		BusBWGbps:     asFloat(perf["coll_busbw_gbs"]),
		FormatVersion: fmtVer,
		GraphCaptured: graph,
	}
	if rec.CommHash == "" || rec.Collective == "" {
		return CollPerf{}, false
	}
	sn, _ := perf["event_trace_sn"].(map[string]any)
	ts, _ := perf["event_trace_ts"].(map[string]any)
	if sn != nil || ts != nil {
		rec.EventTrace = map[string]any{"sn": sn, "ts": ts}
	}
	return rec, true
}

func Aggregate(records []CollPerf) Snapshot {
	var bws []float64
	var slow *int
	minV, maxV, sum := 0.0, 0.0, 0.0
	first := true
	breakdown := map[string]int{}
	for _, r := range records {
		breakdown[r.Collective]++
		if r.BusBWGbps <= 0 {
			continue
		}
		bws = append(bws, r.BusBWGbps)
		if first || r.BusBWGbps < minV {
			minV = r.BusBWGbps
			v := r.Rank
			slow = &v
			first = false
		}
		if r.BusBWGbps > maxV {
			maxV = r.BusBWGbps
		}
		sum += r.BusBWGbps
	}
	latest := map[string]CollPerf{}
	for _, r := range records {
		key := r.CommHash + "/" + fmt.Sprintf("%d", r.Rank)
		if ex, ok := latest[key]; !ok || r.SequenceNum > ex.SequenceNum {
			latest[key] = r
		}
	}
	display := make([]CollPerf, 0, len(latest))
	for _, r := range latest {
		display = append(display, r)
	}
	sort.Slice(display, func(i, j int) bool {
		if display[i].Rank != display[j].Rank {
			return display[i].Rank < display[j].Rank
		}
		return display[i].CommHash < display[j].CommHash
	})
	ts := float64(time.Now().UnixNano()) / 1e9
	if len(records) > 0 {
		ts = records[0].Timestamp
	}
	snap := Snapshot{Timestamp: ts, Records: display, CollectiveBreakdown: breakdown}
	if len(bws) > 0 {
		avg := sum / float64(len(bws))
		snap.AvgBusBWGbps = &avg
		snap.MinBusBWGbps = &minV
		snap.MaxBusBWGbps = &maxV
		snap.SlowestRank = slow
	}
	return snap
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asInt(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	}
	return 0
}

func asInt64(v any) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case int:
		return int64(t)
	case int64:
		return t
	}
	return 0
}

func asFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	}
	return 0
}
