package clustermon

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

// A representative amd-smi metric --json payload (list form), with the
// {value,unit} object shape amd-smi uses, plus a WARNING banner to strip.
const sampleMetricList = `WARNING: amd_smi not in render group
[
  {
    "gpu": 1,
    "usage": {"gfx_activity": {"value": 73, "unit": "%"}, "umc_activity": {"value": 12, "unit": "%"}},
    "mem_usage": {"total_vram": {"value": 196592, "unit": "MB"}, "used_vram": {"value": 49148, "unit": "MB"}},
    "temperature": {"edge": {"value": 45, "unit": "C"}, "hotspot": {"value": 61, "unit": "C"}, "mem": {"value": 58, "unit": "C"}},
    "power": {"socket_power": {"value": 142, "unit": "W"}}
  },
  {
    "gpu": 0,
    "usage": {"gfx_activity": {"value": 0, "unit": "%"}},
    "mem_usage": {"total_vram": {"value": 196592}, "used_vram": {"value": 0}},
    "temperature": {"edge": {"value": 30}, "hotspot": "N/A"},
    "power": {"average_socket_power": {"value": 90}}
  }
]`

// gpu_data wrapper form, bare-number value fields.
const sampleMetricWrapped = `{"gpu_data":[{"gpu":0,"usage":{"gfx_activity":50},"mem_usage":{"total_vram":1000,"used_vram":250},"temperature":{"edge":40},"power":{"socket_power":100}}]}`

func TestParseGPUMetric_ListForm(t *testing.T) {
	gpus, perr := parseGPUMetric(sampleMetricList)
	if perr != "" {
		t.Fatalf("unexpected parse error: %s", perr)
	}
	if len(gpus) != 2 {
		t.Fatalf("want 2 gpus, got %d", len(gpus))
	}
	// Sorted by index, so gpu 0 first.
	if gpus[0].Index != 0 || gpus[1].Index != 1 {
		t.Fatalf("not sorted by index: %+v", gpus)
	}
	g1 := gpus[1]
	if !approx(g1.UtilizationPct, 73) || !approx(g1.PowerW, 142) {
		t.Fatalf("gpu1 util/power wrong: %+v", g1)
	}
	if !approx(g1.MemTotalMB, 196592) || !approx(g1.MemUsedMB, 49148) {
		t.Fatalf("gpu1 mem wrong: %+v", g1)
	}
	if !approx(g1.MemUsedPct, 49148.0/196592.0*100) {
		t.Fatalf("gpu1 mem pct wrong: %v", g1.MemUsedPct)
	}
	if !approx(g1.TempEdgeC, 45) || !approx(g1.TempHotspotC, 61) || !approx(g1.TempMemC, 58) {
		t.Fatalf("gpu1 temps wrong: %+v", g1)
	}
	// gpu0 uses average_socket_power fallback and "N/A" hotspot -> 0.
	g0 := gpus[0]
	if !approx(g0.PowerW, 90) || !approx(g0.TempHotspotC, 0) {
		t.Fatalf("gpu0 power/hotspot fallback wrong: %+v", g0)
	}
}

func TestParseGPUMetric_WrappedBareNumbers(t *testing.T) {
	gpus, perr := parseGPUMetric(sampleMetricWrapped)
	if perr != "" {
		t.Fatalf("unexpected parse error: %s", perr)
	}
	if len(gpus) != 1 {
		t.Fatalf("want 1 gpu, got %d", len(gpus))
	}
	g := gpus[0]
	if !approx(g.UtilizationPct, 50) || !approx(g.MemUsedPct, 25) || !approx(g.PowerW, 100) {
		t.Fatalf("wrapped parse wrong: %+v", g)
	}
}

const samplePCIe = `[
  {"gpu": 0, "pcie": {"width": 16, "speed": {"value": 32, "unit": "GT/s"}, "bandwidth": {"value": 25, "unit": "Mb/s"}, "replay_count": 3, "l0_to_recovery_count": 1, "nak_sent_count": 0, "nak_received_count": 2}}
]`

const sampleECC = `{"gpu_data": [
  {"gpu": 0, "ecc": {"total_correctable_count": 5, "total_uncorrectable_count": 0, "total_deferred_count": 1}}
]}`

const sampleXGMIString = `[{"gpu": 0, "xgmi_err": "No errors detected"}]`
const sampleXGMIObj = `[{"gpu": 0, "xgmi_err": {"read_error_count": 2, "write_error_count": 1, "data_kb": 999}}]`

func TestParsePCIeBlock(t *testing.T) {
	m := parseGPUBlocks(samplePCIe, parsePCIeBlock)
	p := m[0]
	if p == nil {
		t.Fatal("no pcie for gpu 0")
	}
	if p.Width != "x16" || p.Speed != "32 GT/s" || p.Bandwidth != "25 Mb/s" {
		t.Fatalf("pcie link wrong: %+v", p)
	}
	if p.ReplayCount != 3 || p.NakReceivedCount != 2 || p.L0ToRecoveryCount != 1 {
		t.Fatalf("pcie counters wrong: %+v", p)
	}
}

func TestParseECCBlock(t *testing.T) {
	m := parseGPUBlocks(sampleECC, parseECCBlock)
	e := m[0]
	if e == nil {
		t.Fatal("no ecc for gpu 0")
	}
	if e.Correctable != 5 || e.Uncorrectable != 0 || e.Deferred != 1 {
		t.Fatalf("ecc wrong: %+v", e)
	}
}

func TestParseXGMIBlock(t *testing.T) {
	ms := parseGPUBlocks(sampleXGMIString, parseXGMIBlock)
	if ms[0] == nil || ms[0].Status != "No errors detected" {
		t.Fatalf("xgmi string wrong: %+v", ms[0])
	}
	mo := parseGPUBlocks(sampleXGMIObj, parseXGMIBlock)
	if mo[0] == nil || mo[0].ErrorCount != 3 {
		t.Fatalf("xgmi obj error count wrong: %+v", mo[0])
	}
}

