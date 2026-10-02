package collectors

import "testing"

func TestPythonGPUPayloadKeys(t *testing.T) {
	in := map[string]NodeGPUMetrics{
		"n1": {Host: "n1", GPUs: []GPU{{
			Index: 0, UtilizationPct: 12, MemTotalMB: 1024, MemUsedMB: 256,
			TempEdgeC: 40, TempHotspotC: 50, PowerW: 200,
			PCIe: &PCIeInfo{Width: "x16", Speed: "32 GT/s", ReplayCount: 1},
			ECC:  &ECCInfo{Correctable: 2, Uncorrectable: 0},
		}}},
	}
	p := PythonGPUPayload(in)
	for _, k := range []string{"utilization", "memory", "temperature", "power", "pcie", "xgmi", "ras_errors"} {
		if _, ok := p[k]; !ok {
			t.Fatalf("missing %s", k)
		}
	}
	util := p["utilization"].(map[string]any)["n1"].(map[string]any)
	card := util["card0"].(map[string]any)
	if _, ok := card["GPU use (%)"]; !ok {
		t.Fatal("missing GPU use (%)")
	}
}

func TestPythonGPUPayloadXGMIGpuData(t *testing.T) {
	in := map[string]NodeGPUMetrics{
		"n1": {Host: "n1", GPUs: []GPU{{
			Index: 0, PowerW: 200,
			XGMI: &XGMIInfo{Status: "OK", ErrorCount: 3},
		}}},
	}
	p := PythonGPUPayload(in)
	xg := p["xgmi"].(map[string]any)["n1"].(map[string]any)
	rows, ok := xg["gpu_data"].([]any)
	if !ok || len(rows) != 1 {
		t.Fatalf("xgmi gpu_data=%v", xg)
	}
	row := rows[0].(map[string]any)
	if row["gpu"] != 0 {
		t.Fatalf("gpu index %v", row["gpu"])
	}
	errf := row["xgmi_err"].(map[string]any)
	if errf["error_count"] != 3 {
		t.Fatalf("xgmi_err=%v", errf)
	}
	pwr := p["power"].(map[string]any)["n1"].(map[string]any)
	if _, ok := pwr["gpu_data"]; !ok {
		t.Fatal("power missing gpu_data")
	}
}

func TestPythonNICPayloadLLDP(t *testing.T) {
	in := map[string]NodeNICMetrics{
		"n1": {Host: "n1", RDMALinks: []RDMALink{{Device: "mlx5_0", Netdev: "eth0", State: "ACTIVE"}}, Interfaces: []NICInterface{{Name: "eth0", IPv4: []string{"10.0.0.1/24"}}}},
	}
	raw := map[string]any{
		"n1": map[string]any{"lldp": map[string]any{"interface": []any{map[string]any{"eth0": map[string]any{"chassis": "sw1"}}}}},
	}
	p := PythonNICPayload(in, raw)
	ip := p["ip_addr"].(map[string]any)["n1"].(map[string]any)["eth0"].(map[string]any)
	if ip["ipv4_addr_list"] == nil {
		t.Fatal("ipv4_addr_list missing")
	}
}
