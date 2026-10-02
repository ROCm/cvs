package httpserver

import "testing"

func TestNodeHealthECCAndPCIe(t *testing.T) {
	gpu := map[string]any{
		"ras_errors": map[string]any{
			"n1": map[string]any{"gpu_data": []any{
				map[string]any{"gpu": 0, "ecc": map[string]any{"total_uncorrectable": 1, "total_correctable": 0}},
			}},
		},
	}
	st, issues := nodeHealth("n1", gpu, 85)
	if st != "unhealthy" || len(issues) != 1 {
		t.Fatalf("%s %v", st, issues)
	}
	gpu = map[string]any{
		"pcie": map[string]any{
			"n1": map[string]any{"gpu_data": []any{
				map[string]any{"gpu": 1, "pcie": map[string]any{"replay_count": 101, "nak_count": 0}},
			}},
		},
	}
	st, issues = nodeHealth("n1", gpu, 85)
	if st != "unhealthy" || len(issues) != 1 {
		t.Fatalf("%s %v", st, issues)
	}
}

func TestNodeHealthXGMIBoundary(t *testing.T) {
	xgmi := func(n int) map[string]any {
		return map[string]any{"xgmi": map[string]any{"n1": map[string]any{"gpu_data": []any{
			map[string]any{"gpu": 0, "xgmi": map[string]any{"error_count": n}},
		}}}}
	}
	st, _ := nodeHealth("n1", xgmi(10), 85)
	if st != "healthy" {
		t.Fatal(st)
	}
	st, issues := nodeHealth("n1", xgmi(11), 85)
	if st != "unhealthy" || len(issues) != 1 {
		t.Fatalf("%s %v", st, issues)
	}
}

func TestNodeHealthTemperature(t *testing.T) {
	hot := map[string]any{"temperature": map[string]any{"n1": map[string]any{
		"card0": map[string]any{"Temperature (Sensor junction) (C)": "86"},
	}}}
	st, issues := nodeHealth("n1", hot, 85)
	if st != "unhealthy" || len(issues) != 1 {
		t.Fatalf("%s %v", st, issues)
	}
	st, _ = nodeHealth("n1", hot, 90)
	if st != "healthy" {
		t.Fatal("override should allow 86 C")
	}
	edgeOnly := map[string]any{"temperature": map[string]any{"n1": map[string]any{
		"card0": map[string]any{
			"Temperature (Sensor junction) (C)": "0",
			"Temperature (Sensor edge) (C)":     "90",
		},
	}}}
	st, _ = nodeHealth("n1", edgeOnly, 85)
	if st != "unhealthy" {
		t.Fatal("edge temp should count when junction is missing")
	}
	missing := map[string]any{"temperature": map[string]any{"n1": map[string]any{
		"card0": map[string]any{"GPU use (%)": "99"},
	}}}
	st, issues = nodeHealth("n1", missing, 85)
	if st != "healthy" || len(issues) != 0 {
		t.Fatalf("util must not mark unhealthy: %s %v", st, issues)
	}
}
