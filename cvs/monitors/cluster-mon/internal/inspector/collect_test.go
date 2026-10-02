package inspector

import "testing"

func TestInspectorSSHCmdHostnameAndPIDs(t *testing.T) {
	cmd := inspectorSSHCmd("/tmp/insp", 50, nil)
	if !containsAll(cmd, "${HOSTNAME}-pid*.log", "__INSP_EOF__") {
		t.Fatalf("cmd=%s", cmd)
	}
	cmd = inspectorSSHCmd("/tmp/insp", 20, map[int]struct{}{12345: {}, 9: {}})
	if !containsAll(cmd, "${HOSTNAME}-(pid9|pid12345)", "tail -n 20") {
		t.Fatalf("filtered cmd=%s", cmd)
	}
}

func TestInspectorV5Hosts(t *testing.T) {
	yes := true
	recs := []CollPerf{
		{Hostname: "a", GraphCaptured: &yes},
		{Hostname: "a", GraphCaptured: &yes},
		{Hostname: "b"},
	}
	hosts := InspectorV5Hosts(recs)
	if len(hosts) != 1 || hosts[0] != "a" {
		t.Fatalf("%v", hosts)
	}
}

func containsAll(s string, parts ...string) bool {
	for _, p := range parts {
		if len(p) == 0 || !contains(s, p) {
			return false
		}
	}
	return true
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		(func() bool {
			for i := 0; i+len(sub) <= len(s); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
			return false
		})())
}
