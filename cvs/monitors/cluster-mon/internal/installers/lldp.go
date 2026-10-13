package installers

import (
	"context"
	"strings"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

// LLDPInstall runs lldpd install per OS on reachable hosts using ExecHosts
// so Ubuntu nodes do not receive yum and RHEL nodes do not receive apt.
func LLDPInstall(ctx context.Context, pool *pssh.Pool, hosts []string) map[string]any {
	if pool == nil {
		return map[string]any{"package": "lldpd", "total_nodes": 0, "successful": 0, "failed": 0, "results": map[string]any{}}
	}
	if len(hosts) == 0 {
		hosts = pool.Reachable()
	} else {
		allow := map[string]struct{}{}
		for _, h := range pool.Reachable() {
			allow[h] = struct{}{}
		}
		var filtered []string
		for _, h := range hosts {
			if _, ok := allow[h]; ok {
				filtered = append(filtered, h)
			}
		}
		hosts = filtered
	}
	osMap := detectOS(ctx, pool, hosts)
	installed := checkLLDP(ctx, pool, hosts)

	results := map[string]any{}
	already := 0
	var ubuntu, rhel, fedora []string
	for _, h := range hosts {
		if installed[h] {
			already++
			results[h] = map[string]any{"success": true, "already_installed": true, "os_type": osMap[h]}
			continue
		}
		switch osMap[h] {
		case "ubuntu", "debian":
			ubuntu = append(ubuntu, h)
		case "rhel", "centos", "rocky", "almalinux":
			rhel = append(rhel, h)
		case "fedora":
			fedora = append(fedora, h)
		default:
			results[h] = map[string]any{"success": false, "os_type": osMap[h], "error": "Unsupported OS: " + osMap[h]}
		}
	}

	run := func(hosts []string, cmd string) {
		if len(hosts) == 0 {
			return
		}
		out := pool.ExecHosts(ctx, cmd, hosts)
		for _, h := range hosts {
			r := out[h]
			ok := r.Err == nil && !containsFail(r.Output)
			entry := map[string]any{"success": ok, "os_type": osMap[h], "output": r.Output}
			if r.Err != nil {
				entry["error"] = r.Err.Error()
				entry["success"] = false
			}
			results[h] = entry
		}
	}
	run(ubuntu, "bash -c 'sudo apt-get update && sudo apt-get install -y lldpd && sudo systemctl enable lldpd && sudo systemctl start lldpd'")
	run(rhel, "bash -c 'sudo yum install -y lldpd && sudo systemctl enable lldpd && sudo systemctl start lldpd'")
	run(fedora, "bash -c 'sudo dnf install -y lldpd && sudo systemctl enable lldpd && sudo systemctl start lldpd'")

	ok, fail := 0, 0
	for _, v := range results {
		m, _ := v.(map[string]any)
		if m["success"] == true {
			ok++
		} else {
			fail++
		}
	}
	return map[string]any{
		"package":           "lldpd",
		"total_nodes":       len(hosts),
		"successful":        ok,
		"failed":            fail,
		"already_installed": already,
		"results":           results,
	}
}

// LLDPStatus reports whether lldpcli is present on each reachable host.
func LLDPStatus(ctx context.Context, pool *pssh.Pool) map[string]any {
	if pool == nil {
		return map[string]any{"package": "lldp", "total_nodes": 0, "installed_count": 0, "not_installed_count": 0, "installed_nodes": []string{}, "not_installed_nodes": []string{}, "status_by_node": map[string]any{}}
	}
	hosts := pool.Reachable()
	installed := checkLLDP(ctx, pool, hosts)
	var yes, no []string
	status := map[string]any{}
	for _, h := range hosts {
		status[h] = installed[h]
		if installed[h] {
			yes = append(yes, h)
		} else {
			no = append(no, h)
		}
	}
	return map[string]any{
		"package":             "lldp",
		"total_nodes":         len(hosts),
		"installed_count":     len(yes),
		"not_installed_count": len(no),
		"installed_nodes":     yes,
		"not_installed_nodes": no,
		"status_by_node":      status,
	}
}

func detectOS(ctx context.Context, pool *pssh.Pool, hosts []string) map[string]string {
	cmd := `bash -c 'if [ -f /etc/os-release ]; then . /etc/os-release; echo "$ID"; elif [ -f /etc/redhat-release ]; then echo "rhel"; else echo "unknown"; fi'`
	out := pool.ExecHosts(ctx, cmd, hosts)
	m := map[string]string{}
	for _, h := range hosts {
		m[h] = strings.ToLower(strings.TrimSpace(out[h].Output))
		if m[h] == "" {
			m[h] = "unknown"
		}
	}
	return m
}

func checkLLDP(ctx context.Context, pool *pssh.Pool, hosts []string) map[string]bool {
	out := pool.ExecHosts(ctx, "which lldpcli", hosts)
	m := map[string]bool{}
	for _, h := range hosts {
		s := strings.ToLower(out[h].Output)
		m[h] = out[h].Err == nil && strings.TrimSpace(out[h].Output) != "" &&
			!strings.Contains(s, "not found") && !strings.Contains(s, "no such")
	}
	return m
}

func containsFail(s string) bool {
	l := strings.ToLower(s)
	return strings.Contains(l, "error") || strings.Contains(l, "failed") || strings.Contains(l, "unable to") || strings.Contains(l, "could not")
}
