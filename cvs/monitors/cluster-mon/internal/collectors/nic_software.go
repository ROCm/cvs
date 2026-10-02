package collectors

import (
	"context"
	"encoding/json"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

const cmdLinkNames = `bash -c "ip -o link show | awk -F': ' '{print \$2}' | grep -v lo"`

// NICSoftwarePayload is GET /api/software/nic (Python collect_all_software_info).
func CollectNICSoftware(ctx context.Context, pool *pssh.Pool, logger *slog.Logger) map[string]any {
	if logger == nil {
		logger = slog.Default()
	}
	if pool == nil {
		return map[string]any{"timestamp": time.Now().UTC().Format(time.RFC3339)}
	}

	fw := collectEthtoolInfo(ctx, pool, "sudo ethtool -i %s 2>/dev/null")
	ethS := collectEthtoolStats(ctx, pool)
	pci := execMap(ctx, pool, `bash -c "lspci -nn | grep -i 'network\|ethernet'"`)
	mlx := execMap(ctx, pool, "modinfo mlx5_core 2>/dev/null | grep -E '^version|^firmware' | head -3")
	bnxt := execMap(ctx, pool, "modinfo bnxt_en 2>/dev/null | grep -E '^version|^firmware' | head -3")
	amd := execMap(ctx, pool, "modinfo amd-ainic 2>/dev/null | grep -E '^version|^firmware' | head -3 || echo 'Not loaded'")
	rdma := execMap(ctx, pool, `bash -c 'rdma statistic show --json 2>/dev/null || echo "[]"'`)

	drivers := map[string]any{}
	for _, host := range pool.Reachable() {
		d := map[string]any{}
		if info := parseModinfo(mlx[host]); len(info) > 0 {
			d["mlx5_core"] = info
		}
		if info := parseModinfo(bnxt[host]); len(info) > 0 {
			d["bnxt_en"] = info
		}
		if raw := amd[host]; raw != "" && !strings.Contains(raw, "Not loaded") {
			if info := parseModinfo(raw); len(info) > 0 {
				d["amd-ainic"] = info
			}
		}
		drivers[host] = d
	}

	rdmaStats := map[string]any{}
	for host, raw := range rdma {
		var data any
		if json.Unmarshal([]byte(extractJSON(raw)), &data) == nil {
			rdmaStats[host] = data
		} else {
			rdmaStats[host] = map[string]any{}
		}
	}

	pciDevs := map[string]any{}
	rePCI := regexp.MustCompile(`(?i)^([0-9a-f:.]+)\s+(.*)$`)
	for host, raw := range pci {
		var devices []map[string]any
		for _, line := range strings.Split(raw, "\n") {
			line = strings.TrimSpace(line)
			m := rePCI.FindStringSubmatch(line)
			if m != nil {
				devices = append(devices, map[string]any{"pci_address": m[1], "description": strings.TrimSpace(m[2])})
			}
		}
		pciDevs[host] = map[string]any{"devices": devices}
	}

	return map[string]any{
		"timestamp":          time.Now().UTC().Format(time.RFC3339),
		"nic_firmware":       fw,
		"nic_drivers":        drivers,
		"rdma_statistics":    rdmaStats,
		"ethtool_statistics": ethS,
		"pci_devices":        pciDevs,
	}
}

func execMap(ctx context.Context, pool *pssh.Pool, cmd string) map[string]string {
	out := map[string]string{}
	for host, r := range pool.Exec(ctx, cmd) {
		if r.Err != nil {
			continue
		}
		out[host] = r.Output
	}
	return out
}

func parseModinfo(raw string) map[string]string {
	if raw == "" || strings.Contains(raw, "modinfo") {
		return nil
	}
	info := map[string]string{}
	for _, line := range strings.Split(raw, "\n") {
		if i := strings.Index(line, ":"); i > 0 {
			info[strings.TrimSpace(line[:i])] = strings.TrimSpace(line[i+1:])
		}
	}
	return info
}

func collectEthtoolInfo(ctx context.Context, pool *pssh.Pool, tmpl string) map[string]any {
	ipOut := pool.Exec(ctx, cmdLinkNames)
	hostIfaces := map[string][]string{}
	all := map[string]struct{}{}
	result := map[string]any{}
	for host, r := range ipOut {
		if r.Err != nil {
			result[host] = map[string]any{"error": r.Err.Error()}
			continue
		}
		var ifaces []string
		for _, line := range strings.Split(r.Output, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.Contains(line, "@") {
				continue
			}
			ifaces = append(ifaces, line)
			if len(ifaces) >= 10 {
				break
			}
		}
		hostIfaces[host] = ifaces
		for _, i := range ifaces {
			all[i] = struct{}{}
		}
		result[host] = map[string]any{}
	}

	type keyed struct {
		iface string
		res   map[string]pssh.Result
	}
	ch := make(chan keyed, len(all))
	var wg sync.WaitGroup
	for iface := range all {
		wg.Add(1)
		go func(iface string) {
			defer wg.Done()
			cmd := "sudo ethtool -i " + iface + " 2>/dev/null"
			if strings.Contains(tmpl, "-S") {
				cmd = "sudo ethtool -S " + iface + " 2>/dev/null"
			}
			ch <- keyed{iface: iface, res: pool.Exec(ctx, cmd)}
		}(iface)
	}
	go func() { wg.Wait(); close(ch) }()
	byIface := map[string]map[string]pssh.Result{}
	for k := range ch {
		byIface[k.iface] = k.res
	}

	for host, ifaces := range hostIfaces {
		hostMap := map[string]any{}
		for _, iface := range ifaces {
			raw := ""
			if m := byIface[iface]; m != nil {
				if r, ok := m[host]; ok && r.Err == nil {
					raw = r.Output
				}
			}
			info := map[string]any{}
			for _, line := range strings.Split(raw, "\n") {
				if i := strings.Index(line, ":"); i > 0 {
					key := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(line[:i]), " ", "_"))
					info[key] = strings.TrimSpace(line[i+1:])
				}
			}
			if len(info) > 0 {
				hostMap[iface] = info
			}
		}
		result[host] = hostMap
	}
	return result
}

func collectEthtoolStats(ctx context.Context, pool *pssh.Pool) map[string]any {
	ipOut := pool.Exec(ctx, cmdLinkNames)
	hostIfaces := map[string][]string{}
	all := map[string]struct{}{}
	result := map[string]any{}
	for host, r := range ipOut {
		if r.Err != nil {
			result[host] = map[string]any{"error": r.Err.Error()}
			continue
		}
		var ifaces []string
		for _, line := range strings.Split(r.Output, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.Contains(line, "@") {
				continue
			}
			ifaces = append(ifaces, line)
			if len(ifaces) >= 10 {
				break
			}
		}
		hostIfaces[host] = ifaces
		for _, i := range ifaces {
			all[i] = struct{}{}
		}
		result[host] = map[string]any{}
	}

	byIface := map[string]map[string]pssh.Result{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for iface := range all {
		wg.Add(1)
		go func(iface string) {
			defer wg.Done()
			res := pool.Exec(ctx, "sudo ethtool -S "+iface+" 2>/dev/null")
			mu.Lock()
			byIface[iface] = res
			mu.Unlock()
		}(iface)
	}
	wg.Wait()

	reStat := regexp.MustCompile(`^\s+([\w_]+):\s+(\d+)`)
	for host, ifaces := range hostIfaces {
		hostMap := map[string]any{}
		for _, iface := range ifaces {
			raw := ""
			if m := byIface[iface]; m != nil {
				if r, ok := m[host]; ok && r.Err == nil {
					raw = r.Output
				}
			}
			if strings.Contains(raw, "NOT_AVAILABLE") {
				continue
			}
			stats := map[string]any{}
			for _, line := range strings.Split(raw, "\n") {
				if m := reStat.FindStringSubmatch(line); m != nil {
					stats[m[1]] = atoi(m[2])
				}
			}
			if len(stats) > 0 {
				hostMap[iface] = stats
			}
		}
		result[host] = hostMap
	}
	return result
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
	}
	return n
}
