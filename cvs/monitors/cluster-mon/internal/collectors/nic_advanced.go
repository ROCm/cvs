package collectors

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

const cmdNICPCIe = `sudo lspci -vvv 2>/dev/null | egrep -A 30 -i 'ethernet|network' | egrep '^[0-9a-f]{2}:|Ethernet|Network|LnkCap:|LnkSta:' || true`

var (
	reBDF     = regexp.MustCompile(`^([0-9a-fA-F:.]+)`)
	reLnkCap  = regexp.MustCompile(`LnkCap:.*Speed\s+([^,]+),\s*Width\s+([^,]+)`)
	reLnkSta  = regexp.MustCompile(`LnkSta:.*Speed\s+([^,]+),\s*Width\s+([^,\s]+)`)
	reCongest = regexp.MustCompile(`(?i)(pfc|pause|ecn|cnp|drop|err|timeout)`)
)

// CollectNICAdvanced matches GET /api/software/nic/advanced.
func CollectNICAdvanced(ctx context.Context, pool *pssh.Pool, logger *slog.Logger) map[string]any {
	if logger == nil {
		logger = slog.Default()
	}
	if pool == nil {
		return map[string]any{"timestamp": time.Now().UTC().Format(time.RFC3339), "nic_pcie": map[string]any{}, "congestion": map[string]any{}, "mellanox": map[string]any{}, "broadcom": map[string]any{}}
	}
	pcie := collectNICPCIe(ctx, pool)
	cong := collectCongestion(ctx, pool)
	return map[string]any{
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
		"nic_pcie":   pcie,
		"congestion": cong,
		"mellanox":   map[string]any{},
		"broadcom":   map[string]any{},
	}
}

func collectNICPCIe(ctx context.Context, pool *pssh.Pool) map[string]any {
	out := map[string]any{}
	for host, r := range pool.Exec(ctx, cmdNICPCIe) {
		if r.Err != nil || strings.TrimSpace(r.Output) == "" {
			continue
		}
		out[host] = parseNICPCIe(r.Output)
	}
	return out
}

func parseNICPCIe(output string) map[string]any {
	devs := map[string]any{}
	var bdf, device, capSpeed, capWidth, staSpeed, staWidth string
	flush := func() {
		if bdf == "" {
			return
		}
		devs[bdf] = map[string]any{
			"device":             device,
			"link_speed_cap":     capSpeed,
			"link_width_cap":     capWidth,
			"link_speed_current": staSpeed,
			"link_width_current": staWidth,
			"pcie_gen":           pcieGen(staSpeed),
		}
	}
	for _, line := range strings.Split(output, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if m := reBDF.FindStringSubmatch(trimmed); m != nil && strings.Contains(trimmed, ":") {
			if strings.Contains(lower, "ethernet") || strings.Contains(lower, "network") {
				flush()
				bdf = m[1]
				device = trimmed
				capSpeed, capWidth, staSpeed, staWidth = "", "", "", ""
			}
		}
		if m := reLnkCap.FindStringSubmatch(line); m != nil {
			capSpeed, capWidth = strings.TrimSpace(m[1]), strings.TrimSpace(m[2])
		}
		if m := reLnkSta.FindStringSubmatch(line); m != nil {
			staSpeed, staWidth = strings.TrimSpace(m[1]), strings.TrimSpace(m[2])
		}
	}
	flush()
	return devs
}

func pcieGen(speed string) string {
	f := leadingFloat(speed)
	switch {
	case f >= 32:
		return "Gen5"
	case f >= 16:
		return "Gen4"
	case f >= 8:
		return "Gen3"
	case f >= 5:
		return "Gen2"
	case f > 0:
		return "Gen1"
	default:
		return ""
	}
}

func leadingFloat(s string) float64 {
	s = strings.TrimSpace(s)
	end := 0
	for end < len(s) && (s[end] == '.' || (s[end] >= '0' && s[end] <= '9')) {
		end++
	}
	if end == 0 {
		return 0
	}
	var f float64
	fmt.Sscan(s[:end], &f)
	return f
}

func collectCongestion(ctx context.Context, pool *pssh.Pool) map[string]any {
	out := map[string]any{}
	for host, r := range pool.Exec(ctx, `bash -c 'rdma statistic show --json 2>/dev/null || echo "[]"'`) {
		if r.Err != nil {
			continue
		}
		js := extractJSON(r.Output)
		var data []map[string]any
		if err := json.Unmarshal([]byte(js), &data); err != nil {
			out[host] = map[string]any{}
			continue
		}
		devs := map[string]any{}
		for _, entry := range data {
			ifname, _ := entry["ifname"].(string)
			if ifname == "" {
				continue
			}
			port := entry["port"]
			key := ifname
			if port != nil {
				key = ifname + "/" + stringify(port)
			}
			stats := map[string]any{}
			for k, v := range entry {
				if k == "ifname" || k == "port" || k == "ifindex" {
					continue
				}
				if reCongest.MatchString(k) {
					stats[k] = v
				}
			}
			if len(stats) > 0 {
				devs[key] = stats
			}
		}
		out[host] = devs
	}
	return out
}

func stringify(v any) string {
	return fmt.Sprintf("%v", v)
}
