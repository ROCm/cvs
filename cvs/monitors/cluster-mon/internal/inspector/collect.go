package inspector

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

const inspSentinel = "__INSP_EOF__"

func Collect(ctx context.Context, pool *pssh.Pool, cfg config.InspectorConfig, activePIDs map[int]struct{}) Snapshot {
	if !cfg.Enabled {
		return Snapshot{Records: []CollPerf{}, CollectiveBreakdown: map[string]int{}}
	}
	var recs []CollPerf
	if strings.ToLower(cfg.Mode) == "ssh" && pool != nil {
		recs = collectSSH(ctx, pool, cfg, activePIDs)
	} else {
		recs = collectFile(cfg)
	}
	return Aggregate(recs)
}

func collectFile(cfg config.InspectorConfig) []CollPerf {
	if cfg.DumpDir == "" {
		return nil
	}
	matches, err := filepath.Glob(filepath.Join(cfg.DumpDir, "*.log"))
	if err != nil {
		return nil
	}
	tail := cfg.MaxRecordsPerFile
	if tail <= 0 {
		tail = 100
	}
	var recs []CollPerf
	for _, p := range matches {
		recs = append(recs, ParseFile(p, tail)...)
	}
	return recs
}

func collectSSH(ctx context.Context, pool *pssh.Pool, cfg config.InspectorConfig, activePIDs map[int]struct{}) []CollPerf {
	if cfg.DumpDir == "" {
		return nil
	}
	hosts := pool.Reachable()
	cmd := inspectorSSHCmd(cfg.DumpDir, cfg.MaxRecordsPerFile, activePIDs)
	var recs []CollPerf
	for _, host := range hosts {
		out := pool.ExecHosts(ctx, cmd, []string{host})
		raw := stripInspectorSSH(out[host].Output)
		recs = append(recs, ParseLines(raw, 0)...)
	}
	return recs
}

// inspectorSSHCmd matches the Python collector: each node tails only
// ${HOSTNAME}-pid*.log so a shared NFS dump_dir is not double-read.
func inspectorSSHCmd(dumpDir string, tail int, activePIDs map[int]struct{}) string {
	if tail <= 0 {
		tail = 100
	}
	if len(activePIDs) == 0 {
		return fmt.Sprintf("tail -n %d %s/${HOSTNAME}-pid*.log 2>/dev/null; echo %s", tail, dumpDir, inspSentinel)
	}
	pids := make([]int, 0, len(activePIDs))
	for pid := range activePIDs {
		pids = append(pids, pid)
	}
	sort.Ints(pids)
	parts := make([]string, 0, len(pids))
	for _, pid := range pids {
		parts = append(parts, fmt.Sprintf("pid%d", pid))
	}
	pat := strings.Join(parts, "|")
	return fmt.Sprintf(
		"ls %s/ 2>/dev/null | grep -E '^${HOSTNAME}-(%s)\\.log$' | xargs -I{} tail -n %d %s/{} 2>/dev/null; echo %s",
		dumpDir, pat, tail, dumpDir, inspSentinel,
	)
}

func stripInspectorSSH(output string) string {
	var lines []string
	for _, line := range strings.Split(output, "\n") {
		if strings.TrimSpace(line) == inspSentinel || strings.HasPrefix(line, "==>") {
			continue
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func DumpDirExists(dir string) bool {
	if dir == "" {
		return false
	}
	st, err := os.Stat(dir)
	return err == nil && st.IsDir()
}

func InspectorV5Hosts(records []CollPerf) []string {
	seen := map[string]struct{}{}
	var hosts []string
	for _, r := range records {
		if r.GraphCaptured == nil || r.Hostname == "" {
			continue
		}
		if _, ok := seen[r.Hostname]; ok {
			continue
		}
		seen[r.Hostname] = struct{}{}
		hosts = append(hosts, r.Hostname)
	}
	return hosts
}
