package collectors

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

// logsCollectTimeout bounds a full dmesg log sweep (three commands per host).
const logsCollectTimeout = 90 * time.Second

// logsTTL is how long a completed snapshot is served before a background
// re-sweep is triggered. Matches cluster-mon's software_cache_ttl (180s).
const logsTTL = 180 * time.Second

// Logs collector commands (verbatim parity with the cluster-mon Python logs
// collector). Each pipes through `|| echo ""` so a missing tool yields an empty
// string rather than a shell error.
// dedupAWK is the awk program (ready to embed inside bash -c '...') that
// strips the [timestamp] bracket from each line to form a dedup key, keeping
// the first occurrence of each distinct message and discarding repetitions.
// Single quotes are expressed as '"'"' so they survive the outer bash -c '...'.
const dedupAWK = `awk '"'"'{ key=$0; gsub(/\[[^]]*\]/,"[]",key); if (!seen[key]++) print }'"'"' | tail -200`

const (
	cmdAMDLogs = `bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | grep -iE "PCIe|XGMI|amdgpu|epyc|cpu|ionic|bnxt|mlnx|mellanox|Link|error|fail" 2>/dev/null | grep -iv "vital buffer" 2>/dev/null | ` + dedupAWK + ` || echo ""'`

	cmdDmesgErrors = `bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err 2>/dev/null | ` + dedupAWK + ` || echo ""'`

	cmdUserspaceErrors = `bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | egrep -i "oom|out of memory|killed process|segfault|general protection|call trace|bug:|hardware error|mce|stack trace|pytorch|torch|tensorflow|megatron|jax|vllm|sglang|triton.*error|triton.*exception|triton.*failed" 2>/dev/null | ` + dedupAWK + ` || echo ""'`
)

// NodeLogs is one node's collected dmesg log buckets. Empty strings mean a clean
// node (no matching lines).
type NodeLogs struct {
	Host            string `json:"host"`
	AMDLogs         string `json:"amd_logs"`
	DmesgErrors     string `json:"dmesg_errors"`
	UserspaceErrors string `json:"userspace_errors"`
	Error           string `json:"error,omitempty"`
}

// LogsSnapshot is one fleet-wide log sweep.
type LogsSnapshot struct {
	CollectedAt time.Time  `json:"collected_at"`
	Nodes       []NodeLogs `json:"nodes"`
	Collecting  bool       `json:"collecting,omitempty"` // background sweep in progress
}

// SearchResult is one node's first matching lines for a grep search.
type SearchResult struct {
	Host   string `json:"host"`
	Output string `json:"output"`
}

// SearchResponse wraps an ad-hoc dmesg grep search across the fleet.
type SearchResponse struct {
	GrepCommand      string         `json:"grep_command"`
	Results          []SearchResult `json:"results"`
	TotalNodes       int            `json:"total_nodes_searched"`
	NodesWithResults int            `json:"nodes_with_results"`
}

// LogsService runs the dmesg log collectors + ad-hoc grep search over the fleet
// singleton SSH pool. Results are cached for logsTTL. Cold-cache HTTP calls wait
// for the first sweep so the Logs page does not store an empty payload.
type LogsService struct {
	getPool func() *pssh.Pool
	logger  *slog.Logger

	mu         sync.Mutex
	cache      *LogsSnapshot
	collecting bool
	waiters    []chan error
}

// NewLogsService wires the logs collectors to the fleet singleton pool getter.
func NewLogsService(getPool func() *pssh.Pool, logger *slog.Logger) *LogsService {
	if logger == nil {
		logger = slog.Default()
	}
	return &LogsService{getPool: getPool, logger: logger}
}

// Logs returns a fresh cached snapshot, or waits for a sweep when the cache is
// empty or older than logsTTL.
func (s *LogsService) Logs(ctx context.Context) (*LogsSnapshot, error) {
	if snap := s.logsIfFresh(); snap != nil {
		return snap, nil
	}
	wait := s.kickLogsCollect()
	select {
	case <-ctx.Done():
		if snap := s.logsCached(); snap != nil {
			return snap, nil
		}
		return nil, ctx.Err()
	case err := <-wait:
		if snap := s.logsCached(); snap != nil {
			return snap, nil
		}
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("logs collection produced no data")
	}
}

func (s *LogsService) logsIfFresh() *LogsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cache != nil && time.Since(s.cache.CollectedAt) < logsTTL {
		return s.cache
	}
	return nil
}

func (s *LogsService) logsCached() *LogsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cache
}

func (s *LogsService) kickLogsCollect() <-chan error {
	ch := make(chan error, 1)
	s.mu.Lock()
	if s.cache != nil && time.Since(s.cache.CollectedAt) < logsTTL {
		s.mu.Unlock()
		ch <- nil
		return ch
	}
	s.waiters = append(s.waiters, ch)
	if s.collecting {
		s.mu.Unlock()
		return ch
	}
	s.collecting = true
	s.mu.Unlock()
	go s.runLogsCollect()
	return ch
}

func (s *LogsService) runLogsCollect() {
	ctx, cancel := context.WithTimeout(context.Background(), logsCollectTimeout)
	defer cancel()
	snap, err := s.collect(ctx)
	s.mu.Lock()
	if err != nil {
		s.logger.Warn("logs_collect_failed", "err", err)
	} else {
		s.cache = snap
		s.logger.Info("logs_collected", "nodes", len(snap.Nodes))
	}
	waiters := s.waiters
	s.waiters = nil
	s.collecting = false
	s.mu.Unlock()
	for _, w := range waiters {
		w <- err
	}
}

// collect runs the three dmesg commands fleet-wide and returns a fresh snapshot.
func (s *LogsService) collect(ctx context.Context) (*LogsSnapshot, error) {
	pool := s.getPool()
	if pool == nil {
		return nil, fmt.Errorf("no SSH pool: inventory not configured")
	}

	var amdRes, dmesgRes, userspaceRes map[string]pssh.Result
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); amdRes = pool.Exec(ctx, cmdAMDLogs) }()
	go func() { defer wg.Done(); dmesgRes = pool.Exec(ctx, cmdDmesgErrors) }()
	go func() { defer wg.Done(); userspaceRes = pool.Exec(ctx, cmdUserspaceErrors) }()
	wg.Wait()

	nodes := make([]NodeLogs, 0, len(amdRes))
	for host, r := range amdRes {
		m := NodeLogs{Host: host}
		if r.Err != nil {
			m.Error = r.Err.Error()
		} else {
			m.AMDLogs = strings.TrimSpace(r.Output)
		}
		if dr, ok := dmesgRes[host]; ok && dr.Err == nil {
			m.DmesgErrors = strings.TrimSpace(dr.Output)
		}
		if ur, ok := userspaceRes[host]; ok && ur.Err == nil {
			m.UserspaceErrors = strings.TrimSpace(ur.Output)
		}
		nodes = append(nodes, m)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Host < nodes[j].Host })
	return &LogsSnapshot{CollectedAt: time.Now().UTC(), Nodes: nodes}, nil
}

// Search runs a validated grep pipeline against `sudo dmesg -T` on each node,// returning the first 5 matching lines per node.
func (s *LogsService) Search(ctx context.Context, grepCommand string) (*SearchResponse, error) {
	if err := validateGrepCommand(grepCommand); err != nil {
		return nil, &invalidGrepError{msg: err.Error()}
	}
	pool := s.getPool()
	if pool == nil {
		return nil, fmt.Errorf("no SSH pool: inventory not configured")
	}

	// Escape single quotes for the outer bash -c '...': ' -> '\''.
	escaped := strings.ReplaceAll(grepCommand, "'", `'\''`)
	cmd := fmt.Sprintf(`bash -c 'sudo dmesg -T 2>/dev/null | %s | head -5 || echo ""'`, escaped)

	execResults := pool.Exec(ctx, cmd)
	results := make([]SearchResult, 0, len(execResults))
	withResults := 0
	for host, r := range execResults {
		if r.Err != nil {
			continue
		}
		out := strings.TrimSpace(r.Output)
		results = append(results, SearchResult{Host: host, Output: out})
		if out != "" {
			withResults++
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Host < results[j].Host })
	return &SearchResponse{
		GrepCommand:      grepCommand,
		Results:          results,
		TotalNodes:       len(results),
		NodesWithResults: withResults,
	}, nil
}

// invalidGrepError marks a rejected grep command so the handler can return 400.
type invalidGrepError struct{ msg string }

func (e *invalidGrepError) Error() string { return e.msg }

// IsInvalidGrep reports whether err is a rejected grep command (HTTP 400).
func IsInvalidGrep(err error) bool {
	var e *invalidGrepError
	return errors.As(err, &e)
}

var (
	grepDangerousChars = []string{";", "&", "$", "`", "(", ")", "{", "}", "<", ">", "\n", "\r"}
	grepDangerousWords = []string{
		"bash", "sh", "exec", "eval", "sudo", "rm", "mv", "cp", "dd", "cat", "tee", "chmod", "chown",
	}
	grepAllowedFlags = map[string]bool{
		"-i": true, "-v": true, "-E": true, "-A": true, "-B": true, "-C": true,
		"-w": true, "-x": true, "-o": true, "-n": true, "-c": true, "-m": true,
	}
)

// validateGrepCommand mirrors the cluster-mon Python validate_grep_command:
// only grep/egrep segments joined by pipes, a safe-flag allowlist, and a
// dangerous-character/keyword denylist. The keyword check is substring-based
// (parity), so words containing e.g. "sh"/"cat" are rejected.
func validateGrepCommand(grepCmd string) error {
	if strings.TrimSpace(grepCmd) == "" {
		return fmt.Errorf("empty command")
	}
	for _, c := range grepDangerousChars {
		if strings.Contains(grepCmd, c) {
			return fmt.Errorf("invalid character %q in command", c)
		}
	}
	lower := strings.ToLower(grepCmd)
	for _, kw := range grepDangerousWords {
		if strings.Contains(lower, kw) {
			return fmt.Errorf("command contains forbidden keyword: %s", kw)
		}
	}

	for _, seg := range strings.Split(grepCmd, "|") {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		if !strings.HasPrefix(seg, "grep ") && !strings.HasPrefix(seg, "egrep ") {
			return fmt.Errorf("invalid segment (must start with 'grep' or 'egrep'): %s", seg)
		}
		parts := strings.Fields(seg)
		for _, part := range parts[1:] {
			if !strings.HasPrefix(part, "-") {
				continue
			}
			if grepAllowedFlags[part] {
				continue
			}
			if combinedFlagsAllowed(part) {
				continue
			}
			base := part
			if i := strings.IndexByte(part, '='); i >= 0 {
				base = part[:i]
			}
			if grepAllowedFlags[base] || grepAllowedFlags[strings.TrimRight(base, "0123456789")] {
				continue
			}
			return fmt.Errorf("invalid flag: %s", part)
		}
	}
	return nil
}

// combinedFlagsAllowed checks bundled short flags like -iE (every alpha char
// must be an allowed single flag).
func combinedFlagsAllowed(part string) bool {
	for _, c := range part[1:] {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			if !grepAllowedFlags["-"+string(c)] {
				return false
			}
		}
	}
	return true
}
