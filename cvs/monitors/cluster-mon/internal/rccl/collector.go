package rccl

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

const defaultRASPort = 28028

type rasKind int

const (
	rasOK rasKind = iota
	rasRefused
	rasTimeout
	rasProtocol
	rasOther
)

type nodeCap struct {
	JSONRAS     bool
	InspectorV5 bool
	Version     string
	ProbedAt    time.Time
	TTL         time.Duration
}

// Collector polls rcclras over SSH LocalForward and emits timeline events.
type Collector struct {
	mu           sync.Mutex
	pool         *pssh.Pool
	store        *Store
	logger       *slog.Logger
	rasPort      int
	rasMaxBytes  int
	jobState     string
	bootstrapped bool
	caps         map[string]*nodeCap
}

func NewCollector(pool *pssh.Pool, store *Store, rasPort int, logger *slog.Logger) *Collector {
	if logger == nil {
		logger = slog.Default()
	}
	if rasPort <= 0 {
		rasPort = defaultRASPort
	}
	return &Collector{
		pool:        pool,
		store:       store,
		logger:      logger,
		rasPort:     rasPort,
		rasMaxBytes: defaultRASMaxBytes,
		jobState:    "no_job",
		caps:        map[string]*nodeCap{},
	}
}

func (c *Collector) SetPool(pool *pssh.Pool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pool = pool
}

func (c *Collector) SetRASPort(port int) {
	if port <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.rasPort = port
}

func (c *Collector) SetRASMaxBytes(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.rasMaxBytes = NormalizeRASMax(n)
}

func (c *Collector) JobState() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.jobState
}

// MarkInspectorV5 records Inspector v5 (graphCaptured present) per hostname
// and warns when that node last probed as text-only RAS.
func (c *Collector) MarkInspectorV5(hosts []string) {
	if c == nil || len(hosts) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.caps == nil {
		c.caps = map[string]*nodeCap{}
	}
	seen := map[string]struct{}{}
	for _, h := range hosts {
		if h == "" {
			continue
		}
		if _, ok := seen[h]; ok {
			continue
		}
		seen[h] = struct{}{}
		cap := c.caps[h]
		if cap == nil {
			continue
		}
		if !cap.InspectorV5 {
			cap.InspectorV5 = true
			c.logger.Info("inspector_v5_oracle", "host", h)
		}
		if !cap.JSONRAS {
			c.logger.Warn("inspector_v5_ras_mismatch",
				"host", h,
				"msg", "Inspector v5 fields present but json_ras=false; node may be mid-upgrade")
		}
	}
}

// Collect tries each reachable node for an rcclras listener.
// Connection refused → next node → no_job if none listen.
// Timeout on every candidate → unreachable.
// Protocol/handshake errors abort the cycle as error.
func (c *Collector) Collect(ctx context.Context, timeout time.Duration) Snapshot {
	if c == nil {
		return emptySnap("no_job")
	}
	c.bootstrap()
	if timeout <= 0 {
		timeout = 8 * time.Second
	}
	c.mu.Lock()
	pool := c.pool
	port := c.rasPort
	maxBytes := c.rasMaxBytes
	prev := c.jobState
	c.mu.Unlock()
	if pool == nil {
		return c.finish(prev, emptySnap("unreachable"), "", "No SSH pool")
	}
	hosts := pool.Reachable()
	if len(hosts) == 0 {
		return c.finish(prev, emptySnap("unreachable"), "", "No healthy nodes available for RCCL polling")
	}

	var timedOut []string
	for _, host := range hosts {
		select {
		case <-ctx.Done():
			return c.finish(prev, emptySnap("unreachable"), host, ctx.Err().Error())
		default:
		}
		snap, kind, err := c.queryHost(ctx, pool, host, port, timeout, maxBytes)
		switch kind {
		case rasOK:
			c.mu.Lock()
			cap := c.caps[host]
			c.mu.Unlock()
			c.checkSkew(&snap)
			if cap != nil && snap.JobSummary != nil && (cap.Version == "" || cap.Version == "unknown") {
				if snap.JobSummary.RCCLVersion != "" && snap.JobSummary.RCCLVersion != "unknown" {
					c.mu.Lock()
					cap.Version = snap.JobSummary.RCCLVersion
					c.mu.Unlock()
				}
			}
			return c.finish(prev, snap, host, "")
		case rasRefused:
			c.logger.Debug("rccl_ras_refused", "host", host, "port", port)
			continue
		case rasTimeout:
			c.logger.Debug("rccl_ras_timeout", "host", host, "err", err)
			timedOut = append(timedOut, host)
			continue
		case rasProtocol, rasOther:
			msg := ""
			if err != nil {
				msg = err.Error()
			}
			c.logger.Error("rccl_ras_error", "host", host, "err", err)
			return c.finish(prev, emptySnap("error"), host, msg)
		}
	}
	if len(timedOut) > 0 {
		return c.finish(prev, emptySnap("unreachable"), "", "RAS collective timed out on all nodes: "+strings.Join(timedOut, ","))
	}
	return c.finish(prev, emptySnap("no_job"), "", "Port not listening on any healthy node — no RCCL job running")
}

func (c *Collector) bootstrap() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.bootstrapped {
		return
	}
	c.bootstrapped = true
	if c.store == nil {
		return
	}
	last := c.store.Current()
	if last == nil {
		return
	}
	st, _ := last["state"].(string)
	switch st {
	case "healthy", "degraded", "no_job", "unreachable", "error":
		c.jobState = st
		c.logger.Info("rccl_bootstrapped", "state", st)
	}
	if summary, ok := last["job_summary"].(map[string]any); ok {
		if v, _ := summary["rccl_version"].(string); v != "" {
			c.store.SeedLastVersion(v)
		}
	}
}

func (c *Collector) finish(prev string, snap Snapshot, leader, errMsg string) Snapshot {
	curr := snap.State
	c.mu.Lock()
	c.jobState = curr
	c.mu.Unlock()
	c.pushStateEvent(prev, curr, leader)
	if errMsg != "" && (curr == "error" || curr == "unreachable") {
		c.logger.Warn("rccl_collect", "state", curr, "err", errMsg, "leader", leader)
	}
	return snap
}

func (c *Collector) pushStateEvent(prev, curr, leader string) {
	et := EventType(prev, curr)
	if et == "" || c.store == nil {
		return
	}
	ev := Event{
		"event_type":  et,
		"timestamp":   NowUnix(),
		"from_state":  prev,
		"to_state":    curr,
		"leader_node": leader,
	}
	c.store.PushEvent(ev)
	c.logger.Info("rccl_state_transition", "from", prev, "to", curr, "event", et, "leader", leader)
}

func (c *Collector) checkSkew(snap *Snapshot) {
	c.mu.Lock()
	versions := map[string]string{}
	for host, cap := range c.caps {
		if cap != nil && cap.Version != "" && cap.Version != "unknown" {
			versions[host] = cap.Version
		}
	}
	c.mu.Unlock()
	uniq := map[string]struct{}{}
	for _, v := range versions {
		uniq[v] = struct{}{}
	}
	if len(uniq) <= 1 {
		return
	}
	if snap.JobSummary != nil {
		snap.JobSummary.InconsistentTopology = true
	}
	var unique []string
	for v := range uniq {
		unique = append(unique, v)
	}
	c.logger.Warn("rccl_version_skew", "versions", versions)
	if c.store != nil {
		c.store.PushEvent(Event{
			"event_type":       "version_skew",
			"timestamp":        NowUnix(),
			"versions_by_node": versions,
			"unique_versions":  unique,
		})
	}
}

func (c *Collector) queryHost(ctx context.Context, pool *pssh.Pool, host string, port int, timeout time.Duration, maxBytes int) (Snapshot, rasKind, error) {
	qctx, cancel := context.WithTimeout(ctx, timeout+2*time.Second)
	defer cancel()
	conn, err := pool.LocalForward(qctx, host, "::1", port)
	if err != nil {
		conn, err = pool.LocalForward(qctx, host, "127.0.0.1", port)
	}
	if err != nil {
		return Snapshot{}, classifyRASErr(err), err
	}
	defer conn.Close()

	cli := NewClient(conn)
	if err := cli.Handshake(timeout); err != nil {
		return Snapshot{}, classifyRASErr(err), err
	}
	_ = cli.SetTimeout(int(timeout.Seconds()), timeout)

	useJSON := c.ensureJSON(cli, host, timeout)
	raw, err := cli.VerboseStatus(timeout, maxBytes)
	if err != nil {
		if errors.Is(err, ErrResponseTooLarge) {
			c.logger.Warn("rccl_ras_response_truncated", "host", host, "max", maxBytes, "err", err)
		}
		return Snapshot{}, classifyRASErr(err), err
	}
	var snap Snapshot
	if useJSON {
		snap = ParseJSON(raw)
	} else {
		snap = ParseText(raw)
	}
	return snap, rasOK, nil
}

func (c *Collector) ensureJSON(cli *Client, host string, timeout time.Duration) bool {
	c.mu.Lock()
	if c.caps == nil {
		c.caps = map[string]*nodeCap{}
	}
	if cap := c.caps[host]; cap != nil && time.Since(cap.ProbedAt) < cap.TTL {
		jsonOK := cap.JSONRAS
		c.mu.Unlock()
		if jsonOK && cli.Protocol >= protoJSONFormat {
			_ = cli.SetFormatJSON(timeout)
		}
		return jsonOK
	}
	c.mu.Unlock()

	jsonOK := false
	if cli.Protocol >= protoJSONFormat {
		if err := cli.SetFormatJSON(timeout); err == nil {
			jsonOK = true
		}
	}
	ttl := 5 * time.Minute
	if jsonOK {
		ttl = time.Hour
	}
	c.mu.Lock()
	prev := c.caps[host]
	cap := &nodeCap{JSONRAS: jsonOK, ProbedAt: time.Now(), TTL: ttl}
	if prev != nil {
		cap.InspectorV5 = prev.InspectorV5
		cap.Version = prev.Version
	}
	c.caps[host] = cap
	c.mu.Unlock()
	return jsonOK
}

func classifyRASErr(err error) rasKind {
	if err == nil {
		return rasOK
	}
	if errors.Is(err, ErrResponseTooLarge) {
		return rasProtocol
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, os.ErrDeadlineExceeded) {
		return rasTimeout
	}
	s := strings.ToLower(err.Error())
	switch {
	case strings.Contains(s, "timeout"), strings.Contains(s, "deadline exceeded"), strings.Contains(s, "i/o timeout"):
		return rasTimeout
	case strings.Contains(s, "connection refused"), strings.Contains(s, "connect refused"),
		strings.Contains(s, "no route"), strings.Contains(s, "administratively prohibited"),
		strings.Contains(s, "open failed"), strings.Contains(s, "connection reset"):
		return rasRefused
	case strings.Contains(s, "unexpected handshake"), strings.Contains(s, "set format"),
		strings.Contains(s, "protocol"):
		return rasProtocol
	default:
		return rasOther
	}
}
