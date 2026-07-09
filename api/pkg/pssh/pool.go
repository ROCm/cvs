// Package pssh provides a long-lived, application-level parallel SSH pool that
// holds one persistent TCP+SSH connection per inventory node, tracks reachability
// in memory, and exposes a fleet-wide Exec primitive.
//
// Design parity with cvs/lib/parallel/pssh.py:
//   - reachable_hosts / unreachable_hosts maintained in memory
//   - post-Exec double-check before pruning (check_connectivity)
//   - only SSH connection/session errors trigger a prune; command exit-code
//     failures do not (parity with ConnectionError/Timeout/SessionError check)
//   - pruned hosts receive "\nABORT: Host Unreachable Error" in their output
//
// Jump host / bastion support is intentionally omitted: the server binary is
// always deployed inside the cluster network (or port-forwarded to the UI), so
// every node is directly reachable.
//
// The Pool starts a background goroutine that reprobes all nodes (including
// currently unreachable ones) on a fixed interval, re-promoting nodes that
// recover. The initial probe must be triggered by the caller immediately after
// New returns.
package pssh

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	xssh "golang.org/x/crypto/ssh"
)

const (
	defaultDialTimeout        = 10 * time.Second
	defaultMaxSessionsPerConn = 10 // matches sshd MaxSessions default
	defaultReprobeInterval    = 300 * time.Second
	defaultKeepaliveInterval  = 30 * time.Second
	defaultKeepaliveMisses    = 3  // consecutive misses before pruning
	defaultCommandTimeout     = 60 * time.Second
	sshPort                   = 22
)

// Result is the outcome of running a command on one host.
type Result struct {
	Output string
	Err    error
}

// ProbeResult summarises one fleet-wide connectivity sweep.
type ProbeResult struct {
	Reachable   []string
	Unreachable []string
	Duration    time.Duration
}

// connEntry holds a persistent SSH client, its own per-connection session
// semaphore, and a cancel func that stops its keepalive goroutine.
// The semaphore capacity matches the SSH server's MaxSessions limit
// (defaultMaxSessionsPerConn = 10), so each node is throttled independently.
type connEntry struct {
	client *xssh.Client
	sem    chan struct{}
	cancel context.CancelFunc // stops the per-connection keepalive goroutine
}

// Pool is the application-level persistent SSH connection pool. It holds one
// TCP+SSH connection per node, tracks reachable/unreachable state in memory,
// and runs a background goroutine that reprobes the fleet on a fixed interval.
//
// The zero value is not usable; construct via New.
type Pool struct {
	// SSH auth config — immutable after construction.
	user               string
	auth               []xssh.AuthMethod
	dialTimeout        time.Duration
	maxSessionsPerConn int // semaphore cap per connection (mirrors sshd MaxSessions)

	// Persistent connections, keyed by host. Protected by connMu.
	// RWMutex: reads (cache hits) are parallel; writes (new dial, drop) are exclusive.
	// The dial itself happens outside the lock so 165 goroutines run in parallel.
	connMu sync.RWMutex
	conns  map[string]*connEntry

	// Reachability state. Protected by stateMu.
	stateMu     sync.RWMutex
	all         []string            // ordered, matches inventory.Nodes
	reachable   map[string]struct{} // subset of all
	unreachable map[string]struct{} // subset of all
	probeErrs   map[string]string   // last error per unreachable host

	// Keepalive config — immutable after construction.
	keepaliveInterval  time.Duration
	keepaliveMisses    int // consecutive miss threshold before pruning

	// commandTimeout caps every individual SSH command. Applied inside
	// runSession by wrapping the caller's ctx with WithTimeout.
	commandTimeout time.Duration

	logger *slog.Logger

	// reprobe is a buffered channel (cap 1) used to nudge the background loop.
	reprobe chan struct{}
	// cancel stops the background goroutine.
	cancel context.CancelFunc
}

// New builds a Pool for the given user, key file, and node list.
// It starts the background re-probe goroutine immediately; the caller is
// responsible for running the first Probe (typically via go p.Probe(ctx)).
//
// reprobeInterval  <= 0 falls back to 300 s.
// maxSessionsPerConn <= 0 falls back to 10 (matches sshd MaxSessions default).
// keepaliveInterval  <= 0 falls back to 30 s.
// commandTimeout    <= 0 falls back to 60 s (applied per Exec call).
func New(
	user, keyPath string,
	nodes []string,
	maxSessionsPerConn int,
	keepaliveInterval time.Duration,
	reprobeInterval time.Duration,
	commandTimeout time.Duration,
	logger *slog.Logger,
) (*Pool, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if maxSessionsPerConn <= 0 {
		maxSessionsPerConn = defaultMaxSessionsPerConn
	}
	if keepaliveInterval <= 0 {
		keepaliveInterval = defaultKeepaliveInterval
	}
	if reprobeInterval <= 0 {
		reprobeInterval = defaultReprobeInterval
	}
	if commandTimeout <= 0 {
		commandTimeout = defaultCommandTimeout
	}

	auth, err := loadKeyAuth(keyPath)
	if err != nil {
		return nil, err
	}

	all := make([]string, len(nodes))
	copy(all, nodes)

	reachable := make(map[string]struct{}, len(nodes))
	for _, h := range nodes {
		reachable[h] = struct{}{}
	}

	ctx, cancel := context.WithCancel(context.Background())

	p := &Pool{
		user:               user,
		auth:               auth,
		dialTimeout:        defaultDialTimeout,
		maxSessionsPerConn: maxSessionsPerConn,
		keepaliveInterval:  keepaliveInterval,
		keepaliveMisses:    defaultKeepaliveMisses,
		commandTimeout:     commandTimeout,
		conns:              make(map[string]*connEntry),
		all:                all,
		reachable:          reachable,
		unreachable:        make(map[string]struct{}),
		probeErrs:          make(map[string]string),
		logger:             logger,
		reprobe:            make(chan struct{}, 1),
		cancel:             cancel,
	}

	go p.runBackground(ctx, reprobeInterval)

	logger.Info("pssh_pool_created",
		"nodes", len(nodes),
		"user", user,
		"max_sessions_per_conn", maxSessionsPerConn,
		"keepalive_interval", keepaliveInterval.String(),
		"reprobe_interval", reprobeInterval.String(),
		"command_timeout", commandTimeout.String(),
	)
	return p, nil
}

// Refresh updates the pool's node list. It adds newly listed nodes
// (optimistically marking them reachable), removes deleted nodes (closing their
// connections), and nudges an immediate background re-probe.
// Auth credentials are NOT updated by Refresh; call Close + New when credentials
// change.
func (p *Pool) Refresh(nodes []string) {
	newSet := make(map[string]struct{}, len(nodes))
	for _, h := range nodes {
		newSet[h] = struct{}{}
	}

	p.stateMu.Lock()
	oldSet := make(map[string]struct{}, len(p.all))
	for _, h := range p.all {
		oldSet[h] = struct{}{}
	}

	var removed []string
	for h := range oldSet {
		if _, exists := newSet[h]; !exists {
			removed = append(removed, h)
			delete(p.reachable, h)
			delete(p.unreachable, h)
		}
	}

	// Newly added nodes start optimistically reachable — the next probe confirms.
	for _, h := range nodes {
		if _, exists := oldSet[h]; !exists {
			p.reachable[h] = struct{}{}
		}
	}

	p.all = make([]string, len(nodes))
	copy(p.all, nodes)
	p.stateMu.Unlock()

	for _, h := range removed {
		p.drop(h)
	}

	// Nudge background re-probe (non-blocking).
	select {
	case p.reprobe <- struct{}{}:
	default:
	}

	if len(removed) > 0 {
		p.logger.Info("pssh_refresh",
			"added", len(nodes)-len(oldSet)+len(removed),
			"removed", len(removed),
			"total", len(nodes),
		)
	}
}

// Close cancels the background goroutine and tears down all cached connections.
func (p *Pool) Close() {
	p.cancel()

	p.connMu.Lock()
	for h, e := range p.conns {
		e.cancel()           // stop keepalive goroutine
		_ = e.client.Close()
		delete(p.conns, h)
	}
	p.connMu.Unlock()

	p.logger.Info("pssh_pool_closed")
}

// RemoveNodes permanently removes the given hosts from the pool.
// Their live SSH connections are closed, and they are removed from all
// state maps so they never appear in Reachable(), Unreachable(), or All().
// Safe to call concurrently with Exec/Probe.
func (p *Pool) RemoveNodes(hosts []string) {
	if len(hosts) == 0 {
		return
	}
	rm := make(map[string]struct{}, len(hosts))
	for _, h := range hosts {
		rm[h] = struct{}{}
	}

	// Close and delete live connections.
	p.connMu.Lock()
	for h := range rm {
		if e, ok := p.conns[h]; ok {
			e.cancel()
			_ = e.client.Close()
			delete(p.conns, h)
		}
	}
	p.connMu.Unlock()

	// Remove from bookkeeping maps.
	p.stateMu.Lock()
	newAll := p.all[:0:0]
	for _, h := range p.all {
		if _, skip := rm[h]; !skip {
			newAll = append(newAll, h)
		}
	}
	p.all = newAll
	for h := range rm {
		delete(p.reachable, h)
		delete(p.unreachable, h)
	}
	p.stateMu.Unlock()

	p.logger.Info("pssh_nodes_removed", "hosts", hosts, "count", len(hosts))
}

// AddNodes appends new hosts to the pool as unreachable pending a probe.
// Call ProbeSubset(ctx, hosts) immediately after to dial and classify them.
// Safe to call concurrently with Exec/Probe.
func (p *Pool) AddNodes(hosts []string) {
	if len(hosts) == 0 {
		return
	}
	p.stateMu.Lock()
	for _, h := range hosts {
		p.all = append(p.all, h)
		p.unreachable[h] = struct{}{} // pessimistic until ProbeSubset confirms
	}
	p.stateMu.Unlock()

	p.logger.Info("pssh_nodes_added", "hosts", hosts, "count", len(hosts))
}

// Exec runs cmd on every currently-reachable node in parallel, collects results,
// and then runs a post-exec connectivity double-check for any hosts that returned
// a connection error, pruning confirmed-down nodes from the reachable set.
//
// The returned map contains one entry per node that was attempted; nodes that
// were already unreachable before Exec are excluded (use Unreachable() for those).
// Exec runs cmd on every reachable node in parallel and returns a result per host.
// To cap execution time, pass a context with a deadline; runSession also enforces
// the pool's commandTimeout as a backstop.
func (p *Pool) Exec(ctx context.Context, cmd string) map[string]Result {
	// Snapshot reachable set under read lock.
	p.stateMu.RLock()
	targets := make([]string, 0, len(p.reachable))
	for h := range p.reachable {
		targets = append(targets, h)
	}
	p.stateMu.RUnlock()

	results := make(map[string]Result, len(targets))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, host := range targets {
		wg.Add(1)
		go func(h string) {
			defer wg.Done()
			out, err := p.runSession(ctx, h, cmd)
			mu.Lock()
			results[h] = Result{Output: out, Err: err}
			mu.Unlock()
		}(host)
	}
	wg.Wait()

	// Post-exec: double-check and prune connection-failed nodes.
	p.pruneAfterExec(ctx, results)

	return results
}

// Reachable returns a snapshot of the currently reachable nodes in inventory order.
func (p *Pool) Reachable() []string {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	out := make([]string, 0, len(p.reachable))
	for _, h := range p.all {
		if _, ok := p.reachable[h]; ok {
			out = append(out, h)
		}
	}
	return out
}

// Unreachable returns a snapshot of the currently unreachable nodes in inventory order.
func (p *Pool) Unreachable() []string {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	out := make([]string, 0, len(p.unreachable))
	for _, h := range p.all {
		if _, ok := p.unreachable[h]; ok {
			out = append(out, h)
		}
	}
	return out
}

// NodeError returns the last probe error for host, or "" if the host is
// reachable or has never been probed.
func (p *Pool) NodeError(host string) string {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	return p.probeErrs[host]
}

// All returns a copy of the full node list in inventory order.
func (p *Pool) All() []string {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	out := make([]string, len(p.all))
	copy(out, p.all)
	return out
}

// ─── internal helpers ─────────────────────────────────────────────────────────

// conn returns the cached connEntry for host, dialing on a cache miss.
// The dial happens outside the lock so all goroutines run in parallel.
// Caller must NOT hold connMu.
func (p *Pool) conn(host string) (*connEntry, error) {
	// Fast path: parallel read — all goroutines proceed simultaneously on hits.
	p.connMu.RLock()
	if e, ok := p.conns[host]; ok {
		p.connMu.RUnlock()
		return e, nil
	}
	p.connMu.RUnlock()

	// Slow path: dial outside the lock — each goroutine dials a different host
	// so they all run in parallel with no serialization.
	c, err := xssh.Dial("tcp", addrFor(host), &xssh.ClientConfig{
		User:            p.user,
		Auth:            p.auth,
		HostKeyCallback: xssh.InsecureIgnoreHostKey(), // parity with cluster-mon
		Timeout:         p.dialTimeout,
	})
	if err != nil {
		return nil, err
	}

	kctx, cancel := context.WithCancel(context.Background())
	e := &connEntry{
		client: c,
		sem:    make(chan struct{}, p.maxSessionsPerConn),
		cancel: cancel,
	}

	// Write lock only to insert — held for nanoseconds (map write).
	p.connMu.Lock()
	p.conns[host] = e
	p.connMu.Unlock()

	// Start per-connection SSH keepalive goroutine. It sends
	// "keepalive@openssh.com" every keepaliveInterval and prunes the host
	// after keepaliveMisses consecutive failures.
	go p.keepalive(kctx, host, c)

	return e, nil
}

// drop cancels the keepalive goroutine, closes, and removes a cached connEntry
// so the next call to conn redials.
func (p *Pool) drop(host string) {
	p.connMu.Lock()
	defer p.connMu.Unlock()
	if e, ok := p.conns[host]; ok {
		e.cancel()           // stop keepalive goroutine
		_ = e.client.Close()
		delete(p.conns, host)
	}
}

// keepalive sends SSH keepalive requests on the given client every
// keepaliveInterval. A failed request increments a miss counter; once
// keepaliveMisses consecutive failures accumulate the host is pruned.
//
// The goroutine exits immediately when ctx is cancelled (i.e. when drop()
// tears down the connection from outside). This prevents the goroutine from
// racing a concurrent drop+redial for the same host.
func (p *Pool) keepalive(ctx context.Context, host string, client *xssh.Client) {
	t := time.NewTicker(p.keepaliveInterval)
	defer t.Stop()
	missed := 0

	for {
		select {
		case <-ctx.Done():
			// drop() already handled cleanup; exit silently.
			return

		case <-t.C:
			_, _, err := client.SendRequest("keepalive@openssh.com", true, nil)
			if err != nil {
				// If the context was already cancelled while SendRequest was in
				// flight, drop() already ran — don't double-prune.
				select {
				case <-ctx.Done():
					return
				default:
				}

				missed++
				p.logger.Debug("pssh_keepalive_miss",
					"host", host, "missed", missed, "threshold", p.keepaliveMisses)

				if missed >= p.keepaliveMisses {
					p.logger.Info("pssh_keepalive_dead",
						"host", host, "missed", missed)
					p.drop(host)
					p.pruneNodes([]string{host})
					return
				}
			} else {
				missed = 0
			}
		}
	}
}

// runSession acquires the per-connection semaphore slot, opens an SSH session
// on the persistent connection for host, and runs cmd, returning combined
// stdout+stderr. The semaphore cap equals maxSessionsPerConn, which mirrors
// the SSH server's MaxSessions limit — so each node is throttled independently
// and a slow node does not delay others.
//
// Connection and session errors drop the cached entry (so the next call
// redials). Command exit-code errors do NOT drop the connection.
func (p *Pool) runSession(ctx context.Context, host, cmd string) (string, error) {
	// Enforce the per-command timeout. If the caller's context already has a
	// tighter deadline it wins; otherwise we bound every SSH command to
	// commandTimeout so a hung remote process can't leak goroutines forever.
	ctx, cancel := context.WithTimeout(ctx, p.commandTimeout)
	defer cancel()

	entry, err := p.conn(host)
	if err != nil {
		return "", err
	}

	// Acquire the per-connection semaphore slot.
	select {
	case entry.sem <- struct{}{}:
		defer func() { <-entry.sem }()
	case <-ctx.Done():
		return "", ctx.Err()
	}

	// NewSession opens an SSH channel — a blocking network call that is not
	// context-aware. Race it against ctx.Done() so a degraded node can't hold
	// the goroutine past the per-host deadline (which was causing 69s probe times).
	type sessOrErr struct {
		sess *xssh.Session
		err  error
	}
	sessCh := make(chan sessOrErr, 1)
	go func() {
		s, e := entry.client.NewSession()
		sessCh <- sessOrErr{s, e}
	}()
	var sess *xssh.Session
	select {
	case <-ctx.Done():
		p.drop(host)
		return "", ctx.Err()
	case r := <-sessCh:
		if r.err != nil {
			p.drop(host)
			return "", r.err
		}
		sess = r.sess
	}
	defer sess.Close()

	type res struct {
		out []byte
		err error
	}
	done := make(chan res, 1)
	go func() {
		out, err := sess.CombinedOutput(cmd)
		done <- res{out, err}
	}()

	select {
	case <-ctx.Done():
		_ = sess.Close()
		p.drop(host)
		return "", ctx.Err()
	case r := <-done:
		// Only drop the connection on real connection/IO failures; exit-code
		// errors mean the command ran and the connection is healthy.
		if r.err != nil && isConnError(r.err) {
			p.drop(host)
		}
		return string(r.out), r.err
	}
}

// ─── auth helpers ─────────────────────────────────────────────────────────────

func loadKeyAuth(keyPath string) ([]xssh.AuthMethod, error) {
	pem, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, fmt.Errorf("read key %q: %w", keyPath, err)
	}
	signer, err := xssh.ParsePrivateKey(pem)
	if err != nil {
		return nil, fmt.Errorf("parse key %q: %w", keyPath, err)
	}
	return []xssh.AuthMethod{xssh.PublicKeys(signer)}, nil
}

// addrFor appends :22 when host has no explicit port.
func addrFor(host string) string {
	if _, _, err := net.SplitHostPort(host); err == nil {
		return host
	}
	return net.JoinHostPort(host, strconv.Itoa(sshPort))
}
