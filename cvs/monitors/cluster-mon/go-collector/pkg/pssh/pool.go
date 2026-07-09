// Package pssh provides a long-lived, application-level parallel SSH pool that
// holds one persistent TCP+SSH connection per inventory node, tracks reachability
// in memory, and exposes a fleet-wide Exec primitive.
//
// Adapted from github.com/ROCm/cvs/api/pkg/pssh (branch ichristo/design-cvs-api-server).
// Local additions:
//   - dialSem (chan struct{}, cap 256): limits concurrent SSH handshakes so we
//     don't overwhelm sshd MaxStartups during the initial fleet connect.
//   - dialFunc (optional): when non-nil, called instead of xssh.Dial to create
//     the raw net.Conn; used to tunnel connections through a jump host.
//   - Refresh() now returns (added []string, removed []string) so the daemon
//     can respond to refresh_nodes messages with the diff and probe only new hosts.
package pssh

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	xssh "golang.org/x/crypto/ssh"
)

const (
	defaultDialTimeout        = 30 * time.Second // generous for slow SSH banners
	defaultMaxSessionsPerConn = 10 // matches sshd MaxSessions default
	defaultReprobeInterval    = 300 * time.Second
	defaultKeepaliveInterval  = 30 * time.Second
	defaultKeepaliveMisses    = 3  // consecutive misses before pruning
	defaultCommandTimeout     = 60 * time.Second
	sshPort                   = 22
	dialSemCap                = 256 // max concurrent SSH handshakes
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
type connEntry struct {
	client *xssh.Client
	sem    chan struct{}
	cancel context.CancelFunc
}

// Pool is the application-level persistent SSH connection pool.
// The zero value is not usable; construct via New.
type Pool struct {
	// SSH credentials — immutable after construction.
	// keyBytes takes priority over keyPath when non-nil (in-memory key, never
	// written to disk). keyPath is used lazily when keyBytes is nil so the pool
	// can start even when the key file does not yet exist (e.g. fresh container).
	// password is used when both keyPath and keyBytes are empty/nil.
	user               string
	keyPath            string
	keyBytes           []byte // in-memory key PEM; takes priority over keyPath
	password           string // SSH password; used only when no key is available
	dialTimeout        time.Duration
	maxSessionsPerConn int

	// dialSem caps concurrent SSH handshakes to dialSemCap (256).
	dialSem chan struct{}

	// dialFunc, when non-nil, replaces xssh.Dial. Set for jump-host tunnelling.
	dialFunc func(network, addr string) (net.Conn, error)

	// Persistent connections, keyed by host. Protected by connMu.
	connMu sync.RWMutex
	conns  map[string]*connEntry

	// Reachability state. Protected by stateMu.
	stateMu     sync.RWMutex
	all         []string
	reachable   map[string]struct{}
	unreachable map[string]struct{}
	probeErrs   map[string]string

	keepaliveInterval time.Duration
	keepaliveMisses   int
	commandTimeout    time.Duration

	logger *slog.Logger

	reprobe   chan struct{}
	cancel    context.CancelFunc
	probedOnce int32 // set to 1 after the first reprobeUnreachable completes in runBackground
}

// New builds a Pool for the given user, key path, and node list.
// keyPath is read lazily from disk at each new connection so the pool starts
// immediately even if the file does not yet exist. The in-memory key path
// (keyBytes in Pool) is set later via UpdateCredentials when Python delivers
// the key over the UDS refresh_nodes message.
// dialFunc may be nil (direct TCP dial) or a function returning a net.Conn
// already tunnelled through a jump host.
func New(
	user, keyPath, password string,
	nodes []string,
	maxSessionsPerConn int,
	keepaliveInterval time.Duration,
	reprobeInterval time.Duration,
	commandTimeout time.Duration,
	logger *slog.Logger,
	dialFunc func(network, addr string) (net.Conn, error),
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

	all := make([]string, len(nodes))
	copy(all, nodes)

	// Pessimistic default: all nodes start unreachable. runBackground fires an
	// immediate t=0 reprobeUnreachable sweep that classifies the whole fleet
	// before the first ticker interval. This means Exec() returns an empty
	// result set until the first sweep completes rather than optimistically
	// racing 165 SSH dials on the first call.
	unreachable := make(map[string]struct{}, len(nodes))
	for _, h := range nodes {
		unreachable[h] = struct{}{}
	}

	ctx, cancel := context.WithCancel(context.Background())

	p := &Pool{
		user:               user,
		keyPath:            keyPath,
		password:           password,
		dialTimeout:        defaultDialTimeout,
		maxSessionsPerConn: maxSessionsPerConn,
		dialSem:            make(chan struct{}, dialSemCap),
		dialFunc:           dialFunc,
		keepaliveInterval:  keepaliveInterval,
		keepaliveMisses:    defaultKeepaliveMisses,
		commandTimeout:     commandTimeout,
		conns:              make(map[string]*connEntry),
		all:                all,
		reachable:          make(map[string]struct{}),
		unreachable:        unreachable,
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

// Refresh updates the pool's node list to exactly the given set.
// Returns (added, removed) — the diff relative to the previous list.
// Added nodes start optimistically reachable pending a ProbeSubset call.
// Removed nodes have their connections closed immediately.
// Connections for unchanged nodes are preserved.
func (p *Pool) Refresh(nodes []string) (added []string, removed []string) {
	newSet := make(map[string]struct{}, len(nodes))
	for _, h := range nodes {
		newSet[h] = struct{}{}
	}

	p.stateMu.Lock()
	oldSet := make(map[string]struct{}, len(p.all))
	for _, h := range p.all {
		oldSet[h] = struct{}{}
	}

	for h := range oldSet {
		if _, exists := newSet[h]; !exists {
			removed = append(removed, h)
			delete(p.reachable, h)
			delete(p.unreachable, h)
		}
	}

	for _, h := range nodes {
		if _, exists := oldSet[h]; !exists {
			added = append(added, h)
			p.reachable[h] = struct{}{} // optimistic — ProbeSubset confirms
		}
	}

	p.all = make([]string, len(nodes))
	copy(p.all, nodes)
	p.stateMu.Unlock()

	for _, h := range removed {
		p.drop(h)
	}

	p.TriggerReprobe()

	if len(added) > 0 || len(removed) > 0 {
		p.logger.Info("pssh_refresh",
			"added", len(added),
			"removed", len(removed),
			"total", len(nodes),
		)
	}
	return added, removed
}

// TriggerReprobe sends a non-blocking nudge to the background reprobe goroutine,
// waking it immediately outside the normal ticker cycle. Safe to call from any
// goroutine; if a nudge is already pending the extra send is silently dropped.
func (p *Pool) TriggerReprobe() {
	select {
	case p.reprobe <- struct{}{}:
	default: // nudge already queued; reprobe will run shortly
	}
}

// UpdateCredentials replaces the SSH username, key material, and/or password
// in the running pool. Pass empty/nil to keep the current value for that field.
// keyBytes, when non-nil, replaces any existing in-memory key and clears keyPath.
// password is used when both keyPath and keyBytes are empty/nil.
//
// All cached connections are dropped under connMu (they authenticated with the old
// credentials and cannot be re-used). All currently reachable nodes are moved back
// to unreachable so the background reprobe re-dials them with the new credentials.
// TriggerReprobe is called at the end so recovery begins immediately.
func (p *Pool) UpdateCredentials(user, keyPath string, keyBytes []byte, password string) {
	p.connMu.Lock()
	if user != "" {
		p.user = user
	}
	if len(keyBytes) > 0 {
		cp := make([]byte, len(keyBytes))
		copy(cp, keyBytes)
		p.keyBytes = cp
		p.keyPath = ""    // in-memory key takes over
		p.password = ""   // key wins over password
	} else if keyPath != "" {
		p.keyPath = keyPath
		p.keyBytes = nil  // switch back to file-based
		p.password = ""   // key wins over password
	} else if password != "" {
		p.password = password
		p.keyPath = ""
		p.keyBytes = nil
	}
	for h, e := range p.conns {
		e.cancel()
		_ = e.client.Close()
		delete(p.conns, h)
	}
	p.connMu.Unlock()

	// Move all reachable nodes to unreachable so reprobeUnreachable visits them.
	p.stateMu.Lock()
	for h := range p.reachable {
		p.unreachable[h] = struct{}{}
	}
	p.reachable = make(map[string]struct{})
	p.stateMu.Unlock()

	p.TriggerReprobe()
	p.logger.Info("pssh_credentials_updated",
		"user_changed", user != "",
		"key_path_changed", keyPath != "",
		"key_bytes_changed", len(keyBytes) > 0,
		"password_changed", password != "",
	)
}

// Close cancels the background goroutine and tears down all cached connections.
func (p *Pool) Close() {
	p.cancel()

	p.connMu.Lock()
	for h, e := range p.conns {
		e.cancel()
		_ = e.client.Close()
		delete(p.conns, h)
	}
	p.connMu.Unlock()

	p.logger.Info("pssh_pool_closed")
}

// RemoveNodes permanently removes the given hosts from the pool.
func (p *Pool) RemoveNodes(hosts []string) {
	if len(hosts) == 0 {
		return
	}
	rm := make(map[string]struct{}, len(hosts))
	for _, h := range hosts {
		rm[h] = struct{}{}
	}

	p.connMu.Lock()
	for h := range rm {
		if e, ok := p.conns[h]; ok {
			e.cancel()
			_ = e.client.Close()
			delete(p.conns, h)
		}
	}
	p.connMu.Unlock()

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
func (p *Pool) AddNodes(hosts []string) {
	if len(hosts) == 0 {
		return
	}
	p.stateMu.Lock()
	for _, h := range hosts {
		p.all = append(p.all, h)
		p.unreachable[h] = struct{}{}
	}
	p.stateMu.Unlock()

	p.logger.Info("pssh_nodes_added", "hosts", hosts, "count", len(hosts))
}

// Exec runs cmd on every currently-reachable node in parallel.
// Returns one Result per attempted host. Pre-existing unreachable nodes are
// excluded — use Unreachable() for those.
func (p *Pool) Exec(ctx context.Context, cmd string) map[string]Result {
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

	p.pruneAfterExec(ctx, results)

	return results
}

// Reachable returns a snapshot of currently reachable nodes in inventory order.
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

// Unreachable returns a snapshot of currently unreachable nodes in inventory order.
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

// InitialProbeDone reports whether the first fleet-wide reprobeUnreachable sweep
// (fired at t=0 inside runBackground) has completed. Use this to gate
// probe_status="complete" in health responses instead of a separate package-level
// atomic in main.go.
func (p *Pool) InitialProbeDone() bool {
	return atomic.LoadInt32(&p.probedOnce) == 1
}

// NodeError returns the last probe error for host, or "" if reachable.
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
// dialSem limits concurrent handshakes. The dial itself is outside all locks
// so goroutines for different hosts run in parallel.
func (p *Pool) conn(host string) (*connEntry, error) {
	// Fast path: all goroutines proceed in parallel on cache hits.
	p.connMu.RLock()
	if e, ok := p.conns[host]; ok {
		p.connMu.RUnlock()
		return e, nil
	}
	p.connMu.RUnlock()

	// Acquire dial semaphore BEFORE dialing — bounds concurrent handshakes.
	p.dialSem <- struct{}{}
	defer func() { <-p.dialSem }()

	// Double-check under read lock: another goroutine may have dialed while
	// we waited for the semaphore.
	p.connMu.RLock()
	if e, ok := p.conns[host]; ok {
		p.connMu.RUnlock()
		return e, nil
	}
	p.connMu.RUnlock()

	// Load auth methods at connection time — not at pool creation time.
	// Supports key-based (file or in-memory bytes) and password-based auth.
	// If the key file is missing, conn() returns an error, the node is marked
	// unreachable, and the background reprobe retries every 5 minutes.
	auth, err := loadAuth(p.keyPath, p.keyBytes, p.password)
	if err != nil {
		return nil, fmt.Errorf("auth load: %w", err)
	}

	sshCfg := &xssh.ClientConfig{
		User:            p.user,
		Auth:            auth,
		HostKeyCallback: xssh.InsecureIgnoreHostKey(),
		Timeout:         p.dialTimeout,
	}

	var c *xssh.Client
	if p.dialFunc != nil {
		addr := addrFor(host)
		netConn, err := p.dialFunc("tcp", addr)
		if err != nil {
			return nil, err
		}
		cc, chans, reqs, err := xssh.NewClientConn(netConn, addr, sshCfg)
		if err != nil {
			netConn.Close()
			return nil, err
		}
		c = xssh.NewClient(cc, chans, reqs)
	} else {
		var err error
		c, err = xssh.Dial("tcp", addrFor(host), sshCfg)
		if err != nil {
			return nil, err
		}
	}

	kctx, cancel := context.WithCancel(context.Background())
	e := &connEntry{
		client: c,
		sem:    make(chan struct{}, p.maxSessionsPerConn),
		cancel: cancel,
	}

	p.connMu.Lock()
	// Final check under write lock to handle the concurrent-dial race.
	if existing, ok := p.conns[host]; ok {
		p.connMu.Unlock()
		cancel()
		_ = c.Close()
		return existing, nil
	}
	p.conns[host] = e
	p.connMu.Unlock()

	go p.keepalive(kctx, host, c)

	return e, nil
}

// drop cancels the keepalive goroutine, closes, and removes a cached connEntry.
func (p *Pool) drop(host string) {
	p.connMu.Lock()
	defer p.connMu.Unlock()
	if e, ok := p.conns[host]; ok {
		e.cancel()
		_ = e.client.Close()
		delete(p.conns, host)
	}
}

// keepalive sends SSH keepalive requests every keepaliveInterval.
// Prunes the host after keepaliveMisses consecutive failures.
func (p *Pool) keepalive(ctx context.Context, host string, client *xssh.Client) {
	t := time.NewTicker(p.keepaliveInterval)
	defer t.Stop()
	missed := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_, _, err := client.SendRequest("keepalive@openssh.com", true, nil)
			if err != nil {
				select {
				case <-ctx.Done():
					return
				default:
				}
				missed++
				p.logger.Debug("pssh_keepalive_miss",
					"host", host, "missed", missed, "threshold", p.keepaliveMisses)
				if missed >= p.keepaliveMisses {
					p.logger.Info("pssh_keepalive_dead", "host", host, "missed", missed)
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

// runSession acquires a per-connection session slot and runs cmd via SSH.
// Returns combined stdout+stderr.
func (p *Pool) runSession(ctx context.Context, host, cmd string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, p.commandTimeout)
	defer cancel()

	entry, err := p.conn(host)
	if err != nil {
		return "", err
	}

	select {
	case entry.sem <- struct{}{}:
		defer func() { <-entry.sem }()
	case <-ctx.Done():
		return "", ctx.Err()
	}

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
		if r.err != nil && isConnError(r.err) {
			p.drop(host)
		}
		return string(r.out), r.err
	}
}

// ─── auth helpers ─────────────────────────────────────────────────────────────

// loadAuth returns SSH auth methods for connecting to a cluster node.
//
// Priority order:
//  1. keyBytes (in-memory PEM, never written to disk)
//  2. keyPath  (file-based, read lazily with ~ expansion)
//  3. password (plain-text; used when no key is available)
//
// Returns an error only when a key is specified but cannot be parsed.
// If all three are empty, returns an error (no auth available).
func loadAuth(keyPath string, keyBytes []byte, password string) ([]xssh.AuthMethod, error) {
	// In-memory key takes top priority.
	if len(keyBytes) > 0 {
		signer, err := xssh.ParsePrivateKey(keyBytes)
		if err != nil {
			return nil, fmt.Errorf("parse in-memory key: %w", err)
		}
		return []xssh.AuthMethod{xssh.PublicKeys(signer)}, nil
	}

	// File-based key next.
	if keyPath != "" {
		// Expand ~ to the actual home directory.
		if len(keyPath) >= 2 && keyPath[0] == '~' && (keyPath[1] == '/' || keyPath[1] == '\\') {
			home, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("resolve home dir for key %q: %w", keyPath, err)
			}
			keyPath = home + keyPath[1:]
		}
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

	// Password auth as fallback.
	if password != "" {
		return []xssh.AuthMethod{xssh.Password(password)}, nil
	}

	return nil, fmt.Errorf("no SSH credentials available (set key or password)")
}

// loadKeyAuth is kept for backward-compatibility — delegates to loadAuth.
func loadKeyAuth(keyPath string, keyBytes []byte) ([]xssh.AuthMethod, error) {
	return loadAuth(keyPath, keyBytes, "")
}

// addrFor appends :22 when host has no explicit port.
func addrFor(host string) string {
	if _, _, err := net.SplitHostPort(host); err == nil {
		return host
	}
	return net.JoinHostPort(host, strconv.Itoa(sshPort))
}
