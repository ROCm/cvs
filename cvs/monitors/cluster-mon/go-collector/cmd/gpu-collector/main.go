// gpu-collector: persistent Go SSH daemon for CVS cluster-mon.
//
// Runs a Unix-socket server that accepts JSON requests from Python collectors
// and executes SSH commands across the cluster using a persistent connection pool.
//
// Message types:
//   exec          – run a command on all reachable hosts
//   health        – return fleet SSH health status
//   refresh_nodes – update the host list in-place (diff + background probe)
//
// All nodes start pessimistically unreachable. The pool's background goroutine
// fires an immediate t=0 sweep then reprobes only unreachable hosts every
// --probe-interval seconds (reachable hosts are covered by SSH keepalives).
//
// SSH key delivery: two modes.
//   --ssh-key <path>  File-based: key read lazily at each new handshake.
//                     Daemon starts even when the file does not yet exist.
//   (no flag)         UDS delivery: Python fetches the node key from the jump
//                     host via SFTP and sends it as base64 key_bytes inside a
//                     refresh_nodes message after the socket opens. The key is
//                     held only in pool.keyBytes — never written to disk.
//                     Use UpdateCredentials to update key bytes at runtime.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	xssh "golang.org/x/crypto/ssh"

	"github.com/ROCm/cvs/monitors/cluster-mon/go-collector/pkg/pssh"
)

// IncomingMsg is the JSON request sent by Python over UDS.
type IncomingMsg struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Command  string   `json:"command,omitempty"`
	TimeoutS int      `json:"timeout_s,omitempty"`
	Hosts    []string `json:"hosts,omitempty"`
	// Credential update fields for refresh_nodes messages.
	// KeyBytes (base64-encoded PEM) takes priority over KeyPath when both are set.
	// Password is used when neither KeyPath nor KeyBytes is provided.
	// Sending KeyBytes lets Python deliver the node key in-memory via UDS at any
	// time — no daemon restart required, key never touches the container filesystem.
	User     string `json:"user,omitempty"`
	KeyPath  string `json:"key_path,omitempty"`
	KeyBytes []byte `json:"key_bytes,omitempty"`
	Password string `json:"password,omitempty"`
	// Jumphost fields for refresh_nodes messages.
	// When JumpHost is non-empty, the pool's dial function is updated to tunnel
	// through the jump host. All existing connections are dropped and re-established.
	JumpHost     string `json:"jump_host,omitempty"`
	JumpUser     string `json:"jump_user,omitempty"`
	JumpKey      string `json:"jump_key,omitempty"`
	JumpPassword string `json:"jump_password,omitempty"`
}

var (
	pool   *pssh.Pool
	logger *slog.Logger

	// Jumphost state — protected by jumpMu for concurrent access from
	// handleRefreshNodes. Stores the current jump host SSH client and
	// configuration so we can detect changes and reconnect as needed.
	jumpMu       sync.Mutex
	jumpClient   *xssh.Client // current jump host SSH client, nil if direct
	jumpHostAddr string       // "host:port" of current jump host
	jumpUserVal  string       // current jump user
	jumpKeyPath  string       // current jump key path
	jumpPassVal  string       // current jump password (in memory only)
)

func main() {
	socketPath    := flag.String("socket", "/tmp/go-collector.sock", "Unix socket path")
	sshUser       := flag.String("ssh-user", "", "SSH username for cluster nodes")
	sshKey        := flag.String("ssh-key", "", "Path to SSH private key for cluster nodes (file-based, lazy; omit to use UDS key delivery)")
	sshPassword   := flag.String("ssh-password", "", "SSH password for cluster nodes (alternative to --ssh-key; can also be delivered via refresh_nodes UDS message)")
	hostsFile     := flag.String("hosts-file", "", "Path to file with one hostname per line")
	probeInterval := flag.Duration("probe-interval", 300*time.Second, "Reprobe interval for unreachable hosts")
	jumpHost      := flag.String("jump-host", "", "Jump host hostname (optional)")
	jumpUser      := flag.String("jump-user", "", "Jump host SSH username")
	jumpKey       := flag.String("jump-key", "", "Path to jump host SSH private key")
	jumpPassword  := flag.String("jump-password", "", "Jump host SSH password (alternative to --jump-key)")
	flag.Parse()

	logger = slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *sshUser == "" || *hostsFile == "" {
		fmt.Fprintln(os.Stderr, "required: --ssh-user, --hosts-file")
		flag.Usage()
		os.Exit(1)
	}

	if *sshKey == "" && *sshPassword == "" {
		logger.Warn("--ssh-key and --ssh-password not set; credentials must be delivered via refresh_nodes UDS message")
	}

	hosts, err := loadHosts(*hostsFile)
	if err != nil {
		logger.Error("failed to load hosts", "path", *hostsFile, "error", err)
		os.Exit(1)
	}
	logger.Info("hosts loaded", "count", len(hosts), "file", *hostsFile)

	// Build optional jump-host dial function.
	// Store jumpClient in package-level state so handleRefreshNodes can update it.
	if *jumpHost != "" {
		jumpMu.Lock()
		var err error
		jumpClient, err = connectJumpHost(*jumpHost, *jumpUser, *jumpKey, *jumpPassword)
		if err != nil {
			jumpMu.Unlock()
			logger.Error("failed to connect jump host", "host", *jumpHost, "error", err)
			os.Exit(1)
		}
		jumpHostAddr = *jumpHost
		jumpUserVal = *jumpUser
		jumpKeyPath = *jumpKey
		jumpPassVal = *jumpPassword
		jumpMu.Unlock()
		logger.Info("jump host connected", "host", *jumpHost)
	}

	// Create dialFunc that reads from package-level jumpClient.
	// This closure is called for each new node connection and will use
	// the current jumpClient state at call time.
	dialFunc := makeDialFunc()

	// keepalive interval is capped at 30s; cannot exceed probeInterval.
	effectiveKeepalive := 30 * time.Second
	if *probeInterval < effectiveKeepalive {
		effectiveKeepalive = *probeInterval
	}

	// pssh.New() stores the key path and reads lazily in conn() — always
	// succeeds regardless of key presence. All nodes start unreachable;
	// runBackground fires an immediate t=0 sweep to classify the fleet.
	pool, err = pssh.New(
		*sshUser, *sshKey, *sshPassword, hosts,
		10,             // maxSessionsPerConn — matches sshd MaxSessions
		effectiveKeepalive,
		*probeInterval,
		60*time.Second, // commandTimeout backstop
		logger,
		dialFunc,
	)
	if err != nil {
		logger.Error("failed to create pool", "error", err)
		os.Exit(1)
	}

	// Graceful shutdown on SIGTERM / SIGINT.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		sig := <-quit
		logger.Info("shutdown signal received", "signal", sig.String())
		pool.Close()
		os.Remove(*socketPath)
		os.Exit(0)
	}()

	logger.Info("daemon starting", "socket", *socketPath)
	if err := runServer(*socketPath); err != nil {
		logger.Error("server error", "error", err)
		os.Exit(1)
	}
}

// runServer listens on a Unix socket and spawns a goroutine per connection.
func runServer(socketPath string) error {
	os.Remove(socketPath)
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("listen %s: %w", socketPath, err)
	}
	defer os.Remove(socketPath)
	defer ln.Close()

	for {
		conn, err := ln.Accept()
		if err != nil {
			// ln.Close() from the signal handler causes this; exit cleanly.
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			logger.Error("accept error", "error", err)
			continue
		}
		go handleConn(conn)
	}
}

// handleConn reads one JSON request and writes one JSON response per connection.
// Each Python _exec_one / query_daemon_health / _refresh_nodes call opens a fresh
// connection, so there is never cross-client response mixing.
func handleConn(conn net.Conn) {
	defer conn.Close()
	var msg IncomingMsg
	if err := json.NewDecoder(conn).Decode(&msg); err != nil {
		logger.Debug("decode error", "error", err)
		return
	}
	switch msg.Type {
	case "exec":
		handleExec(conn, msg)
	case "health":
		handleHealth(conn, msg)
	case "refresh_nodes":
		handleRefreshNodes(conn, msg)
	default:
		logger.Warn("unknown message type", "type", msg.Type, "id", msg.ID)
		writeJSON(conn, map[string]any{
			"id":    msg.ID,
			"type":  msg.Type,
			"error": "unknown message type",
		})
	}
}

// handleExec runs the command on all reachable hosts and returns results.
func handleExec(conn net.Conn, msg IncomingMsg) {
	timeout := time.Duration(msg.TimeoutS) * time.Second
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	// Give the pool a bit of headroom beyond the per-command timeout so the
	// context doesn't fire before every host has had a chance to finish.
	ctx, cancel := context.WithTimeout(context.Background(), timeout+30*time.Second)
	defer cancel()

	t0 := time.Now()
	rawResults := pool.Exec(ctx, msg.Command)

	// Convert to string map. pruneAfterExec already appended abortMsg to
	// confirmed-down hosts; we normalise the format for Python callers that
	// check startswith("ABORT") or startswith("ERROR").
	results := make(map[string]string, len(rawResults))
	for host, r := range rawResults {
		switch {
		case r.Output != "":
			// Trim leading newline from abortMsg (Python callers use .strip())
			// but keep the ABORT prefix for callers that don't strip.
			results[host] = strings.TrimLeft(r.Output, "\n")
		case r.Err != nil:
			results[host] = "ERROR: " + r.Err.Error()
		default:
			results[host] = ""
		}
	}

	unreachable := pool.Unreachable()

	writeJSON(conn, map[string]any{
		"id":          msg.ID,
		"type":        "exec",
		"results":     results,
		"unreachable": unreachable,
		"duration_ms": time.Since(t0).Milliseconds(),
	})
}

// handleHealth returns the current fleet SSH health status.
func handleHealth(conn net.Conn, msg IncomingMsg) {
	probeStatus := "in-progress"
	if pool.InitialProbeDone() {
		probeStatus = "complete"
	}

	reachable := pool.Reachable()
	unreachableList := pool.Unreachable()
	all := pool.All()

	unreachableMap := make(map[string]string, len(unreachableList))
	for _, h := range unreachableList {
		unreachableMap[h] = pool.NodeError(h)
	}

	writeJSON(conn, map[string]any{
		"id":                msg.ID,
		"type":              "health",
		"probe_status":      probeStatus,
		"reachable":         reachable,
		"unreachable":       unreachableMap,
		"reachable_count":   len(reachable),
		"unreachable_count": len(unreachableList),
		"total_nodes":       len(all),
	})
}

// handleRefreshNodes updates SSH credentials, jumphost config, and/or the fleet
// host list in-place.
//
// Credential change (User or KeyPath non-empty): pool.UpdateCredentials drops all
// cached connections, updates the fields, and nudges an immediate reprobe so nodes
// re-connect with the new credentials — no daemon restart needed.
//
// Jumphost change: handleJumphostUpdate detects whether the jumphost config has
// changed and reconnects if needed. This allows runtime switching between direct
// and tunnelled connections without restarting the daemon.
//
// Node list change: pool.Refresh diffs against the current list and probes any
// newly added hosts. The reprobe nudge from Refresh() is a no-op if credentials
// were also updated (channel already has a pending nudge).
//
// Key-content-only (same path, no cred fields): Refresh() with the same list
// still nudges TriggerReprobe, causing an immediate sweep of unreachable nodes
// which now succeed because the key file has appeared on disk.
func handleRefreshNodes(conn net.Conn, msg IncomingMsg) {
	// Handle node credential updates (existing logic).
	if msg.User != "" || msg.KeyPath != "" || len(msg.KeyBytes) > 0 || msg.Password != "" {
		pool.UpdateCredentials(msg.User, msg.KeyPath, msg.KeyBytes, msg.Password)
	}

	// Handle jumphost updates (new logic).
	jumpErr := handleJumphostUpdate(msg)
	if jumpErr != nil {
		logger.Error("jumphost update failed", "error", jumpErr)
		// Continue with node refresh even if jumphost fails — nodes will be
		// unreachable until jumphost is fixed, but the host list is updated.
	}

	added, removed := pool.Refresh(msg.Hosts)

	if len(added) > 0 {
		go pool.ProbeSubset(context.Background(), added)
	}

	resp := map[string]any{
		"id":      msg.ID,
		"type":    "refresh_nodes",
		"added":   added,
		"removed": removed,
		"total":   len(msg.Hosts),
	}
	if jumpErr != nil {
		resp["jump_error"] = jumpErr.Error()
	}
	writeJSON(conn, resp)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

// makeDialFunc returns a dial function that uses the current jumpClient state.
// When jumpClient is nil, connections are made directly. When non-nil, connections
// are tunnelled through the jump host.
func makeDialFunc() func(network, addr string) (net.Conn, error) {
	return func(network, addr string) (net.Conn, error) {
		jumpMu.Lock()
		jc := jumpClient
		currentJumpAddr := jumpHostAddr
		jumpMu.Unlock()
		if jc == nil {
			// Direct connection (no jump host)
			logger.Debug("dialing direct", "network", network, "addr", addr)
			conn, err := net.DialTimeout(network, addr, 30*time.Second)
			if err != nil {
				logger.Warn("direct dial failed", "addr", addr, "error", err)
			} else {
				logger.Debug("direct dial succeeded", "addr", addr)
			}
			return conn, err
		}
		// Tunnelled connection through jump host
		logger.Info("dialing through jumphost", "jumphost", currentJumpAddr, "target", addr)
		conn, err := jc.Dial(network, addr)
		if err != nil {
			logger.Error("jumphost tunnel dial failed", "jumphost", currentJumpAddr, "target", addr, "error", err)
		} else {
			logger.Info("jumphost tunnel dial succeeded", "jumphost", currentJumpAddr, "target", addr)
		}
		return conn, err
	}
}

// handleJumphostUpdate detects jumphost configuration changes and reconnects
// the jump host if needed. Returns nil if no change or success, error on failure.
//
// Transition cases handled:
//  1. No jumphost -> jumphost (establish new tunnel)
//  2. Jumphost -> different jumphost (reconnect)
//  3. Same jumphost, different credentials (reconnect)
//  4. Jumphost -> no jumphost (switch to direct)
func handleJumphostUpdate(msg IncomingMsg) error {
	jumpMu.Lock()
	defer jumpMu.Unlock()

	newHost := msg.JumpHost
	newUser := msg.JumpUser
	newKey := msg.JumpKey
	newPassword := msg.JumpPassword

	// Log incoming jumphost params for debugging.
	logger.Info("handleJumphostUpdate called",
		"jump_host", newHost,
		"jump_user", newUser,
		"jump_key", newKey,
		"jump_password_set", newPassword != "",
		"current_jumphost", jumpHostAddr,
		"current_client_exists", jumpClient != nil,
	)

	// Case 4: Switching from jumphost to direct connection.
	// Empty JumpHost with existing jumpClient means "remove jumphost".
	if newHost == "" && jumpClient != nil {
		logger.Info("removing jump host, switching to direct connections")
		_ = jumpClient.Close()
		jumpClient = nil
		jumpHostAddr = ""
		jumpUserVal = ""
		jumpKeyPath = ""
		jumpPassVal = ""
		pool.UpdateDialFunc(nil) // signals direct connection
		return nil
	}

	// No jumphost requested and none currently configured — nothing to do.
	if newHost == "" {
		logger.Debug("no jumphost requested and none configured, skipping")
		return nil
	}

	// Check if jumphost config changed.
	changed := jumpHostAddr != newHost ||
		jumpUserVal != newUser ||
		jumpKeyPath != newKey ||
		jumpPassVal != newPassword

	if !changed && jumpClient != nil {
		// Same config and client exists — verify it's still alive.
		_, _, err := jumpClient.SendRequest("keepalive@openssh.com", true, nil)
		if err == nil {
			return nil // Connection still good, no action needed.
		}
		logger.Warn("jump host connection dead, reconnecting", "error", err)
		// Fall through to reconnect.
	}

	// Close existing connection if any.
	if jumpClient != nil {
		_ = jumpClient.Close()
		jumpClient = nil
	}

	// Establish new jump host connection.
	jc, err := connectJumpHost(newHost, newUser, newKey, newPassword)
	if err != nil {
		return fmt.Errorf("connect jump host %q: %w", newHost, err)
	}

	// Update package-level state.
	jumpClient = jc
	jumpHostAddr = newHost
	jumpUserVal = newUser
	jumpKeyPath = newKey
	jumpPassVal = newPassword

	logger.Info("jump host connected/reconnected", "host", newHost, "user", newUser)

	// Update pool's dial function. The new dialFunc will use the updated
	// jumpClient. Since we still hold jumpMu, this ensures the pointer is
	// consistent when makeDialFunc() is called by pool connections.
	pool.UpdateDialFunc(makeDialFunc())

	return nil
}

func writeJSON(conn net.Conn, v any) {
	if err := json.NewEncoder(conn).Encode(v); err != nil {
		logger.Debug("write error", "error", err)
	}
}

// loadHosts reads one hostname per non-blank, non-comment line from path.
func loadHosts(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var hosts []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		hosts = append(hosts, line)
	}
	return hosts, scanner.Err()
}

// connectJumpHost establishes a persistent SSH connection to the jump host.
// The returned *xssh.Client's Dial method is used as the dialFunc for the pool.
// Supports key-based auth (keyPath), password auth (password), or both.
func connectJumpHost(host, user, keyPath, password string) (*xssh.Client, error) {
	logger.Info("connectJumpHost starting",
		"host", host,
		"user", user,
		"keyPath", keyPath,
		"password_set", password != "",
	)

	var authMethods []xssh.AuthMethod

	if keyPath != "" {
		logger.Info("reading jump key file", "path", keyPath)
		pem, err := os.ReadFile(keyPath)
		if err != nil {
			logger.Error("failed to read jump key file", "path", keyPath, "error", err)
			return nil, fmt.Errorf("read jump key %q: %w", keyPath, err)
		}
		logger.Info("jump key file read successfully", "path", keyPath, "size", len(pem))
		signer, err := xssh.ParsePrivateKey(pem)
		if err != nil {
			logger.Error("failed to parse jump key", "path", keyPath, "error", err)
			return nil, fmt.Errorf("parse jump key %q: %w", keyPath, err)
		}
		logger.Info("jump key parsed successfully", "path", keyPath)
		authMethods = append(authMethods, xssh.PublicKeys(signer))
	}

	if password != "" {
		logger.Info("using password auth for jumphost")
		authMethods = append(authMethods, xssh.Password(password))
	}

	if len(authMethods) == 0 {
		logger.Error("no auth methods available for jumphost", "keyPath", keyPath, "password_set", password != "")
		return nil, fmt.Errorf("jump host requires key file or password (got keyPath=%q, password_set=%v)", keyPath, password != "")
	}

	cfg := &xssh.ClientConfig{
		User:            user,
		Auth:            authMethods,
		HostKeyCallback: xssh.InsecureIgnoreHostKey(),
		Timeout:         30 * time.Second,
	}
	addr := host
	if _, _, err := net.SplitHostPort(host); err != nil {
		addr = net.JoinHostPort(host, "22")
	}
	logger.Info("dialing jumphost", "addr", addr, "user", user)
	client, err := xssh.Dial("tcp", addr, cfg)
	if err != nil {
		logger.Error("failed to dial jumphost", "addr", addr, "error", err)
		return nil, err
	}
	logger.Info("jumphost SSH connection established", "addr", addr, "user", user)
	return client, nil
}
