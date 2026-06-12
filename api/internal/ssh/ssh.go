// Package ssh is the shared SSH primitive for the platform foundation (F2):
// a persistent connection pool with bounded concurrency that every tile reuses
// (probe/basic-info now, Cluster Monitor collectors later).
//
// It is a Go reimplementation of the cluster-mon Python SSH layer
// (cvs_parallel_ssh_reliable.py / jump_host_pssh.py). Parity decisions carried
// over: host keys are NOT verified (the Python code used AutoAddPolicy /
// StrictHostKeyChecking=no), connections are cached per host and recreated on
// failure, and an optional jump host proxies otherwise-unroutable nodes.
package ssh

import (
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

// defaultPort is the SSH port used when a host carries no explicit port.
const defaultPort = 22

// JumpConfig describes an optional bastion used to reach segmented nodes.
type JumpConfig struct {
	Host     string
	User     string
	KeyFile  string
	Password string
	Port     int
}

// Config configures a Pool. Auth uses the key file when Password is empty,
// matching the inventory's "key" vs "password" auth method.
type Config struct {
	User     string
	KeyFile  string
	Password string
	// DialTimeout bounds a single TCP+handshake; defaults to 10s.
	DialTimeout time.Duration
	// MaxParallel bounds concurrent sessions; defaults to 16.
	MaxParallel int
	// JumpHost, when set, proxies all node connections through a bastion.
	JumpHost *JumpConfig
}

// Result is the outcome of running a command on one host.
type Result struct {
	Host   string
	Output string
	Err    error
}

// Pool is a concurrency-safe, lazily-dialed SSH connection cache.
type Pool struct {
	cfg         Config
	auth        []ssh.AuthMethod
	dialTimeout time.Duration

	mu      sync.Mutex
	clients map[string]*ssh.Client
	jump    *ssh.Client

	sem chan struct{}
}

// NewPool validates the config (parsing the key / requiring a credential) and
// returns a ready pool. It does not dial until the first Run.
func NewPool(cfg Config) (*Pool, error) {
	auth, err := authMethods(cfg.User, cfg.KeyFile, cfg.Password)
	if err != nil {
		return nil, err
	}
	dialTimeout := cfg.DialTimeout
	if dialTimeout <= 0 {
		dialTimeout = 10 * time.Second
	}
	maxPar := cfg.MaxParallel
	if maxPar <= 0 {
		maxPar = 16
	}
	return &Pool{
		cfg:         cfg,
		auth:        auth,
		dialTimeout: dialTimeout,
		clients:     make(map[string]*ssh.Client),
		sem:         make(chan struct{}, maxPar),
	}, nil
}

// authMethods builds SSH auth: private key when no password, else password.
func authMethods(_ string, keyFile, password string) ([]ssh.AuthMethod, error) {
	if password != "" {
		return []ssh.AuthMethod{ssh.Password(password)}, nil
	}
	if keyFile == "" {
		return nil, fmt.Errorf("ssh: no credentials (need a key file or password)")
	}
	pem, err := os.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("ssh: read key %q: %w", keyFile, err)
	}
	signer, err := ssh.ParsePrivateKey(pem)
	if err != nil {
		return nil, fmt.Errorf("ssh: parse key %q: %w", keyFile, err)
	}
	return []ssh.AuthMethod{ssh.PublicKeys(signer)}, nil
}

func clientConfig(user string, auth []ssh.AuthMethod, timeout time.Duration) *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User:            user,
		Auth:            auth,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // parity with cluster-mon
		Timeout:         timeout,
	}
}

func withPort(host string, port int) string {
	if port <= 0 {
		port = defaultPort
	}
	if _, _, err := net.SplitHostPort(host); err == nil {
		return host // already host:port
	}
	return net.JoinHostPort(host, strconv.Itoa(port))
}

// client returns a cached client for host, dialing (optionally via the jump
// host) on a cache miss.
func (p *Pool) client(host string) (*ssh.Client, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if c, ok := p.clients[host]; ok {
		return c, nil
	}

	addr := withPort(host, defaultPort)
	cfg := clientConfig(p.cfg.User, p.auth, p.dialTimeout)

	var (
		c   *ssh.Client
		err error
	)
	if p.cfg.JumpHost != nil {
		c, err = p.dialViaJumpLocked(addr, cfg)
	} else {
		c, err = ssh.Dial("tcp", addr, cfg)
	}
	if err != nil {
		return nil, err
	}
	p.clients[host] = c
	return c, nil
}

// dialViaJumpLocked dials a node through the bastion. Caller holds p.mu.
func (p *Pool) dialViaJumpLocked(addr string, cfg *ssh.ClientConfig) (*ssh.Client, error) {
	if p.jump == nil {
		j := p.cfg.JumpHost
		jAuth, err := authMethods(j.User, j.KeyFile, j.Password)
		if err != nil {
			return nil, fmt.Errorf("ssh: jump host auth: %w", err)
		}
		jc := clientConfig(j.User, jAuth, p.dialTimeout)
		jump, err := ssh.Dial("tcp", withPort(j.Host, j.Port), jc)
		if err != nil {
			return nil, fmt.Errorf("ssh: dial jump host: %w", err)
		}
		p.jump = jump
	}

	conn, err := p.jump.Dial("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("ssh: jump->node dial: %w", err)
	}
	ncc, chans, reqs, err := ssh.NewClientConn(conn, addr, cfg)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("ssh: node handshake via jump: %w", err)
	}
	return ssh.NewClient(ncc, chans, reqs), nil
}

// drop removes (and closes) a cached client so the next Run redials it. This is
// the Go equivalent of the Python "recreate client on failure" behavior.
func (p *Pool) drop(host string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if c, ok := p.clients[host]; ok {
		_ = c.Close()
		delete(p.clients, host)
	}
}

// Run executes cmd on host and returns combined stdout+stderr. A failure drops
// the cached connection so the next call reconnects. The command is bounded by
// ctx; on cancellation the session is closed.
func (p *Pool) Run(ctx context.Context, host, cmd string) (string, error) {
	select {
	case p.sem <- struct{}{}:
		defer func() { <-p.sem }()
	case <-ctx.Done():
		return "", ctx.Err()
	}

	c, err := p.client(host)
	if err != nil {
		return "", err
	}
	sess, err := c.NewSession()
	if err != nil {
		p.drop(host)
		return "", err
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
		if r.err != nil {
			// A broken pipe/connection should force a redial next time.
			p.drop(host)
		}
		return string(r.out), r.err
	}
}

// RunAll runs cmd on every host concurrently (bounded by MaxParallel) and
// returns one Result per host.
func (p *Pool) RunAll(ctx context.Context, hosts []string, cmd string) map[string]Result {
	out := make(map[string]Result, len(hosts))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, h := range hosts {
		wg.Add(1)
		go func(host string) {
			defer wg.Done()
			s, err := p.Run(ctx, host, cmd)
			mu.Lock()
			out[host] = Result{Host: host, Output: s, Err: err}
			mu.Unlock()
		}(h)
	}
	wg.Wait()
	return out
}

// Close tears down all cached connections, including the jump host.
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for h, c := range p.clients {
		_ = c.Close()
		delete(p.clients, h)
	}
	if p.jump != nil {
		_ = p.jump.Close()
		p.jump = nil
	}
}
