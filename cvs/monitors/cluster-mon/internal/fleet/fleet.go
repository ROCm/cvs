package fleet

import (
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/pkg/sftp"
	xssh "golang.org/x/crypto/ssh"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

// Fleet owns the in-process SSH pool and optional jump-host dialer.
type Fleet struct {
	mu     sync.Mutex
	pool   *pssh.Pool
	jump   *jumpDialer
	logger *slog.Logger
}

func New(logger *slog.Logger) *Fleet {
	if logger == nil {
		logger = slog.Default()
	}
	return &Fleet{logger: logger}
}

func (f *Fleet) Pool() *pssh.Pool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pool
}

// Close tears down the current pool and jump connection.
func (f *Fleet) Close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pool != nil {
		f.pool.Close()
		f.pool = nil
	}
	if f.jump != nil {
		f.jump.close()
		f.jump = nil
	}
}

// Rebuild replaces the pool from cfg. Jump-host node keys are fetched via SFTP
// and kept in memory (never written to the container disk).
func (f *Fleet) Rebuild(cfg *config.Config, sshPassword, jumpPassword string) error {
	if cfg == nil || len(cfg.Nodes) == 0 {
		return fmt.Errorf("no nodes configured")
	}

	user := cfg.SSH.Username
	keyPath := config.ExpandPath(cfg.SSH.KeyFile)
	var dialFunc func(network, addr string) (net.Conn, error)
	var keyBytes []byte
	var jd *jumpDialer

	jh := cfg.SSH.JumpHost
	if jh.Enabled && jh.Host != "" {
		user = jh.NodeUsername
		if user == "" {
			user = cfg.SSH.Username
		}
		keyPath = ""
		var err error
		jd, err = newJumpDialer(jh.Host, jh.Username, config.ExpandPath(jh.KeyFile), jumpPassword, f.logger)
		if err != nil {
			return fmt.Errorf("jump host: %w", err)
		}
		var fetchErr error
		var pem []byte
		if jh.NodeKeyFile != "" {
			pem, fetchErr = jd.fetchFile(jh.NodeKeyFile)
		}
		keyBytes, err = resolveJumpNodeKey(jh.NodeKeyFile, pem, fetchErr, sshPassword, f.logger)
		if err != nil {
			jd.close()
			return err
		}
		if len(keyBytes) > 0 {
			f.logger.Info("node_key_fetched", "bytes", len(keyBytes))
		}
		dialFunc = jd.Dial
	}

	p, err := pssh.New(user, keyPath, sshPassword, cfg.Nodes, 10, 0, 300*time.Second, 60*time.Second, f.logger, dialFunc)
	if err != nil {
		if jd != nil {
			jd.close()
		}
		return err
	}
	if len(keyBytes) > 0 {
		p.UpdateCredentials("", "", keyBytes)
	}

	f.mu.Lock()
	oldPool, oldJump := f.pool, f.jump
	f.pool = p
	f.jump = jd
	f.mu.Unlock()
	if oldPool != nil {
		oldPool.Close()
	}
	if oldJump != nil {
		oldJump.close()
	}
	f.logger.Info("fleet_ready", "nodes", len(cfg.Nodes), "user", user, "jump", jh.Enabled)
	return nil
}

// jumpDialer keeps a reconnecting SSH client to the bastion.
type jumpDialer struct {
	mu       sync.Mutex
	client   *xssh.Client
	host     string
	user     string
	keyPath  string
	password string
	logger   *slog.Logger
}

func newJumpDialer(host, user, keyPath, password string, logger *slog.Logger) (*jumpDialer, error) {
	j := &jumpDialer{host: host, user: user, keyPath: keyPath, password: password, logger: logger}
	if err := j.connect(); err != nil {
		return nil, err
	}
	return j, nil
}

func (j *jumpDialer) Dial(network, addr string) (net.Conn, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.client != nil {
		c, err := j.client.Dial(network, addr)
		if err == nil {
			return c, nil
		}
		j.logger.Warn("jump_dial_failed_reconnecting", "err", err)
		_ = j.client.Close()
		j.client = nil
	}
	if err := j.connectLocked(); err != nil {
		return nil, err
	}
	return j.client.Dial(network, addr)
}

func (j *jumpDialer) connect() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.connectLocked()
}

// resolveJumpNodeKey decides which node credential a jump-host reload may
// install. A failed fetch or unparseable key is fatal only when no node
// password can authenticate instead.
func resolveJumpNodeKey(path string, pem []byte, fetchErr error, password string, logger *slog.Logger) ([]byte, error) {
	if path == "" {
		if password == "" {
			return nil, fmt.Errorf("jump host node auth requires a node key or password")
		}
		return nil, nil
	}
	if fetchErr != nil {
		if password == "" {
			return nil, fmt.Errorf("fetch node key %q: %w", path, fetchErr)
		}
		logger.Warn("node_key_sftp_failed", "err", fetchErr, "path", path, "fallback", "password")
		return nil, nil
	}
	if _, err := xssh.ParsePrivateKey(pem); err != nil {
		if password == "" {
			return nil, fmt.Errorf("parse node key %q: %w", path, err)
		}
		logger.Warn("node_key_parse_failed", "err", err, "path", path, "fallback", "password")
		return nil, nil
	}
	return pem, nil
}

// jumpAuthMethods offers the jump key when it parses, and the jump password
// when a key is missing or unusable. The password is an SSH login password,
// not a key passphrase.
func jumpAuthMethods(keyPath, password string, logger *slog.Logger) ([]xssh.AuthMethod, error) {
	var auth []xssh.AuthMethod
	if keyPath != "" {
		pem, err := os.ReadFile(keyPath)
		if err != nil {
			if password == "" {
				return nil, fmt.Errorf("read jump key %q: %w", keyPath, err)
			}
			logger.Warn("jump_key_read_failed_using_password", "err", err, "path", keyPath)
		} else if signer, perr := xssh.ParsePrivateKey(pem); perr != nil {
			if password == "" {
				return nil, fmt.Errorf("parse jump key: %w", perr)
			}
			logger.Warn("jump_key_parse_failed_using_password", "err", perr, "path", keyPath)
		} else {
			auth = append(auth, xssh.PublicKeys(signer))
		}
	}
	if password != "" {
		auth = append(auth, xssh.Password(password))
	}
	if len(auth) == 0 {
		return nil, fmt.Errorf("jump host requires a key file or password")
	}
	return auth, nil
}

func (j *jumpDialer) connectLocked() error {
	auth, err := jumpAuthMethods(j.keyPath, j.password, j.logger)
	if err != nil {
		return err
	}
	cfg := &xssh.ClientConfig{
		User:            j.user,
		Auth:            auth,
		HostKeyCallback: xssh.InsecureIgnoreHostKey(),
		Timeout:         30 * time.Second,
	}
	addr := j.host
	if _, _, err := net.SplitHostPort(j.host); err != nil {
		addr = net.JoinHostPort(j.host, "22")
	}
	c, err := xssh.Dial("tcp", addr, cfg)
	if err != nil {
		return err
	}
	j.client = c
	j.logger.Info("jump_host_connected", "host", j.host)
	return nil
}

func (j *jumpDialer) fetchFile(remotePath string) ([]byte, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.client == nil {
		if err := j.connectLocked(); err != nil {
			return nil, err
		}
	}
	path := remotePath
	if path == "~" || strings.HasPrefix(path, "~/") {
		path = "/home/" + j.user + strings.TrimPrefix(path, "~")
	}
	s, err := sftp.NewClient(j.client)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	f, err := s.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

func (j *jumpDialer) close() {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.client != nil {
		_ = j.client.Close()
		j.client = nil
	}
}
