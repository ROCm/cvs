// Package probe implements the foundation connectivity + basic-info layer (F2):
// a parallel TCP reachability sweep and an SSH-based collector for per-node GPU
// type, GPU count, and ROCm version. Results feed the inventory node statuses
// that gate and inform the tiles.
package probe

import (
	"context"
	"net"
	"strconv"
	"sync"
	"time"
)

// SSHPort is the port probed for reachability (SSH), matching cluster-mon.
const SSHPort = 22

// DefaultTCPTimeout mirrors the Python host_probe 5s per-host timeout.
const DefaultTCPTimeout = 5 * time.Second

// defaultWorkers bounds the parallel TCP sweep.
const defaultWorkers = 100

// TCP reports whether a TCP connection to host:port succeeds within timeout.
func TCP(host string, port int, timeout time.Duration) bool {
	d := net.Dialer{Timeout: timeout}
	conn, err := d.Dial("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

// TCPAll probes every host in parallel (bounded by workers) and partitions them
// into reachable and unreachable, preserving the input order in each slice.
func TCPAll(hosts []string, port int, timeout time.Duration, workers int) (reachable, unreachable []string) {
	if port <= 0 {
		port = SSHPort
	}
	if timeout <= 0 {
		timeout = DefaultTCPTimeout
	}
	if workers <= 0 {
		workers = defaultWorkers
	}

	results := make([]bool, len(hosts))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i, h := range hosts {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, host string) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = TCP(host, port, timeout)
		}(i, h)
	}
	wg.Wait()

	for i, h := range hosts {
		if results[i] {
			reachable = append(reachable, h)
		} else {
			unreachable = append(unreachable, h)
		}
	}
	return reachable, unreachable
}

// Runner is the SSH capability the basic-info collector needs. *ssh.Pool
// satisfies it.
type Runner interface {
	Run(ctx context.Context, host, cmd string) (string, error)
}
