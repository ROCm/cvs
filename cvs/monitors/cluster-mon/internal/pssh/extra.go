package pssh

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"sync"
)

// ExecHosts runs cmd on the given hosts only (not the full reachable set).
// Hosts that are currently unreachable are still attempted so installers can
// target a subset after a probe. Results are pruned the same way as Exec.
func (p *Pool) ExecHosts(ctx context.Context, cmd string, hosts []string) map[string]Result {
	results := make(map[string]Result, len(hosts))
	if len(hosts) == 0 {
		return results
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, host := range hosts {
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

// LocalForward opens a TCP connection to destHost:destPort via the existing
// SSH session to host. destHost is typically "::1" for services bound to the
// node's loopback (rcclras). Works for both direct SSH and jump-host pools
// because Dial runs on the already-established node client.
func (p *Pool) LocalForward(ctx context.Context, host, destHost string, destPort int) (net.Conn, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	entry, err := p.conn(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("ssh conn %s: %w", host, err)
	}
	addr := net.JoinHostPort(destHost, strconv.Itoa(destPort))
	c, err := entry.client.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("direct-tcpip %s %s: %w", host, addr, err)
	}
	return c, nil
}
