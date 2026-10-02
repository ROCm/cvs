package pssh

import (
	"context"
	"errors"

	xssh "golang.org/x/crypto/ssh"
)

// abortMsg is appended to the output of hosts that are pruned after a failed
// Exec, mirroring the Python Pssh.inform_unreachability message.
const abortMsg = "\nABORT: Host Unreachable Error"

// pruneAfterExec inspects the results of an Exec call, runs a connectivity
// double-check on any hosts that returned a connection/session error, and prunes
// those that are still unreachable (Python prune_unreachable_hosts parity).
//
// Hosts whose error is only a command exit-code failure (ssh.ExitError) are
// never pruned — parity with Python which only prunes on ConnectionError /
// Timeout / SessionError.
//
// The results map is mutated in place: pruned hosts have abortMsg appended to
// their Output so the caller can distinguish them from transient failures.
func (p *Pool) pruneAfterExec(ctx context.Context, results map[string]Result) {
	// Collect hosts whose failure is a connection/session error.
	var connFailed []string
	for host, r := range results {
		if r.Err != nil && isConnError(r.Err) {
			connFailed = append(connFailed, host)
		}
	}
	if len(connFailed) == 0 {
		return
	}

	// Double-check connectivity — only prune hosts that are still unreachable.
	// (Python: check_connectivity with timeout=2, num_retries=0)
	stillDown := p.checkConnectivity(ctx, connFailed)
	if len(stillDown) == 0 {
		return
	}

	// Prune confirmed-down hosts from the reachable set.
	p.pruneNodes(stillDown)

	// Annotate results for the caller (Python inform_unreachability parity).
	for _, host := range stillDown {
		if r, ok := results[host]; ok {
			r.Output += abortMsg
			results[host] = r
		}
	}
}

// checkConnectivity runs probeCmd on each of the given hosts using the shared
// pool connections (which were already dropped by runSession on failure, so this
// triggers a fresh re-dial). It returns only those hosts that still fail,
// indicating confirmed unreachability.
func (p *Pool) checkConnectivity(ctx context.Context, hosts []string) []string {
	if len(hosts) == 0 {
		return nil
	}

	// Use ProbeTimeout as the per-host deadline, independent of the caller's
	// context.
	cctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
	defer cancel()

	type item struct {
		host string
		down bool
	}
	ch := make(chan item, len(hosts))

	for _, host := range hosts {
		go func(h string) {
			_, err := p.runSession(cctx, h, probeCmd)
			ch <- item{host: h, down: err != nil}
		}(host)
	}

	var down []string
	for range hosts {
		it := <-ch
		if it.down {
			down = append(down, it.host)
		}
	}
	return down
}

// pruneNodes removes hosts from the reachable set, adds them to the unreachable
// set, and drops their cached connections.
func (p *Pool) pruneNodes(hosts []string) {
	p.stateMu.Lock()
	for _, h := range hosts {
		delete(p.reachable, h)
		p.unreachable[h] = struct{}{}
	}
	p.stateMu.Unlock()

	for _, h := range hosts {
		p.drop(h)
	}

	p.logger.Info("pssh_pruned", "hosts", hosts, "count", len(hosts))
}

// isConnError reports whether err represents a connection or session failure
// (as opposed to a command exit-code error or a caller-driven cancellation).
// Mirrors the Python check for ConnectionError / Timeout / SessionError.
func isConnError(err error) bool {
	if err == nil {
		return false
	}
	// The command ran and exited non-zero — the SSH connection is healthy.
	var exitErr *xssh.ExitError
	if errors.As(err, &exitErr) {
		return false
	}
	// Caller cancelled — not a node failure.
	if errors.Is(err, context.Canceled) {
		return false
	}
	// Per-host or fleet-wide deadline exceeded — the timeout may have been
	// too short; don't prune healthy nodes that simply ran long.
	if errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	return true
}
