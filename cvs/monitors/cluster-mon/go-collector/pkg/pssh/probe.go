package pssh

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// ProbeTimeout is the per-host deadline for a single connectivity check.
	ProbeTimeout = 30 * time.Second

	// probeCmd is the command used to verify a node is reachable and the SSH
	// session is functional. It must be universally available on target nodes.
	probeCmd = "echo OK"
)

// Probe runs probeCmd on ALL inventory nodes (including nodes currently in the
// unreachable set) and rebuilds the reachable/unreachable sets from scratch.
// Nodes that were unreachable but now respond are re-promoted to reachable.
//
// Use this for the initial fleet sweep after New() returns. The background
// reprobe loop uses reprobeUnreachable instead — reachable nodes are already
// monitored by per-connection keepalives and do not need re-sweeping.
//
// Probe is safe to call concurrently with Exec; both compete for the shared
// semaphore so they never overwhelm the cluster with simultaneous sessions.
func (p *Pool) Probe(ctx context.Context) ProbeResult {
	t0 := time.Now()

	p.stateMu.RLock()
	targets := make([]string, len(p.all))
	copy(targets, p.all)
	p.stateMu.RUnlock()

	type item struct {
		host string
		ok   bool
		err  error
	}
	ch := make(chan item, len(targets))

	var wg sync.WaitGroup
	for _, host := range targets {
		wg.Add(1)
		go func(h string) {
			defer wg.Done()
			pctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
			defer cancel()
			_, err := p.runSession(pctx, h, probeCmd)
			ch <- item{host: h, ok: err == nil, err: err}
		}(host)
	}

	// Close ch once all goroutines have sent their results.
	go func() {
		wg.Wait()
		close(ch)
	}()

	okSet := make(map[string]bool, len(targets))
	errMap := make(map[string]string, len(targets))
	for it := range ch {
		okSet[it.host] = it.ok
		if !it.ok {
			errMap[it.host] = cleanProbeErr(it.err)
			// Drop the connection so the next attempt re-dials.
			p.drop(it.host)
		}
	}

	// Rebuild state maps in inventory order, preserving node ordering in the
	// returned slices.
	newReachable := make(map[string]struct{}, len(targets))
	newUnreachable := make(map[string]struct{}, len(targets))
	var reachable, unreachable []string

	for _, h := range targets {
		if okSet[h] {
			newReachable[h] = struct{}{}
			reachable = append(reachable, h)
		} else {
			newUnreachable[h] = struct{}{}
			unreachable = append(unreachable, h)
		}
	}

	p.stateMu.Lock()
	p.reachable = newReachable
	p.unreachable = newUnreachable
	for h, e := range errMap {
		p.probeErrs[h] = e
	}
	// Clear errors for nodes that became reachable.
	for _, h := range reachable {
		delete(p.probeErrs, h)
	}
	p.stateMu.Unlock()

	elapsed := time.Since(t0)
	p.logger.Info("pssh_probe_done",
		"reachable", len(reachable),
		"unreachable", len(unreachable),
		"total", len(targets),
		"elapsed_ms", elapsed.Milliseconds(),
	)

	return ProbeResult{
		Reachable:   reachable,
		Unreachable: unreachable,
		Duration:    elapsed,
	}
}

// ProbeSubset is like Probe but only targets the given hosts — it does not
// touch any other node's reachability state. Use this after AddNodes to dial
// and classify a small set of newly-added nodes without re-probing the whole
// fleet. Unreachable nodes in the subset have their cached connection dropped.
func (p *Pool) ProbeSubset(ctx context.Context, hosts []string) ProbeResult {
	if len(hosts) == 0 {
		return ProbeResult{}
	}
	t0 := time.Now()

	type item struct {
		host string
		ok   bool
		err  error
	}
	ch := make(chan item, len(hosts))

	var wg sync.WaitGroup
	for _, host := range hosts {
		wg.Add(1)
		go func(h string) {
			defer wg.Done()
			pctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
			defer cancel()
			_, err := p.runSession(pctx, h, probeCmd)
			ch <- item{host: h, ok: err == nil, err: err}
		}(host)
	}
	go func() { wg.Wait(); close(ch) }()

	var reached, missed []string
	errMap := make(map[string]string, len(hosts))
	for it := range ch {
		if it.ok {
			reached = append(reached, it.host)
		} else {
			missed = append(missed, it.host)
			errMap[it.host] = cleanProbeErr(it.err)
			p.drop(it.host)
		}
	}

	// Update only the subset's entries in the reachability maps.
	p.stateMu.Lock()
	for _, h := range reached {
		delete(p.unreachable, h)
		delete(p.probeErrs, h)
		p.reachable[h] = struct{}{}
	}
	for _, h := range missed {
		delete(p.reachable, h)
		p.unreachable[h] = struct{}{}
		p.probeErrs[h] = errMap[h]
	}
	p.stateMu.Unlock()

	elapsed := time.Since(t0)
	p.logger.Info("pssh_probe_subset_done",
		"reachable", len(reached),
		"unreachable", len(missed),
		"total", len(hosts),
		"elapsed_ms", elapsed.Milliseconds(),
	)

	return ProbeResult{
		Reachable:   reached,
		Unreachable: missed,
		Duration:    elapsed,
	}
}

// reprobeUnreachable attempts to re-establish connectivity to every node
// currently in the unreachable set. Nodes that respond are re-promoted to
// reachable; nodes that still fail have their stale cached connection dropped
// so the next attempt gets a fresh dial.
//
// Reachable nodes are intentionally skipped — they are already monitored by
// per-connection SSH keepalives (every 30 s) and by pruneAfterExec on every
// Exec call. Running echo OK on 150 healthy nodes every 300 s just to confirm
// what keepalive already knows would generate unnecessary SSH sessions.
func (p *Pool) reprobeUnreachable(ctx context.Context) {
	p.stateMu.RLock()
	targets := make([]string, 0, len(p.unreachable))
	for h := range p.unreachable {
		targets = append(targets, h)
	}
	p.stateMu.RUnlock()

	if len(targets) == 0 {
		return
	}

	p.logger.Info("pssh_reprobe_unreachable", "count", len(targets))

	type item struct {
		host string
		ok   bool
		err  error
	}
	ch := make(chan item, len(targets))

	var wg sync.WaitGroup
	for _, host := range targets {
		wg.Add(1)
		go func(h string) {
			defer wg.Done()
			pctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
			defer cancel()
			_, err := p.runSession(pctx, h, probeCmd)
			ch <- item{host: h, ok: err == nil, err: err}
		}(host)
	}
	go func() {
		wg.Wait()
		close(ch)
	}()

	var recovered []string
	errMap := make(map[string]string)
	for it := range ch {
		if it.ok {
			recovered = append(recovered, it.host)
		} else {
			reason := cleanProbeErr(it.err)
			errMap[it.host] = reason
			// Log each failing host so operators can see the actual SSH error.
			p.logger.Info("pssh_host_unreachable", "host", it.host, "reason", reason)
			// Drop the stale cached connection so the next attempt re-dials.
			p.drop(it.host)
		}
	}

	p.stateMu.Lock()
	for h, e := range errMap {
		p.probeErrs[h] = e
	}
	for _, h := range recovered {
		delete(p.unreachable, h)
		delete(p.probeErrs, h)
		p.reachable[h] = struct{}{}
	}
	p.stateMu.Unlock()

	if len(recovered) > 0 {
		p.logger.Info("pssh_reprobe_recovered", "hosts", recovered, "count", len(recovered))
	}
}

// runBackground is the background re-probe goroutine started by New.
// On every interval tick it calls reprobeUnreachable (not the full Probe) —
// reachable nodes are already covered by per-connection keepalives.
// It also responds immediately to nudges sent via p.reprobe (e.g. from Refresh).
// It exits when ctx is cancelled.
//
// Because New() initialises all nodes as unreachable (pessimistic default),
// the very first thing runBackground does is a t=0 sweep of the whole fleet.
// This classifies every node before the first ticker interval fires, so the
// initial Exec() call gets real reachability data instead of an empty set.
func (p *Pool) runBackground(ctx context.Context, interval time.Duration) {
	p.logger.Info("pssh_background_started", "reprobe_interval", interval.String())
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// t=0 initial fleet classification. All nodes start unreachable so this
	// sweep touches every node. probedOnce is set immediately after so that
	// InitialProbeDone() / handleHealth probe_status flips to "complete".
	p.reprobeUnreachable(ctx)
	atomic.StoreInt32(&p.probedOnce, 1)

	for {
		select {
		case <-ctx.Done():
			p.logger.Info("pssh_background_stopped")
			return

		case <-ticker.C:
			p.logger.Info("pssh_background_reprobe", "trigger", "interval")
			p.reprobeUnreachable(ctx)

		case <-p.reprobe:
			// Drain any extra nudges that piled up while we were probing.
			for len(p.reprobe) > 0 {
				<-p.reprobe
			}
			p.logger.Info("pssh_background_reprobe", "trigger", "nudge")
			p.reprobeUnreachable(ctx)
		}
	}
}

// cleanProbeErr converts a raw SSH dial/session error into a short, readable
// reason string suitable for display in the UI. The goal is to preserve the
// essential cause without leaking verbose internal Go error chains.
func cleanProbeErr(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()

	switch {
	case strings.Contains(msg, "connection refused"):
		return "connection refused (port 22 closed)"
	case strings.Contains(msg, "no route to host"):
		return "no route to host"
	case strings.Contains(msg, "network is unreachable"):
		return "network unreachable"
	case strings.Contains(msg, "i/o timeout"), strings.Contains(msg, "deadline exceeded"):
		return "connection timed out"
	case strings.Contains(msg, "unable to authenticate"):
		return "SSH authentication failed"
	case strings.Contains(msg, "handshake failed"):
		return "SSH handshake failed"
	case strings.Contains(msg, "host key verification failed"):
		return "host key mismatch"
	default:
		// Trim noisy prefix added by Go's net package and x/crypto/ssh.
		for _, pfx := range []string{"ssh: ", "dial tcp "} {
			if idx := strings.Index(msg, pfx); idx >= 0 {
				msg = msg[idx+len(pfx):]
			}
		}
		// Cap length to keep the UI tidy.
		if len(msg) > 120 {
			msg = msg[:120] + "…"
		}
		return msg
	}
}
