package clustermon

import (
	"testing"
	"time"

	"github.com/ROCm/cvs/api/internal/inventory"
)

func newTestPoller(threshold int) *Poller {
	return &Poller{failureThreshold: threshold, fails: map[string]int{}}
}

func statusOf(ss []inventory.NodeStatus, host string) (inventory.NodeStatus, bool) {
	for _, s := range ss {
		if s.Host == host {
			return s, true
		}
	}
	return inventory.NodeStatus{}, false
}

// A node that was reachable must not flip unreachable until it has missed
// FailureThreshold consecutive reprobes; a single success resets the counter.
func TestDebounceFlipsAfterThreshold(t *testing.T) {
	p := newTestPoller(5)
	host := "10.0.0.1"

	// Last-good state: reachable with basic info.
	prev := []inventory.NodeStatus{{
		Host: host, Reachable: true, GPUType: "MI300X", GPUCount: 8, ROCmVersion: "6.2",
	}}

	t0 := time.Now().UTC()
	unreachable := []inventory.NodeStatus{{Host: host, Reachable: false, Error: "tcp: port 22 unreachable", CheckedAt: t0}}

	// Misses 1..4 stay reachable (debounced), preserving the last-good info but
	// refreshing CheckedAt.
	for i := 1; i <= 4; i++ {
		out := p.debounce(prev, unreachable)
		st, ok := statusOf(out, host)
		if !ok {
			t.Fatalf("miss %d: host missing from output", i)
		}
		if !st.Reachable {
			t.Fatalf("miss %d: expected still-reachable (debounced), got unreachable", i)
		}
		if st.GPUType != "MI300X" {
			t.Fatalf("miss %d: expected last-good GPUType retained, got %q", i, st.GPUType)
		}
		if !st.CheckedAt.Equal(t0) {
			t.Fatalf("miss %d: expected refreshed CheckedAt, got %v", i, st.CheckedAt)
		}
		prev = out
	}

	// 5th consecutive miss confirms the node down.
	out := p.debounce(prev, unreachable)
	st, _ := statusOf(out, host)
	if st.Reachable {
		t.Fatalf("5th miss: expected unreachable, got reachable")
	}
	if st.Error == "" {
		t.Fatalf("5th miss: expected error to surface")
	}
}

func TestDebounceResetsOnSuccess(t *testing.T) {
	p := newTestPoller(3)
	host := "10.0.0.2"
	prev := []inventory.NodeStatus{{Host: host, Reachable: true, GPUType: "MI300X"}}
	unreachable := []inventory.NodeStatus{{Host: host, Reachable: false}}
	reachable := []inventory.NodeStatus{{Host: host, Reachable: true, GPUType: "MI300X"}}

	// Two misses (below threshold of 3).
	prev = p.debounce(prev, unreachable)
	prev = p.debounce(prev, unreachable)
	if p.fails[host] != 2 {
		t.Fatalf("expected 2 consecutive fails, got %d", p.fails[host])
	}

	// A success resets the counter.
	prev = p.debounce(prev, reachable)
	if p.fails[host] != 0 {
		t.Fatalf("expected fails reset to 0 after success, got %d", p.fails[host])
	}

	// Now it takes a fresh full threshold to flip again.
	for i := 1; i < 3; i++ {
		prev = p.debounce(prev, unreachable)
		if st, _ := statusOf(prev, host); !st.Reachable {
			t.Fatalf("post-reset miss %d: expected debounced reachable, got unreachable", i)
		}
	}
	prev = p.debounce(prev, unreachable)
	if st, _ := statusOf(prev, host); st.Reachable {
		t.Fatalf("post-reset 3rd miss: expected unreachable")
	}
}

// A node with no prior reachable status surfaces as unreachable immediately
// (nothing good to fall back to).
func TestDebounceNoPriorGoodStatus(t *testing.T) {
	p := newTestPoller(5)
	host := "10.0.0.3"
	out := p.debounce(nil, []inventory.NodeStatus{{Host: host, Reachable: false, Error: "boom"}})
	st, ok := statusOf(out, host)
	if !ok || st.Reachable {
		t.Fatalf("expected immediate unreachable when no prior good status, got %+v", st)
	}
}
