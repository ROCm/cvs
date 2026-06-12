package probe

import (
	"context"
	"net"
	"testing"
	"time"
)

func TestTCP_ReachableAndUnreachable(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	_, portStr, _ := net.SplitHostPort(ln.Addr().String())
	port := atoiTest(t, portStr)

	if !TCP("127.0.0.1", port, time.Second) {
		t.Fatal("expected open port to be reachable")
	}

	// Closed listener -> unreachable.
	ln2, _ := net.Listen("tcp", "127.0.0.1:0")
	_, p2, _ := net.SplitHostPort(ln2.Addr().String())
	ln2.Close()
	if TCP("127.0.0.1", atoiTest(t, p2), 500*time.Millisecond) {
		t.Fatal("expected closed port to be unreachable")
	}
}

func TestTCPAll_Partitions(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	_, portStr, _ := net.SplitHostPort(ln.Addr().String())
	port := atoiTest(t, portStr)

	// Two entries point at the open port; one at an unroutable address.
	hosts := []string{"127.0.0.1", "192.0.2.1", "127.0.0.1"}
	reachable, unreachable := TCPAll(hosts, port, 500*time.Millisecond, 10)

	if len(reachable) != 2 {
		t.Fatalf("reachable = %v, want 2", reachable)
	}
	if len(unreachable) != 1 || unreachable[0] != "192.0.2.1" {
		t.Fatalf("unreachable = %v, want [192.0.2.1]", unreachable)
	}
}

// fakeRunner returns canned outputs keyed by command substring.
type fakeRunner struct {
	byCmd map[string]string
}

func (f *fakeRunner) Run(_ context.Context, _ string, cmd string) (string, error) {
	if out, ok := f.byCmd[cmd]; ok {
		return out, nil
	}
	return "", nil
}

func TestCollect_UsesRunnerOutputs(t *testing.T) {
	r := &fakeRunner{byCmd: map[string]string{
		cmdROCmVersion: `[{"rocm_version":"7.0.2"}]`,
		cmdProductName: `{"card0":{"Card Series":"AMD Instinct MI300X"},"card1":{"Card Series":"AMD Instinct MI300X"}}`,
	}}
	got := Collect(context.Background(), r, []string{"node1"})
	info := got["node1"]
	if info.ROCmVersion != "7.0.2" {
		t.Fatalf("rocm = %q, want 7.0.2", info.ROCmVersion)
	}
	if info.GPUCount != 2 {
		t.Fatalf("count = %d, want 2", info.GPUCount)
	}
	if info.GPUType != "AMD Instinct MI300X" {
		t.Fatalf("type = %q", info.GPUType)
	}
}

func atoiTest(t *testing.T, s string) int {
	t.Helper()
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			t.Fatalf("bad port %q", s)
		}
		n = n*10 + int(c-'0')
	}
	return n
}
