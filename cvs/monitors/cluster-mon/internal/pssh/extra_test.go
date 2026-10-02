package pssh

import (
	"context"
	"log/slog"
	"testing"
	"time"
)

func TestLocalForwardUnknownHost(t *testing.T) {
	p, err := New("u", "", "", []string{"no-such-host.invalid"}, 1, time.Hour, time.Hour, time.Second, slog.Default(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	_, err = p.LocalForward(context.Background(), "no-such-host.invalid", "::1", 28028)
	if err == nil {
		t.Fatal("expected error for unreachable host")
	}
}

func TestExecHostsEmpty(t *testing.T) {
	p, err := New("u", "", "", nil, 1, time.Hour, time.Hour, time.Second, slog.Default(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	out := p.ExecHosts(context.Background(), "echo ok", nil)
	if len(out) != 0 {
		t.Fatalf("got %d results", len(out))
	}
}
