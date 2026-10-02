package rccl

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"
)

func TestEventTypeMap(t *testing.T) {
	if EventType("no_job", "healthy") != "job_start" {
		t.Fatal(EventType("no_job", "healthy"))
	}
	if EventType("healthy", "no_job") != "job_end" {
		t.Fatal(EventType("healthy", "no_job"))
	}
	if EventType("healthy", "healthy") != "" {
		t.Fatal("same state should emit nothing")
	}
	if EventType("healthy", "degraded") != "job_degraded" {
		t.Fatal(EventType("healthy", "degraded"))
	}
}

func TestBootstrapSkipsSpuriousJobStart(t *testing.T) {
	s := NewStore()
	s.PushSnapshot(map[string]any{"state": "healthy", "job_summary": map[string]any{"rccl_version": "2.28.9"}})
	c := NewCollector(nil, s, 28028, slog.Default())
	c.bootstrap()
	if c.JobState() != "healthy" {
		t.Fatalf("bootstrapped %s", c.JobState())
	}
	before := len(s.Events(0, NowUnix()+10, ""))
	c.pushStateEvent("healthy", "healthy", "")
	if len(s.Events(0, NowUnix()+10, "")) != before {
		t.Fatal("same-state event leaked")
	}
	c.pushStateEvent("healthy", "no_job", "n1")
	ev := s.Events(0, NowUnix()+10, "job_end")
	if len(ev) != 1 {
		t.Fatalf("job_end=%d", len(ev))
	}
}

func TestCollectNilPoolUnreachable(t *testing.T) {
	s := NewStore()
	c := NewCollector(nil, s, 0, slog.Default())
	snap := c.Collect(context.Background(), time.Second)
	if snap.State != "unreachable" {
		t.Fatalf("state=%s", snap.State)
	}
	if EventType("no_job", "unreachable") != "nodes_unreachable" {
		t.Fatal("expected nodes_unreachable")
	}
}

func TestClassifyRASErr(t *testing.T) {
	if classifyRASErr(errors.New("direct-tcpip n1 [::1]:28028: connection refused")) != rasRefused {
		t.Fatal("refused")
	}
	if classifyRASErr(context.DeadlineExceeded) != rasTimeout {
		t.Fatal("timeout")
	}
	if classifyRASErr(errors.New("unexpected handshake: nope")) != rasProtocol {
		t.Fatal("protocol")
	}
}

func TestMarkInspectorV5Mismatch(t *testing.T) {
	c := NewCollector(nil, NewStore(), 28028, slog.Default())
	c.caps["gpu-node-01"] = &nodeCap{JSONRAS: false, TTL: time.Hour, ProbedAt: time.Now()}
	c.MarkInspectorV5([]string{"gpu-node-01"})
	if !c.caps["gpu-node-01"].InspectorV5 {
		t.Fatal("inspector_v5 not set")
	}
}

func TestVersionSkewEvent(t *testing.T) {
	s := NewStore()
	c := NewCollector(nil, s, 28028, slog.Default())
	c.caps["a"] = &nodeCap{Version: "2.28.3"}
	c.caps["b"] = &nodeCap{Version: "2.28.9"}
	snap := emptySnap("healthy")
	snap.JobSummary = &JobSummary{RCCLVersion: "2.28.9"}
	c.checkSkew(&snap)
	if !snap.JobSummary.InconsistentTopology {
		t.Fatal("expected inconsistent_topology")
	}
	ev := s.Events(0, NowUnix()+10, "version_skew")
	if len(ev) != 1 {
		t.Fatalf("skew events=%d", len(ev))
	}
}
