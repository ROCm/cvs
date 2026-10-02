package rccl

import (
	"os"
	"path/filepath"
	"testing"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join("..", "..", "testdata", name)
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestParseJSONHealthy(t *testing.T) {
	snap := ParseJSON(fixture(t, "rccl_v2289_json_healthy.json"))
	if snap.State != "healthy" {
		t.Fatalf("state=%s", snap.State)
	}
	if snap.JobSummary == nil || snap.JobSummary.RCCLVersion != "2.28.9" {
		t.Fatalf("summary=%+v", snap.JobSummary)
	}
	if len(snap.Communicators) != 1 {
		t.Fatalf("comms=%d", len(snap.Communicators))
	}
	c := snap.Communicators[0]
	if c.TotalRanks != 8 || c.MissingRanks != 0 || len(c.Ranks) != 8 {
		t.Fatalf("comm=%+v", c)
	}
	if c.Health != "healthy" {
		t.Fatalf("health=%s", c.Health)
	}
}

func TestParseJSONDegraded(t *testing.T) {
	snap := ParseJSON(fixture(t, "rccl_v2289_json_degraded.json"))
	if snap.State != "degraded" {
		t.Fatalf("state=%s", snap.State)
	}
}

func TestParseJSONEmpty(t *testing.T) {
	snap := ParseJSON("")
	if snap.State != "no_job" {
		t.Fatalf("state=%s", snap.State)
	}
}

func TestParseTextHealthy(t *testing.T) {
	snap := ParseText(fixture(t, "rccl_v2283_text_healthy.txt"))
	if snap.State != "healthy" {
		t.Fatalf("state=%s errors=%v", snap.State, snap.Errors)
	}
	if snap.JobSummary == nil || snap.JobSummary.RCCLVersion != "2.28.3" {
		t.Fatalf("summary=%+v", snap.JobSummary)
	}
	if snap.JobSummary.TotalNodes != 1 || snap.JobSummary.TotalGPUs != 8 {
		t.Fatalf("nodes/gpus %+v", snap.JobSummary)
	}
	if len(snap.Communicators) != 1 || snap.Communicators[0].MissingRanks != 0 {
		t.Fatalf("comms=%+v", snap.Communicators)
	}
}

func TestParseTextDegraded(t *testing.T) {
	snap := ParseText(fixture(t, "rccl_v2283_text_degraded.txt"))
	if snap.State != "degraded" {
		t.Fatalf("state=%s", snap.State)
	}
	if len(snap.Errors) == 0 {
		t.Fatal("expected Errors section")
	}
}

func TestParseTextConnectionReset(t *testing.T) {
	snap := ParseText(fixture(t, "rccl_v2283_text_connection_reset.txt"))
	if snap.State != "no_job" {
		t.Fatalf("state=%s", snap.State)
	}
}

func TestParseTextJSONIsError(t *testing.T) {
	snap := ParseText(`{"nccl_version":"2.28.9"}`)
	if snap.State != "error" {
		t.Fatalf("state=%s", snap.State)
	}
}

func TestParseTextDeadPeers(t *testing.T) {
	raw := "RCCL version 2.28.3 compiled with ROCm \"x\"\nJob summary\n===========\n\n  1 8 1 8 8\nDead peers: 10.0.0.1:28028, 10.0.0.2:28028\nCommunicators\n=============\n"
	snap := ParseText(raw)
	if len(snap.DeadPeers) != 2 {
		t.Fatalf("dead=%v", snap.DeadPeers)
	}
	if snap.State != "degraded" {
		t.Fatalf("state=%s", snap.State)
	}
}

func TestHipVersionStr(t *testing.T) {
	if hipVersionStr(70226015) == "unknown" {
		t.Fatal("expected decoded version")
	}
	if hipVersionStr(0) != "unknown" {
		t.Fatal("zero should be unknown")
	}
}
