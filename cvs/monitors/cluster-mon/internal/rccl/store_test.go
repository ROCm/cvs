package rccl

import "testing"

func TestMemoryEventsSurviveInProcess(t *testing.T) {
	s := NewStore()
	s.PushEvent(Event{"event_type": "training_marker", "timestamp": 100.0, "step": 1})
	s.PushEvent(Event{"event_type": "training_marker", "timestamp": 200.0, "step": 2})
	got := s.Events(50, 150, "")
	if len(got) != 1 {
		t.Fatalf("got %d events", len(got))
	}
	typed := s.Events(0, 300, "training_marker")
	if len(typed) != 2 {
		t.Fatalf("typed %d", len(typed))
	}
	if s.IsCapped() {
		t.Fatal("empty store should not be capped")
	}
	if len(s.Events(0, 0, "")) != 2 {
		t.Fatal("until=0 should be unbounded")
	}
	if len(s.Events(300, 100, "")) != 0 {
		t.Fatal("inverted range should be empty")
	}
}

func TestStoreCurrent(t *testing.T) {
	s := NewStore()
	s.PushSnapshot(map[string]any{"state": "healthy", "n": 1})
	cur := s.Current()
	if cur["state"] != "healthy" {
		t.Fatalf("current=%v", cur)
	}
}

func TestSoftwareUpgradeEvent(t *testing.T) {
	s := NewStore()
	s.PushSnapshot(map[string]any{"job_summary": map[string]any{"rccl_version": "2.28.3"}})
	s.PushSnapshot(map[string]any{"job_summary": map[string]any{"rccl_version": "2.28.9"}})
	ev := s.Events(0, NowUnix()+10, "software_upgrade")
	if len(ev) != 1 {
		t.Fatalf("upgrade events=%d", len(ev))
	}
	if ev[0]["from_version"] != "2.28.3" || ev[0]["to_version"] != "2.28.9" {
		t.Fatalf("%v", ev[0])
	}
}

func TestInspectorMemoryRing(t *testing.T) {
	s := NewStore()
	s.PushInspector(map[string]any{"timestamp": 1.0, "n": 1})
	s.PushInspector(map[string]any{"timestamp": 2.0, "n": 2})
	cur := s.InspectorCurrent()
	if cur["n"] != 2 {
		t.Fatalf("current=%v", cur)
	}
	hist := s.InspectorHistory(10)
	if len(hist) != 2 || hist[0]["n"] != 2 {
		t.Fatalf("hist=%v", hist)
	}
}
