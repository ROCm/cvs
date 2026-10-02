package rccl

import (
	"log/slog"
	"testing"

	"github.com/alicebob/miniredis/v2"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
)

func TestRedisEventsMatchMemoryBounds(t *testing.T) {
	t.Setenv("STORAGE__REDIS__URL", "")
	t.Setenv("STORAGE__REDIS__PASSWORD", "")
	mr := miniredis.RunT(t)
	s := NewStore()
	s.ConnectRedis(config.RedisConfig{URL: "redis://" + mr.Addr()}, slog.Default())
	t.Cleanup(s.Close)
	if s.client() == nil {
		t.Fatal("redis not connected")
	}
	events := []Event{
		{"event_type": "a", "timestamp": 100.0},
		{"event_type": "b", "timestamp": 200.5},
	}
	mem := NewStore()
	for _, e := range events {
		s.PushEvent(e)
		mem.PushEvent(e)
	}
	checks := []struct {
		since, until float64
		typ          string
	}{
		{0, 0, ""},
		{100, 200.5, ""},
		{100.5, 0, ""},
		{0, 0, "a"},
		{300, 100, ""},
	}
	for _, c := range checks {
		gotR := s.Events(c.since, c.until, c.typ)
		gotM := mem.Events(c.since, c.until, c.typ)
		if len(gotR) != len(gotM) {
			t.Fatalf("since=%v until=%v type=%s redis=%d memory=%d", c.since, c.until, c.typ, len(gotR), len(gotM))
		}
	}
	if len(s.Events(0, 0, "")) != 2 {
		t.Fatal("until=0 dropped redis events")
	}
}
