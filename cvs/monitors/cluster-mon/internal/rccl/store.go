package rccl

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
)

const (
	snapshotStream      = "rccl:snapshots"
	eventStream         = "rccl:events"
	currentKey          = "rccl:current"
	inspectorStream     = "rccl:inspector:snapshots"
	inspectorCurrentKey = "rccl:inspector:current"

	memoryEventMax    = 500
	memorySnapshotMax = 100
	memoryInspMax     = 100
)

type Event map[string]any

// Store is the RCCL ring buffer. With Redis it matches the Python
// RCCLDataStore streams so history survives a monitor restart. Without Redis
// (or on Redis errors) it uses bounded in-memory deques.
type Store struct {
	mu     sync.Mutex
	r      *redis.Client
	logger *slog.Logger

	snapshotMax int
	eventMax    int

	events      []Event
	snapshots   []map[string]any
	current     map[string]any
	insp        []map[string]any
	currentInsp map[string]any
	lastVersion string
}

func NewStore() *Store {
	return &Store{
		logger:      slog.Default(),
		snapshotMax: 1000,
		eventMax:    10000,
		events:      []Event{},
		snapshots:   []map[string]any{},
		insp:        []map[string]any{},
	}
}

func (s *Store) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.r != nil {
		_ = s.r.Close()
		s.r = nil
	}
}

// ConnectRedis pings Redis using yaml + STORAGE__REDIS__* env (Python parity).
// On failure the store stays in-memory; callers do not fail startup.
func (s *Store) ConnectRedis(cfg config.RedisConfig, logger *slog.Logger) {
	if logger != nil {
		s.logger = logger
	}
	if cfg.SnapshotMaxEntries > 0 {
		s.snapshotMax = cfg.SnapshotMaxEntries
	}
	if cfg.EventMaxEntries > 0 {
		s.eventMax = cfg.EventMaxEntries
	}
	url := os.Getenv("STORAGE__REDIS__URL")
	if url == "" {
		url = cfg.URL
	}
	if url == "" {
		return
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		s.logger.Warn("redis_url_invalid", "err", err, "url", url)
		return
	}
	pass := os.Getenv("STORAGE__REDIS__PASSWORD")
	if pass == "" {
		pass = cfg.Password
	}
	if pass != "" {
		opts.Password = pass
	}
	if cfg.DB != 0 && opts.DB == 0 {
		opts.DB = cfg.DB
	}
	c := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := c.Ping(ctx).Err(); err != nil {
		s.logger.Warn("redis_unavailable_memory_fallback", "err", err)
		_ = c.Close()
		return
	}
	s.mu.Lock()
	s.r = c
	s.mu.Unlock()
	s.logger.Info("redis_connected", "addr", opts.Addr)
}

func (s *Store) client() *redis.Client {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.r
}

func (s *Store) PushSnapshot(snap map[string]any) {
	s.maybeVersionEvent(snap)
	if s.xaddHash(snapshotStream, currentKey, snap, s.snapshotMax) {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.current = snap
	s.snapshots = append(s.snapshots, snap)
	if len(s.snapshots) > memorySnapshotMax {
		s.snapshots = s.snapshots[len(s.snapshots)-memorySnapshotMax:]
	}
}

func (s *Store) PushEvent(e Event) {
	if s.xadd(eventStream, e, s.eventMax, true) {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, e)
	if len(s.events) > memoryEventMax {
		s.events = s.events[len(s.events)-memoryEventMax:]
	}
}

func (s *Store) PushInspector(snap map[string]any) {
	if s.xaddHash(inspectorStream, inspectorCurrentKey, snap, s.snapshotMax) {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.currentInsp = snap
	s.insp = append(s.insp, snap)
	if len(s.insp) > memoryInspMax {
		s.insp = s.insp[len(s.insp)-memoryInspMax:]
	}
}

func (s *Store) Events(since, until float64, eventType string) []Event {
	if since > 0 && until > 0 && since > until {
		return []Event{}
	}
	if c := s.client(); c != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		startID := "-"
		if since > 0 {
			startID = strconv.FormatInt(int64(since*1000), 10) + "-0"
		}
		entries, err := c.XRange(ctx, eventStream, startID, "+").Result()
		if err != nil {
			s.logger.Warn("redis_xrange_events", "err", err)
		} else {
			out := []Event{}
			for _, e := range entries {
				ev, err := decodeMap(e.Values["data"])
				if err != nil || !eventInRange(ev, since, until, eventType) {
					continue
				}
				out = append(out, ev)
			}
			return out
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []Event{}
	for _, e := range s.events {
		if eventInRange(e, since, until, eventType) {
			out = append(out, e)
		}
	}
	return out
}

// eventInRange applies the same bounds to memory and Redis. since or until
// <= 0 means that side is unbounded. Positive bounds are inclusive.
func eventInRange(e Event, since, until float64, eventType string) bool {
	ts := asFloat(e["timestamp"])
	if since > 0 && ts < since {
		return false
	}
	if until > 0 && ts > until {
		return false
	}
	if eventType != "" {
		if t, _ := e["event_type"].(string); t != eventType {
			return false
		}
	}
	return true
}

func (s *Store) Current() map[string]any {
	if c := s.client(); c != nil {
		if m := s.hgetJSON(c, currentKey); m != nil {
			return m
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

func (s *Store) SeedLastVersion(v string) {
	if v == "" || v == "unknown" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastVersion == "" {
		s.lastVersion = v
	}
}

func (s *Store) InspectorCurrent() map[string]any {
	if c := s.client(); c != nil {
		if m := s.hgetJSON(c, inspectorCurrentKey); m != nil {
			return m
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentInsp
}

func (s *Store) InspectorHistory(count int) []map[string]any {
	if count <= 0 {
		count = 50
	}
	if c := s.client(); c != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		entries, err := c.XRevRangeN(ctx, inspectorStream, "+", "-", int64(count)).Result()
		if err != nil {
			s.logger.Warn("redis_xrevrange_inspector", "err", err)
		} else {
			out := make([]map[string]any, 0, len(entries))
			for _, e := range entries {
				if m, err := decodeMap(e.Values["data"]); err == nil {
					out = append(out, m)
				}
			}
			return out
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := count
	if n > len(s.insp) {
		n = len(s.insp)
	}
	out := make([]map[string]any, n)
	copy(out, s.insp[len(s.insp)-n:])
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func (s *Store) IsCapped() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.r == nil && len(s.events) >= memoryEventMax
}

func (s *Store) maybeVersionEvent(snap map[string]any) {
	summary, _ := snap["job_summary"].(map[string]any)
	newV, _ := summary["rccl_version"].(string)
	if newV == "" || newV == "unknown" {
		return
	}
	s.mu.Lock()
	prev := s.lastVersion
	s.lastVersion = newV
	s.mu.Unlock()
	if prev == "" || prev == newV {
		return
	}
	s.logger.Info("rccl_software_upgrade", "from", prev, "to", newV)
	s.PushEvent(Event{
		"event_type":   "software_upgrade",
		"timestamp":    NowUnix(),
		"from_version": prev,
		"to_version":   newV,
	})
}

func (s *Store) xadd(stream string, payload any, maxlen int, approx bool) bool {
	c := s.client()
	if c == nil {
		return false
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	args := &redis.XAddArgs{Stream: stream, Values: map[string]any{"data": string(b)}, MaxLen: int64(maxlen)}
	if approx {
		args.Approx = true
	}
	if err := c.XAdd(ctx, args).Err(); err != nil {
		s.logger.Warn("redis_xadd_failed", "stream", stream, "err", err)
		return false
	}
	return true
}

func (s *Store) xaddHash(stream, hashKey string, payload map[string]any, maxlen int) bool {
	c := s.client()
	if c == nil {
		return false
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := c.XAdd(ctx, &redis.XAddArgs{Stream: stream, Values: map[string]any{"data": string(b)}, MaxLen: int64(maxlen)}).Err(); err != nil {
		s.logger.Warn("redis_xadd_failed", "stream", stream, "err", err)
		return false
	}
	ts := stringify(payload["timestamp"])
	if err := c.HSet(ctx, hashKey, "data", string(b), "ts", ts).Err(); err != nil {
		s.logger.Warn("redis_hset_failed", "key", hashKey, "err", err)
	}
	return true
}

func (s *Store) hgetJSON(c *redis.Client, key string) map[string]any {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	raw, err := c.HGet(ctx, key, "data").Result()
	if err != nil {
		return nil
	}
	m, err := decodeMap(raw)
	if err != nil {
		return nil
	}
	return m
}

func decodeMap(v any) (map[string]any, error) {
	var raw []byte
	switch t := v.(type) {
	case string:
		raw = []byte(t)
	case []byte:
		raw = t
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func asFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	}
	return 0
}

func stringify(v any) string {
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	default:
		return ""
	}
}

func NowUnix() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}
