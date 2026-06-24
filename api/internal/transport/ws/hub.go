// Package ws implements the shared WebSocket transport: a small pub/sub Hub and
// HTTP handlers that stream execution logs/status and global notifications to
// the UI. Messages use one envelope `{type, execution_id, timestamp, data}`
// with type in log|status|error|completion.
//
// Each execution gets a session that retains the log/status history emitted so
// far, so a client that connects mid-run receives the backlog atomically before
// live frames (no missed or duplicated lines). Sessions are removed once the
// execution reaches a terminal status; late viewers fall back to the persisted
// log file via the HTTP handler.
package ws

import (
	"sync"
	"time"
)

// Message is the wire envelope shared by all WS streams.
type Message struct {
	Type        string    `json:"type"` // log | status | error | completion | metrics
	ExecutionID string    `json:"execution_id,omitempty"`
	Timestamp   time.Time `json:"timestamp"`
	Data        any       `json:"data,omitempty"`
}

// subBuffer bounds a slow client; if it overflows we drop the subscriber rather
// than block publishers.
const subBuffer = 2048

// Hub multiplexes execution events to subscribers and keeps per-execution
// history for late joiners.
type Hub struct {
	mu       sync.Mutex
	sessions map[string]*session
	notif    *topic
	// metrics is the Cluster Monitor live-metrics broadcast (S8): a single
	// historyless fan-out of the latest fleet GPU snapshot to all /clustermon
	// subscribers. The server-side poller publishes here each interval.
	metrics *topic
}

// NewHub returns an empty hub.
func NewHub() *Hub {
	return &Hub{sessions: map[string]*session{}, notif: newTopic(), metrics: newTopic()}
}

// session holds the history + live subscribers for one execution.
type session struct {
	mu      sync.Mutex
	history []Message
	subs    map[int]chan Message
	nextID  int
	closed  bool
}

func newSession() *session { return &session{subs: map[int]chan Message{}} }

// publish appends to history and fans out to live subscribers.
func (s *session) publish(m Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.history = append(s.history, m)
	for id, ch := range s.subs {
		select {
		case ch <- m:
		default:
			// Slow consumer: drop it so it cannot stall the executor.
			close(ch)
			delete(s.subs, id)
		}
	}
}

// subscribe returns a channel preloaded with the current history followed by
// live frames, plus an unsubscribe func.
func (s *session) subscribe() (<-chan Message, func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ch := make(chan Message, len(s.history)+subBuffer)
	for _, m := range s.history {
		ch <- m
	}
	if s.closed {
		close(ch)
		return ch, func() {}
	}
	id := s.nextID
	s.nextID++
	s.subs[id] = ch
	return ch, func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if c, ok := s.subs[id]; ok {
			delete(s.subs, id)
			close(c)
		}
	}
}

// finish fans out the final frame, closes subscribers, and seals the session.
func (s *session) finish() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	for id, ch := range s.subs {
		close(ch)
		delete(s.subs, id)
	}
}

func (h *Hub) session(execID string) *session {
	h.mu.Lock()
	defer h.mu.Unlock()
	s, ok := h.sessions[execID]
	if !ok {
		s = newSession()
		h.sessions[execID] = s
	}
	return s
}

func (h *Hub) dropSession(execID string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.sessions, execID)
}

// DropSession removes an execution's session (used by the WS handler to clean up
// an empty session created while racing a just-finished run).
func (h *Hub) DropSession(execID string) { h.dropSession(execID) }

// SubscribeExecution returns the execution's history-then-live message stream
// and an unsubscribe func.
func (h *Hub) SubscribeExecution(execID string) (<-chan Message, func()) {
	return h.session(execID).subscribe()
}

// SubscribeNotifications returns the global notification stream (no history).
func (h *Hub) SubscribeNotifications() (<-chan Message, func()) {
	return h.notif.subscribe()
}

// SubscribeMetrics returns the Cluster Monitor live-metrics stream (no history;
// the handler sends the cached latest snapshot on connect).
func (h *Hub) SubscribeMetrics() (<-chan Message, func()) {
	return h.metrics.subscribe()
}

// PublishMetrics broadcasts a fleet GPU metrics snapshot to all /clustermon
// subscribers (data is the *MetricsSnapshot the poller just collected).
func (h *Hub) PublishMetrics(data any) {
	h.metrics.publish(Message{
		Type:      "metrics",
		Timestamp: time.Now().UTC(),
		Data:      data,
	})
}

// PublishLog emits a log line for an execution.
func (h *Hub) PublishLog(execID, line string) {
	h.session(execID).publish(Message{
		Type:        "log",
		ExecutionID: execID,
		Timestamp:   time.Now().UTC(),
		Data:        map[string]string{"line": line},
	})
}

// PublishStatus emits a status frame (data is the caller's status payload).
func (h *Hub) PublishStatus(execID string, data any) {
	h.session(execID).publish(Message{
		Type:        "status",
		ExecutionID: execID,
		Timestamp:   time.Now().UTC(),
		Data:        data,
	})
}

// PublishCompletion emits the terminal status to the execution stream, a
// completion frame to the global notifications stream, then seals the session
// (late viewers fall back to the persisted log file).
func (h *Hub) PublishCompletion(execID string, data any) {
	s := h.session(execID)
	s.publish(Message{
		Type:        "status",
		ExecutionID: execID,
		Timestamp:   time.Now().UTC(),
		Data:        data,
	})
	h.notif.publish(Message{
		Type:        "completion",
		ExecutionID: execID,
		Timestamp:   time.Now().UTC(),
		Data:        data,
	})
	s.finish()
	h.dropSession(execID)
}

// topic is a simple historyless fan-out (used for global notifications and the
// Cluster Monitor metrics broadcast). It supports optional lifecycle callbacks
// fired when the subscriber count transitions 0→1 (onFirst) or 1→0 (onLast).
type topic struct {
	mu      sync.Mutex
	subs    map[int]chan Message
	nextID  int
	onFirst func() // called (outside the lock) when subs goes 0 → 1
	onLast  func() // called (outside the lock) when subs goes 1 → 0
}

func newTopic() *topic { return &topic{subs: map[int]chan Message{}} }

func (t *topic) publish(m Message) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for id, ch := range t.subs {
		select {
		case ch <- m:
		default:
			close(ch)
			delete(t.subs, id)
		}
	}
}

func (t *topic) subscribe() (<-chan Message, func()) {
	t.mu.Lock()
	ch := make(chan Message, subBuffer)
	id := t.nextID
	t.nextID++
	t.subs[id] = ch
	first := len(t.subs) == 1
	onFirst := t.onFirst
	t.mu.Unlock()

	// Fire outside the lock to avoid deadlocks if the callback re-enters.
	if first && onFirst != nil {
		onFirst()
	}

	return ch, func() {
		t.mu.Lock()
		if c, ok := t.subs[id]; ok {
			delete(t.subs, id)
			close(c)
		}
		last := len(t.subs) == 0
		onLast := t.onLast
		t.mu.Unlock()

		if last && onLast != nil {
			onLast()
		}
	}
}

// SetMetricsLifecycle wires subscriber-lifecycle hooks onto the metrics topic.
// onFirst is called when the first live-updates subscriber connects, triggering
// an immediate poll; onLast is called when the last subscriber disconnects so
// the poller can idle until someone is watching again.
// Must be called before any client connects (typically in main, after wiring
// the hub and poller).
func (h *Hub) SetMetricsLifecycle(onFirst, onLast func()) {
	h.metrics.onFirst = onFirst
	h.metrics.onLast = onLast
}
