package ws

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/go-chi/chi/v5"
)

// ExecutionSnapshot is the persisted state of an execution, used for the
// terminal/late-joiner fallback when no live session exists.
type ExecutionSnapshot struct {
	Terminal bool
	Logs     string
	// Status is the payload sent as a "status" frame's data (the execution).
	Status any
}

// SnapshotProvider resolves an execution's persisted state (decouples this
// package from the Test Execution tile).
type SnapshotProvider interface {
	Snapshot(execID string) (ExecutionSnapshot, bool)
}

// MetricsProvider supplies the latest cached Cluster Monitor snapshot, sent as
// the first frame to a newly connected /clustermon subscriber so it paints
// immediately instead of waiting for the next poll tick.
type MetricsProvider interface {
	LatestMetrics() (any, bool)
}

// Handler serves the WebSocket streams.
type Handler struct {
	hub     *Hub
	snaps   SnapshotProvider
	metrics MetricsProvider
	logger  *slog.Logger
}

// NewHandler constructs the WS handler.
func NewHandler(hub *Hub, snaps SnapshotProvider, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{hub: hub, snaps: snaps, logger: logger}
}

// SetMetricsProvider wires the latest-snapshot source for /clustermon. Optional;
// without it the stream still works but the first frame waits for the next poll.
func (h *Handler) SetMetricsProvider(m MetricsProvider) { h.metrics = m }

// Routes mounts the WS endpoints (expected under /ws).
func (h *Handler) Routes(r chi.Router) {
	r.Get("/executions/{id}", h.execution)
	r.Get("/notifications", h.notifications)
	r.Get("/clustermon", h.clusterMetrics)
}

// writeTimeout bounds a single frame write to a (possibly slow) client.
const writeTimeout = 10 * time.Second

func (h *Handler) execution(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	snap, ok := h.snaps.Snapshot(id)
	if !ok {
		http.Error(w, "execution not found", http.StatusNotFound)
		return
	}

	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return // Accept already wrote the error
	}
	defer c.Close(websocket.StatusNormalClosure, "")
	ctx := r.Context()

	// Already finished before connect: replay the persisted log + status, done.
	if snap.Terminal {
		h.sendFallback(ctx, c, id, snap)
		return
	}

	ch, unsub := h.hub.SubscribeExecution(id)
	defer unsub()

	// Re-check after subscribing to close the race where the run finished (and
	// its session was dropped) between the snapshot and the subscribe.
	if snap2, ok := h.snaps.Snapshot(id); ok && snap2.Terminal {
		h.hub.DropSession(id)
		h.sendFallback(ctx, c, id, snap2)
		return
	}

	// Current status first, then the history-then-live stream.
	h.write(ctx, c, Message{Type: "status", ExecutionID: id, Timestamp: time.Now().UTC(), Data: snap.Status})
	for m := range ch {
		if err := h.write(ctx, c, m); err != nil {
			return
		}
	}
}

// sendFallback replays a completed execution from persisted state.
func (h *Handler) sendFallback(ctx context.Context, c *websocket.Conn, id string, snap ExecutionSnapshot) {
	if snap.Logs != "" {
		_ = h.write(ctx, c, Message{
			Type: "log", ExecutionID: id, Timestamp: time.Now().UTC(),
			Data: map[string]string{"line": snap.Logs},
		})
	}
	_ = h.write(ctx, c, Message{
		Type: "status", ExecutionID: id, Timestamp: time.Now().UTC(), Data: snap.Status,
	})
}

// clusterMetrics streams fleet GPU snapshots: the cached latest immediately on
// connect, then each subsequent poll the server broadcasts.
func (h *Handler) clusterMetrics(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer c.Close(websocket.StatusNormalClosure, "")
	ctx := r.Context()

	ch, unsub := h.hub.SubscribeMetrics()
	defer unsub()

	if h.metrics != nil {
		if snap, ok := h.metrics.LatestMetrics(); ok {
			if err := h.write(ctx, c, Message{Type: "metrics", Timestamp: time.Now().UTC(), Data: snap}); err != nil {
				return
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		case m, ok := <-ch:
			if !ok {
				return
			}
			if err := h.write(ctx, c, m); err != nil {
				return
			}
		}
	}
}

func (h *Handler) notifications(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer c.Close(websocket.StatusNormalClosure, "")

	ch, unsub := h.hub.SubscribeNotifications()
	defer unsub()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case m, ok := <-ch:
			if !ok {
				return
			}
			if err := h.write(ctx, c, m); err != nil {
				return
			}
		}
	}
}

func (h *Handler) write(ctx context.Context, c *websocket.Conn, m Message) error {
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	return wsjson.Write(ctx, c, m)
}
