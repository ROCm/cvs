package ws_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/go-chi/chi/v5"
	nethttptest "net/http/httptest"

	"github.com/ROCm/cvs/api/internal/testexec"
	"github.com/ROCm/cvs/api/internal/transport/ws"
)

// wsEvents bridges the executor to the hub (mirrors cmd/server wiring).
type wsEvents struct{ hub *ws.Hub }

func (e wsEvents) Log(id, line string)                     { e.hub.PublishLog(id, line) }
func (e wsEvents) Status(id string, ex testexec.Execution) { e.hub.PublishStatus(id, ex) }
func (e wsEvents) Complete(ex testexec.Execution)          { e.hub.PublishCompletion(ex.ID, ex) }

type execSnapshots struct{ store testexec.ExecutionStore }

func (s execSnapshots) Snapshot(id string) (ws.ExecutionSnapshot, bool) {
	ex, ok := s.store.Get(id)
	if !ok {
		return ws.ExecutionSnapshot{}, false
	}
	logs := ""
	if ex.LogPath != "" {
		if b, err := os.ReadFile(ex.LogPath); err == nil {
			logs = string(b)
		}
	}
	return ws.ExecutionSnapshot{Terminal: ex.Status.Terminal(), Logs: logs, Status: ex}, true
}

// TestLiveExecutionStreamEndToEnd runs the fake runner and asserts a WS client
// receives streamed log lines followed by a terminal status frame.
func TestLiveExecutionStreamEndToEnd(t *testing.T) {
	dir := t.TempDir()
	store, err := testexec.NewFileExecutionStore(filepath.Join(dir, "exec.json"))
	if err != nil {
		t.Fatal(err)
	}
	hub := ws.NewHub()
	exec := testexec.NewExecutor(store, testexec.FakeRunner{Delay: 50 * time.Millisecond}, wsEvents{hub: hub}, 1, 4, nil)
	defer exec.Shutdown()

	r := chi.NewRouter()
	r.Route("/ws", func(wr chi.Router) {
		ws.NewHandler(hub, execSnapshots{store: store}, nil).Routes(wr)
	})
	srv := nethttptest.NewServer(r)
	defer srv.Close()

	const id = "run-e2e"
	logPath := filepath.Join(dir, id, "output.log")
	if err := store.Save(testexec.Execution{ID: id, Suite: "rccl_perf", Status: testexec.StatusQueued, LogPath: logPath, CreatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	if err := exec.Submit(testexec.Job{ID: id, Suite: "rccl_perf", LogPath: logPath}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws/executions/" + id
	c, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer c.Close(websocket.StatusNormalClosure, "")

	var gotLog bool
	var finalStatus string
	for {
		var msg ws.Message
		if err := wsjson.Read(ctx, c, &msg); err != nil {
			break // server seals the connection on terminal status
		}
		switch msg.Type {
		case "log":
			gotLog = true
		case "status":
			if d, ok := msg.Data.(map[string]any); ok {
				if s, ok := d["status"].(string); ok {
					finalStatus = s
				}
			}
		}
	}

	if !gotLog {
		t.Error("expected at least one streamed log frame")
	}
	if finalStatus != string(testexec.StatusPassed) {
		t.Fatalf("final status = %q, want passed", finalStatus)
	}
}
