package ws

import (
	"testing"
	"time"
)

func drain(ch <-chan Message) []Message {
	var out []Message
	for {
		select {
		case m, ok := <-ch:
			if !ok {
				return out
			}
			out = append(out, m)
		default:
			return out
		}
	}
}

func TestSubscribeReceivesHistoryThenLive(t *testing.T) {
	h := NewHub()
	h.PublishLog("e1", "line-1")
	h.PublishLog("e1", "line-2")

	ch, unsub := h.SubscribeExecution("e1")
	defer unsub()

	// History is delivered immediately on subscribe.
	got := drain(ch)
	if len(got) != 2 || got[0].Data.(map[string]string)["line"] != "line-1" {
		t.Fatalf("history = %#v, want 2 backlog lines", got)
	}

	// Subsequent publishes stream live.
	h.PublishLog("e1", "line-3")
	select {
	case m := <-ch:
		if m.Data.(map[string]string)["line"] != "line-3" {
			t.Fatalf("live line = %#v", m)
		}
	case <-time.After(time.Second):
		t.Fatal("did not receive live line")
	}
}

func TestCompletionSealsSessionAndNotifies(t *testing.T) {
	h := NewHub()
	notif, unsub := h.SubscribeNotifications()
	defer unsub()

	ch, _ := h.SubscribeExecution("e2")
	h.PublishLog("e2", "working")
	h.PublishCompletion("e2", map[string]string{"status": "passed"})

	// Execution channel gets the final status frame then closes.
	var sawStatus, closed bool
	for {
		m, ok := <-ch
		if !ok {
			closed = true
			break
		}
		if m.Type == "status" {
			sawStatus = true
		}
	}
	if !sawStatus || !closed {
		t.Fatalf("exec stream: sawStatus=%v closed=%v", sawStatus, closed)
	}

	// Notifications stream gets a completion frame.
	select {
	case m := <-notif:
		if m.Type != "completion" || m.ExecutionID != "e2" {
			t.Fatalf("notification = %#v", m)
		}
	case <-time.After(time.Second):
		t.Fatal("no completion notification")
	}

	// Session was dropped: a late subscriber gets a fresh session with no
	// replayed history (the HTTP handler serves these from the persisted log).
	late, unsubLate := h.SubscribeExecution("e2")
	defer unsubLate()
	if got := drain(late); len(got) != 0 {
		t.Fatalf("late subscriber replayed stale history: %#v", got)
	}
}
