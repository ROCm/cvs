package ws

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestDropIsIdempotent(t *testing.T) {
	h := NewHub()
	c := &client{send: make(chan []byte, 8), done: make(chan struct{})}
	h.clients[c] = struct{}{}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.drop(c)
			h.enqueue(c, []byte("pong"))
		}()
	}
	wg.Wait()
	if h.Count() != 0 {
		t.Fatalf("clients=%d", h.Count())
	}
}

func TestBroadcastFullBufferDoesNotPanic(t *testing.T) {
	h := NewHub()
	srv := httptest.NewServer(http.HandlerFunc(h.ServeWS))
	defer srv.Close()
	url := "ws" + strings.TrimPrefix(srv.URL, "http")
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	time.Sleep(50 * time.Millisecond)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			h.Broadcast(map[string]int{"n": i})
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_ = conn.WriteMessage(websocket.TextMessage, []byte("ping"))
		}
	}()
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("broadcast or ping wedged")
	}
}
