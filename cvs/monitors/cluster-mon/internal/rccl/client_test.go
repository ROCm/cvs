package rccl

import (
	"errors"
	"io"
	"net"
	"testing"
	"time"
)

func pipePair(t *testing.T) (net.Conn, net.Conn) {
	t.Helper()
	srv, cli := net.Pipe()
	t.Cleanup(func() {
		_ = srv.Close()
		_ = cli.Close()
	})
	return srv, cli
}

func consumeCommand(conn net.Conn) {
	buf := make([]byte, 64)
	_, _ = conn.Read(buf)
}

func TestVerboseStatusEOF(t *testing.T) {
	srv, cli := pipePair(t)
	go func() {
		consumeCommand(srv)
		_, _ = srv.Write([]byte("ok-body"))
		_ = srv.Close()
	}()
	got, err := NewClient(cli).VerboseStatus(2*time.Second, 64)
	if err != nil || got != "ok-body" {
		t.Fatalf("got %q err %v", got, err)
	}
}

func TestVerboseStatusExactLimit(t *testing.T) {
	srv, cli := pipePair(t)
	body := make([]byte, 8)
	for i := range body {
		body[i] = 'a'
	}
	go func() {
		consumeCommand(srv)
		_, _ = srv.Write(body)
		_ = srv.Close()
	}()
	got, err := NewClient(cli).VerboseStatus(2*time.Second, 8)
	if err != nil || len(got) != 8 {
		t.Fatalf("len=%d err=%v", len(got), err)
	}
}

func TestVerboseStatusOverLimit(t *testing.T) {
	srv, cli := pipePair(t)
	go func() {
		consumeCommand(srv)
		_, _ = srv.Write(make([]byte, 9))
		_ = srv.Close()
	}()
	got, err := NewClient(cli).VerboseStatus(2*time.Second, 8)
	if !errors.Is(err, ErrResponseTooLarge) || got != "" {
		t.Fatalf("got %q err %v", got, err)
	}
	if classifyRASErr(err) != rasProtocol {
		t.Fatalf("kind %v", classifyRASErr(err))
	}
}

func TestVerboseStatusMultiBurst(t *testing.T) {
	srv, cli := pipePair(t)
	go func() {
		consumeCommand(srv)
		_, _ = srv.Write([]byte("aaa"))
		time.Sleep(30 * time.Millisecond)
		_, _ = srv.Write([]byte("bbb"))
		_ = srv.Close()
	}()
	got, err := NewClient(cli).VerboseStatus(2*time.Second, 64)
	if err != nil || got != "aaabbb" {
		t.Fatalf("got %q err %v", got, err)
	}
}

func TestVerboseStatusTimeoutAfterPartial(t *testing.T) {
	srv, cli := pipePair(t)
	go func() {
		consumeCommand(srv)
		_, _ = srv.Write([]byte("partial"))
	}()
	got, err := NewClient(cli).VerboseStatus(50*time.Millisecond, 64)
	if err == nil || got != "" {
		t.Fatalf("partial success got %q err %v", got, err)
	}
	if errors.Is(err, io.EOF) {
		t.Fatal("timeout reported as EOF")
	}
}

func TestVerboseStatusTimeoutBeforeData(t *testing.T) {
	srv, cli := pipePair(t)
	go func() {
		consumeCommand(srv)
	}()
	got, err := NewClient(cli).VerboseStatus(50*time.Millisecond, 64)
	if err == nil || got != "" {
		t.Fatalf("got %q err %v", got, err)
	}
}
