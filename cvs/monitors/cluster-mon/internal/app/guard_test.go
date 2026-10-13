package app

import (
	"io"
	"log/slog"
	"testing"
)

func TestGuardContainsPanic(t *testing.T) {
	a := &App{
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
		results: map[string]CollectorResult{},
	}
	if ok := a.guard("metrics", true, func() { panic("boom") }); ok {
		t.Fatal("panicking collector reported success")
	}
	got := a.CollectorResults()["metrics"]
	if got.State != "error" || got.Error != "panic: boom" || !got.Critical {
		t.Fatalf("panic result=%+v", got)
	}
	if ok := a.guard("metrics", true, func() {}); !ok {
		t.Fatal("successful collector reported failure")
	}
}
