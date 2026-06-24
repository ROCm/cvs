package clustermon

import (
	"context"
	"strings"
	"testing"

	"github.com/ROCm/cvs/api/pkg/pssh"
)

func TestValidateGrepCommand(t *testing.T) {
	cases := []struct {
		name string
		cmd  string
		ok   bool
	}{
		{"simple grep", "grep -i error", true},
		{"egrep combined flags", "egrep -iE oom", true},
		// A pipe inside the pattern is split naively (parity with Python), so the
		// second half is no longer a grep segment -> rejected.
		{"pipe inside pattern rejected", "egrep -iE 'oom|mce'", false},
		{"piped grep chain", "grep -i error | grep -v warn", true},
		{"flag with number -A5 base", "grep -A5 panic", true},
		{"flag with arg style -A=5", "grep -A=5 panic", true},
		{"empty", "   ", false},
		{"semicolon", "grep -i error; rm -rf /", false},
		{"command substitution", "grep -i $(whoami)", false},
		{"backtick", "grep -i `id`", false},
		{"redirect", "grep -i error > /tmp/x", false},
		{"forbidden keyword cat", "grep -i error | cat", false},
		{"not grep segment", "awk '{print}'", false},
		{"invalid flag", "grep -z error", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateGrepCommand(c.cmd)
			if c.ok && err != nil {
				t.Fatalf("expected valid, got error: %v", err)
			}
			if !c.ok && err == nil {
				t.Fatalf("expected invalid, got nil error")
			}
		})
	}
}

func TestSearch_RejectsInvalidGrep(t *testing.T) {
	svc := NewLogsService(func() *pssh.Pool { return nil }, nil)
	_, err := svc.Search(context.Background(), "rm -rf /")
	if err == nil {
		t.Fatal("expected rejection")
	}
	var bad *invalidGrepError
	if !strings.Contains(err.Error(), "forbidden") && !asInvalid(err, &bad) {
		t.Fatalf("expected invalidGrepError, got %v", err)
	}
}

func asInvalid(err error, target **invalidGrepError) bool {
	e, ok := err.(*invalidGrepError)
	if ok {
		*target = e
	}
	return ok
}

