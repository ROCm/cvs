package testexec

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// writeFakeCvs creates an executable stub that emits the given stdout for
// `list --json`. Returns its path.
func writeFakeCvs(t *testing.T, stdout string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake shell script not supported on windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "cvs")
	script := "#!/bin/sh\ncat <<'JSON'\n" + stdout + "\nJSON\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCLISuiteLister_ParsesJSON(t *testing.T) {
	bin := writeFakeCvs(t, `{"cvs":{"rccl_perf":"cvs.tests.rccl.rccl_perf","rccl_regression":"cvs.tests.rccl.rccl_regression"}}`)

	suites, err := NewCLISuiteLister(bin).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(suites) != 2 {
		t.Fatalf("got %d suites, want 2", len(suites))
	}
	// Sorted by module then name.
	got := suites[0]
	if got.Name != "rccl_perf" || got.ModulePath != "cvs.tests.rccl.rccl_perf" ||
		got.Module != "cvs.tests.rccl" || got.Package != "cvs" || got.Category != "rccl" {
		t.Fatalf("unexpected derived suite: %+v", got)
	}
}

func TestCLISuiteLister_EmptySuites(t *testing.T) {
	bin := writeFakeCvs(t, `{}`)
	suites, err := NewCLISuiteLister(bin).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if suites == nil || len(suites) != 0 {
		t.Fatalf("want empty non-nil slice, got %#v", suites)
	}
}

func TestCLISuiteLister_BadBinary(t *testing.T) {
	_, err := NewCLISuiteLister(filepath.Join(t.TempDir(), "does-not-exist")).List(context.Background())
	if err == nil {
		t.Fatal("expected error for missing binary")
	}
}
