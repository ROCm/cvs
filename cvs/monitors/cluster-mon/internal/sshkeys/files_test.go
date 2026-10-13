package sshkeys

import (
	"os"
	"testing"
)

func TestEnsureDirIsPrivate(t *testing.T) {
	dir := t.TempDir()
	prev := sshDirOverride
	sshDirOverride = dir
	t.Cleanup(func() { sshDirOverride = prev })
	if err := os.Chmod(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := EnsureDir(); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o700 {
		t.Fatalf("perm %o", st.Mode().Perm())
	}
}
