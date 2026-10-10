package sshkeys

import (
	"os"
	"path/filepath"
	"testing"
)

func withSSHDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	prev := sshDirOverride
	sshDirOverride = dir
	t.Cleanup(func() { sshDirOverride = prev })
	return dir
}

func keyBody() []byte {
	return append(append(pemBegin("OPENSSH"), '\n'), []byte("AAAA\n")...)
}

func TestSaveForcesModeOnExistingKey(t *testing.T) {
	dir := withSSHDir(t)
	path := filepath.Join(dir, "id_test")
	body := keyBody()
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Save("id_test", body); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("perm %o", st.Mode().Perm())
	}
}

func TestSaveNewKeyIsPrivate(t *testing.T) {
	dir := withSSHDir(t)
	body := keyBody()
	if err := Save("id_new", body); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(filepath.Join(dir, "id_new"))
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("perm %o", st.Mode().Perm())
	}
}

func TestSaveKnownHostsStaysWorldReadable(t *testing.T) {
	dir := withSSHDir(t)
	path := filepath.Join(dir, "known_hosts")
	if err := os.WriteFile(path, []byte("old\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Save("known_hosts", []byte("host key\n")); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o644 {
		t.Fatalf("perm %o", st.Mode().Perm())
	}
}

func TestSaveRejectsSymlink(t *testing.T) {
	dir := withSSHDir(t)
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "id_link")); err != nil {
		t.Fatal(err)
	}
	if err := Save("id_link", keyBody()); err == nil {
		t.Fatal("expected symlink rejection")
	}
	b, err := os.ReadFile(outside)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "secret" {
		t.Fatalf("symlink target changed: %q", b)
	}
}
