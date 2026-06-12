package ssh

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"

	gossh "golang.org/x/crypto/ssh"
)

func writeTestKey(t *testing.T) string {
	t.Helper()
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}
	block, err := gossh.MarshalPrivateKey(priv, "")
	if err != nil {
		t.Fatalf("marshal key: %v", err)
	}
	path := filepath.Join(t.TempDir(), "id_ed25519")
	if err := os.WriteFile(path, pem.EncodeToMemory(block), 0o600); err != nil {
		t.Fatalf("write key: %v", err)
	}
	return path
}

func TestAuthMethods_Password(t *testing.T) {
	auth, err := authMethods("u", "", "secret")
	if err != nil {
		t.Fatalf("password auth: %v", err)
	}
	if len(auth) != 1 {
		t.Fatalf("want 1 auth method, got %d", len(auth))
	}
}

func TestAuthMethods_Key(t *testing.T) {
	path := writeTestKey(t)
	auth, err := authMethods("u", path, "")
	if err != nil {
		t.Fatalf("key auth: %v", err)
	}
	if len(auth) != 1 {
		t.Fatalf("want 1 auth method, got %d", len(auth))
	}
}

func TestAuthMethods_None(t *testing.T) {
	if _, err := authMethods("u", "", ""); err == nil {
		t.Fatal("expected error when no credentials provided")
	}
}

func TestAuthMethods_BadKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad")
	_ = os.WriteFile(path, []byte("not a key"), 0o600)
	if _, err := authMethods("u", path, ""); err == nil {
		t.Fatal("expected parse error for invalid key")
	}
}

func TestWithPort(t *testing.T) {
	cases := map[string]string{
		"node1":         "node1:22",
		"10.0.0.1":      "10.0.0.1:22",
		"10.0.0.1:2222": "10.0.0.1:2222",
		"[::1]:22":      "[::1]:22",
	}
	for in, want := range cases {
		if got := withPort(in, 22); got != want {
			t.Errorf("withPort(%q) = %q, want %q", in, got, want)
		}
	}
}
