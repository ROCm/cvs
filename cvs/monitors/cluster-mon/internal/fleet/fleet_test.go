package fleet

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	xssh "golang.org/x/crypto/ssh"
)

func testKeyPEM(t *testing.T) []byte {
	t.Helper()
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	block, err := xssh.MarshalPrivateKey(priv, "")
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(block)
}

func TestResolveJumpNodeKey(t *testing.T) {
	logger := slog.Default()
	pem := testKeyPEM(t)
	fetchFail := errors.New("sftp down")

	if _, err := resolveJumpNodeKey("", nil, nil, "", logger); err == nil {
		t.Fatal("missing key and password should fail")
	}
	got, err := resolveJumpNodeKey("", nil, nil, "secret", logger)
	if err != nil || got != nil {
		t.Fatalf("password only: %v %v", got, err)
	}
	if _, err := resolveJumpNodeKey("/k", nil, fetchFail, "", logger); err == nil {
		t.Fatal("fetch failure without password should fail")
	}
	if _, err := resolveJumpNodeKey("/k", nil, fetchFail, "secret", logger); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveJumpNodeKey("/k", []byte("not a key"), nil, "", logger); err == nil {
		t.Fatal("bad key without password should fail")
	}
	if _, err := resolveJumpNodeKey("/k", []byte("not a key"), nil, "secret", logger); err != nil {
		t.Fatal(err)
	}
	got, err = resolveJumpNodeKey("/k", pem, nil, "", logger)
	if err != nil || string(got) != string(pem) {
		t.Fatalf("valid key: %v %v", err, len(got))
	}
}

func TestJumpAuthMethods(t *testing.T) {
	logger := slog.Default()
	dir := t.TempDir()
	good := filepath.Join(dir, "good")
	if err := os.WriteFile(good, testKeyPEM(t), 0o600); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(dir, "bad")
	if err := os.WriteFile(bad, []byte("not a key"), 0o600); err != nil {
		t.Fatal(err)
	}
	missing := filepath.Join(dir, "missing")

	if _, err := jumpAuthMethods("", "", logger); err == nil {
		t.Fatal("expected no auth")
	}
	auth, err := jumpAuthMethods("", "secret", logger)
	if err != nil || len(auth) != 1 {
		t.Fatalf("password only: %v %d", err, len(auth))
	}
	auth, err = jumpAuthMethods(good, "", logger)
	if err != nil || len(auth) != 1 {
		t.Fatalf("key only: %v %d", err, len(auth))
	}
	auth, err = jumpAuthMethods(good, "secret", logger)
	if err != nil || len(auth) != 2 {
		t.Fatalf("key and password: %v %d", err, len(auth))
	}
	auth, err = jumpAuthMethods(missing, "secret", logger)
	if err != nil || len(auth) != 1 {
		t.Fatalf("missing key with password: %v %d", err, len(auth))
	}
	auth, err = jumpAuthMethods(bad, "secret", logger)
	if err != nil || len(auth) != 1 {
		t.Fatalf("bad key with password: %v %d", err, len(auth))
	}
	if _, err := jumpAuthMethods(bad, "", logger); err == nil {
		t.Fatal("bad key without password should fail")
	}
	if _, err := jumpAuthMethods(missing, "", logger); err == nil {
		t.Fatal("missing key without password should fail")
	}
}
