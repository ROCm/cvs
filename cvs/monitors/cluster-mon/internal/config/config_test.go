package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalizeSSHKeyPath(t *testing.T) {
	if got := NormalizeSSHKeyPath("/home/alice/.ssh/id_rsa"); got != "~/.ssh/id_rsa" {
		t.Fatalf("got %s", got)
	}
	if got := NormalizeSSHKeyPath("/root/.ssh/cluster_id_ed25519"); got != "~/.ssh/cluster_id_ed25519" {
		t.Fatalf("got %s", got)
	}
	if got := NormalizeSSHKeyPath("~/.ssh/id_ed25519"); got != "~/.ssh/id_ed25519" {
		t.Fatalf("got %s", got)
	}
	t.Setenv("HOME", "/home/ichristo")
	if got := ExpandPath("~/.ssh/id_ed25519_tw"); got != "/home/ichristo/.ssh/id_ed25519_tw" {
		t.Fatalf("expand got %s", got)
	}
	jump := "/home/user@example.com/.ssh/cluster_key"
	if got := NormalizeSSHKeyPath(jump); got != jump {
		t.Fatalf("jump path rewritten: %s", got)
	}
}

func TestLoadSaveRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "nodes.txt"), []byte("node-a\n# comment\nnode-b\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	yaml := "cluster:\n  ssh:\n    username: tester\n    key_file: /root/.ssh/id_rsa\n    timeout: 15\n"
	if err := os.WriteFile(filepath.Join(dir, "cluster.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SSH.Username != "tester" {
		t.Fatalf("username=%s", cfg.SSH.Username)
	}
	if len(cfg.Nodes) != 2 {
		t.Fatalf("nodes=%v", cfg.Nodes)
	}
	cfg.SSH.Password = "secret"
	if err := SaveYAML(dir, cfg); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "cluster.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "secret") {
		t.Fatal("password persisted")
	}
}

func TestLoadLegacyCORSOrigins(t *testing.T) {
	dir := t.TempDir()
	yaml := "cluster:\n  cors_origins:\n    - http://monitor.example:8005\n"
	if err := os.WriteFile(filepath.Join(dir, "cluster.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.HTTP.CORSOrigins) != 1 || cfg.HTTP.CORSOrigins[0] != "http://monitor.example:8005" {
		t.Fatalf("legacy cors=%v", cfg.HTTP.CORSOrigins)
	}
}

func TestLoadHTTPCorsWinsOverLegacy(t *testing.T) {
	dir := t.TempDir()
	yaml := "cluster:\n  cors_origins:\n    - http://old.example\n  http:\n    cors_origins:\n      - http://new.example\n"
	if err := os.WriteFile(filepath.Join(dir, "cluster.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.HTTP.CORSOrigins) != 1 || cfg.HTTP.CORSOrigins[0] != "http://new.example" {
		t.Fatalf("http cors=%v", cfg.HTTP.CORSOrigins)
	}
}
