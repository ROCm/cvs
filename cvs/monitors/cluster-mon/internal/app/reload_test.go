package app

import (
	"testing"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
)

func TestClassifyReload(t *testing.T) {
	base := &config.Config{
		Nodes: []string{"a", "b"},
		SSH: config.SSHConfig{
			Username: "root",
			KeyFile:  "/root/.ssh/id_ed25519",
		},
	}
	same := *base
	same.Nodes = []string{"a", "b"}
	if g := classifyReload(base, &same, "", "", "", ""); g != reloadCfg {
		t.Fatalf("got %s", g)
	}

	nodes := *base
	nodes.Nodes = []string{"a", "b", "c"}
	if g := classifyReload(base, &nodes, "", "", "", ""); g != reloadNodes {
		t.Fatalf("got %s", g)
	}

	creds := *base
	creds.SSH.Username = "ubuntu"
	if g := classifyReload(base, &creds, "", "", "", ""); g != reloadCreds {
		t.Fatalf("got %s", g)
	}
	if g := classifyReload(base, base, "", "", "", "newjump"); g != reloadCfg {
		t.Fatalf("jump password ignored when jump disabled, got %s", g)
	}

	jump := *base
	jump.SSH.JumpHost.Enabled = true
	jump.SSH.JumpHost.Host = "bastion"
	if g := classifyReload(base, &jump, "", "", "", ""); g != reloadRebuild {
		t.Fatalf("got %s", g)
	}
}
