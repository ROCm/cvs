package app

import (
	"context"
	"strings"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/config"
)

type reloadKind string

const (
	reloadRebuild reloadKind = "rebuild"
	reloadCreds   reloadKind = "credentials"
	reloadNodes   reloadKind = "nodes"
	reloadCfg     reloadKind = "config"
)

func (a *App) Reload() (map[string]any, error) {
	cfg, err := config.Load(a.ConfigDir)
	if err != nil {
		return nil, err
	}
	ssh, jump := a.Passwords()
	old := a.Cfg()
	kind := classifyReload(old, cfg, a.appliedSSH, a.appliedJump, ssh, jump)

	a.mu.Lock()
	a.cfg = cfg
	a.mu.Unlock()

	if len(cfg.Nodes) == 0 {
		a.Fleet.Close()
		a.appliedSSH, a.appliedJump = ssh, jump
		return map[string]any{"success": true, "message": "Configuration reloaded (no nodes)", "nodes": 0, "reload": string(reloadCfg)}, nil
	}

	pool := a.Pool()
	if pool == nil {
		kind = reloadRebuild
	}

	switch kind {
	case reloadRebuild:
		if err := a.Fleet.Rebuild(cfg, ssh, jump); err != nil {
			return map[string]any{"success": false, "error": err.Error(), "reload": string(kind)}, err
		}
	case reloadCreds:
		user, key := nodeUserKey(cfg)
		pool.UpdateCredentials(user, key, nil)
		pool.SetPassword(ssh)
	case reloadNodes:
		added, _ := pool.Refresh(cfg.Nodes)
		if len(added) > 0 {
			ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
			pool.ProbeSubset(ctx, added)
			cancel()
		}
	default:
		if pool != nil {
			pool.TriggerReprobe()
		}
	}
	a.appliedSSH, a.appliedJump = ssh, jump
	return map[string]any{
		"success": true,
		"message": "Configuration reloaded",
		"nodes":   len(cfg.Nodes),
		"jump":    cfg.SSH.JumpHost.Enabled,
		"reload":  string(kind),
	}, nil
}

func classifyReload(old, new *config.Config, oldSSH, oldJump, newSSH, newJump string) reloadKind {
	if old == nil || new == nil {
		return reloadRebuild
	}
	if jumpIdentity(old, oldJump) != jumpIdentity(new, newJump) {
		return reloadRebuild
	}
	if nodeIdentity(old, oldSSH) != nodeIdentity(new, newSSH) {
		return reloadCreds
	}
	if !nodesEqual(old.Nodes, new.Nodes) {
		return reloadNodes
	}
	return reloadCfg
}

func jumpIdentity(cfg *config.Config, jumpPw string) string {
	j := cfg.SSH.JumpHost
	if !j.Enabled {
		return "direct"
	}
	return strings.Join([]string{
		j.Host, j.Username, j.KeyFile, j.NodeUsername, j.NodeKeyFile, jumpPw,
	}, "\x00")
}

func nodeIdentity(cfg *config.Config, sshPw string) string {
	user, key := nodeUserKey(cfg)
	return user + "\x00" + key + "\x00" + sshPw
}

func nodeUserKey(cfg *config.Config) (string, string) {
	if cfg.SSH.JumpHost.Enabled {
		user := cfg.SSH.JumpHost.NodeUsername
		if user == "" {
			user = cfg.SSH.Username
		}
		return user, ""
	}
	return cfg.SSH.Username, config.ExpandPath(cfg.SSH.KeyFile)
}

func nodesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
