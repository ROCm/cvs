package cluster

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// GenerateParams are the inputs to the cluster_json generator, mirroring the
// `cvs generate cluster_json` CLI flags.
type GenerateParams struct {
	Hosts    []string // --hosts (comma-joined); a subset of the inventory
	Username string   // --username
	KeyFile  string   // --key_file (absolute path of the uploaded key)
	Output   string   // --output_json_file
	HeadNode string   // --head_node (optional)
}

// Generator renders a cluster_json file from a node subset.
type Generator interface {
	Generate(ctx context.Context, p GenerateParams) error
}

// CLIGenerator shells out to the authoritative `cvs generate cluster_json`,
// which is a pure Jinja2 template render (no SSH). We do not hand-roll the
// cluster JSON in Go so the format stays owned by the CLI.
type CLIGenerator struct {
	Bin     string // cvs executable (name on PATH or absolute path)
	Timeout time.Duration
}

// NewCLIGenerator returns a generator that runs `<bin> generate cluster_json`.
func NewCLIGenerator(bin string) *CLIGenerator {
	if bin == "" {
		bin = "cvs"
	}
	return &CLIGenerator{Bin: bin, Timeout: 30 * time.Second}
}

// Generate invokes the CLI to write the cluster_json to p.Output.
func (g *CLIGenerator) Generate(ctx context.Context, p GenerateParams) error {
	timeout := g.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := []string{
		"generate", "cluster_json",
		"--hosts", strings.Join(p.Hosts, ","),
		"--username", p.Username,
		"--key_file", p.KeyFile,
		"--output_json_file", p.Output,
	}
	if p.HeadNode != "" {
		args = append(args, "--head_node", p.HeadNode)
	}

	cmd := exec.CommandContext(ctx, g.Bin, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cvs generate cluster_json failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
