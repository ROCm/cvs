// Package testexec implements the Test Execution tile: discovering CVS test
// suites and (in later slices) generating forms and running them via the CVS CLI.
//
// Suite discovery is delegated to the authoritative source — the CVS CLI's
// `cvs list --json` — rather than scanning config files. This keeps the suite
// list correct (it includes suites without configs, and extension-package
// suites) and avoids duplicating the CLI's pytest-based discovery logic.
package testexec

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"
	"time"
)

// Suite mirrors one entry from `cvs list --json`.
type Suite struct {
	Name       string `json:"name"`        // suite/test module name, e.g. "rccl_perf"
	ModulePath string `json:"module_path"` // e.g. "cvs.tests.rccl.rccl_perf"
	Module     string `json:"module"`      // parent module, e.g. "cvs.tests.rccl"
	Package    string `json:"package"`     // e.g. "cvs"
	Category   string `json:"category"`    // module leaf, e.g. "rccl"
}

// SuiteLister discovers the available CVS test suites.
type SuiteLister interface {
	List(ctx context.Context) ([]Suite, error)
}

// CLISuiteLister discovers suites by invoking the CVS CLI.
type CLISuiteLister struct {
	// Bin is the cvs executable (name on PATH or absolute path).
	Bin string
	// Timeout bounds the CLI invocation.
	Timeout time.Duration
}

// NewCLISuiteLister returns a lister that runs `<bin> list --json`.
func NewCLISuiteLister(bin string) *CLISuiteLister {
	if bin == "" {
		bin = "cvs"
	}
	return &CLISuiteLister{Bin: bin, Timeout: 30 * time.Second}
}

// List runs `cvs list --json` and flattens its output into suites.
func (l *CLISuiteLister) List(ctx context.Context) ([]Suite, error) {
	timeout := l.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, l.Bin, "list", "--json")
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("cvs list failed: %w: %s", err, string(ee.Stderr))
		}
		return nil, fmt.Errorf("cvs list failed: %w", err)
	}
	return parseSuites(out)
}

// parseSuites converts `cvs list --json` output — a nested
// {package: {test_name: module_path}} map — into a flat, sorted suite list,
// deriving the parent module and its leaf category from module_path.
func parseSuites(data []byte) ([]Suite, error) {
	var testMap map[string]map[string]string
	if err := json.Unmarshal(data, &testMap); err != nil {
		return nil, fmt.Errorf("parse cvs list output: %w", err)
	}

	suites := []Suite{}
	for pkg, tests := range testMap {
		for name, modulePath := range tests {
			module := modulePath
			if i := strings.LastIndex(modulePath, "."); i >= 0 {
				module = modulePath[:i]
			}
			category := module
			if i := strings.LastIndex(module, "."); i >= 0 {
				category = module[i+1:]
			}
			suites = append(suites, Suite{
				Name:       name,
				ModulePath: modulePath,
				Module:     module,
				Package:    pkg,
				Category:   category,
			})
		}
	}

	sort.Slice(suites, func(i, j int) bool {
		if suites[i].Module != suites[j].Module {
			return suites[i].Module < suites[j].Module
		}
		return suites[i].Name < suites[j].Name
	})
	return suites, nil
}
