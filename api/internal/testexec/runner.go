package testexec

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// FakeRunner simulates a test run for S3: it emits a few log lines with small
// delays and exits 0. It lets the execute -> poll -> terminal lifecycle be
// demoed and tested without real nodes or the Python CLI (that arrives in S5).
type FakeRunner struct {
	// Delay between emitted lines (kept short; tests may set it to 0).
	Delay time.Duration
}

// Run writes a deterministic transcript and returns exit code 0.
func (r FakeRunner) Run(ctx context.Context, job Job, out io.Writer) (int, error) {
	lines := []string{
		fmt.Sprintf("[cvs] (fake runner) starting suite %q", job.Suite),
		fmt.Sprintf("[cvs] cluster_file=%s", job.ClusterFile),
		fmt.Sprintf("[cvs] config_file=%s", job.ConfigFile),
		"[cvs] collecting tests ...",
		"[cvs] running ...",
		"[cvs] 1 passed",
		"[cvs] done (exit 0)",
	}
	for _, l := range lines {
		select {
		case <-ctx.Done():
			fmt.Fprintln(out, "[cvs] cancelled")
			return 0, ctx.Err()
		default:
		}
		fmt.Fprintln(out, l)
		if r.Delay > 0 {
			timer := time.NewTimer(r.Delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				fmt.Fprintln(out, "[cvs] cancelled")
				return 0, ctx.Err()
			case <-timer.C:
			}
		}
	}
	return 0, nil
}

// CLIRunner runs the real `cvs run <suite> --cluster_file <f> --config_file <f>`
// (S5). The Python CLI owns its own SSH; no Go SSH is involved. It is wired
// behind the Runner interface now so S5 only flips which runner is constructed.
type CLIRunner struct {
	Bin     string
	Timeout time.Duration
	Logger  *slog.Logger
}

// NewCLIRunner returns a runner that shells out to the cvs CLI.
func NewCLIRunner(bin string) *CLIRunner {
	if bin == "" {
		bin = "cvs"
	}
	return &CLIRunner{Bin: bin, Timeout: 6 * time.Hour, Logger: slog.Default()}
}

// Run spawns the CLI, streaming combined stdout/stderr to out, and maps the
// process exit code. The exact command is logged (app log) and written as a
// header line into out (so it appears in the UI live log + persisted output.log).
func (r *CLIRunner) Run(ctx context.Context, job Job, out io.Writer) (int, error) {
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}
	args := []string{"run", job.Suite, "--cluster_file", job.ClusterFile}
	if job.ConfigFile != "" {
		args = append(args, "--config_file", job.ConfigFile)
	}
	// Always generate the pytest HTML report + per-test logs alongside the
	// streamed output, in the per-execution dir (derived from the log path).
	// CVS's HtmlReportManager (conftest) writes report.html, a <suite>_html/
	// sidecar of per-test logs, and a zip bundle; all served via the artifacts
	// endpoint and surfaced as a report link in the execution history.
	if dir := filepath.Dir(job.LogPath); job.LogPath != "" && dir != "." {
		args = append(args,
			"--html", filepath.Join(dir, "report.html"),
			"--self-contained-html",
			"--log-file", filepath.Join(dir, "pytest.log"),
		)
	}

	cmdline := r.Bin + " " + strings.Join(args, " ")
	if r.Logger != nil {
		r.Logger.Info("cvs_run_exec", "id", job.ID, "suite", job.Suite, "command", cmdline)
	}
	fmt.Fprintf(out, "[cvs] $ %s\n", cmdline)

	cmd := exec.CommandContext(ctx, r.Bin, args...)
	cmd.Stdout = out
	cmd.Stderr = out
	err := cmd.Run()
	if err == nil {
		return 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), nil // ran, non-zero exit -> failed
	}
	return 0, err // could not run -> error
}
