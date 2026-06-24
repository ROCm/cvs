package testexec

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

// Job is the unit of work handed to a Runner.
type Job struct {
	ID          string
	Suite       string
	ClusterFile string
	ConfigFile  string // may be empty for suites without a config
	LogPath     string
}

// Runner executes a job, streaming combined output to out, and reports the
// process exit code. A non-nil error means the job could not be run at all
// (spawn/setup failure -> StatusError); a non-zero exit code with a nil error
// means the test ran and failed.
type Runner interface {
	Run(ctx context.Context, job Job, out io.Writer) (exitCode int, err error)
}

// Events receives lifecycle notifications for live streaming (WebSocket). All
// methods must be safe for concurrent use and non-blocking.
type Events interface {
	// Log is called once per complete output line.
	Log(execID, line string)
	// Status is called on each lifecycle transition (running, terminal).
	Status(execID string, ex Execution)
	// Complete is called once when the execution reaches a terminal status.
	Complete(ex Execution)
}

// nopEvents is the default when no Events sink is wired.
type nopEvents struct{}

func (nopEvents) Log(string, string)       {}
func (nopEvents) Status(string, Execution) {}
func (nopEvents) Complete(Execution)       {}

// Executor runs jobs on a bounded worker pool, persisting each lifecycle
// transition to the ExecutionStore.
type Executor struct {
	store  ExecutionStore
	runner Runner
	events Events
	logger *slog.Logger
	jobs   chan Job
	ctx    context.Context
	cancel context.CancelFunc
}

// NewExecutor starts `workers` goroutines draining a queue of `queueSize`. The
// caller submits jobs after persisting the corresponding queued Execution.
// events may be nil to disable live streaming.
func NewExecutor(s ExecutionStore, runner Runner, events Events, workers, queueSize int, logger *slog.Logger) *Executor {
	if logger == nil {
		logger = slog.Default()
	}
	if events == nil {
		events = nopEvents{}
	}
	if workers <= 0 {
		workers = 2
	}
	if queueSize <= 0 {
		queueSize = 1024
	}
	ctx, cancel := context.WithCancel(context.Background())
	e := &Executor{
		store:  s,
		runner: runner,
		events: events,
		logger: logger,
		jobs:   make(chan Job, queueSize),
		ctx:    ctx,
		cancel: cancel,
	}
	for i := 0; i < workers; i++ {
		go e.worker()
	}
	return e
}

// Submit enqueues a job. It returns an error only if the queue is full.
func (e *Executor) Submit(j Job) error {
	select {
	case e.jobs <- j:
		return nil
	default:
		return fmt.Errorf("execution queue is full")
	}
}

// Shutdown stops accepting and signals running jobs to cancel.
func (e *Executor) Shutdown() { e.cancel() }

func (e *Executor) worker() {
	for {
		select {
		case <-e.ctx.Done():
			return
		case j := <-e.jobs:
			e.run(j)
		}
	}
}

func (e *Executor) run(j Job) {
	ex, ok := e.store.Get(j.ID)
	if !ok {
		e.logger.Error("execution_missing", "id", j.ID)
		return
	}

	now := time.Now().UTC()
	ex.Status = StatusRunning
	ex.StartedAt = &now
	if err := e.store.Save(ex); err != nil {
		e.logger.Error("execution_save_failed", "id", j.ID, "err", err)
	}
	e.events.Status(j.ID, ex)

	code, runErr := e.execute(j)

	fin := time.Now().UTC()
	ex, _ = e.store.Get(j.ID)
	ex.FinishedAt = &fin
	switch {
	case runErr != nil:
		ex.Status = StatusError
		ex.Error = runErr.Error()
	default:
		ex.ExitCode = &code
		if code == 0 {
			ex.Status = StatusPassed
		} else {
			ex.Status = StatusFailed
		}
	}
	if err := e.store.Save(ex); err != nil {
		e.logger.Error("execution_save_failed", "id", j.ID, "err", err)
	}
	e.events.Complete(ex)
	e.logger.Info("execution_finished", "id", j.ID, "status", ex.Status)
}

// execute opens the log file and runs the job, returning the exit code. Output
// is tee'd to the log file and the event sink (one event per complete line).
func (e *Executor) execute(j Job) (int, error) {
	emitter := &lineEmitter{emit: func(line string) { e.events.Log(j.ID, line) }}
	defer emitter.flush()

	if j.LogPath == "" {
		return e.runner.Run(e.ctx, j, emitter)
	}
	if err := os.MkdirAll(filepath.Dir(j.LogPath), 0o755); err != nil {
		return 0, fmt.Errorf("create log dir: %w", err)
	}
	f, err := os.Create(j.LogPath)
	if err != nil {
		return 0, fmt.Errorf("create log file: %w", err)
	}
	defer f.Close()
	return e.runner.Run(e.ctx, j, io.MultiWriter(f, emitter))
}

// lineEmitter splits a byte stream into complete lines and forwards each to a
// callback (used to stream log lines to the event sink).
type lineEmitter struct {
	buf  []byte
	emit func(string)
}

func (w *lineEmitter) Write(p []byte) (int, error) {
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		w.emit(string(w.buf[:i]))
		w.buf = w.buf[i+1:]
	}
	return len(p), nil
}

// flush emits any trailing partial line (output not newline-terminated).
func (w *lineEmitter) flush() {
	if len(w.buf) > 0 {
		w.emit(string(w.buf))
		w.buf = nil
	}
}
