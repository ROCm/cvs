// GPU Cluster Collector
// Reads a JSON request from stdin, SSHes into all nodes simultaneously,
// runs all commands in parallel per node, writes JSON results to stdout.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

// ---- Input schema ----

type Input struct {
	Hosts          []string          `json:"hosts"`
	SSHUser        string            `json:"ssh_user"`
	SSHKeyPath     string            `json:"ssh_key_path"`
	SSHPassword    string            `json:"ssh_password"`
	SSHPort        int               `json:"ssh_port"`
	Commands       map[string]string `json:"commands"`        // name -> shell command
	PerHostTimeout int               `json:"per_host_timeout_s"`
	GlobalTimeout  int               `json:"global_timeout_s"`
}

// ---- Output schema ----

type CmdResult struct {
	Status string `json:"status"` // "ok", "timeout", "ssh_error", "command_error"
	Raw    string `json:"raw"`
	Error  string `json:"error,omitempty"`
}

type HostResult map[string]CmdResult // cmd name -> result

type Output struct {
	Timestamp          string                `json:"timestamp"`
	Results            map[string]HostResult `json:"results"`
	Unreachable        []string              `json:"unreachable"`
	CollectionDurationMs int64               `json:"collection_duration_ms"`
}

// ---- Core logic ----

func runCommand(ctx context.Context, client *ssh.Client, cmdStr string, timeout time.Duration) CmdResult {
	session, err := client.NewSession()
	if err != nil {
		return CmdResult{Status: "ssh_error", Error: fmt.Sprintf("new session: %v", err)}
	}
	defer session.Close()

	var stdoutBuf, stderrBuf bytes.Buffer
	session.Stdout = &stdoutBuf
	session.Stderr = &stderrBuf

	type done struct {
		err error
	}
	ch := make(chan done, 1)

	go func() {
		ch <- done{session.Run(cmdStr)}
	}()

	select {
	case d := <-ch:
		raw := stdoutBuf.String()
		if d.err != nil {
			if raw != "" {
				// amd-smi sometimes exits non-zero but still emits valid JSON
				return CmdResult{Status: "ok", Raw: raw}
			}
			return CmdResult{
				Status: "command_error",
				Raw:    raw,
				Error:  fmt.Sprintf("%v stderr: %s", d.err, stderrBuf.String()),
			}
		}
		return CmdResult{Status: "ok", Raw: raw}

	case <-time.After(timeout):
		session.Signal(ssh.SIGTERM)
		return CmdResult{Status: "timeout", Error: fmt.Sprintf("timed out after %v", timeout)}

	case <-ctx.Done():
		session.Signal(ssh.SIGTERM)
		return CmdResult{Status: "timeout", Error: "global timeout"}
	}
}

func collectNode(
	ctx context.Context,
	host string,
	port int,
	authMethods []ssh.AuthMethod,
	user string,
	commands map[string]string,
	perHostTimeout time.Duration,
) (HostResult, bool) {

	cfg := &ssh.ClientConfig{
		User:            user,
		Auth:            authMethods,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), //nolint:gosec
		Timeout:         15 * time.Second,
	}

	addr := fmt.Sprintf("%s:%d", host, port)
	client, err := ssh.Dial("tcp", addr, cfg)
	if err != nil {
		return nil, false // unreachable
	}
	defer client.Close()

	hr := make(HostResult)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for name, cmd := range commands {
		wg.Add(1)
		go func(n, c string) {
			defer wg.Done()
			result := runCommand(ctx, client, c, perHostTimeout)
			mu.Lock()
			hr[n] = result
			mu.Unlock()
		}(name, cmd)
	}

	wg.Wait()
	return hr, true
}

func main() {
	start := time.Now()

	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("read stdin: %v", err)
	}

	var inp Input
	if err := json.Unmarshal(data, &inp); err != nil {
		log.Fatalf("parse input: %v", err)
	}

	// Defaults
	if inp.SSHPort == 0 {
		inp.SSHPort = 22
	}
	if inp.PerHostTimeout == 0 {
		inp.PerHostTimeout = 90
	}
	if inp.GlobalTimeout == 0 {
		inp.GlobalTimeout = 110
	}

	// Build auth methods
	var authMethods []ssh.AuthMethod
	if inp.SSHPassword != "" {
		authMethods = append(authMethods, ssh.Password(inp.SSHPassword))
	}
	if inp.SSHKeyPath != "" {
		keyData, err := os.ReadFile(inp.SSHKeyPath)
		if err != nil {
			log.Fatalf("read key file %s: %v", inp.SSHKeyPath, err)
		}
		signer, err := ssh.ParsePrivateKey(keyData)
		if err != nil {
			log.Fatalf("parse private key: %v", err)
		}
		authMethods = append(authMethods, ssh.PublicKeys(signer))
	}
	if len(authMethods) == 0 {
		log.Fatal("no auth method: provide ssh_key_path or ssh_password")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(inp.GlobalTimeout)*time.Second)
	defer cancel()

	perHostTimeout := time.Duration(inp.PerHostTimeout) * time.Second

	results := make(map[string]HostResult)
	var unreachable []string
	var mu sync.Mutex
	var wg sync.WaitGroup

	log.Printf("Starting collection: %d hosts, %d commands, timeout %ds",
		len(inp.Hosts), len(inp.Commands), inp.GlobalTimeout)

	for _, host := range inp.Hosts {
		wg.Add(1)
		go func(h string) {
			defer wg.Done()
			hr, reachable := collectNode(ctx, h, inp.SSHPort, authMethods, inp.SSHUser, inp.Commands, perHostTimeout)
			mu.Lock()
			defer mu.Unlock()
			if reachable {
				results[h] = hr
			} else {
				unreachable = append(unreachable, h)
			}
		}(host)
	}

	wg.Wait()

	durationMs := time.Since(start).Milliseconds()
	log.Printf("Collection complete: %d succeeded, %d unreachable, took %dms",
		len(results), len(unreachable), durationMs)

	out := Output{
		Timestamp:            time.Now().UTC().Format(time.RFC3339),
		Results:              results,
		Unreachable:          unreachable,
		CollectionDurationMs: durationMs,
	}

	if unreachable == nil {
		out.Unreachable = []string{}
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(out); err != nil {
		log.Fatalf("encode output: %v", err)
	}
}
