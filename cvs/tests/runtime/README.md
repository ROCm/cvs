# `cvs/tests/runtime/` — docker-mode lifecycle tests (P6)

Two pytest tests that bring CVS-runner containers up on every cluster node
(`prepare_runtime`) and tear them down (`cleanup_runtime`). They are skipped
in host mode, so they're invisible to existing host-mode users.

## Files

| File | Purpose |
|---|---|
| `prepare_runtime.py` | Single `test_prepare_runtime` function. Runs preflight, image staging, container start, arch detection, AGFHC tarball staging, `cvs run install_*` orchestration, and in-container smoke. Writes per-node JSON artifacts to `/tmp/cvs/prepare_runtime/<host>.json`. |
| `cleanup_runtime.py` | Single `test_cleanup_runtime` function. `docker rm -f cvs-runner` on every node, writes `/tmp/cvs/cleanup_runtime/cleanup.json`. |

## Usage

```bash
# bring containers up + install components
cvs run prepare_runtime --cluster_file cluster.json --config_file config.json

# ... run validation tests (transferbench_cvs, agfhc_cvs, rvs_cvs, ...) ...

# tear containers down
cvs run cleanup_runtime --cluster_file cluster.json --config_file config.json
```

`prepare_runtime` exits successfully when:

* All preflight checks pass (amdgpu live, `/dev/kfd`, `/dev/dri`, docker daemon).
* Image staged (pulled or loaded) on every node.
* Container started on every node.
* Cluster's GPU arch is uniform (single-arch invariant) and matches `runtime.expected_gfx_arch` if set.
* Each `runtime.installs` entry was attempted (success/failure recorded per-install in artifact; downstream tests skip cleanly on failed installs).

`prepare_runtime` hard-fails when:

* preflight fails on any node.
* image stage fails (pull error, load file missing).
* container fails to start.
* arch invariant is violated (single-arch mismatch or `expected_gfx_arch` mismatch).

It does NOT hard-fail on a per-install failure -- those are recorded in the
artifact and downstream tests skip cleanly. This means a missing AGFHC tarball
or a flaky RVS install doesn't block the rest of the test suite.

## Per-node artifact schema

`/tmp/cvs/prepare_runtime/<host>.json`:

```json
{
  "phase": "ready",
  "started_at": "2026-04-20T22:00:00Z",
  "finished_at": "2026-04-20T22:06:00Z",
  "cluster_file": "/path/to/cluster.json",
  "config_file": "/path/to/config.json",
  "image": "cvs-runner:7.13.0a-gfx942",
  "container_name": "cvs-runner",
  "gfx_arch": "gfx942",
  "checks": {
    "amdgpu_live": true,
    "dri_render_count": 8,
    "docker_present": true,
    "image_pulled": true,
    "container_running": true,
    "in_container_rocminfo": true,
    "rvs_version": "1.3.0",
    "agfhc_version": "agfhc version: 1.30.2",
    "transferbench_path": "/opt/INSTALL/TransferBench/TransferBench"
  },
  "installs": {
    "install_rvs":           {"exit": 0, "ok": true,  "stderr_tail": ""},
    "install_transferbench": {"exit": 0, "ok": true,  "stderr_tail": ""},
    "install_agfhc":         {"exit": 0, "ok": true,  "stderr_tail": ""}
  },
  "agfhc_staged": true,
  "agfhc_tarball_remote_path": "/tmp/cvs/agfhc.tar.bz2",
  "errors": []
}
```

When `runtime.agfhc_tarball: null`:

```json
{
  ...
  "agfhc_staged": false,
  "agfhc_skip_reason": "no agfhc_tarball configured",
  "installs": {
    "install_rvs":           {"exit": 0, "ok": true},
    "install_transferbench": {"exit": 0, "ok": true}
    // install_agfhc not attempted
  }
}
```

## Idempotency

`prepare_runtime` always does `docker rm -f cvs-runner` before `docker run -d`.
Re-running is safe and produces a fresh container + fresh artifact.

`cleanup_runtime` is idempotent too -- if no container is present, it still
writes a cleanup artifact noting the no-op.

## Test machine quirks

* Conductor SSH banner is automatically tolerated -- `prepare_runtime` parses
  CVS-emitted local artifacts, not raw SSH stdout.
* If `atnair` is not in the docker group on a node, every docker command runs
  via `sudo docker` (the test always uses sudo to be safe).
