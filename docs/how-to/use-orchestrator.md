# Use the CVS orchestrator

`orch` is CVS's distributed execution interface. A suite calls one object to
run commands on its selected hosts; configuration decides whether those commands
run on the host operating system or in a long-lived container on each host.

```python
out = orch.exec("rocm-smi", timeout=30)
```

This keeps suite code focused on workload behavior instead of parallel host
execution, Docker invocation, and connection cleanup.

## How `orch` works

The orchestrator has three layers:

1. The suite calls `orch`.
2. The `orchestrator` cluster-file setting chooses the execution environment:
   - `baremetal` runs commands on the selected hosts.
   - `container` runs commands in the named container on the selected hosts.
3. Container mode delegates container creation, inspection, execution, and
   removal to the selected runtime.

In container mode, CVS starts one named, long-lived container on every selected
host. Commands then run through `docker exec`:

```text
Suite -> orch.exec("vllm serve ...")
      -> host transport to selected nodes
      -> docker exec <container-name> bash -c "vllm serve ..."
```

The container holds its environment, bind mounts, and workload processes for
the suite lifecycle.

## Backend and runtime selection

The cluster file chooses the orchestrator backend:

```json
"orchestrator": "baremetal"
```

or:

```json
"orchestrator": "container"
```

If omitted, CVS uses `baremetal`. Container mode selects its runtime under the
`container` block:

```json
"container": {
  "runtime": {
    "name": "docker"
  }
}
```

Docker is the currently supported runtime. `enroot` is registered as a
placeholder but cannot execute workloads.

The separation is intentional:

- The orchestrator decides *where* a suite command executes: host or container.
- The runtime knows *how* to start, inspect, execute in, and remove containers.

A future runtime can be added behind the same interface. Suites that use the
`orch` execution and lifecycle API should not need runtime-specific logic.

There is also a separate transport decision. In normal runs, CVS reaches hosts
via SSH. In a scheduler-managed job step, it can use the managed HTTP agent.
Suite code should not depend on which transport is active.

## Cluster-file contract

A minimal container-oriented cluster file looks like this:

```json
{
  "orchestrator": "container",
  "username": "{user-id}",
  "priv_key_file": "/home/{user-id}/.ssh/id_rsa",

  "head_node_dict": {
    "mgmt_ip": "node-1"
  },

  "node_dict": {
    "node-1": {
      "bmc_ip": "NA",
      "vpc_ip": "node-1"
    },
    "node-2": {
      "bmc_ip": "NA",
      "vpc_ip": "node-2"
    }
  },

  "container": {
    "lifetime": "per_run",
    "name": "my_suite",
    "image": "registry.example/my-workload:tag",
    "runtime": {
      "name": "docker",
      "args": {
        "network": "host",
        "ipc": "host",
        "privileged": true
      }
    }
  }
}
```

The cluster-file fields most relevant to suites are:

- `username`, `priv_key_file`, and optional `password` provide access to remote
  hosts. Never put credential values in a configuration file.
- `node_dict` is the execution set. Its mapping order matters: the first entry
  becomes `orch.head_node`.
- `orch.exec()` targets every selected host by default. `orch.exec_on_head()`
  targets the first selected host.
- `head_node_dict` should remain aligned with the first `node_dict` host for
  compatibility with suites that consume it. The core orchestrator itself
  selects its head from the first `node_dict` entry.
- `vpc_ip` is a node's peer-reachable address for suites that need inter-node
  connectivity.
- `bmc_ip` is hardware-management metadata and is not consumed by the core
  orchestrator.
- `agent_token_file` and node `agent_port` are managed-compute transport
  settings. Most suite owners do not set these directly.

`env_vars` in the cluster file is for older direct parallel-SSH paths. It is
not automatically injected into `orch` commands. Use `container.env` for
container workload environment, or export the environment explicitly in a
bare-metal command.

## Container configuration

The `container` object defines the workload image and its lifecycle:

```json
"container": {
  "lifetime": "per_run",
  "name": "vllm_example",
  "image": "registry.example/vllm:tag",
  "env": {
    "HF_HUB_OFFLINE": "1",
    "NCCL_DEBUG": "ERROR"
  },
  "runtime": {
    "name": "docker",
    "args": {
      "network": "host",
      "ipc": "host",
      "privileged": true,
      "volumes": [
        "/shared/models:/models:ro",
        "/shared/results:/results"
      ]
    }
  }
}
```

### Image identity

`container.image` is the Docker image reference to run, such as
`registry.example/vllm:tag`. If that exact reference is already present on
every selected host, CVS uses it. Otherwise, CVS pulls it.

To distribute a pre-staged image tarball, specify `image_tar` separately:

```json
"container": {
  "image": "registry.example/vllm:tag",
  "image_tar": "/shared/images/vllm-tag.tar"
}
```

If the image is absent, CVS loads the tarball on every selected host before
starting the containers. The tarball path must exist on every host, and it must
contain an image tagged exactly as `container.image`.

### Container lifetime

`container.lifetime` controls who owns the container lifecycle.

#### `per_run`

CVS removes a stale same-named container, creates fresh containers, runs the
suite, and removes them at teardown. This is the normal choice for reproducible
validation suites.

#### `persistent`

CVS attaches when the named container is running on every selected host. If it
is absent on every host, CVS creates it. CVS leaves it running afterward.

Pin `container.name` explicitly. If the container is running on only some
hosts, CVS fails rather than destroying the healthy hosts' state.

#### `no_launch`

CVS does not create, pull, load, or remove containers. It verifies only that a
container with the configured name is running on every selected host.

It does not verify that the running container was created from
`container.image`; the external owner is responsible for consistent container
identity across hosts.

## Docker arguments

Put workload environment at `container.env`:

```json
"env": {
  "HF_HUB_OFFLINE": "1",
  "TRANSFORMERS_OFFLINE": "1",
  "NCCL_SOCKET_IFNAME": "eth0"
}
```

These values become container environment variables and are available to later
`orch.exec()` calls. Do not use `runtime.args.env` for workload environment;
use `container.env`.

Use `runtime.args` for Docker launch behavior:

```json
"runtime": {
  "name": "docker",
  "args": {
    "network": "host",
    "ipc": "host",
    "privileged": true,
    "volumes": [
      "/host/models:/models:ro",
      "/host/logs:/logs"
    ],
    "devices": [
      "/dev/kfd",
      "/dev/dri"
    ],
    "cap_add": [
      "IPC_LOCK"
    ],
    "security_opt": [
      "seccomp=unconfined"
    ],
    "group_add": [
      "video"
    ],
    "ulimit": [
      "memlock=-1"
    ]
  }
}
```

The relevant arguments are:

- `volumes`: bind mounts in `host-path:container-path[:ro]` form. Model mounts
  should normally be read-only. Logs and result paths must be writable. Every
  mount source must exist on every selected host.
- `devices`: extra device passthroughs. CVS includes standard ROCm and
  RDMA-oriented device defaults, including `/dev/kfd`, `/dev/dri`, and
  InfiniBand device discovery.
- `cap_add`: extra Linux capabilities.
- `security_opt`: extra Docker security options.
- `group_add`: extra supplementary groups.
- `ulimit`: per-process resource limits.
- `network`: defaults to `host`. A server port therefore binds directly in the
  host network namespace; choose ports that do not collide with another
  workload on that host.
- `ipc`: defaults to `host`.
- `privileged`: defaults to `true`.

List arguments append to CVS's ROCm/RDMA-oriented defaults. Scalar arguments
such as `network`, `ipc`, and `privileged` override their defaults.

For a private registry, configure:

```json
"registry": {
  "username": "registry-user",
  "password_file": "/path/on/every/remote-host/registry-token",
  "server": "registry.example"
}
```

`password_file` is a path on each remote host, not the machine launching CVS.
CVS supplies the secret through standard input rather than placing it in a
command line.

CVS also injects an SSH-directory mount:

```text
/home/<CVS-process-user>/.ssh:/host_ssh
```

This assumes that path exists on the remote hosts. Keep account and
home-directory layouts consistent between the run controller and target hosts.

## Bare metal and container lifecycle

Bare-metal and container mode share an execution API but have different setup.

### Bare metal

After fixture construction, a bare-metal orchestrator is ready:

```python
def test_gpu_visible(orch):
    out = orch.exec("rocm-smi", timeout=30)
```

`orch.exec()` runs on the host operating system. There is no container
lifecycle for the suite to launch or remove.

### Container

Container commands require a running container first:

```python
def test_launch_container(orch):
    assert orch.setup_containers()


def test_workload(orch):
    out = orch.exec("vllm --version", timeout=30)


def test_teardown(orch):
    assert orch.teardown_containers()
```

`orch.exec()` now runs inside the container. To inspect or prepare the host
operating system from a container suite, use:

```python
out = orch.exec_on_host("docker ps", timeout=30)
```

For `per_run`, retain a fixture-finalizer cleanup guard so a failure cannot
leave containers running. More involved suites often make launch and teardown
explicit lifecycle tests so reports identify the failed stage and its duration.

## Execution API

Broadcast to every selected host:

```python
out = orch.exec("hostname", timeout=30)
```

Normal results are a host-to-output map:

```python
{
    "node-1": "node-1\n",
    "node-2": "node-2\n",
}
```

Use a targeted call for a host-specific role:

```python
out = orch.exec(
    "vllm serve ...",
    hosts=["node-2"],
    timeout=300,
)
```

Use the effective head for a single coordinator, benchmark client, or result
collector:

```python
out = orch.exec_on_head(
    "vllm bench serve ...",
    timeout=600,
)
```

Request detailed results whenever a command's exit code matters:

```python
out = orch.exec("test -d /models", detailed=True)

failed = [
    host
    for host, result in out.items()
    if result.get("exit_code") != 0
]

if failed:
    pytest.fail(f"Model directory missing on: {failed}")
```

Do not interpret the presence of output as command success.

Container orchestrators also support one different command per host:

```python
commands = [
    "start-rank-0",
    "start-rank-1",
]
out = orch.exec_cmd_list(commands)
```

Commands correspond positionally to `orch.hosts`. Prefer targeted
`orch.exec(..., hosts=[host])` calls when the role mapping needs to be obvious
to a reader.

## vLLM as an example

vLLM demonstrates the recommended pattern for an inference suite. Its relevant
implementation lives in:

- `cvs/tests/inference/vllm/conftest.py`
- `cvs/tests/inference/vllm/_common.py`
- `cvs/lib/inference/vllm_job.py`
- `cvs/input/config_file/inference/vllm/`

The vLLM fixture does not use the generic fixture unchanged. It first resolves
the cluster and typed variant configuration, then scopes the execution set:

- `vllm_single` retains only the first cluster host.
- `vllm_distributed` retains all selected hosts.

This prevents a single-node suite from creating containers or starting workload
processes on every node merely because the cluster file contains multiple
nodes.

The vLLM fixture also deep-merges its variant container settings over the
cluster's `container` block before creating the orchestrator. This is needed
because `OrchestratorConfig.from_configs()` performs a top-level merge. A
suite-level `container` object would otherwise replace the entire cluster-level
object.

The suite makes launch, topology checks, model-cache verification, serving,
benchmarking, result validation, and teardown separately reportable stages.
Its fixture finalizer remains a cleanup safety net.

vLLM uses the execution API deliberately:

- Server commands are targeted to the host for each server role.
- The benchmark client uses `orch.exec_on_head()`. Broadcasting it would launch
  competing clients against the same endpoint.
- Cleanup broadcasts process shutdown so no workload process remains on a
  worker.

Keep vLLM paths consistent with its mounts:

```json
"paths": {
  "models_dir": "/models",
  "log_dir": "/home/{user-id}/LOGS"
},
"container": {
  "runtime": {
    "args": {
      "volumes": [
        "/home/{user-id}:/home/{user-id}",
        "/shared/model-cache:/models:ro"
      ]
    }
  }
}
```

`models_dir` must be visible inside every target container. `log_dir` must be
writable in the container and bind-mounted or shared when results must survive
a `per_run` teardown.

For distributed vLLM, configure the container network environment consistently
across all hosts:

```json
"env": {
  "NCCL_IB_HCA": "rdma0,rdma1",
  "NCCL_SOCKET_IFNAME": "eth0",
  "GLOO_SOCKET_IFNAME": "eth0",
  "TP_SOCKET_IFNAME": "eth0",
  "NCCL_IB_GID_INDEX": "3"
}
```

These values are cluster-specific and belong in configuration, not suite
Python.

## Suite-owner checklist

Before adopting `orch`:

- Decide whether the suite validates the host environment (`baremetal`) or a
  workload image (`container`).
- Put selected hosts in `node_dict` in the intended head-first order.
- Choose `per_run`, `persistent`, or `no_launch` deliberately.
- Give persistent or concurrent runs an explicit, unique container name.
- Put workload environment in `container.env`.
- Bind-mount model, log, and output paths that the container needs.
- Ensure mount sources and any `image_tar` path exist on every target host.
- Use broadcast, targeted, and head-only execution according to workload roles.
- Request detailed results and check exit codes when command success matters.
- Keep a `per_run` cleanup guard in the fixture finalizer.
- Keep cluster-specific paths, interfaces, ports, and credentials in
  configuration rather than suite Python.
