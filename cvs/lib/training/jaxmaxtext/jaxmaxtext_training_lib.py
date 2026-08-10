'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone JAX MaxText training job driven by a ContainerOrchestrator.

This class talks only to `orch.exec`, which already routes into the running
container, and to a typed `TrainingVariantConfig` (see
`cvs.lib.training.jaxmaxtext.utils.training_config_loader`).

All container interaction goes through `orch.exec()`. No direct Pssh or
docker_lib. The training command, env script, and MaxText YAML config are
built in Python and written into the container by the driver — no external
.sh scripts from the MAD repo.

Both single-node and distributed training use this same class; the config's
`training.distributed` field drives the branching.
'''

from __future__ import annotations

import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import (
    parse_training_log,
    extract_step_metrics,
    extract_eval_metrics,
)

log = globals.log

# Default training-log error signatures (name -> regex). Used as the fallback
# when a config does not define `training.error_patterns`; a config's patterns
# fully replace this set. Kept here so the suite still detects common failures
# out of the box.
_TRAINING_ERR_PATTERNS = {
    'NCCL ERROR': r'NCCL ERROR|NCCL timeout|local work queue catastrophic error',
    'GPU HW ERROR': r'HW Exception by GPU|GPU Hang|Uncorrectable error|GPU Reset',
    'AssertionError': r'AssertionError|ValueError:|JaxStackTrace|During handling of the above exception|triggered the following exception',
    'rocm Err': r'FAILED_PRECONDITION: No visible GPU devices|failed call to hipInit: HIP_ERROR_NoDevice|librocm reported version is: NOT_FOUND',
    'python err': r'ModuleNotFoundError: No module named|Fatal Python error:',
    'tensorflow': r'tensorflow.CoordinationServiceError|tensorflow.BarrierError|CoordinationServiceError',
    'resource': r'RESOURCE_EXHAUSTED: Out of memory|failed: RESOURCE_EXHAUSTED',
}

_COMPLETION_RE = re.compile(r'completed step:\s*(\d+)')
_NAN_INF_RE = re.compile(r'(TFLOP/s/device|Tokens/s/device):\s*(NaN|Inf|-Inf)', re.I)


def _sanitize(name):
    """Filesystem/run-name-safe token from a sweep name (non-alnum -> '_')."""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(name)).strip('_') or "default"


class MaxTextTrainingJob:
    """JAX MaxText training job driven by an injected ContainerOrchestrator.

    All container/SSH plumbing belongs to `orch`. This class composes the
    env script, MaxText YAML config, launches training in the background
    inside the container, polls until complete, and parses the resulting log.

    The `orch` instance is expected to already have `setup_containers()`
    called against it (by the test fixture); lifecycle is explicitly NOT
    owned here.
    """

    def __init__(self, orch, variant, hf_token, sweep=None):
        self.orch = orch
        self.variant = variant
        self.hf_token = hf_token
        self.training = variant.training

        # Per-sweep run: merge the sweep's maxtext overrides onto the base config,
        # and namespace the output dir by the sweep so parallel sweeps' logs never
        # clobber each other (is_complete/parse_results read this per-sweep dir).
        self.sweep = sweep
        self.sweep_tag = _sanitize(sweep.name) if sweep is not None else None
        merged = dict(self.training.maxtext_config)
        if sweep is not None and getattr(sweep, "maxtext_overrides", None):
            merged.update(sweep.maxtext_overrides)
        self.maxtext_config = merged

        self.log_dir = variant.paths.log_dir
        self.out_dir = f"{self.log_dir}/jaxmaxtext/{self.sweep_tag}" if self.sweep_tag else f"{self.log_dir}/jaxmaxtext"
        self.num_nodes = len(orch.hosts)
        self.num_gpus = self.num_nodes * 8

        # Training-log error signatures scanned during polling. Sourced from the
        # config (`training.error_patterns`) so users can add/remove signatures
        # without code changes; falls back to the built-in defaults when the
        # config omits them.
        self.error_patterns = dict(getattr(self.training, "error_patterns", None) or {}) or dict(_TRAINING_ERR_PATTERNS)

        self.step_metrics = []
        self.eval_metrics = []
        self.summary_metrics = {}

        self._poll_wait_s = 60
        self._poll_count = int(self.training.steps * 10)
        self._initial_wait_s = 60

    # ---------- setup ----------

    def setup_training_env(self):
        """Write env script and MaxText YAML config into the container."""
        self.orch.exec("mkdir -p /tmp/jax")
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")
        for i in range(self.num_nodes):
            self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}/out-node{i}")

        self._write_env_script()
        self._write_maxtext_yaml()

    def _build_xla_flags_str(self):
        parts = []
        for k, v in self.training.xla_flags.items():
            parts.append(f"--{k}={v}")
        return " ".join(parts)

    def _write_env_script(self):
        """Write the env script sourced before training launch."""
        lines = []

        lines.append(f"export HF_TOKEN={shlex.quote(self.hf_token)}")
        lines.append(f"export HF_HOME={shlex.quote(self.variant.paths.models_dir)}")
        lines.append("export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH")

        for k, v in self.training.env_vars.items():
            lines.append(f"export {k}={shlex.quote(str(v))}")

        xla_flags = self._build_xla_flags_str()
        if xla_flags:
            lines.append(f'export XLA_FLAGS="{xla_flags}"')

        if self.training.distributed:
            nccl = self.training.nccl
            if nccl.ib_hca:
                lines.append(f"export NCCL_IB_HCA={shlex.quote(nccl.ib_hca)}")
            if nccl.ib_hca_list:
                lines.append(f"export NCCL_IB_HCA_LIST={shlex.quote(nccl.ib_hca_list)}")
            if nccl.socket_ifname:
                lines.append(f"export NCCL_SOCKET_IFNAME={shlex.quote(nccl.socket_ifname)}")
            if nccl.gloo_socket_ifname:
                lines.append(f"export GLOO_SOCKET_IFNAME={shlex.quote(nccl.gloo_socket_ifname)}")
        else:
            lines.append("export NCCL_IB_DISABLE=1")
            lines.append("export NCCL_SHM_DISABLE=0")
            lines.append("export NCCL_P2P_DISABLE=0")

        env_script = "\n".join(lines) + "\n"
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/jax/maxtext_env.sh"))

    def _write_maxtext_yaml(self):
        """Write the MaxText YAML config into the container."""
        mc = dict(self.maxtext_config)

        run_name = f"jaxmaxtext_{self.variant.model.id}"
        if self.sweep_tag:
            run_name = f"{run_name}_{self.sweep_tag}"
        mc["run_name"] = run_name
        mc["steps"] = self.training.steps
        mc["enable_checkpointing"] = self.training.enable_checkpointing
        mc["base_output_directory"] = self.out_dir
        mc["tokenizer_path"] = self.training.tokenizer.tokenizer_path

        yml_lines = []
        for k, v in mc.items():
            if isinstance(v, list):
                yml_lines.append(f'{k}: {v}')
            elif isinstance(v, bool):
                yml_lines.append(f"{k}: {'true' if v else 'false'}")
            else:
                yml_lines.append(f"{k}: {v}")

        yml_content = "\n".join(yml_lines) + "\n"
        self.orch.exec("bash -c " + shlex.quote(f"cat > /tmp/jax/maxtext_config.yml <<'YMLEOF'\n{yml_content}YMLEOF"))

    # ---------- RDMA / NIC setup ----------

    def setup_rdma_lib(self):
        """Copy host RDMA library into container (Broadcom/Thor2 NIC workaround)."""
        rdma = self.training.rdma_lib
        if not rdma.container_mount_file or not rdma.container_dest_file:
            log.info("rdma_lib paths not configured, skipping")
            return
        cmd = f"sudo cp {shlex.quote(rdma.container_mount_file)} {shlex.quote(rdma.container_dest_file)}"
        out = self.orch.exec(cmd)
        for host, output in (out or {}).items():
            log.info("[rdma_lib %s] %s", host, (output or "")[:200])

        verify = self.orch.exec("ibv_devinfo 2>/dev/null | head -20")
        for host, output in (verify or {}).items():
            if not re.search(r'hca_id:\s+(bnxt_|rocep|rdma)', output or "", re.I):
                raise RuntimeError(f"RDMA library not properly configured on {host}: {(output or '')[:300]}")

    def exec_nic_setup_scripts(self):
        """Run NIC setup scripts inside container (distributed only)."""
        log.info("NIC setup for nic_type=%s", self.training.nic_type)

    # ---------- tokenizer ----------

    def setup_tokenizer(self):
        """Download HuggingFace tokenizer into the models dir."""
        tok = self.training.tokenizer
        models_dir = self.variant.paths.models_dir
        self.orch.exec(f"mkdir -p {shlex.quote(models_dir)}")

        hf_model = tok.hf_model_id
        if not hf_model:
            log.info("tokenizer.hf_model_id not set, skipping download")
            return

        # Export the credentials inline rather than sourcing /tmp/jax/maxtext_env.sh:
        # the tokenizer stage runs before setup_training_env() writes that env
        # script, so sourcing it here fails with "No such file or directory".
        dl_cmd = (
            f"export HF_TOKEN={shlex.quote(self.hf_token)} && "
            f"export HF_HOME={shlex.quote(self.variant.paths.models_dir)} && "
            f"huggingface-cli download {shlex.quote(hf_model)} --local-dir {shlex.quote(tok.tokenizer_path)}"
        )
        log.info("downloading tokenizer: %s -> %s", hf_model, tok.tokenizer_path)
        self.orch.exec("bash -c " + shlex.quote(dl_cmd))

    # ---------- training launch ----------

    def build_training_cmd(self):
        """Build the per-node training launcher scripts and write them into the
        container.

        Each node gets its own script with a distinct JAX_PROCESS_INDEX/NODE_RANK.
        The scripts are written across nodes in a single parallel
        ``orch.exec_cmd_list`` call where ``cmd_list[i]`` runs on ``hosts[i]`` --
        so rank i's launcher only ever lands on host i.
        """
        write_cmds = []
        for i in range(self.num_nodes):
            launcher_lines = [
                "#!/bin/bash",
                "source /tmp/jax/maxtext_env.sh",
            ]

            if self.training.distributed:
                jax_dist = self.training.jax_distributed
                # "auto" (or empty) -> use the first cluster node (node_dict order,
                # i.e. orch.hosts[0]) as the JAX coordinator; an explicit IP in the
                # config overrides it.
                coordinator_ip = (getattr(jax_dist, "coordinator_ip", "") or "").strip()
                if not coordinator_ip or coordinator_ip.lower() == "auto":
                    coordinator_ip = self.orch.hosts[0]
                launcher_lines.extend(
                    [
                        f"export JAX_COORDINATOR_IP={shlex.quote(coordinator_ip)}",
                        f"export JAX_COORDINATOR_PORT={shlex.quote(jax_dist.coordinator_port)}",
                        f"export NNODES={self.num_nodes}",
                        f"export NODE_RANK={i}",
                        f"export JAX_PROCESS_INDEX={i}",
                        f"export JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS={jax_dist.initialization_timeout_seconds}",
                        f"export JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS={jax_dist.heartbeat_timeout_seconds}",
                    ]
                )
            else:
                launcher_lines.extend(
                    [
                        "export JAX_COORDINATOR_IP=localhost",
                        "export JAX_COORDINATOR_PORT=12346",
                        "export NNODES=1",
                        "export NODE_RANK=0",
                        "export JAX_PROCESS_INDEX=0",
                    ]
                )

            launcher_lines.append("export PYTHONPATH=$PYTHONPATH:/workspace/maxtext/")
            log_file = f"{self.out_dir}/out-node{i}/training.log"
            launcher_lines.append(
                f"cd /workspace/maxtext && python {shlex.quote(self.training.train_script)} "
                f"/tmp/jax/maxtext_config.yml 2>&1 | tee {shlex.quote(log_file)}"
            )

            script_content = "\n".join(launcher_lines) + "\n"
            script_path = f"/tmp/jax/training_launcher_node{i}.sh"
            write_cmds.append(
                "bash -c "
                + shlex.quote(f"printf '%s' {shlex.quote(script_content)} > {script_path} && chmod +x {script_path}")
            )

        # cmd_list[i] -> hosts[i]: write each node's launcher only on its own host.
        self.orch.exec_cmd_list(write_cmds)

    def start_training(self):
        """Launch training in the background on every node in parallel.

        Uses ``orch.exec_cmd_list`` so ``cmd_list[i]`` runs on ``hosts[i]``: each
        node runs only its own rank-i launcher, and all ranks start together so
        JAX distributed init can rendezvous within the timeout. Fanning the same
        command out to every host (plain ``exec``) would start every rank on
        every node, so multiple processes would claim the same JAX_PROCESS_INDEX
        and the coordinator aborts with a "different incarnation" error.
        """
        log.info("starting training on %d node(s)", self.num_nodes)

        launch_cmds = []
        for i in range(self.num_nodes):
            script_path = f"/tmp/jax/training_launcher_node{i}.sh"
            redirect_log = f"{self.out_dir}/out-node{i}/training_redirect_logs"
            inner = f"nohup bash {script_path} > {shlex.quote(redirect_log)} 2>&1 &"
            launch_cmds.append("bash -c " + shlex.quote(inner))

        self.orch.exec_cmd_list(launch_cmds)

        time.sleep(self._initial_wait_s)

    # ---------- polling ----------

    def is_complete(self):
        """Check if training has completed on all nodes.

        Greps each node's own training.log in a single parallel
        ``orch.exec_cmd_list`` call (``cmd_list[i]`` runs on ``hosts[i]``). Uses
        ``|| true`` rather than ``|| echo 0`` so a no-match yields a clean "0":
        ``grep -c`` already prints "0" and exits 1 on no match, so ``|| echo 0``
        would emit "0\\n0" and defeat the equality check below.
        """
        final_step = self.training.steps - 1
        pattern = f"completed step:\\s*{final_step},"
        cmd_list = [
            f"grep -cE {shlex.quote(pattern)} "
            f"{shlex.quote(f'{self.out_dir}/out-node{i}/training.log')} 2>/dev/null || true"
            for i in range(self.num_nodes)
        ]
        out = self.orch.exec_cmd_list(cmd_list)
        if not out or len(out) < self.num_nodes:
            return False
        for _host, result in out.items():
            text = result if isinstance(result, str) else (result or {}).get("output", "")
            text = (text or "").strip()
            if not text or text == "0":
                return False
        return True

    def _scan_for_errors(self):
        """Scan each node's own training log for known error patterns.

        Reads all nodes' logs in one parallel ``orch.exec_cmd_list`` call
        (``cmd_list[i]`` runs on ``hosts[i]``). Raises on the first match.
        """
        cmd_list = [
            f"tail -2000 {shlex.quote(f'{self.out_dir}/out-node{i}/training.log')} 2>/dev/null"
            for i in range(self.num_nodes)
        ]
        out = self.orch.exec_cmd_list(cmd_list)
        node_of = {h: i for i, h in enumerate(self.orch.hosts)}
        for host, text in (out or {}).items():
            text = text if isinstance(text, str) else (text or {}).get("output", "")
            text = text or ""
            i = node_of.get(host, "?")
            if _NAN_INF_RE.search(text):
                raise RuntimeError(f"NaN/Inf in training metrics on {host} (node {i}): {text[-500:]}")
            for err_name, err_pattern in self.error_patterns.items():
                if not err_pattern:
                    continue
                if re.search(err_pattern, text, re.I):
                    raise RuntimeError(f"Training error '{err_name}' on {host} (node {i}): {text[-500:]}")

    def poll_for_completion(self, timeout_s=None):
        """Poll is_complete() with error scanning until training finishes or times out."""
        if timeout_s is None:
            timeout_s = self._poll_count * self._poll_wait_s

        start = time.monotonic()
        for it in range(self._poll_count):
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                raise RuntimeError(f"training did not complete within {timeout_s}s (polled {it} times)")

            self._scan_for_errors()

            if self.is_complete():
                log.info("training complete (poll iter=%d, %.0fs elapsed)", it, elapsed)
                return

            log.info(
                "training in progress (poll iter=%d, %.0fs elapsed)",
                it,
                elapsed,
            )
            time.sleep(self._poll_wait_s)

        raise RuntimeError(f"training did not complete after {self._poll_count} poll iterations")

    # ---------- results ----------

    def parse_results(self):
        """Parse per-step metrics from training log, compute aggregates.

        Reads the training log from node 0 (the coordinator), parses it via
        the pure `parse_training_log`, and stores both per-step and aggregate
        metrics on self.
        """
        log_file = f"{self.out_dir}/out-node0/training.log"
        # Read node 0's (coordinator) log. Only hosts[0] runs the cat; the other
        # nodes get a no-op so cmd_list[i] still lines up with hosts[i].
        cmd_list = [f"cat {shlex.quote(log_file)}" if i == 0 else "true" for i in range(self.num_nodes)]
        out = self.orch.exec_cmd_list(cmd_list) or {}
        raw = out.get(self.orch.hosts[0], "")
        log_text = raw if isinstance(raw, str) else (raw or {}).get("output", "")
        log_text = log_text or ""

        if not log_text.strip():
            raise RuntimeError(f"empty/missing training log: {log_file}")

        self.step_metrics = extract_step_metrics(log_text)
        self.eval_metrics = extract_eval_metrics(log_text)
        self.summary_metrics = parse_training_log(log_text, self.num_gpus)
        return dict(self.summary_metrics)

    # ---------- cleanup ----------

    def stop_training(self):
        """Kill any running training processes."""
        log.info("stopping training processes")
        self.orch.exec("bash -c 'pkill -f \"maxtext\" || true'")
        time.sleep(5)
