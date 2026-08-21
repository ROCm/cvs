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

# Bound lazily to cvs.lib.verify_lib.verify_dmesg_for_errors on first use so this
# module stays importable without the broader utils stack that verify_lib pulls
# in (rocm_plib, node_scraper, pytest, ...). Tests patch this symbol directly.
_verify_dmesg_for_errors = None

# Host-side timestamp used to bound the dmesg scan to this training window.
# Format matches what verify_dmesg_for_errors() expects (dmesg -T style).
_DMESG_TIME_CMD = 'date +"%a %b %e %H:%M"'

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
    'segfault': r'Segmentation fault|SIGSEGV|signal 11|core dumped',
}

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
        # GPUs-per-node is config-driven -- do not assume a uniform 8-GPU topology.
        # It feeds num_gpus -> tokens_per_sec_total -> scaling efficiency, so an
        # implicit constant would silently skew a gated-adjacent metric.
        self.gpus_per_node = int(getattr(self.training, "gpus_per_node", 8) or 8)
        self.num_gpus = self.num_nodes * self.gpus_per_node

        # Training-log error signatures scanned during polling. Sourced from the
        # config (`training.error_patterns`) so users can add/remove signatures
        # without code changes; falls back to the built-in defaults when the
        # config omits them.
        self.error_patterns = dict(getattr(self.training, "error_patterns", None) or {}) or dict(_TRAINING_ERR_PATTERNS)

        self.step_metrics = []
        self.eval_metrics = []
        self.summary_metrics = {}
        self.raw_log = ""

        # Host-side timestamp ({node: str}) captured when training launches, so
        # scan_dmesg_for_errors() can slice the kernel log to this run's window.
        self.training_start_time = None

        self._poll_wait_s = 60
        self._poll_count = int(self.training.steps * 10)
        self._initial_wait_s = 60

        self._scratch_dir = None  # resolved lazily to /tmp/<user>/jax
        self._train_script = None  # resolved lazily to the first existing candidate

        # Per-node cursor (lines already surfaced) so polling STREAMS only the
        # new training-log lines to the console once, instead of re-dumping a
        # `tail -N` window every iteration (which bloated --log-file with
        # repeated content).
        self._log_line_cursor = [0] * self.num_nodes

    def _get_scratch_dir(self):
        """User-namespaced in-container scratch base (``/tmp/<user>/jax``).

        Namespacing by the container user avoids /tmp ownership collisions on
        shared GPU nodes: a scratch dir left behind by one user would otherwise
        block a different user's run with a permission error. Resolved once
        (via ``id -un``) and cached.
        """
        if self._scratch_dir:
            return self._scratch_dir
        user = "cvs"
        try:
            out = self.orch.exec("bash -c " + shlex.quote("id -un 2>/dev/null || true"))
            raw = (out or {}).get(self.orch.hosts[0], "")
            text = raw if isinstance(raw, str) else (raw or {}).get("output", "")
            text = (text or "").strip()
            if text:
                user = text.splitlines()[-1].strip() or "cvs"
        except Exception:  # noqa: BLE001 - fall back to a safe default
            pass
        self._scratch_dir = f"/tmp/{user}/jax"
        return self._scratch_dir

    def _resolve_train_script(self):
        """Return the first configured train-script path that exists in the container.

        MaxText moved the train entrypoint across versions (v26.3 and earlier:
        ``.../src/MaxText/train.py``; v26.4+: ``.../src/maxtext/trainers/pre_train/
        train.py``). The config lists candidates in ``train_script_paths`` and we
        pick whichever the running image ships, so the same config works across
        versions. Resolved once and cached.
        """
        if self._train_script:
            return self._train_script
        candidates = list(getattr(self.training, "train_script_paths", None) or [])
        single = getattr(self.training, "train_script", None)
        if single and single not in candidates:
            candidates.append(single)
        if not candidates:
            raise RuntimeError("no train_script_paths (or train_script) configured")

        probe = "".join(f"if [ -f {shlex.quote(p)} ]; then echo {shlex.quote(p)}; exit 0; fi; " for p in candidates)
        out = self.orch.exec("bash -c " + shlex.quote(probe))
        raw = (out or {}).get(self.orch.hosts[0], "")
        text = raw if isinstance(raw, str) else (raw or {}).get("output", "")
        resolved = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
        if not resolved:
            raise RuntimeError(f"none of the configured train_script_paths exist in the container: {candidates}")
        log.info("resolved train_script: %s", resolved)
        self._train_script = resolved
        return resolved

    # ---------- setup ----------

    def setup_training_env(self):
        """Write env script and MaxText YAML config into the container."""
        self.orch.exec(f"mkdir -p {shlex.quote(self._get_scratch_dir())}")
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
        env_path = f"{self._get_scratch_dir()}/maxtext_env.sh"
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > {env_path}"))

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
            elif isinstance(v, str) and v == "":
                # Emit an explicit empty string ('key: ""'), not a bare 'key:'.
                # A bare value parses as YAML null -> Python None, which breaks
                # MaxText fields that are strict enums over "" (e.g. `profiler`
                # accepts only '', 'xplane', 'nsys'; None raises a ValidationError).
                yml_lines.append(f'{k}: ""')
            else:
                yml_lines.append(f"{k}: {v}")

        yml_content = "\n".join(yml_lines) + "\n"
        yml_path = f"{self._get_scratch_dir()}/maxtext_config.yml"
        self.orch.exec("bash -c " + shlex.quote(f"cat > {yml_path} <<'YMLEOF'\n{yml_content}YMLEOF"))

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
        scratch = self._get_scratch_dir()
        train_script = self._resolve_train_script()
        write_cmds = []
        for i in range(self.num_nodes):
            launcher_lines = [
                "#!/bin/bash",
                f"source {scratch}/maxtext_env.sh",
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
                f"cd /workspace/maxtext && python {shlex.quote(train_script)} "
                f"{scratch}/maxtext_config.yml 2>&1 | tee {shlex.quote(log_file)}"
            )

            script_content = "\n".join(launcher_lines) + "\n"
            script_path = f"{scratch}/training_launcher_node{i}.sh"
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

        # Record the host-side start time so a later dmesg scan only looks at
        # kernel messages emitted during this training run.
        self.training_start_time = self._host_date()

        scratch = self._get_scratch_dir()

        # Clear each node's previous training.log BEFORE launch. The launcher
        # only truncates the log once it reaches `python ... | tee training.log`;
        # if this run dies earlier (env source fails, launcher never starts,
        # etc.) a STALE log with an old "completed step: N" marker would remain
        # and is_complete() would report success on the first poll -- a
        # fail-open that lets smoke/training pass without this run doing the
        # work. Removing it here means a run that never reaches `tee` leaves no
        # log, so is_complete() stays false and the stage correctly times out.
        clear_cmds = [
            "bash -c " + shlex.quote(f"rm -f {shlex.quote(f'{self.out_dir}/out-node{i}/training.log')}")
            for i in range(self.num_nodes)
        ]
        self.orch.exec_cmd_list(clear_cmds)

        launch_cmds = []
        for i in range(self.num_nodes):
            script_path = f"{scratch}/training_launcher_node{i}.sh"
            redirect_log = f"{self.out_dir}/out-node{i}/training_redirect_logs"
            inner = f"nohup bash {script_path} > {shlex.quote(redirect_log)} 2>&1 &"
            launch_cmds.append("bash -c " + shlex.quote(inner))

        self.orch.exec_cmd_list(launch_cmds)

        # Fresh run -> stream the log from the top.
        self._log_line_cursor = [0] * self.num_nodes

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
        out = self.orch.exec_cmd_list(cmd_list, print_console=False)
        if not out or len(out) < self.num_nodes:
            return False
        for _host, result in out.items():
            text = result if isinstance(result, str) else (result or {}).get("output", "")
            text = (text or "").strip()
            if not text or text == "0":
                return False
        return True

    def _scan_chunk_for_errors(self, host, i, text):
        """Raise on the first known error signature (or NaN/Inf) in `text`."""
        if not text:
            return
        if _NAN_INF_RE.search(text):
            raise RuntimeError(f"NaN/Inf in training metrics on {host} (node {i}): {text[-500:]}")
        for err_name, err_pattern in self.error_patterns.items():
            if not err_pattern:
                continue
            if re.search(err_pattern, text, re.I):
                raise RuntimeError(f"Training error '{err_name}' on {host} (node {i}): {text[-500:]}")

    def _drain_new_log_lines(self):
        """Fetch training-log lines written since the last poll, on every node,
        in one parallel call, and advance each node's cursor.

        Returns ``{node_index: new_text}``. ``print_console=False`` so the
        orchestrator does NOT re-echo the bulk output -- the caller decides what
        to surface (we stream node 0 and scan every node for errors). This is
        what makes the console/--log-file carry each log line ONCE instead of a
        repeated ``tail`` window per poll.
        """
        cmd_list = [
            f"tail -n +{self._log_line_cursor[i] + 1} "
            f"{shlex.quote(f'{self.out_dir}/out-node{i}/training.log')} 2>/dev/null || true"
            for i in range(self.num_nodes)
        ]
        out = self.orch.exec_cmd_list(cmd_list, print_console=False)
        node_of = {h: i for i, h in enumerate(self.orch.hosts)}
        new_by_node = {}
        for host, result in (out or {}).items():
            text = result if isinstance(result, str) else (result or {}).get("output", "")
            text = text or ""
            i = node_of.get(host)
            if i is None or not text:
                continue
            # Advance the cursor by the number of newly read lines so the next
            # poll starts right after them.
            self._log_line_cursor[i] += len(text.splitlines())
            new_by_node[i] = text
        return new_by_node

    def poll_for_completion(self, timeout_s=None):
        """Poll until training finishes or times out.

        Each iteration streams only the NEW training-log lines (node 0 to the
        console; all nodes scanned for error signatures), then checks for the
        completion marker. A concise ``[poll]`` heartbeat makes the internal
        polling visible without re-dumping the log.
        """
        if timeout_s is None:
            timeout_s = self._poll_count * self._poll_wait_s

        start = time.monotonic()
        for it in range(self._poll_count):
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                raise RuntimeError(f"training did not complete within {timeout_s}s (polled {it} times)")

            new_by_node = self._drain_new_log_lines()

            # Scan every node's new chunk for errors (raises on the first match).
            for i, text in new_by_node.items():
                self._scan_chunk_for_errors(self.orch.hosts[i], i, text)

            # Stream node 0's (coordinator) new lines to the console once.
            node0_new = (new_by_node.get(0) or "").rstrip()
            if node0_new:
                log.info("[train node0]\n%s", node0_new)

            if self.is_complete():
                # Flush the tail lines written between the drain above and this
                # completion check -- the final "completed step" marker, and
                # anything after it (a shutdown traceback or a last-step NaN),
                # land here. Scan EVERY node's final chunk for error signatures
                # before declaring success (dropping non-0 nodes would hide a
                # worker-only failure in this window), then stream node 0.
                tail = self._drain_new_log_lines()
                for i, text in tail.items():
                    self._scan_chunk_for_errors(self.orch.hosts[i], i, text)
                node0_tail = (tail.get(0) or "").rstrip()
                if node0_tail:
                    log.info("[train node0]\n%s", node0_tail)
                log.info("training complete (poll iter=%d, %.0fs elapsed)", it, elapsed)
                return

            log.info(
                "[poll] iter=%d elapsed=%.0fs (streaming node0 log; scanning %d node(s))", it, elapsed, self.num_nodes
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

        self.raw_log = log_text  # kept so callers (e.g. checkpoint I/O timing) can re-scan it
        self.step_metrics = extract_step_metrics(log_text)
        self.eval_metrics = extract_eval_metrics(log_text)
        self.summary_metrics = parse_training_log(log_text, self.num_gpus)
        return dict(self.summary_metrics)

    # ---------- system checks ----------

    def _host_date(self):
        """Return {host: timestamp} from the cluster host OS (not the container).

        Uses the baremetal fan-out handle (``orch.all``) so the timestamp lines
        up with ``dmesg -T`` on the same hosts. Best-effort: returns None if the
        handle/exec is unavailable so callers can skip the dmesg scan cleanly.
        """
        allh = getattr(self.orch, "all", None)
        if allh is None or not hasattr(allh, "exec"):
            return None
        try:
            return allh.exec(_DMESG_TIME_CMD)
        except Exception as e:  # noqa: BLE001 - infra probe, never fatal
            log.warning("could not capture host time for dmesg scan: %s", e)
            return None

    def scan_dmesg_for_errors(self):
        """Scan host kernel logs (dmesg) on all nodes for GPU/HW/kernel faults.

        Ports the sglang flow to the training suite: over the [start, end] window
        captured around the training loop, the shared ``verify_dmesg_for_errors``
        scanner flags HW/crash/driver/network signatures via ``fail_test`` (these
        roll up into the suite's aggregated failure summary) and logs
        perf-degradation signatures as warnings only.

        Best-effort by design: gated on ``training.verify_dmesg`` (default on) and
        wrapped so an infra failure of the scan itself (no passwordless sudo, an
        unexpected ``date`` format, a missing baremetal handle) is logged and
        swallowed -- it must never mask or replace the actual training result.
        Requires a captured start time (i.e. ``start_training`` ran).
        """
        if not getattr(self.training, "verify_dmesg", True):
            log.info("dmesg verification disabled (training.verify_dmesg=false)")
            return
        if not self.training_start_time:
            log.warning("dmesg verification skipped: no training start time captured")
            return
        allh = getattr(self.orch, "all", None)
        if allh is None or not hasattr(allh, "exec"):
            log.warning("dmesg verification skipped: no baremetal host handle (orch.all)")
            return
        try:
            verify = _verify_dmesg_for_errors
            if verify is None:
                from cvs.lib.verify_lib import verify_dmesg_for_errors as verify
            end_time = self._host_date()
            time.sleep(2)
            verify(allh, self.training_start_time, end_time)
        except Exception as e:  # noqa: BLE001 - scan infra failure is non-fatal
            log.warning("dmesg verification skipped (scan failed): %s", e)

    # ---------- cleanup ----------

    def stop_training(self):
        """Best-effort kill of lingering training processes on every node.

        Called when a sweep fails/times out so the next sweep does not launch on
        top of orphaned ranks (important for persistent containers, where per-run
        teardown does not reap them).

        Uses a bracketed first character in the pattern: a running rank's cmdline
        contains ``maxtext_config.yml``/``training_launcher_node`` and matches,
        but this ``pkill`` wrapper's own cmdline contains the literal
        ``[m]axtext_config.yml`` / ``[t]raining_launcher_node`` which the regex
        does not match -- so pkill never targets itself.
        """
        log.info("stopping lingering training processes")
        self.orch.exec(
            "bash -c "
            + shlex.quote("pkill -9 -f '[m]axtext_config.yml' || true; pkill -9 -f '[t]raining_launcher_node' || true")
        )
        time.sleep(3)
