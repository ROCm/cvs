"""DTNI workload runner — library entrypoint.

`execute_workload(cluster_path, workload_config_path)` runs one DTNI workload
end-to-end and returns the `JobResult`. Consumed by the pytest wrapper at
`cvs/tests/dtni/vllm_single.py`, which powers `cvs run vllm_single ...`.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cvs.frameworks.registry import FRAMEWORK_REGISTRY
from cvs.lib.dtni.arch_detect import detect_arch_via
from cvs.lib.dtni.artifact_writer import artifact_basename, utc_compact_ts, write_artifacts
from cvs.lib.dtni.catalog import load_catalog
from cvs.lib.dtni.config_loader import load_cluster, load_thresholds, load_workload, resolve_paths
from cvs.lib.dtni.errors import WorkloadError
from cvs.lib.dtni.executor import MultiHostExecutor
from cvs.lib.dtni.hashing import workload_hash
from cvs.lib.dtni.job import Job, JobResult
from cvs.lib.dtni.resource_resolver import (
    resolve_dataset_path,
    resolve_image_on_host,
    resolve_model_path,
)
from cvs.lib.dtni.run_context import RunContext
from cvs.lib.dtni.substitution import build_context, substitute
from cvs.lib.dtni.topology import resolve_bindings

DEFAULT_INPUT_DIR = Path(os.environ.get("CVS_INPUT_DIR", "cvs/input"))


@dataclass
class RunOutcome:
    """What `execute_workload` returns: the JobResult plus paths to artifacts.

    `job_result` carries `.passed`, `.failed_phase`, `.verdicts`, etc.
    `artifacts_dir` is the local directory under `./cvs_artifacts/<run_id>/`.
    `run_id` is the unique tag for this invocation.
    """

    job_result: JobResult
    artifacts_dir: Path
    run_id: str


def _gen_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"r_{ts}_{secrets.token_hex(2)}"


def _read_hf_token_from_file(path: str | os.PathLike[str]) -> None:
    """Export HF_TOKEN from file if env unset. No-op if file missing."""
    if os.environ.get("HF_TOKEN"):
        return
    p = Path(path).expanduser()
    if p.exists():
        os.environ["HF_TOKEN"] = p.read_text().strip()


def _resolve_workload_paths(workload_config_path: Path) -> tuple[str, str, Path, Path]:
    """Given a path to .../<framework>/<model>/<workload>/config.json,
    return (framework, wl_name, config_path, threshold_path).
    Framework and workload names come from the directory layout so a single
    positional `--config_file` is enough.
    """
    cfg = workload_config_path.resolve()
    if cfg.name != "config.json":
        raise WorkloadError(
            f"workload config must be a config.json file; got {cfg}"
        )
    wl_dir = cfg.parent
    framework_dir = wl_dir.parent.parent
    threshold_path = wl_dir / "threshold.json"
    if not threshold_path.exists():
        raise WorkloadError(
            f"expected sibling threshold.json next to {cfg} (looked at {threshold_path})"
        )
    return framework_dir.name, wl_dir.name, cfg, threshold_path


def execute_workload(
    cluster_path: str | os.PathLike[str],
    workload_config_path: str | os.PathLike[str],
    input_dir: str | os.PathLike[str] | None = None,
    hf_token_file: str | os.PathLike[str] | None = None,
    log: Any = None,
) -> RunOutcome:
    """Run one DTNI workload end-to-end. Returns the JobResult + artifact paths.

    `cluster_path` and `workload_config_path` are both filesystem paths.
    `input_dir` defaults to `$CVS_INPUT_DIR` or `cvs/input/`.
    `log` is an optional logger; falls back to `print()` for back-compat.
    """
    say = log.info if log is not None else (lambda msg, *a: print(f"[cvs-dtni] {msg % a if a else msg}", flush=True))

    if hf_token_file is None:
        hf_token_file = os.environ.get("HF_TOKEN_FILE", str(Path.home() / ".hf_token"))
    _read_hf_token_from_file(hf_token_file)

    in_dir = Path(input_dir).resolve() if input_dir else DEFAULT_INPUT_DIR.resolve()

    wl_config = Path(workload_config_path)
    framework, wl_name, wl_path, thr_path = _resolve_workload_paths(wl_config)

    cluster = load_cluster(Path(cluster_path))
    catalog = load_catalog(in_dir)

    # Probe arch on head node.
    pre_exec = MultiHostExecutor(
        hosts=[cluster.head_node],
        user=cluster.username,
        priv_key=cluster.priv_key_file or None,
        env_vars=cluster.env_vars or None,
    )
    say("probing arch on head node %s ...", cluster.head_node)
    arch = detect_arch_via(pre_exec.executor_for(cluster.head_node))
    say("arch=%s", arch)

    workload = load_workload(wl_path, catalog=catalog)
    thresholds = load_thresholds(thr_path)
    if workload.framework != framework:
        raise WorkloadError(
            f"path framework={framework} but {wl_path}.framework={workload.framework}"
        )
    if workload.gpu_arch != arch:
        raise WorkloadError(
            f"config gpu_arch={workload.gpu_arch} != detected arch={arch} "
            f"on {cluster.head_node}; wrong workload for this node"
        )

    paths = resolve_paths(workload, user_id=cluster.username)
    run_id = _gen_run_id()

    bindings = resolve_bindings(
        node_dict=cluster.node_dict,
        roles={r: {"count": v.count, "gpus_per_node": v.gpus_per_node}
               for r, v in workload.topology.roles.items()},
    )
    all_hosts: list[str] = []
    for hs in bindings.values():
        for h in hs:
            if h not in all_hosts:
                all_hosts.append(h)
    if cluster.head_node not in all_hosts:
        all_hosts.append(cluster.head_node)
    executor = MultiHostExecutor(
        hosts=all_hosts, user=cluster.username,
        priv_key=cluster.priv_key_file or None,
        env_vars=cluster.env_vars or None,
    )

    if workload.image is not None:
        image_targets = {h: workload.image for h in all_hosts}
    else:
        image_targets = {}
        for role, hosts in bindings.items():
            role_img = workload.roles[role].image
            if role_img is None:
                raise WorkloadError(
                    f"role {role!r} has no image and no top-level image is set"
                )
            for h in hosts:
                image_targets.setdefault(h, role_img)
    tags = sorted({img.tag for img in image_targets.values()})
    say("resolving images %s on %s ...", tags, list(image_targets))
    first_digest = None
    for host, img in image_targets.items():
        digest = resolve_image_on_host(executor.executor_for(host), img)
        first_digest = first_digest or digest

    wl_dict = workload.model_dump(mode="json")
    wh = workload_hash(image_digest=first_digest, workload=wl_dict, thresholds=thresholds)
    say("workload_hash=%s", wh)

    head_exec = executor.executor_for(cluster.head_node)
    say("resolving model %s ...", workload.model.id)
    model_host_path = resolve_model_path(
        executor=head_exec, catalog=catalog, workload=workload, paths=paths,
    )
    say("model on host: %s", model_host_path)
    dataset_host_path = None
    if workload.dataset is not None:
        dataset_host_path = resolve_dataset_path(
            executor=head_exec, catalog=catalog, workload=workload, paths=paths,
        )

    sub_ctx = build_context(
        paths=paths,
        run_id=run_id,
        user_id=cluster.username,
        model_id=workload.model.id,
        model_path=f"/models/{Path(model_host_path).name}",
        model_precision=workload.model.precision,
        params=workload.params,
    )

    role_spec = workload.roles["server"]
    role_spec_dict = role_spec.model_dump()
    vols = dict(role_spec_dict.get("volumes", {}))
    vols[model_host_path] = f"/models/{Path(model_host_path).name}"
    artifacts_root = Path(substitute(paths["artifacts_dir"], sub_ctx)) / run_id
    artifacts_root_remote = str(artifacts_root)
    vols[artifacts_root_remote] = "/output"
    head_exec.exec(f"mkdir -p {artifacts_root_remote}", timeout=10)
    role_spec_dict["volumes"] = vols

    wl_for_ctx = wl_dict
    wl_for_ctx["roles"]["server"] = role_spec_dict

    ctx = RunContext(
        run_id=run_id,
        arch=arch,
        cluster={"username": cluster.username, "head": cluster.head_node, "node_dict": cluster.node_dict},
        workload=wl_for_ctx,
        thresholds=thresholds,
        workload_name=f"{framework}/{wl_name}",
        workload_hash=wh,
        bindings=bindings,
        executor=executor,
        artifacts_dir=artifacts_root,
    )
    ctx.scratch["sub_ctx"] = sub_ctx
    ctx.scratch["model_host_path"] = model_host_path
    ctx.scratch["dataset_host_path"] = dataset_host_path

    adapter_cls = FRAMEWORK_REGISTRY[framework]
    adapter = adapter_cls()

    say("starting Job run_id=%s", run_id)
    t_start = time.time()
    result = Job(adapter, ctx).run()
    elapsed = time.time() - t_start
    say("Job done in %.1fs; passed=%s failed_phase=%s",
        elapsed, result.passed, result.failed_phase)

    log_text = ""
    for k, v in ctx.logs.items():
        log_text += f"\n===== {k} =====\n{v}\n"
    samples_df = None
    if ctx.result.scalars:
        samples_df = pd.DataFrame([ctx.result.scalars])
    base = artifact_basename(
        arch=arch, model_id=workload.model.id, framework=framework,
        workload_hash=wh, ts=utc_compact_ts(),
    )
    local_out_dir = Path(f"./cvs_artifacts/{run_id}")
    paths_written = write_artifacts(
        out_dir=local_out_dir, basename=base,
        samples_df=samples_df, log_text=log_text, verdict=result.verdict_dict,
    )
    say("artifacts at %s/:", local_out_dir)
    for k, p in paths_written.items():
        say("  %s: %s", k, p.name)

    return RunOutcome(job_result=result, artifacts_dir=local_out_dir, run_id=run_id)
