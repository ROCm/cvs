"""`cvs-dtni run <fw>/<wl> --cluster=<name>` entrypoint.

Loads cluster + workload + threshold, builds RunContext, drives Job, writes
artifacts. Returns 0 on pass, 1 on fail (any phase or threshold).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cvs.frameworks.registry import FRAMEWORK_REGISTRY
from cvs.lib.dtni.arch_detect import detect_arch_via
from cvs.lib.dtni.artifact_writer import artifact_basename, utc_compact_ts, write_artifacts
from cvs.lib.dtni.catalog import load_catalog
from cvs.lib.dtni.config_loader import load_cluster, load_thresholds, load_workload, resolve_paths
from cvs.lib.dtni.errors import WorkloadError
from cvs.lib.dtni.executor import MultiHostExecutor
from cvs.lib.dtni.hashing import workload_hash
from cvs.lib.dtni.job import Job
from cvs.lib.dtni.resource_resolver import (
    resolve_dataset_path,
    resolve_image_on_host,
    resolve_model_path,
)
from cvs.lib.dtni.run_context import RunContext
from cvs.lib.dtni.substitution import build_context, substitute
from cvs.lib.dtni.topology import resolve_bindings

DEFAULT_INPUT_DIR = Path(os.environ.get("CVS_INPUT_DIR", "cvs/input"))


def _gen_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"r_{ts}_{secrets.token_hex(2)}"


def _read_hf_token_from_file(path: str) -> None:
    """Read HF token from a file and export to env if HF_TOKEN unset."""
    if os.environ.get("HF_TOKEN"):
        return
    p = Path(path).expanduser()
    if p.exists():
        os.environ["HF_TOKEN"] = p.read_text().strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cvs-dtni run")
    ap.add_argument(
        "workload_selector",
        help="<framework>/<model>/<workload>, e.g. vllm_single/qwen3_next_80b/qwen3_next_80b_bf16_single_8",
    )
    ap.add_argument("--cluster", required=True,
                    help="cluster file name (under input/clusters/) OR absolute path to a cluster JSON")
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"CVS input dir (default {DEFAULT_INPUT_DIR})")
    ap.add_argument(
        "--hf-token-file",
        default=os.environ.get("HF_TOKEN_FILE", str(Path.home() / ".hf_token")),
        help="path to file containing HF_TOKEN (used if env not set)",
    )
    args = ap.parse_args(argv)

    _read_hf_token_from_file(args.hf_token_file)

    input_dir = Path(args.input_dir).resolve()
    sel_parts = args.workload_selector.split("/")
    if len(sel_parts) != 3:
        print(
            f"ERROR: workload selector must be <framework>/<model>/<workload>; "
            f"got {args.workload_selector!r}",
            file=sys.stderr,
        )
        return 2
    framework, model_folder, wl_name = sel_parts

    cluster_arg = Path(args.cluster)
    if cluster_arg.is_absolute() or cluster_arg.suffix == ".json":
        cluster_path = cluster_arg
    else:
        cluster_path = input_dir / "clusters" / f"{args.cluster}.json"
    cluster = load_cluster(cluster_path)
    catalog = load_catalog(input_dir)

    # Pre-arch executor (just to probe arch on head node)
    pre_exec = MultiHostExecutor(
        hosts=[cluster.head_node],
        user=cluster.username,
        priv_key=cluster.priv_key_file or None,
        env_vars=cluster.env_vars or None,
    )
    print(f"[cvs-dtni] probing arch on head node {cluster.head_node} ...", flush=True)
    arch = detect_arch_via(pre_exec.executor_for(cluster.head_node))
    print(f"[cvs-dtni] arch={arch}", flush=True)

    wl_dir = input_dir / "dtni" / framework / model_folder / wl_name
    wl_path = wl_dir / "config.json"
    thr_path = wl_dir / "threshold.json"
    workload = load_workload(wl_path, catalog=catalog)
    thresholds = load_thresholds(thr_path)
    if workload.framework != framework:
        raise WorkloadError(
            f"selector framework={framework} but {wl_path}.framework={workload.framework}"
        )
    if workload.gpu_arch != arch:
        raise WorkloadError(
            f"config gpu_arch={workload.gpu_arch} != detected arch={arch} "
            f"on {cluster.head_node}; wrong workload for this node"
        )

    # Path resolution (cross-references inside paths block first;
    # {user-id} comes from the cluster file).
    paths = resolve_paths(workload, user_id=cluster.username)
    run_id = _gen_run_id()

    # Topology bindings
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
    # Always include head_node in the executor pool — model/dataset probes and
    # mkdir of artifacts_dir run on head_exec even when head_node isn't bound
    # to a role.
    if cluster.head_node not in all_hosts:
        all_hosts.append(cluster.head_node)
    executor = MultiHostExecutor(
        hosts=all_hosts, user=cluster.username,
        priv_key=cluster.priv_key_file or None,
        env_vars=cluster.env_vars or None,
    )

    # Resolve image (on each host) — use first host's digest for the hash.
    # For per-role-only images (e.g. disagg), iterate role specs; for v1
    # single-image workloads the top-level image is the source of truth.
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
    print(f"[cvs-dtni] resolving images {tags} on {list(image_targets)} ...", flush=True)
    first_digest = None
    for host, img in image_targets.items():
        digest = resolve_image_on_host(executor.executor_for(host), img)
        first_digest = first_digest or digest

    # Compute workload_hash
    wl_dict = workload.model_dump(mode="json")
    wh = workload_hash(image_digest=first_digest, workload=wl_dict, thresholds=thresholds)
    print(f"[cvs-dtni] workload_hash={wh}", flush=True)

    # Resolve model/dataset paths on head node (assume shared FS)
    head_exec = executor.executor_for(cluster.head_node)
    print(f"[cvs-dtni] resolving model {workload.model.id} ...", flush=True)
    model_host_path = resolve_model_path(
        executor=head_exec, catalog=catalog, workload=workload, paths=paths,
    )
    print(f"[cvs-dtni] model on host: {model_host_path}", flush=True)
    dataset_host_path = None
    if workload.dataset is not None:
        dataset_host_path = resolve_dataset_path(
            executor=head_exec, catalog=catalog, workload=workload, paths=paths,
        )

    # Build substitution context. model.path is the IN-CONTAINER path; we map
    # the model dir as a volume below.
    sub_ctx = build_context(
        paths=paths,
        run_id=run_id,
        user_id=cluster.username,
        model_id=workload.model.id,
        model_path=f"/models/{Path(model_host_path).name}",
        model_precision=workload.model.precision,
        params=workload.params,
    )

    # Volumes need {models_dir} -> model_host_path's PARENT, so the resolved
    # in-container model path matches. Easier: bind-mount the model_host_path
    # directly to /models/<basename>.
    role_spec = workload.roles["server"]
    role_spec_dict = role_spec.model_dump()
    # Force volume entry for model directory.
    vols = dict(role_spec_dict.get("volumes", {}))
    vols[model_host_path] = f"/models/{Path(model_host_path).name}"
    # Resolve artifacts_dir token and ensure dir
    artifacts_root = Path(substitute(paths["artifacts_dir"], sub_ctx)) / run_id
    artifacts_root_remote = str(artifacts_root)
    # Don't try to mkdir locally — head node may differ. Adapter container creates /output via mount.
    vols[artifacts_root_remote] = "/output"
    head_exec.exec(f"mkdir -p {artifacts_root_remote}", timeout=10)
    role_spec_dict["volumes"] = vols

    # Stash resolved values back into a mutable workload dict for the adapter
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

    print(f"[cvs-dtni] starting Job run_id={run_id}", flush=True)
    t_start = time.time()
    result = Job(adapter, ctx).run()
    elapsed = time.time() - t_start
    print(f"[cvs-dtni] Job done in {elapsed:.1f}s; passed={result.passed} "
          f"failed_phase={result.failed_phase}", flush=True)

    # Write artifacts locally (on whoever runs cvs-dtni). For Step 1 we keep
    # log_text from ctx.logs concatenated.
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
    print(f"[cvs-dtni] artifacts at {local_out_dir}/:", flush=True)
    for k, p in paths_written.items():
        print(f"  {k}: {p.name}", flush=True)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
