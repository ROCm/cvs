"""Resolve image / model / dataset to local paths on bound hosts.

Image:
  remote=1: docker pull (with optional registry login)
  remote=0: docker load -i <tarball>

Model / Dataset:
  remote=1: huggingface-cli download via one-shot container
  remote=0: must already be at {models_dir}/<hf_repo> OR HF cache form
            (models--<owner>--<repo>/snapshots/<ref>/)

Returns dict[host] -> {image_digest: str, model_path_in_container: str | None,
                       dataset_path_in_container: str | None}
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from cvs.lib.catalog import Catalog
from cvs.lib.config_loader import ImageBlock, WorkloadConfig
from cvs.lib.errors import WorkloadError


def resolve_image_on_host(executor, image: ImageBlock) -> str:
    """Ensure image is present locally; return its digest (sha256:...)."""
    if image.remote == 1:
        if image.registry_auth is not None:
            ra = image.registry_auth
            u = os.environ.get(ra.username_env)
            p = os.environ.get(ra.password_env)
            if not u or not p:
                raise WorkloadError(
                    f"image registry_auth: env vars {ra.username_env}/{ra.password_env} not set"
                )
            login = (
                f"echo {shlex.quote(p)} | docker login {shlex.quote(ra.registry)} "
                f"-u {shlex.quote(u)} --password-stdin"
            )
            executor.exec(login, timeout=60)
        executor.exec(f"docker pull {shlex.quote(image.tag)}", timeout=1800)
    else:
        raise WorkloadError("image.remote=0 (tarball load) not implemented in v1 smoke path")
    # Get digest. pssh gives us clean per-host stdout (no ssh banner), so the
    # single non-empty line IS the digest.
    out = executor.exec(
        f"docker inspect --format '{{{{.Id}}}}' {shlex.quote(image.tag)}",
        timeout=30,
    )
    digest = out.strip()
    if not digest.startswith("sha256:"):
        raise WorkloadError(
            f"docker inspect returned no sha256 digest for {image.tag}; out:\n{out[:500]}"
        )
    return digest


def resolve_model_path(
    *,
    executor,
    catalog: Catalog,
    workload: WorkloadConfig,
    paths: dict[str, str],
) -> str:
    """Return the host-side path to the model directory.

    For HF-cache layouts (models--owner--repo/snapshots/<ref>/), returns the
    snapshot dir. For flat layouts ({models_dir}/<hf_repo>), returns that.
    """
    hf_repo = catalog.models[workload.model.id].hf_repo
    models_dir = paths["models_dir"]
    flat = f"{models_dir}/{hf_repo}"
    hf_cache = f"{models_dir}/models--{hf_repo.replace('/', '--')}"

    if workload.model.remote == 0:
        # Probe both layouts on host
        probe = (
            f"if [ -d {shlex.quote(flat)} ]; then echo FLAT={flat}; "
            f"elif [ -d {shlex.quote(hf_cache)} ]; then "
            f"  REF=$(cat {shlex.quote(hf_cache)}/refs/main 2>/dev/null || ls -t {shlex.quote(hf_cache)}/snapshots | head -1); "
            f"  echo HF={hf_cache}/snapshots/$REF; "
            f"else echo MISSING; fi"
        )
        raw = executor.exec(probe, timeout=30).strip()
        if raw.startswith("FLAT="):
            return raw[len("FLAT="):]
        if raw.startswith("HF="):
            return raw[len("HF="):]
        raise WorkloadError(
            f"model {workload.model.id!r} (remote=0) not found: tried {flat} and {hf_cache}; "
            f"raw probe output:\n{raw[:500]}"
        )

    # remote=1: download
    token_env = workload.model.hf_token_env or "HF_TOKEN"
    tok = os.environ.get(token_env)
    if not tok:
        raise WorkloadError(f"model.remote=1 but ${token_env} not set in environment")
    dl_cmd = (
        f"mkdir -p {shlex.quote(flat)} && "
        f"docker run --rm -v {shlex.quote(models_dir)}:/models "
        f"-e HF_TOKEN={shlex.quote(tok)} "
        f"python:3.11-slim sh -c "
        f"'pip install -q huggingface_hub && huggingface-cli download "
        f"{shlex.quote(hf_repo)} --local-dir /models/{shlex.quote(hf_repo)}'"
    )
    executor.exec(dl_cmd, timeout=3600)
    return flat


def resolve_dataset_path(*, executor, catalog: Catalog, workload: WorkloadConfig,
                         paths: dict[str, str]) -> str | None:
    if workload.dataset is None:
        return None
    hf_repo = catalog.datasets[workload.dataset.id].hf_repo
    datasets_dir = paths["datasets_dir"]
    flat = f"{datasets_dir}/{hf_repo}"
    if workload.dataset.remote == 0:
        probe = f"test -d {shlex.quote(flat)} && echo OK || echo MISSING"
        if "OK" not in executor.exec(probe, timeout=10):
            raise WorkloadError(f"dataset {workload.dataset.id!r} (remote=0) not at {flat}")
        return flat
    # remote=1: defer real impl to Step 4 (benchmarks). v1 smoke needs no dataset.
    raise WorkloadError("dataset.remote=1 not implemented in v1 smoke path")
