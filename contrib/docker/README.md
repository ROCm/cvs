# CVS docker mode — image build (P5)

This directory ships the production CVS-runner image used by docker-mode
CVS (P3+). The image is **substrate-only**: it bundles ROCm (from a
TheRock dist tarball) plus all the system prerequisites that CVS's
`install_*` scripts need, but does NOT bundle TransferBench, RVS, AGFHC,
or ibperf-tools as pre-built artifacts. Those are installed inside the
container at prepare_runtime time (P6) by `cvs run install_*` invoked in
docker mode.

## Why substrate-only?

| Concern | Substrate + runtime install (this design) | Bake-everything-at-build |
|---|---|---|
| Source of truth | CVS install scripts (validated for bare metal too) | Custom Dockerfile RUN steps that drift |
| Image size | ~3-5 GB | ~7-10 GB |
| Per-arch flexibility | install_transferbench respects `GPU_TARGETS=<offload_arches>` from manifest | Need to pin arch at build time |
| Per-cluster config | install_* reads paths from user's config_file | Image must hardcode |
| `prepare_runtime` cost | Slower (~4-6 min for installs) | Fast |
| CVS install_* fixes | Both bare-metal and docker users benefit | Docker users diverge |

The "single source of truth" win is decisive: bare metal and docker mode
run the exact same install logic, so a fix to `install_rvs.py` benefits
both modes without any image rebuild gymnastics.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Production image: TheRock + apt prereqs + python deps + ibverbs + OpenMPI + CVS source + AGFHC compatibility shims + manifest. |
| `build.sh` | Build the image, locally or on a remote node via screen. Required: `--rocm <tarball> --offload-arches <gfx-targets> --tag <tag>`. |
| `smoke.sh` | Substrate-level checks against a running container. Asserts ROCm, rccl-tests, build prereqs, python deps, ibverbs, MPI, CVS source, AGFHC shim, manifest. Does NOT check for RVS/TransferBench/AGFHC binaries -- those land at prepare_runtime time (P6). |
| `Dockerfile.spike` | P0 spike image (public ROCm base + TransferBench). Kept for reference; superseded by `Dockerfile` for production. |
| `build-spike.sh` | Builder for the P0 spike image. Kept for reference. |

## Usage

### Build locally

```bash
contrib/docker/build.sh \
    --rocm ~/Downloads/therock-dist-linux-gfx94X-dcgpu-7.13.0a.tar.gz \
    --offload-arches gfx942 \
    --tag cvs-runner:7.13.0a-gfx942
```

### Build on a remote test node (avoids multi-GB image transfer)

```bash
contrib/docker/build.sh \
    --rocm ~/Downloads/therock-dist-linux-gfx94X-dcgpu-7.13.0a.tar.gz \
    --offload-arches gfx942 \
    --tag cvs-runner:7.13.0a-gfx942 \
    --remote atnair@<test-node>
```

The remote build runs in a `screen` session named `cvs_p5_build` so a
dropped SSH session doesn't kill the build. Poll for completion with the
command the script prints.

### Smoke-test the image

```bash
docker run -d --name cvs-runner --privileged --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    -v /sys:/sys:ro -v /tmp/cvs:/tmp/cvs \
    cvs-runner:7.13.0a-gfx942 sleep infinity
contrib/docker/smoke.sh cvs-runner
```

## Choosing OFFLOAD_ARCHES

`--offload-arches` is the GPU_TARGETS string passed to install_transferbench
(and other install_* scripts that compile HIP code) at prepare_runtime.

| Cluster hardware | Recommended value | Image size impact |
|---|---|---|
| Single-arch fleet (only MI300x) | `gfx942` | smallest |
| Single-arch fleet (only MI250x) | `gfx90a` | smallest |
| Mixed-fleet fat-binary build | `"gfx90a;gfx942"` | +50-200 MB per binary per extra arch |

v1 of the design assumes one cluster.json = one arch. The multi-arch
syntax is documented for future heterogeneous-fleet support but is not
the typical v1 use case.

## What's NOT in the image

- `TransferBench` — installed by `cvs run install_transferbench` at prepare_runtime
- `rvs` — installed by `cvs run install_rvs` at prepare_runtime
- `/opt/amd/agfhc/agfhc` — installed by `cvs run install_agfhc` at prepare_runtime if `runtime.agfhc_tarball` is set in cluster.json
- `ib_write_bw` etc. — installed by `cvs run install_ibperf_tools` at prepare_runtime (when re-enabled; see `cvs run --help`)

## Manifest

`/etc/cvs-runner-manifest.json` records:

```json
{
  "offload_arches": "gfx942",
  "rocm_version": "7.13.0",
  "image_build_time": "2026-04-20T22:00:00Z"
}
```

P6's `prepare_runtime` reads this for arch validation and threads
`offload_arches` into install_transferbench/install_rvs as their
`GPU_TARGETS`.
