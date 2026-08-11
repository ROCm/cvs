#!/usr/bin/env bash
# RCCL build-only runner: build reference + candidate librccl.so and template
# their lib paths into CONFIG_JSON. No MPI/detect happens here — see
# run_rccl_ab.sh for the 4-node detection step that follows.
#
# Invoked by sbatch/rccl_build.sbatch. Env contract:
#   CANDIDATE_SRC   rocm-systems checkout at PR head
#   BASE_REF        ref to merge-base against for the reference build
#   CONFIG_JSON     ci_detect.json to template (defaults to configs/ci_detect.json)
#
# Optional (per-run workspace isolation — see sbatch/lib/workspace.sh):
#   RCCL_CI_WORKSPACE=1    build into runs/<run_key>/builds instead of the shared
#                          builds/ dir, so concurrent runs cannot clobber each
#                          other's librccl. Default 0 = legacy shared behaviour.
#   RCCL_CI_BUILD_CACHE=0  disable the rev-keyed shared lib cache (forces a cold
#                          build every run). Default 1.

set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"

: "${CANDIDATE_SRC:?run_rccl_build.sh requires CANDIDATE_SRC (RCCL PR checkout)}"
: "${BASE_REF:?run_rccl_build.sh requires BASE_REF (e.g. origin/develop)}"
export CONFIG_JSON="${CONFIG_JSON:-${RCCL_CI_ROOT}/configs/ci_detect.json}"

[[ -f "${CONFIG_JSON}" ]] || { echo "[ERROR] CONFIG_JSON not found: ${CONFIG_JSON}" >&2; exit 1; }

# --- ROCm dist layout invariant ----------------------------------------------
# The SDK is a pip wheel, and wheel packaging flattens versioned-library symlink
# chains into independent regular files. When that happens, one SONAME maps to
# several inodes; glibc dedups loaded objects by inode, so a dlopen() by
# unversioned name yields a SECOND, uninitialised copy of the library. For ROCr
# that silently disables DMA-BUF export and multi-node collectives hang for the
# full rccl_timeout instead of failing.
#
# This is a ~1s read-only check. It runs before the ~30min build so a re-flattened
# dist (any SDK reinstall reintroduces it) fails here with a named cause, rather
# than surfacing hours later as an unexplained timeout.
_norm="${RCCL_CI_ROOT}/sbatch/lib/normalize_rocm_dist.sh"
if [[ -x "${_norm}" ]]; then
  if ! _norm_out="$("${_norm}" --check 2>&1)"; then
    echo "${_norm_out}" >&2
    echo "[ERROR] ROCm dist layout invariant violated — refusing to build." >&2
    exit 1
  fi
  echo "[INFO] ROCm dist layout OK (one inode per SONAME)."
else
  echo "[WARN] ${_norm} not found; skipping ROCm dist layout check." >&2
fi

mkdir -p "${RCCL_CI_ROOT}/logs"

# --- per-run workspace (no-op unless RCCL_CI_WORKSPACE=1) ---------------------
# shellcheck source=/dev/null
source "${RCCL_CI_ROOT}/sbatch/lib/workspace.sh"

if ws_enabled; then
  ws_init || { echo "[ERROR] workspace init failed" >&2; exit 1; }
  ws_begin
  # Release the in-use marker however we exit, so gc can reap this run later.
  trap 'ws_end' EXIT

  ws_config_retarget "${CONFIG_JSON}" || true

  # Build into the workspace rather than the shared builds/ dir. build_rccl.sh
  # already honours --out; run_rccl_build.sh just points it here.
  BUILD_OUT="$(ws_root)/builds"
  export BUILD_OUT
fi

echo "========================================================================"
echo "RCCL build-only CI"
echo "  Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "  Node         : ${SLURMD_NODENAME:-N/A}"
echo "  Config       : ${CONFIG_JSON}"
if ws_enabled; then
echo "  Workspace    : $(ws_root)"
echo "  Build out    : ${BUILD_OUT}"
echo "  Build cache  : $(ws_cache_enabled && echo enabled || echo disabled)"
fi
echo "========================================================================"

# --- seed both sides from the shared rev-keyed lib cache ----------------------
# build_rccl.sh skips a side when lib/librccl.so exists and lib/.built_rev
# matches the revision it wants. We compute the same two revisions here and, on
# a cache hit, lay down a complete lib/ (stamp included) before calling it — so
# its own cache logic does the skipping and build_rccl.sh needs no changes.
#
# If our revision computation ever disagrees with build_rccl.sh's, the worst
# case is a cache miss and a rebuild: it re-validates .built_rev itself, so a
# mismatched lib can never be silently accepted.
if ws_enabled && ws_cache_enabled; then
  git -C "${CANDIDATE_SRC}" config --global --add safe.directory "${CANDIDATE_SRC}" 2>/dev/null || true
  git -C "${CANDIDATE_SRC}" fetch --no-tags origin "${BASE_REF#origin/}" 2>/dev/null || true

  _cand_rev="$(git -C "${CANDIDATE_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
  _ref_rev="$(git -C "${CANDIDATE_SRC}" merge-base HEAD "${BASE_REF}" 2>/dev/null || echo unknown)"

  echo "[INFO] reference rev=${_ref_rev:0:10}  candidate rev=${_cand_rev:0:10}"
  ws_record reference_rev "${_ref_rev}"
  ws_record candidate_rev "${_cand_rev}"

  ws_cache_fetch reference "${_ref_rev}" || true
  ws_cache_fetch candidate "${_cand_rev}" || true
fi

build_args=(--candidate-src "${CANDIDATE_SRC}" --base-ref "${BASE_REF}" --config "${CONFIG_JSON}")
[[ -n "${BUILD_OUT:-}" ]] && build_args+=(--out "${BUILD_OUT}")

bash "${RCCL_CI_ROOT}/cvs-sbatch/lib/build_rccl.sh" "${build_args[@]}"

# --- publish to the cache, then drop the object trees -------------------------
# Order matters: publish before pruning, and prune only after lib/ is safely
# cached. ws_prune_build refuses to prune a side that has no librccl.so, so a
# failed build keeps its full tree for post-mortem.
if ws_enabled; then
  # Housekeeping must never fail a build that already produced good libraries.
  # Caching and pruning are optimisations: losing either costs disk or a rebuild,
  # whereas aborting here throws away a successful ~10min compile and, via the
  # afterok dependency, silently cancels the detect job behind it.
  ws_cache_publish reference || ws_warn "cache publish (reference) failed — non-fatal"
  ws_cache_publish candidate || ws_warn "cache publish (candidate) failed — non-fatal"
  ws_prune_build reference   || ws_warn "prune (reference) failed — non-fatal"
  ws_prune_build candidate   || ws_warn "prune (candidate) failed — non-fatal"

  echo "[INFO] workspace: $(ws_root)"
  du -sh "$(ws_root)" 2>/dev/null || true

  # Opportunistic: reap old workspaces on the build node, where there is no
  # 4-node allocation being held hostage by the cleanup.
  ws_gc || true
fi
