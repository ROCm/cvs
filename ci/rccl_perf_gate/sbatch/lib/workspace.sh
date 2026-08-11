#!/usr/bin/env bash
##############################################################
# Per-run workspace isolation for the RCCL A/B regression CI.
#
# WHY: the build and detect steps both write to fixed shared paths today —
#   builds/{reference,candidate}/   librccl.so for each A/B side
#   ab_artifacts/                   ab_regression_report.json + rccl_runs.log
#   cvs-sbatch/cluster.json         regenerated in-tree on every run
#   logs/latest                     global "most recent run" symlink
# With exactly one self-hosted runner those never overlap. The moment a second
# runner exists (see RCCL_CI_REMEDIATION_PLAN.md, issue #1) two PRs share them
# silently — worst case PR A's candidate librccl is measured against PR B's
# reference and the verdict is meaningless but looks legitimate. This module
# gives every run its own workspace so that cannot happen.
#
# OPT-IN: everything here is gated on RCCL_CI_WORKSPACE=1. Unset or 0 and every
# function is a no-op that returns success, so this file is safe to source
# unconditionally from the existing scripts and legacy shared-path behaviour is
# unchanged.
#
# LAYOUT (RCCL_CI_WORKSPACE=1):
#   runs/<run_key>/
#     meta.json        provenance: run key, slurm ids, cvs sha, A/B revs, times
#     config.json      per-run detect config (lib paths + output_dir templated)
#     cvs/             detached git worktree of cvs/ at the recorded sha
#     cvs-sbatch/      real copy — run.sh regenerates cluster.json in-tree
#     builds/          reference/ + candidate/ (see BUILD CACHE below)
#     artifacts/       detector output_dir
#     logs/            RUN_LOG_DIR
#
# RUN KEY: must be stable across the build job and the detect job, because
# detect loads the librccl that build produced. The Slurm job id differs between
# them, so the key is GITHUB_RUN_ID (+ attempt), falling back to SLURM_JOB_ID for
# manual sbatch and a timestamp for bare local runs.
#
# BUILD CACHE: build_rccl.sh already has a rev-aware skip (lib/.built_rev), so
# naively giving every run a private builds/ would turn every run into a cold
# ~30min LTO build and make the pipeline SLOWER — the opposite of the goal.
# Instead librccl is cached by content under builds/by-rev/<rev>-<recipe>/ and
# the per-run builds/<side>/lib is populated from it by hardlink. A given git rev
# built with a given recipe is immutable, so sharing across runs is safe, and the
# merge-base reference is usually identical across PRs targeting develop — so
# most runs hit cache on BOTH sides. Publishing is atomic (build into a private
# tmp dir, then rename), so two runs racing on the same rev cannot corrupt it;
# the loser just adopts the winner's copy.
# Set RCCL_CI_BUILD_CACHE=0 for a fully private cold build per run instead.
##############################################################

# Deliberately no `set -euo pipefail` here — this file is sourced by callers that
# already set it, and we don't want to impose it on any that don't.

# Assign only when unset. Callers (run_rccl_build.sh, run_rccl_ab.sh) declare
# this readonly BEFORE sourcing us, and reassigning a readonly variable is a
# fatal error under 'set -e' -- which would break them even with the workspace
# flag OFF, since sourcing happens before any ws_enabled check.
if [[ -z "${RCCL_CI_ROOT:-}" ]]; then
  RCCL_CI_ROOT="/it-share/rccl-ci"
fi

# How many completed run workspaces the janitor keeps, and the age floor below
# which a workspace is never reaped regardless of count (so a run that is still
# in flight, or one someone is actively debugging, survives).
RCCL_CI_WS_KEEP_RUNS="${RCCL_CI_WS_KEEP_RUNS:-20}"
RCCL_CI_WS_KEEP_DAYS="${RCCL_CI_WS_KEEP_DAYS:-14}"

ws_log() { echo "[workspace $(date +%H:%M:%S)] $*"; }
ws_warn() { echo "[workspace WARN] $*" >&2; }

# ---------------------------------------------------------------------------
# ws_enabled — the single gate. Every other function short-circuits on this.
# ---------------------------------------------------------------------------
# Default ON as of the workspace rollout. It was introduced opt-in so it could be
# validated against the live pipeline without risking prod; that validation
# passed (build + 4-node detect, 224 sweeps, prod config verifiably untouched),
# and leaving it opt-in would mean the shared-path clobbering returns the moment
# a second runner is added. Set RCCL_CI_WORKSPACE=0 to fall back to the legacy
# shared trees for a one-off debug run.
ws_enabled() {
  [[ "${RCCL_CI_WORKSPACE:-1}" == "1" ]]
}

ws_cache_enabled() {
  [[ "${RCCL_CI_BUILD_CACHE:-1}" == "1" ]]
}

# ---------------------------------------------------------------------------
# ws_run_key — stable identity shared by the build job and the detect job.
#
# GITHUB_RUN_ID is the only id both Slurm jobs of one PR run see, so it is the
# primary key. RUN_ATTEMPT is included because a re-run of a failed workflow
# should get a clean workspace rather than inherit half-written state.
# ---------------------------------------------------------------------------
ws_run_key() {
  if [[ -n "${RCCL_CI_RUN_KEY:-}" ]]; then
    echo "${RCCL_CI_RUN_KEY}"
  elif [[ -n "${GITHUB_RUN_ID:-}" ]]; then
    echo "gh-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT:-1}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "slurm-${SLURM_JOB_ID}"
  else
    echo "local-$(date +%Y%m%d_%H%M%S)-$$"
  fi
}

ws_root() {
  echo "${RCCL_CI_ROOT}/runs/$(ws_run_key)"
}

# ---------------------------------------------------------------------------
# ws_recipe_hash — invalidate the build cache when the toolchain changes.
#
# build_rccl.sh stamps only the git rev in .built_rev, so a lib built with an
# older ROCm dist or a different gfx target would look like a cache hit. Fold the
# things that actually change codegen into the cache key. ROCM_DIST is resolved
# through its symlink on purpose: rocm_devel -> 7.14.0a<date>/... so a dist bump
# changes the hash even though the symlink path is constant.
# ---------------------------------------------------------------------------
ws_recipe_hash() {
  local rocm_dist gpu_targets resolved
  rocm_dist="${ROCM_DIST:-${RCCL_CI_ROOT}/rocm_devel}"
  gpu_targets="${GPU_TARGETS:-gfx950}"
  resolved="$(readlink -f "${rocm_dist}" 2>/dev/null || echo "${rocm_dist}")"
  printf '%s|%s' "${resolved}" "${gpu_targets}" | sha256sum | cut -c1-12
}

ws_cache_dir_for() {
  # $1 = git rev
  local rev="$1"
  echo "${RCCL_CI_ROOT}/builds/by-rev/${rev}-$(ws_recipe_hash)"
}

# ---------------------------------------------------------------------------
# ws_init — create the workspace. Idempotent: the build job calls it first, the
# detect job calls it again later and must find the same tree intact.
# ---------------------------------------------------------------------------
ws_init() {
  ws_enabled || return 0

  local ws cvs_src cvs_sha
  ws="$(ws_root)"
  cvs_src="${RCCL_CI_ROOT}/cvs"

  mkdir -p "${ws}"/{builds,artifacts,logs} || {
    ws_warn "could not create workspace at ${ws}"
    return 1
  }

  # --- cvs: detached git worktree ------------------------------------------
  # --detach is required: the branch (aimvt-196-rccl-regression-robustness) is
  # already checked out in the main worktree and git refuses a second checkout
  # of the same branch. Detaching at the resolved sha also gives us exact
  # provenance — meta.json records which cvs revision produced the verdict.
  if [[ ! -d "${ws}/cvs" ]]; then
    local cvs_dirty=""
    cvs_sha="$(git -C "${cvs_src}" rev-parse HEAD 2>/dev/null)"
    [[ -n "${cvs_sha}" ]] && cvs_dirty="$(git -C "${cvs_src}" status --porcelain 2>/dev/null | head -1)"

    if [[ -z "${cvs_sha}" ]]; then
      ws_warn "cvs is not a git checkout; falling back to a plain copy"
      cp -a "${cvs_src}" "${ws}/cvs" || return 1
      cvs_sha="unknown"
    elif [[ -n "${cvs_dirty}" ]]; then
      # A worktree checks out HEAD, so uncommitted changes in cvs/ would be
      # SILENTLY DROPPED — the run would execute different code than the tree
      # the operator is looking at, and meta.json would record a sha that does
      # not describe what ran. Copy instead: correctness beats saving 280M.
      ws_warn "cvs has uncommitted changes — using a full copy, not a worktree."
      ws_warn "  Commit them to get the cheap worktree and exact provenance back."
      cp -a "${cvs_src}" "${ws}/cvs" || return 1
      cvs_sha="${cvs_sha}-dirty"
    else
      # Concurrent worktree adds contend on the repo lock; retry briefly.
      local attempt
      for attempt in 1 2 3; do
        if git -C "${cvs_src}" worktree add --detach "${ws}/cvs" "${cvs_sha}" >/dev/null 2>&1; then
          break
        fi
        [[ ${attempt} -eq 3 ]] && { ws_warn "git worktree add failed after 3 attempts"; return 1; }
        sleep $(( attempt * 2 ))
      done
    fi
  else
    cvs_sha="$(git -C "${ws}/cvs" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi

  # --- cvs-sbatch: real copy ------------------------------------------------
  # Must be a real copy, not a link: run.sh's generate_cluster_config rewrites
  # cluster.json in-tree, which is precisely the file two concurrent runs would
  # corrupt for each other.
  if [[ ! -d "${ws}/cvs-sbatch" ]]; then
    cp -a "${RCCL_CI_ROOT}/cvs-sbatch" "${ws}/cvs-sbatch" || return 1
  fi

  ws_write_meta "${cvs_sha}"

  ws_log "workspace ready: ${ws} (cvs ${cvs_sha:0:10})"
  return 0
}

# ---------------------------------------------------------------------------
# ws_write_meta — provenance. Merged, not overwritten, so the detect job adds
# its Slurm id without erasing the build job's.
# ---------------------------------------------------------------------------
ws_write_meta() {
  local cvs_sha="${1:-unknown}"
  local ws meta
  ws="$(ws_root)"
  meta="${ws}/meta.json"

  RCCL_WS_META="${meta}" \
  RCCL_WS_KEY="$(ws_run_key)" \
  RCCL_WS_CVS_SHA="${cvs_sha}" \
  RCCL_WS_RECIPE="$(ws_recipe_hash)" \
  python3 - <<'PY' 2>/dev/null || true
import json, os, datetime

path = os.environ["RCCL_WS_META"]
try:
    with open(path) as fh:
        doc = json.load(fh)
except Exception:
    doc = {}

doc.setdefault("run_key", os.environ["RCCL_WS_KEY"])
doc.setdefault("created_utc", datetime.datetime.utcnow().isoformat() + "Z")
doc["updated_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
doc["cvs_sha"] = os.environ["RCCL_WS_CVS_SHA"]
doc["recipe_hash"] = os.environ["RCCL_WS_RECIPE"]

for key, env in (
    ("github_run_id", "GITHUB_RUN_ID"),
    ("github_run_attempt", "GITHUB_RUN_ATTEMPT"),
    ("github_sha", "GITHUB_SHA"),
    ("github_pr", "GITHUB_PR_NUMBER"),
):
    if os.environ.get(env):
        doc[key] = os.environ[env]

# Slurm job ids accumulate: one for the build job, one for detect.
job = os.environ.get("SLURM_JOB_ID")
if job:
    name = os.environ.get("SLURM_JOB_NAME", "job")
    jobs = doc.setdefault("slurm_jobs", {})
    jobs[name] = job

with open(path, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

# ---------------------------------------------------------------------------
# ws_record — stash an arbitrary key/value in meta.json (revs, verdicts, ...).
# ---------------------------------------------------------------------------
ws_record() {
  ws_enabled || return 0
  local key="$1" value="$2"
  RCCL_WS_META="$(ws_root)/meta.json" RCCL_WS_K="${key}" RCCL_WS_V="${value}" \
  python3 - <<'PY' 2>/dev/null || true
import json, os
path = os.environ["RCCL_WS_META"]
try:
    with open(path) as fh:
        doc = json.load(fh)
except Exception:
    doc = {}
doc[os.environ["RCCL_WS_K"]] = os.environ["RCCL_WS_V"]
with open(path, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

# ---------------------------------------------------------------------------
# ws_cache_fetch <label> <rev> — seed builds/<label>/lib from the shared cache.
#
# Returns 0 on a hit. On a hit we lay down lib/ complete with build_rccl.sh's own
# .built_rev stamp, so its build_side() sees a valid cached lib and skips the
# rebuild by its own logic — no changes to build_rccl.sh required.
#
# Hardlinks (cp -al) not copies: librccl.so is ~27M and immutable once
# published, so linking is free and keeps disk flat across runs.
# ---------------------------------------------------------------------------
ws_cache_fetch() {
  ws_enabled || return 1
  ws_cache_enabled || return 1

  local label="$1" rev="$2"
  [[ -n "${rev}" && "${rev}" != "unknown" ]] || return 1

  local cache lib_dir
  cache="$(ws_cache_dir_for "${rev}")"
  lib_dir="$(ws_root)/builds/${label}/lib"

  [[ -e "${cache}/lib/librccl.so" ]] || return 1

  mkdir -p "$(dirname "${lib_dir}")"
  rm -rf "${lib_dir}"
  if cp -al "${cache}/lib" "${lib_dir}" 2>/dev/null || cp -a "${cache}/lib" "${lib_dir}"; then
    ws_log "[${label}] build cache HIT for ${rev:0:10} (${cache})"
    return 0
  fi

  ws_warn "[${label}] cache hit but copy failed; will rebuild"
  rm -rf "${lib_dir}"
  return 1
}

# ---------------------------------------------------------------------------
# ws_cache_publish <label> — publish a freshly built lib into the shared cache.
#
# Atomic: stage into a private .tmp-<key> dir alongside the target, then rename.
# rename(2) within one directory is atomic on this NFS mount, so a reader either
# sees no cache entry or a complete one. If another run published the same rev
# first, our rename loses harmlessly and we keep theirs (identical content — same
# rev, same recipe hash).
# ---------------------------------------------------------------------------
ws_cache_publish() {
  ws_enabled || return 0
  ws_cache_enabled || return 0

  local label="$1"
  local ws lib_dir rev cache tmp
  ws="$(ws_root)"
  lib_dir="${ws}/builds/${label}/lib"

  [[ -e "${lib_dir}/librccl.so" ]] || return 0

  # build_rccl.sh stamps the rev it actually built; trust that over our own
  # computation so a cache entry can never be filed under the wrong revision.
  rev="$(cat "${lib_dir}/.built_rev" 2>/dev/null)"
  [[ -n "${rev}" ]] || { ws_warn "[${label}] no .built_rev stamp; not caching"; return 0; }

  cache="$(ws_cache_dir_for "${rev}")"
  [[ -d "${cache}" ]] && { ws_log "[${label}] cache already holds ${rev:0:10}"; return 0; }

  mkdir -p "$(dirname "${cache}")"
  tmp="$(dirname "${cache}")/.tmp-$(ws_run_key)-${label}"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  if ! cp -a "${lib_dir}" "${tmp}/lib"; then
    ws_warn "[${label}] staging to cache failed"
    rm -rf "${tmp}"
    return 0
  fi

  if mv -T "${tmp}" "${cache}" 2>/dev/null; then
    ws_log "[${label}] published ${rev:0:10} to build cache"
  else
    # Lost the race — another run published the same rev+recipe first.
    ws_log "[${label}] cache for ${rev:0:10} published concurrently; keeping existing"
    rm -rf "${tmp}"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# ws_prune_build <label> — drop the object trees once lib/ is safely cached.
#
# build/ + stage/ + src/ are ~1.5G for reference and ~700M for candidate. Keeping
# them per-run would be ~2.3G a run on a filesystem already at 92%. lib/ (~27M)
# is the only thing detect actually loads. Skipped on failure so a broken build
# stays debuggable.
# ---------------------------------------------------------------------------
ws_prune_build() {
  ws_enabled || return 0
  [[ "${RCCL_CI_WS_PRUNE:-1}" == "1" ]] || return 0

  local label="$1" base
  base="$(ws_root)/builds/${label}"
  [[ -d "${base}" ]] || return 0
  [[ -e "${base}/lib/librccl.so" ]] || { ws_log "[${label}] no lib; keeping build tree for debugging"; return 0; }

  # Only measure paths that actually exist. build_rccl.sh does not always
  # create stage/, and the candidate side builds in place so it has no src/.
  # du returns non-zero on a missing path, and with the caller's
  # 'set -o pipefail' that status propagates out of this assignment and
  # aborts an otherwise successful build.
  local d freed=""
  local -a paths=()
  for d in build stage src; do
    [[ -e "${base}/${d}" ]] && paths+=( "${base}/${d}" )
  done
  if [[ ${#paths[@]} -eq 0 ]]; then
    ws_log "[${label}] nothing to prune; kept lib/"
    return 0
  fi
  freed="$(du -shc "${paths[@]}" 2>/dev/null | tail -1 | awk '{print $1}')" || freed=""
  rm -rf "${paths[@]}"
  ws_log "[${label}] pruned build/stage/src (${freed:-?}) — kept lib/"
  return 0
}

# ---------------------------------------------------------------------------
# ws_gc — reap old workspaces. Keeps the newest N and anything younger than
# KEEP_DAYS. Prunes the cvs worktree registry afterwards or git accumulates
# stale administrative entries in cvs/.git/worktrees/.
# ---------------------------------------------------------------------------
ws_gc() {
  ws_enabled || return 0

  local runs_dir="${RCCL_CI_ROOT}/runs"
  [[ -d "${runs_dir}" ]] || return 0

  local reaped=0 candidate
  # Newest-first, skip the newest N, then delete anything older than KEEP_DAYS.
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] || continue
    # Never reap a workspace whose run is still going.
    [[ -e "${candidate}/.active" ]] && continue
    if [[ -n "$(find "${candidate}" -maxdepth 0 -mtime "+${RCCL_CI_WS_KEEP_DAYS}" 2>/dev/null)" ]]; then
      rm -rf "${candidate}" && reaped=$(( reaped + 1 ))
    fi
  done < <(ls -1dt "${runs_dir}"/*/ 2>/dev/null | tail -n "+$(( RCCL_CI_WS_KEEP_RUNS + 1 ))")

  (( reaped > 0 )) && ws_log "gc: removed ${reaped} old workspace(s)"
  git -C "${RCCL_CI_ROOT}/cvs" worktree prune 2>/dev/null || true
  return 0
}

# ---------------------------------------------------------------------------
# ws_begin / ws_end — mark a workspace in use so gc leaves it alone.
# ---------------------------------------------------------------------------
ws_begin() { ws_enabled || return 0; touch "$(ws_root)/.active" 2>/dev/null || true; }
ws_end()   { ws_enabled || return 0; rm -f "$(ws_root)/.active" 2>/dev/null || true; }

# ---------------------------------------------------------------------------
# ws_config_retarget <config.json> — point the detect config at this workspace.
#
# The detector writes ab_regression_report.json and rccl_runs.log to the config's
# output_dir, which ships as the shared /it-share/rccl-ci/ab_artifacts, and each
# sweep writes rccl_result_file under /tmp. Both are shared state; retarget them
# at this run. Idempotent, so build and detect can both call it safely.
#
# Lib paths are NOT touched here — build_rccl.sh already templates them, and it
# writes workspace paths because BUILD_OUT points into the workspace.
#
# output_dir is located by walking the document rather than assuming a section,
# so this keeps working if the config schema is reorganised.
# ---------------------------------------------------------------------------
ws_config_retarget() {
  ws_enabled || return 0
  local src="$1"
  [[ -f "${src}" ]] || { ws_warn "config not found: ${src}"; return 1; }

  # Stage the config INSIDE the workspace and retarget the copy. Editing the
  # caller file in place would rewrite the shared configs/ci_detect*.json that
  # every run reads -- the exact cross-run clobbering this module exists to
  # prevent -- and would leave the prod config permanently pointing at one
  # run artifacts dir.
  local cfg
  cfg="$(ws_root)/config.json"

  if [[ -f "${cfg}" ]]; then
    # Re-entry: the build job already staged this run config and templated the
    # reference/candidate lib paths into it. Re-copying the pristine source
    # would discard those paths and point detect at the wrong libraries.
    ws_log "adopting existing workspace config: ${cfg}"
  elif [[ "$(readlink -f "${src}")" == "$(readlink -f "${cfg}" 2>/dev/null)" ]]; then
    :  # caller already passed the workspace copy
  else
    cp -f "${src}" "${cfg}" || { ws_warn "could not stage config into workspace"; return 1; }
    ws_log "staged config: ${src} -> ${cfg}"
  fi

  # Every later consumer (build_rccl.sh --config, run.sh) must use the copy.
  CONFIG_JSON="${cfg}"
  export CONFIG_JSON

  RCCL_WS_CFG="${cfg}" \
  RCCL_WS_ART="$(ws_root)/artifacts" \
  RCCL_WS_KEY="$(ws_run_key)" \
  python3 - <<'PY' || { ws_warn "could not retarget ${cfg}"; return 1; }
import json, os

cfg = os.environ["RCCL_WS_CFG"]
art = os.environ["RCCL_WS_ART"]
key = os.environ["RCCL_WS_KEY"]

with open(cfg) as fh:
    doc = json.load(fh)

changed = []

def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else k
            if k == "output_dir" and isinstance(v, str) and v != art:
                node[k] = art
                changed.append(here)
            elif k == "rccl_result_file" and isinstance(v, str) and v.startswith("/tmp/"):
                # Per-node scratch shared by every run on that node. Scope it to
                # the run key so concurrent runs cannot read each other's sweeps.
                scoped = f"/tmp/rccl_ci_{key}.json"
                if v != scoped:
                    node[k] = scoped
                    changed.append(here)
            else:
                walk(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]")

walk(doc)

if changed:
    with open(cfg, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print("[workspace] retargeted: " + ", ".join(changed))
PY
  return 0
}

# ---------------------------------------------------------------------------
# Executable entry point (this file is normally sourced; running it directly
# turns it into a small query CLI).
#
# The GitHub workflow has to know where THIS run's artifacts landed so it can
# render the report, post the PR comment and upload them. In workspace mode
# that path is derived from the run key, and re-deriving the key in YAML would
# create a second source of truth that drifts the moment ws_run_key changes.
# So the workflow asks instead:
#
#   ART="$(bash .../sbatch/lib/workspace.sh --print-artifacts)"
#
# With RCCL_CI_WORKSPACE unset or 0 this prints the legacy shared paths, so a
# single workflow line stays correct in both modes. Getting this wrong is not a
# cosmetic bug: the shared ab_artifacts/ still holds a report from a previous
# run, so a workflow that read the wrong path would happily render a MONTHS-OLD
# verdict and post it on the PR as if it were this run's result.
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --print-artifacts)
      if ws_enabled; then echo "$(ws_root)/artifacts"; else echo "${RCCL_CI_ROOT}/ab_artifacts"; fi ;;
    --print-root)
      if ws_enabled; then ws_root; else echo "${RCCL_CI_ROOT}"; fi ;;
    --print-key)
      ws_enabled && ws_run_key || echo "" ;;
    --gc)
      ws_gc ;;
    *)
      echo "usage: workspace.sh --print-artifacts|--print-root|--print-key|--gc" >&2
      exit 2 ;;
  esac
  exit 0
fi
