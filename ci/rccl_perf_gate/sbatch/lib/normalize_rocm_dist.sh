#!/usr/bin/env bash
# Normalise a ROCm SDK lib/ directory so each SONAME resolves to exactly ONE inode.
#
# WHY THIS EXISTS
# ---------------
# rocm_devel is a pip-installed wheel (rocm_sdk_devel-*.whl). Wheels are zip
# archives, and the packaging step flattened most versioned-library symlink
# chains into independent regular files. The wheel RECORD confirms it:
#
#     _rocm_sdk_devel/lib/libhsa-runtime64.so,,
#     _rocm_sdk_devel/lib/libhsa-runtime64.so.1,,
#     _rocm_sdk_devel/lib/libhsa-runtime64.so.1.21.0,,
#
# Three separate entries -> three regular files -> three DIFFERENT INODES with
# byte-identical content and the same SONAME.
#
# That breaks a load-bearing glibc invariant: the dynamic loader dedups already
# loaded shared objects by (device, inode), NOT by filename or SONAME. So a
# process that links libhsa-runtime64.so.1 via DT_NEEDED and later
# dlopen()s "libhsa-runtime64.so" gets a SECOND, INDEPENDENT, NEVER-INITIALISED
# copy of ROCr. Every hsa_* call on that handle returns 4107
# (HSA_STATUS_ERROR_NOT_INITIALIZED).
#
# In RCCL that path is rocmwrap.cc: the dlopen fails its version probe, jumps to
# `error:`, and pfn_hsa_amd_portable_export_dmabuf is never resolved -- DMA-BUF
# is silently disabled, with no message unless NCCL_DEBUG>=WARN. Multi-node
# collectives then fall back to a path that hangs on this fabric.
#
# Reinstalling or rebuilding the SDK does NOT fix this: the same wheel produces
# the same flattened layout. The fix belongs here, as a post-install step that
# runs every time the dist is refreshed.
#
# WHAT IT DOES
# ------------
# For each group of byte-identical regular files whose names form a versioned
# chain (libfoo.so, libfoo.so.1, libfoo.so.1.2.3), keep the most-versioned file
# as the single real object and replace the shorter names with relative symlinks
# forming the conventional chain:
#
#     libfoo.so -> libfoo.so.1 -> libfoo.so.1.2.3   (one inode)
#
# This is exactly the layout the ROCm .deb/.tar ships and what ldconfig would
# produce. It is content-preserving: no bytes change, so processes holding the
# old inodes open are unaffected.
#
# USAGE
# -----
#   normalize_rocm_dist.sh                 # dry run against the default dist
#   normalize_rocm_dist.sh --apply         # make the changes (writes a rollback script)
#   normalize_rocm_dist.sh --check         # invariant assertion; exit 1 if violated
#   normalize_rocm_dist.sh --dist /path/to/rocm_devel [--apply|--check]
#
# --check is the CI-facing mode: cheap, read-only, and fails the build with a
# clear message instead of letting a crippled SDK produce a 2h39m hang.

set -euo pipefail

DIST="${ROCM_DIST:-/it-share/rccl-ci/rocm_devel}"
MODE="dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)   MODE="apply"; shift ;;
    --check)   MODE="check"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --dist)    DIST="$2"; shift 2 ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done

LIBDIR="${DIST%/}/lib"
[[ -d "${LIBDIR}" ]] || { echo "[ERROR] not a directory: ${LIBDIR}" >&2; exit 2; }

# Resolve so the rollback script and log are unambiguous even if DIST is a symlink.
REAL_LIBDIR="$(readlink -f "${LIBDIR}")"

echo "=========================================================================="
echo "ROCm dist SONAME normalisation"
echo "  dist    : ${DIST}"
echo "  lib dir : ${REAL_LIBDIR}"
echo "  mode    : ${MODE}"
echo "=========================================================================="

# --- discover groups of content-identical regular .so files -------------------
# Only regular files: existing symlinks are already correct and must be left alone.
tmp_hashes="$(mktemp)"
trap 'rm -f "${tmp_hashes}"' EXIT

find "${REAL_LIBDIR}" -maxdepth 1 -type f -name '*.so*' -exec md5sum {} + 2>/dev/null \
  | sed "s| ${REAL_LIBDIR}/| |" > "${tmp_hashes}"

# Group by hash, emit "hash name1 name2 ..." for groups with >1 member.
groups="$(awk '{h=$1; sub(/^ +/,"",$2); g[h]=g[h]" "$2; n[h]++}
               END {for (k in n) if (n[k]>1) print k g[k]}' "${tmp_hashes}" | sort)"

if [[ -z "${groups}" ]]; then
  echo "[OK] no duplicate-inode SONAME groups found — dist is already normalised."
  exit 0
fi

# --- plan ---------------------------------------------------------------------
declare -a PLAN_LINK PLAN_TARGET
skipped=0
bytes_freed=0
groups_ok=0

while read -r _hash rest; do
  # shellcheck disable=SC2206
  members=( ${rest} )

  # Sort by name length ascending: libfoo.so, libfoo.so.1, libfoo.so.1.2.3
  mapfile -t sorted < <(printf '%s\n' "${members[@]}" | awk '{print length, $0}' | sort -n | cut -d' ' -f2-)

  # SAFETY GUARD: only touch a group whose names form a genuine versioned chain,
  # i.e. each shorter name is a literal prefix of the next longer one. Two
  # unrelated libraries that happen to be byte-identical (vendored duplicates,
  # stubs) must NOT be collapsed into each other — that would silently rewrite
  # the dependency graph.
  chain_ok=1
  for (( i = 0; i < ${#sorted[@]} - 1; i++ )); do
    if [[ "${sorted[i+1]}" != "${sorted[i]}"* ]]; then chain_ok=0; break; fi
  done

  if [[ ${chain_ok} -eq 0 ]]; then
    echo "[SKIP] not a versioned chain, leaving alone: ${sorted[*]}"
    skipped=$(( skipped + 1 ))
    continue
  fi

  groups_ok=$(( groups_ok + 1 ))
  canonical="${sorted[-1]}"

  # Build the conventional chain: each name points at the next longer name.
  for (( i = 0; i < ${#sorted[@]} - 1; i++ )); do
    PLAN_LINK+=( "${sorted[i]}" )
    PLAN_TARGET+=( "${sorted[i+1]}" )
    sz="$(stat -c %s "${REAL_LIBDIR}/${sorted[i]}" 2>/dev/null || echo 0)"
    bytes_freed=$(( bytes_freed + sz ))
  done

  printf '  %-42s <- %s\n' "${canonical}" "$(printf '%s ' "${sorted[@]::${#sorted[@]}-1}")"
done <<< "${groups}"

echo
echo "  chains to normalise : ${groups_ok}"
echo "  files -> symlinks   : ${#PLAN_LINK[@]}"
echo "  groups skipped      : ${skipped}"
printf  "  disk reclaimed      : %.2f GB\n" "$(awk -v b="${bytes_freed}" 'BEGIN{print b/1024/1024/1024}')"
echo

# --- check mode: assert the invariant, change nothing -------------------------
if [[ "${MODE}" == "check" ]]; then
  if [[ ${#PLAN_LINK[@]} -gt 0 ]]; then
    echo "[FAIL] ${#PLAN_LINK[@]} duplicate-inode library file(s) in ${REAL_LIBDIR}." >&2
    echo "       A dlopen() by unversioned name will load a SECOND, uninitialised" >&2
    echo "       copy of these libraries. For ROCr this silently disables DMA-BUF" >&2
    echo "       and hangs multi-node collectives." >&2
    echo "       Fix: ${BASH_SOURCE[0]} --dist ${DIST} --apply" >&2
    exit 1
  fi
  echo "[OK] invariant holds: one inode per SONAME."
  exit 0
fi

if [[ "${MODE}" == "dry-run" ]]; then
  echo "[DRY RUN] nothing changed. Re-run with --apply to make these changes."
  exit 0
fi

# --- apply --------------------------------------------------------------------
[[ -w "${REAL_LIBDIR}" ]] || { echo "[ERROR] ${REAL_LIBDIR} is not writable" >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
rollback="${REAL_LIBDIR}/.normalize_rollback_${stamp}.sh"
{
  echo "#!/usr/bin/env bash"
  echo "# Undo normalize_rocm_dist.sh run of ${stamp}."
  echo "# Replaces each symlink with an independent copy of its target, restoring"
  echo "# the original (broken) multi-inode layout."
  echo "set -euo pipefail"
  echo "cd \"${REAL_LIBDIR}\""
} > "${rollback}"

converted=0
for (( i = 0; i < ${#PLAN_LINK[@]}; i++ )); do
  link="${PLAN_LINK[i]}"
  target="${PLAN_TARGET[i]}"

  # Re-verify identical content at apply time. The plan was computed from a
  # snapshot; refuse to act on anything that changed underneath us.
  if ! cmp -s "${REAL_LIBDIR}/${link}" "${REAL_LIBDIR}/${target}"; then
    echo "[SKIP] content diverged since planning: ${link}" >&2
    continue
  fi

  echo "cp -a --remove-destination \"${target}\" \"${link}\"" >> "${rollback}"

  # Atomic replace: build the symlink under a temp name, then rename over the
  # regular file. There is no instant where ${link} is absent, so a concurrent
  # dlopen either gets the old file or the new symlink — never ENOENT.
  ln -sfn "${target}" "${REAL_LIBDIR}/.${link}.tmp.$$"
  mv -Tf "${REAL_LIBDIR}/.${link}.tmp.$$" "${REAL_LIBDIR}/${link}"
  converted=$(( converted + 1 ))
done

chmod +x "${rollback}"

echo
echo "[OK] converted ${converted} file(s) to symlinks."
echo "[OK] rollback script: ${rollback}"

# --- verify -------------------------------------------------------------------
echo
echo "=== verification: one inode per SONAME chain ==="
fail=0
while read -r _hash rest; do
  # shellcheck disable=SC2206
  members=( ${rest} )
  inodes="$(for m in "${members[@]}"; do stat -Lc %i "${REAL_LIBDIR}/${m}" 2>/dev/null; done | sort -u | wc -l)"
  if [[ "${inodes}" != "1" ]]; then
    echo "  [FAIL] ${members[*]} -> ${inodes} distinct inodes"
    fail=1
  fi
done <<< "${groups}"

[[ ${fail} -eq 0 ]] && echo "  [OK] every normalised chain now resolves to a single inode."
exit "${fail}"
