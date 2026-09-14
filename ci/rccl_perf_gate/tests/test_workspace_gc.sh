#!/usr/bin/env bash
# Isolated GC test for sbatch/lib/workspace.sh.
#
# Every case runs against a throwaway RCCL_CI_ROOT built in a mktemp dir, so a
# bug in ws_gc cannot reach the real /it-share/rccl-ci. That matters more than
# usual here: the code under test is `rm -rf` in a loop.
#
# Asserts that the janitor says what it did in every case (including reclaiming
# nothing -- silence and "removed 0" are indistinguishable in a log you are
# reading precisely because you don't know whether the job ran), and that an
# in-flight run marked .active survives even when it is old and outside the
# keep-newest-N window.
#
# Needs no GPU, no allocation and no cluster. Run it anywhere:
#   bash cvs/ci/rccl_perf_gate/tests/test_workspace_gc.sh
set -uo pipefail
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sbatch/lib" && pwd)/workspace.sh"
[[ -f "$LIB" ]] || { echo "cannot find workspace.sh at $LIB" >&2; exit 1; }
pass=0; fail=0
check() { if grep -qE "$2" <<<"$3"; then echo "  PASS $1"; pass=$((pass+1));
          else echo "  FAIL $1: no match for /$2/"; echo "$3" | sed 's/^/      /'; fail=$((fail+1)); fi; }
nocheck() { if grep -qE "$2" <<<"$3"; then echo "  FAIL $1: unexpected /$2/"; fail=$((fail+1));
            else echo "  PASS $1"; pass=$((pass+1)); fi; }

echo "1. populated root, nothing old enough to reap -> must still say so"
R=$(mktemp -d)
mkdir -p "$R"/runs/run-{a,b,c} "$R"/builds/by-rev/rev-{1,2,3} "$R"/logs
out=$(RCCL_CI_ROOT="$R" RCCL_CI_WORKSPACE=1 RCCL_CI_WS_KEEP_RUNS=1 RCCL_CI_WS_KEEP_DAYS=99 \
      RCCL_CI_CACHE_KEEP=99 bash -c "source $LIB; ws_gc" 2>&1)
check "reports workspaces line" "gc: workspaces: removed 0" "$out"
check "reports how many kept"   "3 kept"                    "$out"
check "reports build cache"     "gc: build cache: removed 0, 3 kept" "$out"
nocheck "did not delete a run"  "^$" "$(ls "$R"/runs)"
[[ $(ls "$R"/runs | wc -l) -eq 3 ]] && { echo "  PASS runs untouched"; pass=$((pass+1)); } \
                                    || { echo "  FAIL runs were deleted"; fail=$((fail+1)); }

echo "2. genuinely stale entries -> reaped, and the count is reported"
R2=$(mktemp -d)
mkdir -p "$R2"/runs/run-{a,b,c,d,e} "$R2"/builds/by-rev/rev-{1,2,3,4,5}
# run-e is the in-flight one. Mark it, THEN age the directory -- marking it last
# would bump its mtime to now and make it the newest, so it would be spared by
# the keep-newest-N rule and the .active guard would never be exercised. run-a
# is left newest so it is the one N protects.
touch "$R2"/runs/run-e/.active
touch -d "10 days ago" "$R2"/runs/run-{b,c,d,e}
touch -d "1 hour ago"  "$R2"/runs/run-a
out=$(RCCL_CI_ROOT="$R2" RCCL_CI_WORKSPACE=1 RCCL_CI_WS_KEEP_RUNS=1 RCCL_CI_WS_KEEP_DAYS=0 \
      RCCL_CI_CACHE_KEEP=2 bash -c "source $LIB; ws_gc" 2>&1)
echo "$out" | sed 's/^/      /'
check "reaped exactly b,c,d" "gc: workspaces: removed 3" "$out"
check "reaped build cache"   "gc: build cache: removed 3, 2 kept" "$out"
# run-e is old AND outside the keep-newest-1 window, so only the .active marker
# can be saving it here.
[[ -d "$R2/runs/run-e" ]] && { echo "  PASS .active run survived despite being old and unprotected by N"; pass=$((pass+1)); } \
                          || { echo "  FAIL .active run was deleted"; fail=$((fail+1)); }
[[ -d "$R2/runs/run-a" && ! -d "$R2/runs/run-b" ]] \
  && { echo "  PASS newest kept, stale removed"; pass=$((pass+1)); } \
  || { echo "  FAIL wrong runs reaped: $(ls "$R2"/runs | tr '\n' ' ')"; fail=$((fail+1)); }

echo "3. empty root (no runs/ or builds/) -> says nothing to reclaim, does not crash"
R3=$(mktemp -d)
out=$(RCCL_CI_ROOT="$R3" RCCL_CI_WORKSPACE=1 bash -c "source $LIB; ws_gc" 2>&1); rc=$?
check "workspaces: nothing to reclaim" "no runs dir" "$out"
check "cache: nothing to reclaim"      "no cache dir" "$out"
[[ $rc -eq 0 ]] && { echo "  PASS rc=0"; pass=$((pass+1)); } || { echo "  FAIL rc=$rc"; fail=$((fail+1)); }

echo "4. workspace mode off -> ws_gc is still a no-op (janitor forces it on itself)"
out=$(RCCL_CI_ROOT="$R" RCCL_CI_WORKSPACE=0 bash -c "source $LIB; ws_gc" 2>&1)
nocheck "silent when disabled" "gc:" "$out"

rm -rf "$R" "$R2" "$R3"
echo; echo "passed=$pass failed=$fail"
[[ $fail -eq 0 ]]
