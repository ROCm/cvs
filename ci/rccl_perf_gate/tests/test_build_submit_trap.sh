#!/usr/bin/env bash
# Functional test for the scancel trap around the build submit.
#
# `sbatch --wait` does not cancel its job when the client dies, so without a
# trap a cancelled or timed-out Actions run leaves the build compiling on an
# --exclusive node until its wall clock expires. This proves the trap fires and
# releases the allocation, by faking sbatch/scancel on PATH and then actually
# killing the step mid-wait.
#
# KEEP IN SYNC: the `step()` function below is a copy of the build submit in
# .github/workflows/rccl_perf_regression.yml (job `build`, step "Build
# reference + candidate librccl"), minus the sbatch arguments. It is duplicated
# rather than extracted because the workflow needs to be readable standalone;
# if you change the trap logic in one, change it in the other. Mirrors GitHub's
# default step shell: bash -e -o pipefail, no -u.
#
# Needs no GPU, no allocation and no cluster:
#   bash cvs/ci/rccl_perf_gate/tests/test_build_submit_trap.sh
BIN=$(mktemp -d); LOG=$(mktemp -d)
export PATH="$BIN:$PATH"

cat > "$BIN/sbatch" <<'EOS'
#!/usr/bin/env bash
echo 99123
if [[ "${FAKE_SBATCH_FAIL:-0}" == "1" ]]; then echo "sbatch: error: nope" >&2; exit 1; fi
sleep "${FAKE_BUILD_SECS:-2}"
exit "${FAKE_BUILD_RC:-0}"
EOS
cat > "$BIN/scancel" <<EOS
#!/usr/bin/env bash
echo "\$@" >> "$LOG/scancel.log"
EOS
chmod +x "$BIN/sbatch" "$BIN/scancel"

# --- the block under test, verbatim from the workflow bar the sbatch args ---
step() {
  set -e -o pipefail
  sb_out="$(mktemp)"
  build_job=""
  cleanup_build() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${build_job}" && "${build_done:-0}" != "1" ]]; then
      echo "::warning::cancelling build job ${build_job}; this step is exiting (rc=${rc}) while it is still allocated"
      scancel "${build_job}" 2>/dev/null || true
    fi
    rm -f "${sb_out}"
    exit "${rc}"
  }
  trap cleanup_build EXIT INT TERM HUP

  sbatch --parsable --wait >"${sb_out}" 2>&1 &
  sb_pid=$!

  for _ in $(seq 1 30); do
    build_job="$(grep -m1 -oE '^[0-9]+' "${sb_out}" 2>/dev/null || true)"
    [[ -n "${build_job}" ]] && break
    sleep 1
  done
  echo "build slurm job: ${build_job:-<unknown>}"

  set +e
  wait "${sb_pid}"
  sb_rc=$?
  set -e
  build_done=1
  cat "${sb_out}"
  [[ "${sb_rc}" -eq 0 ]] || exit "${sb_rc}"
  echo "step reached the end"
}
# ---------------------------------------------------------------------------

pass=0; fail=0
check() { if [[ "$2" == "$3" ]]; then echo "  PASS $1"; pass=$((pass+1));
          else echo "  FAIL $1: expected [$3] got [$2]"; fail=$((fail+1)); fi; }

echo "A. build succeeds -> rc 0, nothing cancelled"
rm -f "$LOG/scancel.log"
( FAKE_BUILD_SECS=2 step ) >/dev/null 2>&1; rc=$?
check "exit code" "$rc" "0"
check "scancel not called" "$(cat "$LOG/scancel.log" 2>/dev/null)" ""

echo "B. build fails in Slurm -> rc propagates, nothing to cancel (job is done)"
rm -f "$LOG/scancel.log"
( FAKE_BUILD_SECS=1 FAKE_BUILD_RC=7 step ) >/dev/null 2>&1; rc=$?
check "exit code" "$rc" "7"
check "scancel not called" "$(cat "$LOG/scancel.log" 2>/dev/null)" ""

echo "C. sbatch cannot submit -> rc 1, no bogus scancel of a nonexistent job"
rm -f "$LOG/scancel.log"
( FAKE_SBATCH_FAIL=1 step ) >/dev/null 2>&1; rc=$?
check "exit code" "$rc" "1"

echo "D. step killed mid-build (the GHA timeout / cancelled-run case)"
rm -f "$LOG/scancel.log"
( FAKE_BUILD_SECS=60 step ) >/dev/null 2>&1 &
victim=$!
sleep 4
kill -TERM "$victim" 2>/dev/null
wait "$victim" 2>/dev/null; rc=$?
check "scancel called for the job" "$(cat "$LOG/scancel.log" 2>/dev/null)" "99123"
check "did not exit 0 (would mask the kill)" "$([[ $rc -ne 0 ]] && echo notzero)" "notzero"

echo "E. same, via SIGHUP (runner shutdown)"
rm -f "$LOG/scancel.log"
( FAKE_BUILD_SECS=60 step ) >/dev/null 2>&1 &
victim=$!
sleep 4
kill -HUP "$victim" 2>/dev/null
wait "$victim" 2>/dev/null
check "scancel called for the job" "$(cat "$LOG/scancel.log" 2>/dev/null)" "99123"

echo
echo "passed=$pass failed=$fail"
rm -rf "$BIN" "$LOG"
[[ $fail -eq 0 ]]
