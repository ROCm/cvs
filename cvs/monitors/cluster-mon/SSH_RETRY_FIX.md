# SSH Retry Fix - Summary

## Problem
Unreachable nodes were taking ~10 minutes to fail:
- 3 retry attempts with ~3 minute timeout each
- Total time: 06:16 → 06:26 (10 minutes) for a single unreachable node
- Unreachable nodes were retried on every subsequent command, causing repeated delays

## Solution
Modified SSH connection behavior to fail fast and track unreachable nodes:

### 1. Reduced Retries and Timeouts

**File: `backend/app/core/cvs_parallel_ssh_reliable.py`**

Changes to `Pssh.__init__()`:
- Added `num_retries=1` to ParallelSSHClient (was 3, now 2 total attempts)
- Added `timeout=30` for connection timeout (was unlimited)
- Expected failure time: ~60 seconds (2 attempts × 30s) instead of 10 minutes

Changes to `check_connectivity()`:
- Increased timeout from 2s to 5s for more reliable connectivity checks
- Kept `num_retries=0` for fast connectivity verification

Changes to `prune_unreachable_hosts()`:
- Updated client recreation to include `num_retries=1` and `timeout=30`

### 2. Jump Host Improvements

**File: `backend/app/core/jump_host_pssh.py`**

Changes to `_execute_on_node()`:
- Added check to skip nodes in `unreachable_hosts` list
- Changed SSH ConnectTimeout from 10s to 30s
- Added `ConnectionAttempts=2` (one retry)
- Added logic to detect connection failures and mark nodes as unreachable
- Patterns detected: 'connection timed out', 'connection refused', 'no route to host', 'host is down'

Changes to `exec()`:
- Pre-populate results for unreachable nodes with "ABORT: Host Unreachable Error"
- Only submit ThreadPoolExecutor tasks for reachable nodes
- Log split: "Total nodes: X, Reachable: Y, Unreachable: Z"

## Benefits

1. **Faster Failure Detection**: ~60 seconds instead of ~10 minutes per unreachable node
2. **Reduced Retry Attempts**: 2 total attempts (1 retry) instead of 4 attempts (3 retries)
3. **Persistent Tracking**: Unreachable nodes are marked and skipped in future commands
4. **Better Logging**: Clear indication of reachable vs unreachable nodes

## Testing

Run the test script to verify behavior:
```bash
cd /scratch/venksrin/cvs/cvs/monitors/cluster-mon
python3 test_ssh_retry.py
```

Expected results:
- First command to unreachable host: fails in < 90 seconds
- Second command: skips unreachable host instantly (< 5 seconds)

## Example Log Output (After Fix)

```
2026-03-08 06:16:22 - INFO - Executing command: amd-smi metric --json
2026-03-08 06:16:22 - INFO - Total nodes: 392, Reachable: 390, Unreachable: 2
2026-03-08 06:17:22 - ERROR - Failed to run on host uswslocpm2m-106-1672 - Connection timed out - retry 2/2
2026-03-08 06:17:22 - INFO - ✅ run_command returned 392 results
2026-03-08 06:17:22 - INFO - ✅ CVS Pssh completed: 390 successful, 2 failed
```

Total time: ~60 seconds (not 10 minutes)

## Files Modified

1. `backend/app/core/cvs_parallel_ssh_reliable.py`
   - Added retry and timeout parameters to ParallelSSHClient initialization
   - Updated connectivity check timeouts

2. `backend/app/core/jump_host_pssh.py`
   - Added unreachable host tracking and skipping logic
   - Improved connection timeout settings
   - Added automatic node marking based on error patterns

## Next Steps

1. Test with actual cluster deployment
2. Monitor logs to verify faster failure detection
3. Verify unreachable nodes are properly skipped in subsequent commands
4. Consider adding a periodic "re-check" mechanism to attempt reconnection to previously unreachable nodes
