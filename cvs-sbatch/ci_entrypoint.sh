#!/usr/bin/env bash
#
# CI-friendly entrypoint for running CVS heatmap tests under an existing SLURM allocation (salloc).
# - Optionally runs setup_nodes.py (requires a Python env with paramiko, e.g. TheRock venv)
# - Runs ./run.sh
# - Archives/publishes artifacts the same way as sbatch/default.sbatch
#
# Usage (inside an allocation):
#   cd /apps/cvs_tests/cvs-sbatch
#   ./ci_entrypoint.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional node setup/cleanup
if [[ "${RUN_SETUP_NODES:-0}" == "1" ]]; then
  echo "[INFO] RUN_SETUP_NODES=1: running setup_nodes.py" >&2
  python3 ./setup_nodes.py
fi

echo "[INFO] Running ./run.sh" >&2
./run.sh
exit_code=$?

# Mirror the sbatch artifact behavior for CI/salloc runs
artifact_src="/tmp/cvs_rccl_${USER}_${SLURM_JOB_ID:-$$}"
archive_dir="/apps/cvs_tests/artifacts_archive/${SLURM_JOB_ID:-local_$$}"
publish_dir="/apps/cvs_tests/test_reports"

mkdir -p "${archive_dir}" "${publish_dir}" 2>/dev/null || true

if [[ -d "${artifact_src}" ]]; then
  cp -a "${artifact_src}/." "${archive_dir}/" 2>/dev/null || true

  perf_html="$(ls -1t "${artifact_src}"/rccl_perf_report_*.html 2>/dev/null | head -n 1 || true)"
  heatmap_html="$(ls -1t "${artifact_src}"/rccl_heatmap_*.html 2>/dev/null | head -n 1 || true)"
  pytest_html="${artifact_src}/pytest_report.html"

  [[ -n "${perf_html}" && -f "${perf_html}" ]] && cp -f "${perf_html}" "${publish_dir}/rccl_perf_report.html" 2>/dev/null || true
  [[ -n "${heatmap_html}" && -f "${heatmap_html}" ]] && cp -f "${heatmap_html}" "${publish_dir}/rccl_heatmap.html" 2>/dev/null || true
  [[ -f "${pytest_html}" ]] && cp -f "${pytest_html}" "${publish_dir}/pytest_report.html" 2>/dev/null || true
else
  echo "[WARNING] Artifact source directory not found: ${artifact_src}" >&2
fi

chmod -R g+rwX "${archive_dir}" "${publish_dir}" 2>/dev/null || true

exit $exit_code



