"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Shared fixtures for the cvs/lib/manifest unit tests. Not a test module (no
# ``test_`` prefix) so unittest discovery ignores it.
#
# ``_full_manifest`` builds a Manifest with every submodel populated, including a
# real G2 ``ThresholdVerdict`` so the schema round-trip exercises the cross-group
# config->manifest contract. Pass ``scalars`` kwargs to add extra fact-table
# scalar columns (used by the export suite).

from cvs.lib.config.thresholds import ThresholdVerdict
from cvs.lib.manifest import (
    ConfigInputs,
    HostFingerprint,
    Identity,
    Manifest,
    PatternMatch,
    PhaseTiming,
    ResourceSummary,
    SidecarPointers,
    SystemFingerprint,
    Verdicts,
)


def _full_manifest(run_id: str = "run-1", **scalars) -> Manifest:
    """A manifest with every submodel populated, incl. a real G2 verdict."""
    return Manifest(
        identity=Identity(
            run_id=run_id,
            test_id="suite",
            cell_id="cell-a",
            config_hash="ch",
            workload_hash="wh",
            verification_hash="vh",
            cvs_git_sha="abc123",
            started_at="2025-01-01T00:00:00+00:00",
            finished_at="2025-01-01T00:01:00+00:00",
            invoker="tester",
        ),
        system=SystemFingerprint(
            hosts=[HostFingerprint(hostname="n0", gpus=["mi300x"] * 8, nics=["mlx5_0"])],
            topology_hash="th",
        ),
        config=ConfigInputs(
            resolved_config_path="/run/config.resolved.yaml",
            model="llama-3.1-70b",
            env={"HF_TOKEN": "hf_plaintext_token"},
            commands=["docker run -e HF_TOKEN=hf_plaintext_token ..."],
            seed=7,
        ),
        phases=[PhaseTiming(phase="prepare", duration_s=1.5, status="complete")],
        verdicts=Verdicts(
            overall_status="complete",
            threshold_verdicts=[
                ThresholdVerdict(
                    threshold_type="rate", metric="throughput", op=">=", expected=1200.0, actual=1500.0, passed=True
                ),
            ],
            pattern_matches=[PatternMatch(id="xgmi_err", severity="fatal", line="bad", node="n0", source="dmesg")],
            scalars={"total_throughput": 1500.0, **{k: float(v) for k, v in scalars.items()}},
        ),
        resources=ResourceSummary(per_host={"n0": {"gpu_util": 0.97}}, oom=False),
        sidecars=SidecarPointers(samples="samples.parquet", logs_dir="logs"),
    )
