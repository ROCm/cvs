"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

BaseWorkloadAdapter.prepare(): role-contract validation + rocm-smi probe.

These tests guard the boundary checks the base prepare() owns: workload
configs whose topology.roles don't match the adapter's required_roles are
rejected at the boundary with a clear message; rocm-smi probe failures
surface as SetupFailure naming the offending host; the per-host GPU count
lands on ctx.scratch where subclasses can read it.
"""

from __future__ import annotations

import types
import unittest

from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.failure_taxonomy import SetupFailure


class _ConcreteAdapter(BaseWorkloadAdapter):
    """Minimal concrete adapter for testing the base prepare path."""

    framework = "concrete"
    required_roles = ("server",)
    optional_roles = ()

    def launch(self, ctx):
        pass

    def progress_predicate(self, ctx):
        from cvs.lib.adapter_protocol import Progress

        return Progress.DONE

    def parse(self, ctx):
        pass


class _MultiRoleAdapter(BaseWorkloadAdapter):
    framework = "multirole"
    required_roles = ("prefill", "decode")
    optional_roles = ("router",)

    def launch(self, ctx):
        pass

    def progress_predicate(self, ctx):
        from cvs.lib.adapter_protocol import Progress

        return Progress.DONE

    def parse(self, ctx):
        pass


class _FakeExecutor:
    """Duck-typed executor returning a canned rocm-smi output."""

    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def exec(self, cmd, timeout=None):
        self.calls.append((cmd, timeout))
        return self.output


class _FakeMultiExecutor:
    """A2 ``_MultiHostExecutor`` test double.

    ``per_host_outputs`` maps host -> canned rocm-smi output (or callable
    raising on exec). Records (host, cmd, timeout) for every per-host exec
    call so tests can assert the probe ran exactly once per host.
    """

    def __init__(self, per_host_outputs):
        self._per_host = per_host_outputs
        self.calls = []  # list[(host, cmd, timeout)]

    def executor_for(self, host):
        outer = self

        class _One:
            def exec(_self, cmd, timeout=None):
                outer.calls.append((host, cmd, timeout))
                out = outer._per_host[host]
                if callable(out):
                    return out()
                return out

        return _One()

    def exec(self, cmd, timeout=None):  # back-compat forward (unused here)
        first = next(iter(self._per_host))
        return self.executor_for(first).exec(cmd, timeout=timeout)


def _ctx(bindings, executor=None):
    return types.SimpleNamespace(
        bindings=bindings,
        executor=executor,
        scratch={},
    )


# ---- role contract -----------------------------------------------------


class TestRequiredRolesContract(unittest.TestCase):
    def test_missing_required_role_raises(self):
        ctx = _ctx(bindings={"worker": ["n0"]})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("missing required roles", str(exc.exception))
        self.assertIn("server", str(exc.exception))

    def test_empty_bindings_list_for_required_role_raises(self):
        ctx = _ctx(bindings={"server": []})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("bound to no hosts", str(exc.exception))

    def test_extra_role_raises_naming_known_set(self):
        ctx = _ctx(bindings={"server": ["n0"], "router": ["n1"]})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("router", str(exc.exception))
        self.assertIn("adapter knows", str(exc.exception))

    def test_optional_role_allowed_but_not_required(self):
        # Multi-role adapter declares router as optional: present is fine,
        # absent is fine.
        with_router = _ctx(bindings={"prefill": ["n0"], "decode": ["n1"], "router": ["n2"]})
        _MultiRoleAdapter().prepare(with_router)  # no raise
        without_router = _ctx(bindings={"prefill": ["n0"], "decode": ["n1"]})
        _MultiRoleAdapter().prepare(without_router)  # no raise

    def test_no_executor_skips_probe_but_validates_contract(self):
        ctx = _ctx(bindings={"server": ["n0"]}, executor=None)
        _ConcreteAdapter().prepare(ctx)
        self.assertNotIn("host_gpus", ctx.scratch)  # probe skipped


# ---- rocm-smi probe ----------------------------------------------------


_ROCM_SMI_8_GPU = """
============================ ROCm System Management Interface ============================
GPU[0]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[1]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[2]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[3]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[4]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[5]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[6]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[7]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
==========================================================================================
"""


class TestRocmSmiProbe(unittest.TestCase):
    def test_probe_caches_count_per_bound_host(self):
        ctx = _ctx(
            bindings={"server": ["n0"]},
            executor=_FakeExecutor(_ROCM_SMI_8_GPU),
        )
        _ConcreteAdapter().prepare(ctx)
        self.assertEqual(ctx.scratch["host_gpus"], {"n0": 8})

    def test_multi_host_keys_every_bound_host(self):
        # A2: per-host probe via _MultiHostExecutor.executor_for(host).
        # Each host returns its own rocm-smi output and is probed exactly
        # once (hosts that appear in multiple roles are deduped).
        n0_out = _ROCM_SMI_8_GPU
        n1_out = "\n".join(f"GPU[{i}]\t: stub" for i in range(4))
        n2_out = "\n".join(f"GPU[{i}]\t: stub" for i in range(2))
        executor = _FakeMultiExecutor({"n0": n0_out, "n1": n1_out, "n2": n2_out})
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1", "n2"]},
            executor=executor,
        )
        _MultiRoleAdapter().prepare(ctx)
        self.assertEqual(ctx.scratch["host_gpus"], {"n0": 8, "n1": 4, "n2": 2})
        probed_hosts = [host for host, _cmd, _to in executor.calls]
        self.assertEqual(sorted(probed_hosts), ["n0", "n1", "n2"])
        self.assertEqual(len(probed_hosts), 3)  # one probe per host, no replication

    def test_multi_host_probe_failure_names_host(self):
        def _boom():
            raise RuntimeError("ssh timeout")

        executor = _FakeMultiExecutor({
            "n0": _ROCM_SMI_8_GPU,
            "n1": _boom,
            "n2": _ROCM_SMI_8_GPU,
        })
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1", "n2"]},
            executor=executor,
        )
        with self.assertRaises(SetupFailure) as exc:
            _MultiRoleAdapter().prepare(ctx)
        msg = str(exc.exception)
        self.assertIn("rocm-smi probe failed on host n1", msg)
        self.assertIn("ssh timeout", msg)

    def test_multi_host_zero_gpus_names_host(self):
        executor = _FakeMultiExecutor({
            "n0": _ROCM_SMI_8_GPU,
            "n1": "No ROCm devices found",
        })
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1"]},
            executor=executor,
        )
        with self.assertRaises(SetupFailure) as exc:
            _MultiRoleAdapter().prepare(ctx)
        self.assertIn("0 GPUs on host n1", str(exc.exception))

    def test_probe_failure_raises_setup_failure_with_hosts(self):
        class _Boom:
            def exec(self, cmd, timeout=None):
                raise RuntimeError("ssh timeout")

        ctx = _ctx(bindings={"server": ["n0"]}, executor=_Boom())
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("rocm-smi probe failed", str(exc.exception))
        self.assertIn("n0", str(exc.exception))
        self.assertIn("ssh timeout", str(exc.exception))

    def test_zero_gpus_raises_setup_failure(self):
        ctx = _ctx(
            bindings={"server": ["n0"]},
            executor=_FakeExecutor("No ROCm devices found"),
        )
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("rocm-smi reported 0 GPUs", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
