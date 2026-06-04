"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fabric round-trip + to_env() lowering + extra=forbid typo gate.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from cvs.lib.config.fabric import Fabric


class TestFabricRoundtrip(unittest.TestCase):
    def test_defaults_all_none(self):
        f = Fabric()
        self.assertIsNone(f.nic_type)
        self.assertIsNone(f.nccl_ib_hca)
        self.assertEqual(f.extra_env, {})

    def test_full_roundtrip(self):
        f = Fabric.model_validate(
            {
                "nic_type": "thor2",
                "nccl_ib_hca": "bnxt_re0,bnxt_re1",
                "nccl_socket_ifname": "ens51f1np1",
                "nccl_ib_gid_index": "3",
                "nccl_ib_sl": "3",
                "ucx_net_devices": "bnxt_re0:1",
                "gloo_socket_ifname": "ens51f1np1",
                "hca_id_pattern": "bnxt_|rocep",
                "extra_env": {"NCCL_IB_TC": "106"},
            }
        )
        self.assertEqual(f.nccl_ib_hca, "bnxt_re0,bnxt_re1")
        self.assertEqual(f.hca_id_pattern, "bnxt_|rocep")
        self.assertEqual(f.extra_env["NCCL_IB_TC"], "106")


class TestFabricExtraForbid(unittest.TestCase):
    """The whole point of typed fabric: typos in well-known knobs fail at
    load instead of silently never reaching the env."""

    def test_typo_in_typed_field_rejected(self):
        with self.assertRaises(ValidationError):
            Fabric.model_validate({"nccl_ib_gid_indx": "3"})  # typo: indx


class TestFabricToEnv(unittest.TestCase):
    def test_typed_fields_lowered_to_canonical_env_names(self):
        f = Fabric(
            nccl_ib_hca="rocep158s0",
            nccl_socket_ifname="ens",
            nccl_ib_gid_index="3",
            nccl_ib_sl="3",
            ucx_net_devices="rocep158s0:1",
            gloo_socket_ifname="ens",
        )
        env = f.to_env()
        self.assertEqual(env["NCCL_IB_HCA"], "rocep158s0")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "ens")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")
        self.assertEqual(env["NCCL_IB_SL"], "3")
        self.assertEqual(env["UCX_NET_DEVICES"], "rocep158s0:1")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "ens")

    def test_none_fields_omitted(self):
        f = Fabric(nccl_ib_hca="rocep158s0")
        env = f.to_env()
        self.assertEqual(env, {"NCCL_IB_HCA": "rocep158s0"})
        # Absence is the signal -- the framework default applies; never emit
        # the empty string for an unset field.
        self.assertNotIn("NCCL_SOCKET_IFNAME", env)

    def test_extra_env_overrides_typed(self):
        # Rare but supported: a site-specific override for a typed key.
        f = Fabric(
            nccl_ib_hca="rocep158s0",
            extra_env={"NCCL_IB_HCA": "bnxt_re0", "NCCL_IB_TC": "106"},
        )
        env = f.to_env()
        # extra_env wins (it's the surgical site override).
        self.assertEqual(env["NCCL_IB_HCA"], "bnxt_re0")
        self.assertEqual(env["NCCL_IB_TC"], "106")

    def test_nic_type_and_hca_pattern_not_lowered(self):
        f = Fabric(nic_type="ainic", hca_id_pattern="rocep")
        self.assertEqual(f.to_env(), {})  # neither becomes an env var


if __name__ == "__main__":
    unittest.main()
