"""Unit tests for the IFoE station-mask admission floor plumbing."""

import unittest

from cvs.lib.preflight.scaleup_fabric import DEFAULT_MIN_UP_PORTS_PER_GPU
from cvs.parsers.schemas import PreflightIfoeConfig
from cvs.tests.preflight.preflight_checks import _station_mask_min_up_ports


class TestStationMaskMinUpPorts(unittest.TestCase):
    def test_schema_default_matches_library_constant(self):
        # The pydantic schema hardcodes the default; guard against it drifting
        # from DEFAULT_MIN_UP_PORTS_PER_GPU, which is the real source of truth.
        self.assertEqual(PreflightIfoeConfig().min_up_ports_per_gpu, DEFAULT_MIN_UP_PORTS_PER_GPU)

    def test_defaults_to_real_floor_when_unset(self):
        config = {'connectivity_check': {'ifoe': {'fabric_checks': True}}}
        self.assertEqual(_station_mask_min_up_ports(config), DEFAULT_MIN_UP_PORTS_PER_GPU)

    def test_defaults_when_ifoe_section_missing(self):
        self.assertEqual(_station_mask_min_up_ports({}), DEFAULT_MIN_UP_PORTS_PER_GPU)

    def test_diagnostic_zero_floor_is_honored(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': 0}}}
        self.assertEqual(_station_mask_min_up_ports(config), 0)

    def test_explicit_floor_is_honored(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': 4}}}
        self.assertEqual(_station_mask_min_up_ports(config), 4)

    def test_negative_floor_is_rejected(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': -1}}}
        with self.assertRaises(ValueError):
            _station_mask_min_up_ports(config)

    def test_non_integer_floor_is_rejected(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': 'two'}}}
        with self.assertRaises(ValueError):
            _station_mask_min_up_ports(config)

    def test_fractional_floor_is_rejected(self):
        # int() would silently truncate 0.5->0 (disabling the floor) or 2.9->2.
        for value in (0.5, -0.5, 2.9):
            config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': value}}}
            with self.assertRaises(ValueError):
                _station_mask_min_up_ports(config)

    def test_boolean_floor_is_rejected(self):
        for value in (True, False):
            config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': value}}}
            with self.assertRaises(ValueError):
                _station_mask_min_up_ports(config)

    def test_integer_valued_float_is_accepted(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': 4.0}}}
        self.assertEqual(_station_mask_min_up_ports(config), 4)

    def test_integer_string_is_accepted(self):
        config = {'connectivity_check': {'ifoe': {'min_up_ports_per_gpu': '4'}}}
        self.assertEqual(_station_mask_min_up_ports(config), 4)

    def test_unknown_ifoe_key_still_rejected(self):
        config = {'connectivity_check': {'ifoe': {'bogus': 1}}}
        with self.assertRaises(ValueError):
            _station_mask_min_up_ports(config)


if __name__ == '__main__':
    unittest.main()
