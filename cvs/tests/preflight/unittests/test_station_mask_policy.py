"""Unit tests for the IFoE station-mask admission floor plumbing."""

import unittest

from cvs.lib.preflight.scaleup_fabric import DEFAULT_MIN_UP_PORTS_PER_GPU
from cvs.tests.preflight.preflight_checks import _station_mask_min_up_ports


class TestStationMaskMinUpPorts(unittest.TestCase):
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

    def test_unknown_ifoe_key_still_rejected(self):
        config = {'connectivity_check': {'ifoe': {'bogus': 1}}}
        with self.assertRaises(ValueError):
            _station_mask_min_up_ports(config)


if __name__ == '__main__':
    unittest.main()
