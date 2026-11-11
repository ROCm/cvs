"""Unit tests for the host inventory readiness check."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from cvs.lib.preflight.host_inventory import (
    HostInventoryCheck,
    export_inventory,
    frontend_nics,
    parse_block_devices,
    parse_dimms,
    parse_nics,
    parse_size_gb,
    profile_config_errors,
    resolve_profile,
)
from cvs.lib.preflight.host_readiness_base import parse_markers

PROFILE = {
    'name': 'gpu-node',
    'nodes': [],
    'dmi_product_name': 'G593-ZX1',
    'cpu_sockets': 2,
    'min_cpu_cores': 128,
    'dimm_count': 2,
    'dimm_size_gb': 64,
    'nvme_count': 2,
    'nvme_min_size_gb': 3000,
    'frontend_nic_count': 1,
    'frontend_nic_interfaces': [],
    'min_ram_gb': 100,
}


def _probe(dimms=('64 GB', '64 GB', 'No Module Installed'), denied=False, product='G593-ZX1'):
    lines = [
        f'PRODUCT_NAME={product}',
        'LSCPU=Model name:            AMD EPYC 9654 96-Core Processor',
        'LSCPU=Socket(s):             2',
        'LSCPU=Core(s) per socket:    96',
        'LSCPU=Thread(s) per core:    2',
        'LSCPU=CPU(s):                384',
        'MEMTOTAL_KB=131072000',
    ]
    if denied:
        lines += ['CVS_CMD_DENIED dmidecode', 'EDAC_DIMM=/sys/devices/system/edac/mc/mc0/dimm0/size 65536']
    else:
        for index, size in enumerate(dimms, start=1):
            lines += [f'DIMM_FIELD={index}|Size: {size}', f'DIMM_FIELD={index}|Locator: DIMM{index}']
    lines += [
        'BLK=NAME="nvme0n1" SIZE="3840755982336" TRAN="nvme" TYPE="disk" MODEL="SAMSUNG MZQL23T8HCLS"',
        'BLK=NAME="nvme1n1" SIZE="3840755982336" TRAN="nvme" TYPE="disk" MODEL="SAMSUNG MZQL23T8HCLS"',
        'BLK=NAME="nvme2n1" SIZE="960197124096" TRAN="nvme" TYPE="disk" MODEL="boot"',
        'BLK=NAME="sda" SIZE="480103981056" TRAN="sata" TYPE="disk" MODEL="x"',
        'NIC=eno1|up|25000|bnxt_en||0',
        'NIC=ens1np0|up|400000|bnxt_en|rocep28s0|0',
        'NIC=ens1v0|up|400000|bnxt_en||1',
    ]
    return '\n'.join(lines)


class TestProfiles(unittest.TestCase):
    def test_profile_config_errors(self):
        profiles = [{'name': 'a', 'nodes': ['n1']}, {'name': 'a', 'nodes': ['n1']}]
        errors = profile_config_errors(profiles, 'missing')
        self.assertEqual(len(errors), 3)
        self.assertEqual(profile_config_errors([{'name': 'a', 'nodes': ['n1']}], 'a'), [])

    def test_resolve_profile_precedence(self):
        explicit = {'name': 'explicit', 'nodes': ['n1']}
        by_product = {'name': 'product', 'dmi_product_name': 'G593-ZX1'}
        default = {'name': 'default'}
        profiles = [by_product, explicit, default]
        self.assertIs(resolve_profile('n1', 'g593-zx1', profiles, 'default'), explicit)
        self.assertIs(resolve_profile('n2', ' G593-ZX1 ', profiles, 'default'), by_product)
        self.assertIs(resolve_profile('n3', 'other', profiles, 'default'), default)
        self.assertIsNone(resolve_profile('n3', 'other', profiles, ''))


class TestParsing(unittest.TestCase):
    def test_parse_size_gb(self):
        self.assertEqual(parse_size_gb('64 GB'), 64.0)
        self.assertEqual(parse_size_gb('32768 MB'), 32.0)
        self.assertEqual(parse_size_gb('1 TB'), 1024.0)
        self.assertIsNone(parse_size_gb('No Module Installed'))

    def test_parse_dimms_sources(self):
        dimms, source = parse_dimms(parse_markers(_probe()))
        self.assertEqual(source, 'dmidecode')
        self.assertEqual([d['size_gb'] for d in dimms], [64.0, 64.0])
        dimms, source = parse_dimms(parse_markers(_probe(denied=True)))
        self.assertEqual((source, dimms[0]['size_gb']), ('edac', 64.0))
        self.assertEqual(parse_dimms({}), ([], 'unavailable'))

    def test_parse_block_devices_keeps_nvme_disks(self):
        drives = parse_block_devices(parse_markers(_probe()))
        self.assertEqual([d['name'] for d in drives], ['nvme0n1', 'nvme1n1', 'nvme2n1'])
        self.assertEqual(drives[0]['size_gb'], 3840.8)

    def test_frontend_nics_excludes_vfs_and_backend(self):
        nics = parse_nics(parse_markers(_probe()))
        self.assertEqual([n['name'] for n in frontend_nics(nics, [])], ['eno1'])
        self.assertEqual([n['name'] for n in frontend_nics(nics, ['mlx5_0'])], ['eno1', 'ens1np0'])
        self.assertEqual([n['name'] for n in frontend_nics(nics, ['rocep28s0'])], ['eno1'])


class TestHostInventoryCheck(unittest.TestCase):
    def test_matching_node_passes_and_records_inventory(self):
        result = HostInventoryCheck(MagicMock(), [PROFILE]).evaluate('n1', _probe())
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['details']['profile'], 'gpu-node')
        inventory = result['details']['inventory']
        self.assertEqual(inventory['cpu']['total_cores'], 192)
        self.assertEqual(inventory['memory']['memtotal_gib'], 125.0)
        roles = {nic['name']: nic['role'] for nic in inventory['nics']}
        self.assertEqual(roles, {'eno1': 'frontend', 'ens1np0': 'backend', 'ens1v0': 'other'})

    def test_mismatches_are_listed(self):
        profile = dict(PROFILE, cpu_sockets=1, dimm_count=4, dimm_size_gb=32, nvme_count=3, frontend_nic_count=2)
        result = HostInventoryCheck(MagicMock(), [profile]).evaluate('n1', _probe())
        joined = '\n'.join(result['errors'])
        self.assertIn('CPU sockets 2 (expected 1)', joined)
        self.assertIn('Populated DIMMs 2 (expected 4)', joined)
        self.assertIn('DIMM sizes 64 GB (expected 32 GB each)', joined)
        self.assertIn('NVMe drives of at least 3000 GB: 2 (expected 3)', joined)
        self.assertIn('Frontend NICs 1 (expected 2)', joined)

    def test_listed_frontend_interfaces_must_exist_and_be_up(self):
        profile = dict(PROFILE, frontend_nic_interfaces=['eno1', 'eno2'], frontend_nic_count=None)
        output = _probe().replace('NIC=eno1|up', 'NIC=eno1|down')
        result = HostInventoryCheck(MagicMock(), [profile]).evaluate('n1', output)
        self.assertIn('Frontend NIC eno1 is down (expected up)', result['errors'])
        self.assertIn('Frontend NIC eno2 is missing', result['errors'])

    def test_low_ram_fails(self):
        result = HostInventoryCheck(MagicMock(), [dict(PROFILE, min_ram_gb=1500)]).evaluate('n1', _probe())
        self.assertIn('Usable RAM 125.0 GiB (MemTotal) is below the required 1500 GiB', result['errors'])

    def test_unreadable_dimms_fail_even_in_warn_mode(self):
        output = _probe(denied=True).replace('EDAC_DIMM=/sys/devices/system/edac/mc/mc0/dimm0/size 65536', '')
        result = HostInventoryCheck(MagicMock(), [PROFILE], on_mismatch='warn').evaluate('n1', output)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('DIMM inventory is unreadable', result['errors'][0])
        self.assertIn("'sudo -n dmidecode' was denied", result['warnings'][0])

    def test_no_profiles_records_inventory_only(self):
        result = HostInventoryCheck(MagicMock()).evaluate('n1', _probe())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['warnings'], ['No SKU profiles configured; inventory recorded only'])

    def test_unmatched_node_and_bad_profile_config_fail(self):
        unmatched = HostInventoryCheck(MagicMock(), [PROFILE]).evaluate('n1', _probe(product='other'))
        self.assertIn('No SKU profile matches node n1', unmatched['errors'][0])
        broken = HostInventoryCheck(MagicMock(), [PROFILE], default_profile='nope').evaluate('n1', _probe())
        self.assertIn("default_profile 'nope' does not name a SKU profile", broken['errors'][0])


class TestExport(unittest.TestCase):
    def test_export_writes_per_node_and_combined_json(self):
        results = {'n/1': HostInventoryCheck(MagicMock(), [PROFILE]).evaluate('n/1', _probe())}
        with tempfile.TemporaryDirectory() as tmp:
            path = export_inventory(results, os.path.join(tmp, 'inv'))
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'inv', 'n_1.json')))
            with open(path, encoding='utf-8') as handle:
                combined = json.load(handle)
        self.assertEqual(combined['n/1']['profile'], 'gpu-node')
        self.assertEqual(combined['n/1']['inventory']['cpu']['sockets'], 2)


if __name__ == '__main__':
    unittest.main()
