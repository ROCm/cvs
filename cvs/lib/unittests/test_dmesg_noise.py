'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest

from cvs.lib import dmesg_noise


class TestDmesgNoise(unittest.TestCase):
    def test_zero_vm_status_fields_are_benign(self):
        walker = 'amdgpu 0000:03:00.0: amdgpu: WALKER_ERROR: 0x0'
        mapping = 'amdgpu 0000:03:00.0: amdgpu: MAPPING_ERROR: 0x0'
        protection = 'amdgpu 0000:03:00.0: amdgpu: VM_L2_PROTECTION_FAULT_STATUS:0x00000000'
        self.assertTrue(dmesg_noise.dmesg_line_is_benign(walker))
        self.assertTrue(dmesg_noise.dmesg_line_is_benign(mapping))
        self.assertTrue(dmesg_noise.dmesg_line_is_benign(protection))
        self.assertFalse(dmesg_noise.line_indicates_driver_error(walker))
        self.assertFalse(dmesg_noise.line_indicates_driver_error(mapping))

    def test_nonzero_status_fields_stay_errors(self):
        walker = 'amdgpu 0000:03:00.0: amdgpu: WALKER_ERROR: 0x1'
        mapping = 'amdgpu 0000:03:00.0: amdgpu: MAPPING_ERROR: 0x1'
        protection = 'amdgpu 0000:03:00.0: amdgpu: VM_L2_PROTECTION_FAULT_STATUS:0x00301030'
        self.assertFalse(dmesg_noise.dmesg_line_is_benign(walker))
        self.assertFalse(dmesg_noise.dmesg_line_is_benign(mapping))
        self.assertFalse(dmesg_noise.dmesg_line_is_benign(protection))
        self.assertTrue(dmesg_noise.line_indicates_driver_error(walker))
        self.assertTrue(dmesg_noise.line_indicates_driver_error(mapping))

    def test_firmware_bug_warning_is_benign(self):
        line = (
            '[Firmware Bug]: cpu 0, try to use APIC520 (LVT offset 2) for vector 0xf4, '
            'but the register is already in use for vector 0x0 on this cpu'
        )
        self.assertTrue(dmesg_noise.dmesg_line_is_benign(line))
        self.assertFalse(dmesg_noise.line_indicates_driver_error(line))

    def test_real_driver_error_and_kernel_bug_stay_failures(self):
        self.assertTrue(dmesg_noise.line_indicates_driver_error('amdgpu: ring timeout error'))
        self.assertFalse(dmesg_noise.dmesg_line_is_benign('kernel BUG: unable to handle'))
        combined = 'amdgpu: page fault WALKER_ERROR: 0x0'
        self.assertFalse(dmesg_noise.dmesg_line_is_benign(combined))


if __name__ == '__main__':
    unittest.main()
