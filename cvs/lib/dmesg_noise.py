'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re

# Zero-valued VM-fault status fields. The names contain ERROR or FAULT, but 0x0
# means that check did not fire. Non-zero values stay failures.
_ZERO_STATUS_RE = re.compile(
    r'(?<![A-Za-z0-9])(?:WALKER_ERROR|MAPPING_ERROR|PROTECTION_FAULT_STATUS)\s*:\s*0x0+\b',
    re.I,
)

# Any *_ERROR / *_FAIL register dumped as zero, including the names above.
_ZERO_ERROR_FIELD_RE = re.compile(
    r'\b\w*(?:error|fail)\w*\s*:\s*0x0+\b',
    re.I,
)

# VMware (and other hypervisors) log this for APIC/TSC quirks. It is a provider
# warning, not a kernel BUG.
_FIRMWARE_BUG_RE = re.compile(r'\[Firmware Bug\]:', re.I)

_RESIDUAL_SIGNAL_RE = re.compile(r'fail|error|fault|hang|reset|bug:|call trace|segfault', re.I)


def dmesg_line_is_benign(line):
    '''Return True for healthy zero status fields and hypervisor firmware warnings.'''
    text = line or ''
    if _FIRMWARE_BUG_RE.search(text):
        remainder = _FIRMWARE_BUG_RE.sub(' ', text)
        return _RESIDUAL_SIGNAL_RE.search(remainder) is None
    if not _ZERO_STATUS_RE.search(text):
        return False
    remainder = _ZERO_STATUS_RE.sub(' ', text)
    return _RESIDUAL_SIGNAL_RE.search(remainder) is None


def line_indicates_driver_error(line):
    '''Return True when a line has a real amdgpu fail/error, not a zero status field.'''
    if dmesg_line_is_benign(line):
        return False
    stripped = _ZERO_ERROR_FIELD_RE.sub(' ', line or '')
    return re.search(r'fail|error', stripped, re.I) is not None
