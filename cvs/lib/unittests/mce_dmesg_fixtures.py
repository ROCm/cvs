'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

CE_HEADER = 'Corrected error, no action required.'
CE_STATUS = 'MC53_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc604000003e0119'

# Real dmesg --time-format iso -x capture of a corrected AMD host-CPU L1 MCE.
# The PPIN is a synthetic placeholder for the processor serial identifier.
CE_LINES = [
    'kern  :info  : 2026-10-15T06:24:34,073039+00:00 mce: [Hardware Error]: Machine check events logged',
    'kern  :emerg : 2026-10-15T06:24:34,073045+00:00 [Hardware Error]: Corrected error, no action required.',
    'kern  :emerg : 2026-10-15T06:24:34,080230+00:00 [Hardware Error]: CPU:24 (1a:51:0) '
    'MC53_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc604000003e0119',
    'kern  :emerg : 2026-10-15T06:24:34,092598+00:00 [Hardware Error]: Error Addr: 0x0000000000000605',
    'kern  :emerg : 2026-10-15T06:24:34,101264+00:00 [Hardware Error]: PPIN: 0x0123456789abcdef',
    'kern  :emerg : 2026-10-15T06:24:34,109395+00:00 [Hardware Error]: IPID: 0x0002200103830404, Syndrome: 0x0000000012010018',
    '',
    'kern  :emerg : 2026-10-15T06:24:34,120433+00:00 [Hardware Error]: Microprocessor 5 Unit Ext. Error Code: 62',
    'kern  :emerg : 2026-10-15T06:24:34,120435+00:00 [Hardware Error]: cache level: L1, tx: GEN, mem-tx: RD',
]


def derive(header, status, lines=CE_LINES):
    '''Same real block with only the decode_error_status() header and status row changed.'''
    return [line.replace(CE_HEADER, header).replace(CE_STATUS, status) for line in lines]


UE_STATUS = 'MC53_STATUS[Over|UE|MiscV|AddrV|-|-|SyndV|UECC|-|-|-]: 0xfc602000003e0119'
FATAL_STATUS = 'MC53_STATUS[Over|UE|MiscV|AddrV|PCC|-|SyndV|UECC|-|-|-]: 0xfe602000003e0119'
DEFERRED_STATUS = 'MC53_STATUS[Over|-|MiscV|AddrV|-|-|SyndV|UECC|Deferred|-|-]: 0xdc603000003e0119'
UE_LINES = derive('Uncorrected, software containable error.', UE_STATUS)
FATAL_LINES = derive('System Fatal error.', FATAL_STATUS)
DEFERRED_LINES = derive('Deferred error, no action required.', DEFERRED_STATUS)
CE_LINES_PREFIXED_EMPTY = [line or 'kern  :emerg : 2026-10-15T06:24:34,109395+00:00 ' for line in CE_LINES]
# The same block one MCE period earlier for time-window tests.
CE_LINES_EARLIER = [line.replace('T06:24:34,', 'T06:18:47,') for line in CE_LINES]
MIXED_CE_UE = CE_LINES + UE_LINES
MIXED_CE_DEFERRED = CE_LINES + DEFERRED_LINES
MIXED_UE_CE = UE_LINES + CE_LINES
GHES_LINE = (
    'kern  :emerg : 2026-10-15T06:24:34,080230+00:00 {1}[Hardware Error]:  section_type: general processor error'
)
EDAC_CE_LINE = (
    'kern  :err   : 2026-10-15T06:24:34,080230+00:00 EDAC MC0: 1 CE on mc#0csrow#2channel#0 '
    '(csrow:2 channel:0 page:0x1234 offset:0x0 grain:64 syndrome:0x5a)'
)
# Built from the kernel's __print_mce() format with the real block's CPU, bank and status.
MCE_CORE_LINE = (
    'kern  :emerg : 2026-10-15T06:24:34,080230+00:00 '
    'mce: [Hardware Error]: CPU 24: Machine Check: 0 Bank 53: dc604000003e0119'
)
WORKQUEUE_LINE = (
    'kern  :warn  : 2026-10-15T06:24:34,080231+00:00 workqueue: mce_gen_pool_process hogged CPU '
    'for >10000us 4 times, consider switching to WQ_UNBOUND'
)
UNRELATED_LINE = (
    'kern  :info  : 2026-10-15T06:24:30,000000+00:00 infiniband rdma0: Changing to default roce '
    'traffic class DSCP 26 and SL 3'
)


def _body(line):
    return line.split('+00:00 ', 1)[1] if line else ''


def as_dmesg_T(lines):
    return [f'[Thu Oct 15 06:24:34 2026] {_body(line)}' if line else '' for line in lines]


def as_journal_short_iso(lines):
    return [f'2026-10-15T06:24:34+0000 node1 kernel: {_body(line)}' if line else '' for line in lines]


def as_journal_short(lines):
    return [f'Oct 15 06:24:34 node1 kernel: {_body(line)}' if line else '' for line in lines]


def text(lines):
    return '\n'.join(lines) + '\n'
