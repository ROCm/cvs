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

_HW_ERR_RE = re.compile(r'\[Hardware Error\]', re.I)
# A bracketed timestamp must contain a digit so this cannot consume [Hardware Error].
_KERNEL_LINE_PREFIX_RE = re.compile(
    r'^\s*(?:(?:kern\s*:\s*[a-z]+\s*:'
    r'|\d{4}-\d\d-\d\dT[\d:.,]+(?:[+-]\d\d:?\d\d|Z)?'
    r'|[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d'
    r'|\[[^\]]*\d[^\]]*\]'
    r'|\S+\s+kernel:)\s*)*',
    re.I,
)
# dmesg -x can put a truly bare empty continuation between decoder rows.
_EMPTY_KERNEL_LINE_RE = re.compile(_KERNEL_LINE_PREFIX_RE.pattern + r'$', re.I)
_MCE_DECODER_LINE_RE = re.compile(r'\[Hardware Error\]:', re.I)
_MCE_CORRECTED_HEADER_RE = re.compile(r'\[Hardware Error\]:\s*Corrected error, no action required', re.I)
_MCE_NOTIFY_RE = re.compile(r'mce:\s*\[Hardware Error\]:\s*Machine check events logged', re.I)
_HW_ERR_RECORD_START_RE = re.compile(
    r'\[Hardware Error\]:\s*(?:Corrected error|Uncorrected|Deferred error|System Fatal error'
    r'|Machine check events logged|Machine Check|Hardware error from APEI|event severity)',
    re.I,
)
_GHES_RECORD_RE = re.compile(r'\{\d+\}\[Hardware Error\]', re.I)
_OTHER_HW_REPORTER_RE = re.compile(r'(?:mce:|EDAC\b)', re.I)
_MCE_STATUS_RE = re.compile(r'\bMC\d+_STATUS\[([^\]]*)\]', re.I)
# These flags mean the kernel did not fully correct the error; TCC means task context corrupt.
_MCE_UNCORRECTED_FLAGS = frozenset({'ue', 'uc', 'pcc', 'tcc', 'uecc', 'deferred', 'poison'})
# mce_amd normally emits about eight lines, so a bound protects later records.
_MCE_BLOCK_MAX_LINES = 16


def _message_body(line):
    '''Return the kernel message text of a dmesg/journalctl line, without its prefix.'''
    return line[_KERNEL_LINE_PREFIX_RE.match(line).end() :]


def _mce_line_kind(line):
    '''Classify a line as empty, notify, ce_header, record_start, decoder, break or other.'''
    if _EMPTY_KERNEL_LINE_RE.match(line):
        return 'empty'
    body = _message_body(line)
    if _MCE_NOTIFY_RE.match(body):
        return 'notify'
    if _MCE_CORRECTED_HEADER_RE.match(body):
        return 'ce_header'
    if _HW_ERR_RECORD_START_RE.match(body):
        return 'record_start'
    if _MCE_DECODER_LINE_RE.match(body):
        return 'decoder'
    if _GHES_RECORD_RE.search(line) or _OTHER_HW_REPORTER_RE.match(body) or _HW_ERR_RE.search(line):
        return 'break'
    return 'other'


def _mce_status_is_corrected(flags):
    '''Return True when an MCn_STATUS flag field has CE and no uncorrected flag.'''
    names = {flag.strip().lower() for flag in flags.split('|')}
    return 'ce' in names and not names & _MCE_UNCORRECTED_FLAGS


def drop_corrected_mce_blocks(text):
    '''Return (text, status_lines) after removing clean corrected host-CPU MCE blocks.

    A block needs a corrected header and exactly one leading clean CE status row.
    Only its plain decoder lines, intervening empty lines and adjacent notifier
    are removed. Other content and later hardware reports are always retained.
    '''
    if not text:
        return '', []
    if not _HW_ERR_RE.search(text):
        return text, []

    lines = text.splitlines()
    kinds = [_mce_line_kind(line) for line in lines]
    drop = set()
    status_lines = []
    pending_notice = None
    i = 0
    while i < len(lines):
        if kinds[i] == 'notify':
            pending_notice = i
            i += 1
            continue
        if kinds[i] == 'ce_header':
            absorbed = [i]
            rows = []
            first_decoder = True
            ok = True
            j = i + 1
            while j < len(lines) and j - i < _MCE_BLOCK_MAX_LINES:
                if kinds[j] in {'notify', 'ce_header', 'record_start', 'break'}:
                    break
                if kinds[j] == 'decoder':
                    match = _MCE_STATUS_RE.search(lines[j])
                    if first_decoder and not match:
                        ok = False
                    first_decoder = False
                    absorbed.append(j)
                    if match:
                        rows.append((j, match.group(1)))
                j += 1
            if ok and len(rows) == 1 and _mce_status_is_corrected(rows[0][1]):
                drop.update(absorbed)
                if pending_notice is not None:
                    drop.add(pending_notice)
                drop.update(k for k in range(i + 1, absorbed[-1]) if kinds[k] == 'empty')
                status_lines.append(lines[rows[0][0]].strip())
            pending_notice = None
            i = j
            continue
        if kinds[i] in {'record_start', 'decoder', 'break'}:
            pending_notice = None
        i += 1

    if not drop:
        return text, []
    return '\n'.join(line for k, line in enumerate(lines) if k not in drop), status_lines


def uncorrected_mce_status_lines(text):
    '''Return stripped MCn_STATUS rows that are not clean corrected errors.'''
    if not text or not _MCE_STATUS_RE.search(text):
        return []
    rows = []
    for line in text.splitlines():
        match = _MCE_STATUS_RE.search(line)
        if match and not _mce_status_is_corrected(match.group(1)):
            rows.append(line.strip())
    return rows


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
