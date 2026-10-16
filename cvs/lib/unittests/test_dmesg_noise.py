'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest

from cvs.lib import dmesg_noise
from cvs.lib.unittests import mce_dmesg_fixtures as mce


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


class TestDropCorrectedMceBlocks(unittest.TestCase):
    def test_real_corrected_block_is_dropped(self):
        kept, rows = dmesg_noise.drop_corrected_mce_blocks(
            mce.text([mce.UNRELATED_LINE] + mce.CE_LINES + [mce.UNRELATED_LINE])
        )
        self.assertEqual(kept, f'{mce.UNRELATED_LINE}\n{mce.UNRELATED_LINE}')
        self.assertEqual(rows, [mce.CE_LINES[2].strip()])

    def test_prefixed_empty_message_inside_block_is_dropped(self):
        kept, rows = dmesg_noise.drop_corrected_mce_blocks(mce.text(mce.CE_LINES_PREFIXED_EMPTY))
        self.assertEqual(kept, '')
        self.assertEqual(rows, [mce.CE_LINES[2].strip()])

    def test_empty_kernel_line_regex(self):
        for line in (
            '',
            mce.CE_LINES_PREFIXED_EMPTY[6],
            '[Thu Oct 15 06:24:34 2026] ',
            '2026-10-15T06:24:34+0000 node1 kernel: ',
            'Oct 15 06:24:34 node1 kernel: ',
        ):
            with self.subTest(line=line):
                self.assertIsNotNone(dmesg_noise._EMPTY_KERNEL_LINE_RE.match(line))
        for line in mce.CE_LINES:
            if line:
                self.assertIsNone(dmesg_noise._EMPTY_KERNEL_LINE_RE.match(line))

    def test_uncorrected_blocks_are_kept(self):
        for lines in (mce.UE_LINES, mce.FATAL_LINES, mce.DEFERRED_LINES):
            with self.subTest(header=lines[1]):
                source = mce.text(lines)
                self.assertEqual(dmesg_noise.drop_corrected_mce_blocks(source), (source, []))

    def test_mixed_blocks_keep_uncorrected_and_deferred(self):
        for lines, remaining in (
            (mce.MIXED_CE_UE, mce.UE_LINES),
            (mce.MIXED_CE_DEFERRED, mce.DEFERRED_LINES),
            (mce.MIXED_UE_CE, mce.UE_LINES),
        ):
            with self.subTest(header=remaining[1]):
                kept, rows = dmesg_noise.drop_corrected_mce_blocks(mce.text(lines))
                self.assertEqual(kept, '\n'.join(remaining))
                self.assertEqual(rows, [mce.CE_LINES[2].strip()])

    def test_corrected_header_with_uncorrected_flag_is_kept(self):
        flags = (
            'Over|UE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-',
            'Over|UC|MiscV|AddrV|-|-|SyndV|CECC|-|-|-',
            'Over|CE|MiscV|AddrV|PCC|-|SyndV|CECC|-|-|-',
            'Over|CE|MiscV|AddrV|-|TCC|SyndV|CECC|-|-|-',
            'Over|CE|MiscV|AddrV|-|-|SyndV|UECC|-|-|-',
            'Over|CE|MiscV|AddrV|-|-|SyndV|CECC|Deferred|-|-',
            'Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|Poison|-',
        )
        for field in flags:
            with self.subTest(flags=field):
                status = mce.CE_STATUS.replace('Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-', field)
                source = mce.text(mce.derive(mce.CE_HEADER, status))
                self.assertEqual(dmesg_noise.drop_corrected_mce_blocks(source), (source, []))

    def test_block_stops_at_ghes_line(self):
        for lines in (
            mce.CE_LINES[:3] + [mce.GHES_LINE] + mce.CE_LINES[3:],
            mce.CE_LINES + [mce.GHES_LINE],
        ):
            with self.subTest(placement=lines.index(mce.GHES_LINE)):
                kept, _ = dmesg_noise.drop_corrected_mce_blocks(mce.text(lines))
                self.assertIn(mce.GHES_LINE, kept)
                if lines.index(mce.GHES_LINE) == 3:
                    for line in mce.CE_LINES[3:]:
                        if line:
                            self.assertIn(line, kept)

    def test_block_stops_at_other_mce_and_edac_lines(self):
        for report in (mce.EDAC_CE_LINE, mce.MCE_CORE_LINE):
            with self.subTest(report=report):
                lines = mce.CE_LINES[:3] + [report] + mce.CE_LINES[3:]
                kept, _ = dmesg_noise.drop_corrected_mce_blocks(mce.text(lines))
                self.assertIn(report, kept)
                for line in mce.CE_LINES[3:]:
                    if line:
                        self.assertIn(line, kept)

    def test_block_needs_one_leading_status_row(self):
        lines = mce.CE_LINES
        variants = (
            lines[:2] + lines[3:],
            lines[:2] + [lines[3], lines[2]] + lines[4:],
            lines[:3] + [lines[2]] + lines[3:],
        )
        for variant in variants:
            with self.subTest(variant=variant[:4]):
                source = mce.text(variant)
                self.assertEqual(dmesg_noise.drop_corrected_mce_blocks(source), (source, []))

    def test_interleaved_unrelated_line_is_kept(self):
        lines = mce.CE_LINES[:3] + [mce.WORKQUEUE_LINE] + mce.CE_LINES[3:]
        self.assertEqual(
            dmesg_noise.drop_corrected_mce_blocks(mce.text(lines)), (mce.WORKQUEUE_LINE, [mce.CE_LINES[2].strip()])
        )

    def test_notifier_handling(self):
        notice = mce.CE_LINES[0]
        kept, rows = dmesg_noise.drop_corrected_mce_blocks(mce.text([notice, mce.UNRELATED_LINE]))
        self.assertEqual((kept, rows), (mce.text([notice, mce.UNRELATED_LINE]), []))
        kept, _ = dmesg_noise.drop_corrected_mce_blocks(mce.text([notice] + mce.CE_LINES))
        self.assertEqual(kept, notice)
        kept, _ = dmesg_noise.drop_corrected_mce_blocks(mce.text([notice, mce.GHES_LINE] + mce.CE_LINES[1:]))
        self.assertIn(notice, kept)
        self.assertIn(mce.GHES_LINE, kept)

    def test_block_is_bounded(self):
        lines = mce.CE_LINES[1:3] + 20 * [mce.UNRELATED_LINE] + [mce.CE_LINES[3]]
        kept, _ = dmesg_noise.drop_corrected_mce_blocks(mce.text(lines))
        self.assertIn(mce.CE_LINES[3], kept)

    def test_other_log_formats(self):
        for formatter in (mce.as_dmesg_T, mce.as_journal_short_iso, mce.as_journal_short):
            with self.subTest(formatter=formatter.__name__):
                kept, rows = dmesg_noise.drop_corrected_mce_blocks(mce.text(formatter(mce.CE_LINES)))
                self.assertNotIn('Hardware Error', kept)
                self.assertEqual(len(rows), 1)

    def test_empty_and_clean_input(self):
        self.assertEqual(dmesg_noise.drop_corrected_mce_blocks(None), ('', []))
        self.assertEqual(dmesg_noise.drop_corrected_mce_blocks(''), ('', []))
        self.assertIs(dmesg_noise.drop_corrected_mce_blocks(mce.UNRELATED_LINE)[0], mce.UNRELATED_LINE)


class TestUncorrectedMceStatusLines(unittest.TestCase):
    def test_status_rows(self):
        self.assertEqual(dmesg_noise.uncorrected_mce_status_lines(mce.text(mce.CE_LINES)), [])
        for lines in (mce.UE_LINES, mce.FATAL_LINES, mce.DEFERRED_LINES):
            with self.subTest(header=lines[1]):
                self.assertEqual(dmesg_noise.uncorrected_mce_status_lines(mce.text(lines)), [lines[2].strip()])
        self.assertEqual(
            dmesg_noise.uncorrected_mce_status_lines(mce.text(mce.MIXED_CE_DEFERRED)),
            [mce.DEFERRED_LINES[2].strip()],
        )
        tcc_status = mce.CE_STATUS.replace('|-|-|SyndV|', '|-|TCC|SyndV|')
        self.assertEqual(
            dmesg_noise.uncorrected_mce_status_lines(mce.text(mce.derive(mce.CE_HEADER, tcc_status))),
            [mce.derive(mce.CE_HEADER, tcc_status)[2].strip()],
        )
        self.assertEqual(dmesg_noise.uncorrected_mce_status_lines(None), [])


if __name__ == '__main__':
    unittest.main()
