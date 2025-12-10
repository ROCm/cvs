# cvs/lib/unittests/test_ibperf_lib.py
import unittest
from unittest.mock import patch, MagicMock
import cvs.lib.ibperf_lib as ibperf_lib


class TestIbperfLib(unittest.TestCase):
    @patch('xlsxwriter.Workbook')
    def test_generate_ibperf_bw_chart(self, mock_workbook_class):
        mock_workbook = MagicMock()
        mock_workbook_class.return_value = mock_workbook
        mock_worksheet = MagicMock()
        mock_workbook.add_worksheet.return_value = mock_worksheet

        res_dict = {
            'ib_write_bw': {
                1024: {1: {'node1': {i: {'pps': str(10.0 + i), 'bw': str(1.0 + i * 0.1)} for i in range(8)}}}
            }
        }
        ibperf_lib.generate_ibperf_bw_chart(res_dict, 'test.xlsx')
        self.assertTrue(mock_workbook.add_worksheet.called)
        self.assertTrue(mock_workbook.close.called)

    @patch('xlsxwriter.Workbook')
    def test_generate_ibperf_lat_chart(self, mock_workbook_class):
        mock_workbook = MagicMock()
        mock_workbook_class.return_value = mock_workbook
        mock_worksheet = MagicMock()
        mock_workbook.add_worksheet.return_value = mock_worksheet

        res_dict = {
            'ib_write_lat': {
                1024: {
                    'node1': {
                        i: {
                            't_min': str(1.0 + i * 0.1),
                            't_max': str(2.0 + i * 0.1),
                            't_avg': str(1.5 + i * 0.1),
                            't_stdev': str(0.1 + i * 0.01),
                            't_99_pct': str(1.9 + i * 0.1),
                        }
                        for i in range(8)
                    }
                }
            }
        }
        ibperf_lib.generate_ibperf_lat_chart(res_dict, 'test.xlsx')
        self.assertTrue(mock_workbook.add_worksheet.called)
        self.assertTrue(mock_workbook.close.called)


if __name__ == '__main__':
    unittest.main()
