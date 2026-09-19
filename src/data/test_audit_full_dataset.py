"""Audit correctness checks using synthetic records; never changes source data."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_full_dataset import audit_file


class AuditTests(unittest.TestCase):
    def test_full_scan_and_duplicate_collision_verification(self):
        columns = ['Src IP', 'Dst IP', 'Timestamp', 'Flow Byts/s', 'TotLen Fwd Pkts', 'Label', 'Constant']
        rows = [
            ['a', 'b', '02/03/2018 08:00:00', 'NaN', '10', 'Benign', '0'],
            ['a', 'b', '02/03/2018 08:00:00', 'NaN', '10', 'Benign', '0'],
            ['b', 'a', '01/03/2018 09:00:00', 'Infinity', '20.5', 'Bot', '0'],
            ['b', 'a', 'bad', '', 'NULL', '', '0'],
            columns,
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
            for collision in [False, True]:
                context = patch('audit_full_dataset.fingerprint', return_value=b'collision') if collision else patch('audit_full_dataset.fingerprint', wraps=__import__('audit_full_dataset').fingerprint)
                with context:
                    summary, classes, inventory, quality = audit_file(path, 2)
                self.assertEqual(summary['row_count'], 5)
                self.assertEqual(summary['duplicate_records'], 1)
                self.assertEqual(summary['repeated_header_records'], 1)
                self.assertEqual(summary['benign_count'], 2)
                self.assertEqual(summary['malicious_count'], 1)
                self.assertEqual(summary['unclassified_count'], 2)
                self.assertEqual(summary['timestamp_min'], '2018-03-01T09:00:00')
                self.assertEqual(summary['timestamp_max'], '2018-03-02T08:00:00')
                self.assertEqual(summary['invalid_timestamp_values'], 2)
                self.assertEqual(summary['nan_values'], 2)
                self.assertEqual(summary['missing_values'], 2)
                self.assertEqual(summary['null_marker_values'], 1)
                self.assertEqual(summary['infinite_values'], 1)
                self.assertTrue(summary['has_byte_counts'])
                self.assertTrue(summary['temporal_endpoint_graph_columns_sufficient'])
                constant = next(r for r in inventory if r['column_name'] == 'Constant')
                self.assertFalse(constant['is_constant'])
                self.assertTrue(constant['constant_non_missing_excluding_header_tokens'])
                self.assertEqual(constant['data_type'], 'mixed numeric/string')
            expected = (summary, classes, inventory, quality)
            with patch('audit_full_dataset.np.asarray', side_effect=ValueError('force scalar inspection')):
                self.assertEqual(audit_file(path, 2), expected)


if __name__ == '__main__':
    unittest.main()
