import unittest
import numpy as np
import pandas as pd
from .pre_split import partition_rows, summarize_split


class ChronologicalSplitTests(unittest.TestCase):
    def test_boundary_ties_and_membership_are_preserved(self):
        frame = pd.DataFrame(dict(
            timestamp=pd.to_datetime(['2018-02-20 10:29:59','2018-02-20 10:30:00',
                                      '2018-02-20 10:30:00','2018-02-20 11:00:00']),
            target_binary=[0,1,0,1], source_row=[40,12,98,3]))
        bounds = ['2018-02-20 10:30:00','2018-02-20 11:00:00']
        np.testing.assert_array_equal(partition_rows(frame,bounds), [0,1,1,2])
        rows = summarize_split(frame,bounds)
        self.assertEqual(sum(r['rows'] for r in rows),len(frame))
        self.assertEqual([r['malicious'] for r in rows],[0,1,1])
        changed_labels = frame.assign(target_binary=1-frame.target_binary)
        self.assertEqual([r['source_row_sequence_sha256'] for r in rows],
                         [r['source_row_sequence_sha256'] for r in summarize_split(changed_labels,bounds)])


if __name__ == '__main__':
    unittest.main()
