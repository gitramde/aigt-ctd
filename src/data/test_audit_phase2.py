import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import audit_phase2 as audit


class Phase2Tests(unittest.TestCase):
    def test_exact_duplicates_anomalies_and_pair_windows(self):
        columns = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp', 'Rate', 'Label']
        rows = [
            ['a','b','1','80','6','20/02/2018 01:00:00','NaN','Benign'],
            ['a','b','1','80','6','20/02/2018 01:00:00','NaN','Benign'],
            ['a','b','2','80','6','20/02/2018 01:01:00','+Infinity','Attack'],
            ['b','a','80','2','6','20/02/2018 01:05:00','-inf','Attack'],
            ['a','b','1','80','6','05/01/1970 03:01:17','1','Benign'],
            columns,
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(columns); w.writerows(rows)
            original=p.read_bytes()
            # Force hash collisions: exact tuple verification must still be correct.
            class FakeDigest:
                def digest(self): return b'x'
            with patch.object(audit.hashlib,'blake2b',return_value=FakeDigest()):
                result=audit.audit(p)
            self.assertEqual(original,p.read_bytes())
        self.assertEqual(result['rows'],6)
        self.assertEqual(result['duplicates'][('Benign','extra_occurrences')],1)
        self.assertEqual(result['duplicates'][('Attack','extra_occurrences')],0)
        self.assertEqual(result['quality'][('Rate','nan','Benign')],2)
        self.assertEqual(result['quality'][('Rate','positive_infinity','Attack')],1)
        self.assertEqual(result['quality'][('Rate','negative_infinity','Attack')],1)
        epochs=[r for r in result['anomalies'] if r['anomaly_type']=='year_1970']
        self.assertEqual(len(epochs),1)
        self.assertEqual(epochs[0]['source_line'],6)
        self.assertEqual(epochs[0]['label'],'Benign')
        self.assertEqual(sum(r['count'] for r in result['anomalies'] if r['anomaly_type']=='unparseable_timestamp'),1)
        five=next(r for r in result['windows'] if r['window_minutes']==5 and r['window_start']=='2018-02-20T01:00:00')
        self.assertEqual(five['flows'],3)
        self.assertEqual(five['unique_directed_pairs'],1)
        self.assertEqual(five['extra_flows_same_pair'],2)
        ten=next(r for r in result['windows'] if r['window_minutes']==10 and r['window_start']=='2018-02-20T01:00:00')
        self.assertEqual(ten['unique_directed_pairs'],2)
        self.assertEqual(ten['flows'],4)


if __name__=='__main__':
    unittest.main()
