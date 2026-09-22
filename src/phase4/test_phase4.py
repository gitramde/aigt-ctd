"""Small independent checks of evaluation arithmetic and data isolation."""
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from .metrics import binary_metrics,family_diagnostics
from .train import Selection

class MetricsTests(unittest.TestCase):
    def test_metrics_against_reference(self):
        y=np.array([0,0,0,1,1,1]);p=np.array([.01,.7,.4,.2,.9,.8]);pred=p>=.5
        result=binary_metrics(y,p)
        for key,reference in [('accuracy',accuracy_score(y,pred)),('precision',precision_score(y,pred)),
                              ('recall',recall_score(y,pred)),('f1',f1_score(y,pred)),
                              ('macro_f1',f1_score(y,pred,average='macro')),('roc_auc',roc_auc_score(y,p)),
                              ('average_precision',average_precision_score(y,p))]:
            self.assertAlmostEqual(result[key],reference)
        self.assertEqual((result['true_negatives'],result['false_positives'],result['false_negatives'],result['true_positives']),(2,1,1,2))
        self.assertAlmostEqual(result['benign_recall_specificity'],2/3)
    def test_no_positive_predictions(self):
        r=binary_metrics([0,1],[.1,.2]);self.assertEqual(r['precision'],0);self.assertEqual(r['false_negative_rate'],1)
    def test_families(self):
        r=family_diagnostics([0,1,1,2],['Benign','Known','Unseen','Absent'],[.8,.1,.9,.4],{'Known':10})
        self.assertEqual(r[0]['training_status'],'KNOWN');self.assertEqual(r[0]['binary_detection_recall'],.5)
        self.assertEqual(r[1]['training_status'],'UNSEEN');self.assertEqual(r[1]['missed'],1)
        self.assertIsNone(r[2]['binary_detection_recall'])
    def test_selection_accesses_validation_only(self):
        cfg={'threshold':.5,'minimum_improvement':.00001,'early_stopping_patience':2}
        with patch('src.phase4.train.diagnostic_labels',return_value=(np.array([0,1]),None,None)) as labels, \
             patch('src.phase4.train.predict_partition',return_value=(np.array([.1,.9]),{})) as predict, \
             patch('src.phase4.train.save_model'),patch('src.phase4.train.json_write'):
            selection=Selection('logistic_regression',cfg);selection.check(object(),1)
            labels.assert_called_once_with('validation');self.assertEqual(predict.call_args.args[2],'validation')

if __name__=='__main__': unittest.main()
