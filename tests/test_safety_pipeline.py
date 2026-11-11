"""Split isolation, score export and import-boundary regressions."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from safety_calib import evaluate_scores, load_split, split_membership, write_predictions


class PipelineTests(unittest.TestCase):
    def test_import_requires_no_gpu_libraries(self):
        proc = subprocess.run([sys.executable, '-c',
            'import safety_calib,sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_splits_reproducible_disjoint_and_complete(self):
        train = [{'id': f'train-{i}'} for i in range(30)]
        test = [{'id': f'test-{i}'} for i in range(4)]
        fit, cal, membership = split_membership(train, test, n_cal=10, seed=0)
        fit2, cal2, membership2 = split_membership(train, test, n_cal=10, seed=0)
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(cal, cal2)
        self.assertEqual(membership, membership2)
        self.assertFalse(set(membership['fit']) & set(membership['calibration']))
        self.assertEqual(set(membership['fit']) | set(membership['calibration']), {x['id'] for x in train})
        with self.assertRaises(ValueError):
            split_membership(train, test, n_cal=30)

    def test_test_labels_cannot_change_fitted_calibration_or_threshold(self):
        sc = np.array([.1,.2,.3,.4,.6,.7,.8,.9])
        yc = np.array([0,0,0,0,1,1,1,1])
        st = np.array([.05,.4,.5,.95])
        groups = np.array([0,1,0,1])
        args = dict(target_fpr=.25, bootstraps=5, seed=0)
        r1,p1 = evaluate_scores(sc, st, yc, np.array([0,0,1,1]), groups, **args)
        r2,p2 = evaluate_scores(sc, st, yc, np.array([1,1,0,0]), groups, **args)
        for variant in p1:
            self.assertEqual(r1[variant]['threshold'], r2[variant]['threshold'])
            for a,b in zip(p1[variant],p2[variant]):
                np.testing.assert_array_equal(a,b)
            self.assertLessEqual(r1[variant]['calibration_empirical_fpr'],.25)

    def test_prediction_export_excludes_prompts_and_refuses_overwrite(self):
        cal = [{'id': 'hashed-cal-id', 'y': 0, 'jailbreak': 0, 'text': 'DO NOT EXPORT THIS'}]
        test = [{'id': 'hashed-test-id', 'y': 1, 'jailbreak': 1, 'text': 'DO NOT EXPORT THIS EITHER'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.jsonl'
            write_predictions(path, cal, test, {'raw': (np.array([.2]), np.array([.8]))})
            text = path.read_text()
            self.assertNotIn('DO NOT EXPORT', text)
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([x['split'] for x in rows], ['calibration','test'])
            self.assertEqual(set(rows[0]), {'id','split','y','jailbreak','probabilities'})
            with self.assertRaises(FileExistsError):
                write_predictions(path,cal,test,{'raw':([.2],[.8])})

    def test_csv_loader_validates_and_pseudonymizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.csv'
            with path.open('w',newline='') as handle:
                writer = csv.DictWriter(handle,fieldnames=['conv_id','user_input','toxicity','jailbreaking'])
                writer.writeheader()
                writer.writerow({'conv_id':'source-1','user_input':'test prompt','toxicity':0,'jailbreaking':0})
            loaded=load_split(path,'train')
            self.assertEqual(len(loaded[0]['id']),64)
            self.assertNotIn('source-1',loaded[0]['id'])
            with path.open('a',newline='') as handle:
                csv.writer(handle).writerow(['source-1','duplicate prompt',1,0])
            with self.assertRaises(ValueError):
                load_split(path,'train')

    def test_csv_loader_rejects_nonbinary_labels_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'
            path.write_text('conv_id,user_input,toxicity,jailbreaking\na,prompt,2,0\n')
            with self.assertRaises(ValueError):load_split(path,'train')
            path.write_text('conv_id,toxicity\na,1\n')
            with self.assertRaises(ValueError):load_split(path,'train')


if __name__ == '__main__':
    unittest.main()
