import json
import shutil
import tempfile
import unittest
from pathlib import Path

from verify_safety_evidence import verify

RUN = Path(__file__).resolve().parents[1] / 'results/reproduction-cpu-20261111'


class RecordedEvidenceTests(unittest.TestCase):
    def test_all_recorded_metrics_recompute_from_scores_without_text(self):
        result = verify(RUN)
        self.assertEqual(result['recomputed_metric_groups'],15)
        self.assertEqual(result['score_rows'],7083)
        self.assertFalse(result['raw_text_required'])

    def test_modified_scores_fail_checksum_before_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'run'
            shutil.copytree(RUN,target)
            path=target/'tfidf_lr_predictions.jsonl'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'predictions SHA-256'):
                verify(target)

    def test_changed_report_calibration_counts_fail_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'run'
            shutil.copytree(RUN,target)
            path=target/'report.json'
            report=json.loads(path.read_text())
            report['judges']['tfidf_lr']['variants']['raw']['calibration_false_positives']+=1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'calibration_false_positives'):
                verify(target)


if __name__=='__main__':
    unittest.main()
