"""Subprocess regression checks for the published fixed-score experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / 'results/threshold-subsets-20261120/results.json'
EVIDENCE = ROOT / 'results/reproduction-cpu-20261111'


class ThresholdSubsetTests(unittest.TestCase):
    def invoke(self, folder, *, optimized=False, repo=ROOT, check=RECORDED):
        out = Path(folder) / 'output'
        out.mkdir(exist_ok=True)
        saved = out / 'results.json'
        saved.write_bytes(b'PRESERVE')
        command = [sys.executable, *(['-O'] if optimized else []),
                   str(ROOT / 'threshold_selection_sensitivity.py'),
                   '--repo', str(repo), '--out', str(out), '--check', str(check)]
        result = subprocess.run(command, capture_output=True, text=True)
        return result, saved

    def fixture(self, folder):
        repo = Path(folder) / 'fixture'
        evidence = repo / 'results/reproduction-cpu-20261111'
        evidence.mkdir(parents=True)
        shutil.copy2(ROOT / 'safety_metrics.py', repo / 'safety_metrics.py')
        for filename in ['report.json', 'tfidf_lr_predictions.jsonl']:
            shutil.copy2(EVIDENCE / filename, evidence / filename)
        return repo, evidence

    def rejected(self, result, saved, message):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn(message, result.stderr)
        self.assertEqual(saved.read_bytes(), b'PRESERVE')

    def test_valid_normal_and_optimized_equal_all_recorded_results(self):
        for optimized in [False, True]:
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as folder:
                result, saved = self.invoke(folder, optimized=optimized)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(saved.read_text()), json.loads(RECORDED.read_text()))

    def test_duplicate_ids_rejected_even_with_matching_fixture_hash(self):
        for optimized in [False, True]:
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as folder:
                repo, evidence = self.fixture(folder)
                path = evidence / 'tfidf_lr_predictions.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[1]['id'] = rows[0]['id']
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                report = json.loads((evidence / 'report.json').read_text())
                report['judges']['tfidf_lr']['predictions']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                (evidence / 'report.json').write_text(json.dumps(report))
                result, saved = self.invoke(folder, optimized=optimized, repo=repo)
                self.rejected(result, saved, 'Duplicate evidence row IDs')

    def test_changed_recorded_summary_or_run_rejected_before_output(self):
        for field in ['summary', 'runs']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                recorded = json.loads(RECORDED.read_text())
                if field == 'summary':
                    recorded['summary'][0]['test_fpr']['max'] = 0.001
                else:
                    recorded['runs'][0]['test_false_positives'] = 0
                path = Path(folder) / 'changed.json'
                path.write_text(json.dumps(recorded))
                result, saved = self.invoke(folder, check=path)
                self.rejected(result, saved, 'Recorded threshold-subset result differs')

    def test_changed_score_bytes_fail_archive_hash_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            repo, evidence = self.fixture(folder)
            path = evidence / 'tfidf_lr_predictions.jsonl'
            path.write_bytes(path.read_bytes() + b'\n')
            result, saved = self.invoke(folder, optimized=True, repo=repo)
            self.rejected(result, saved, 'Prediction archive does not match recorded report')
