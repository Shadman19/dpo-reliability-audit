"""Synthetic end-to-end regression checks for the historical aggregator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AggregatePairingTests(unittest.TestCase):
    def run_fixture(self, folder, drop=False, baseline_mode='normal'):
        baseline = [{'prompt':p, 'rm':n, 'len':j+1}
                    for j, (p,n) in enumerate([('a',0),('b',2),('c',4)])]
        candidate = [{'prompt':p, 'rm':n, 'len':j+2}
                     for j, (p,n) in enumerate([('c',5),('a',1),('b',3)])]
        if drop:
            candidate.pop()
        fixtures = [('beta0_s0',0,baseline),('beta0.1_s0',0.1,candidate)]
        if baseline_mode == 'renamed':
            fixtures[0] = ('control',0,baseline)
        elif baseline_mode == 'missing':
            fixtures = fixtures[1:]
        elif baseline_mode == 'multiple':
            fixtures.append(('second_control',0,baseline))
        for run, beta, records in fixtures:
            path = Path(folder)/'runs'/run
            path.mkdir(parents=True)
            result = {'config':{'beta':beta,'seed':0},
                      'gen':{'kl_mean':0,'rm_mean':0,'len_mean':2,'hit_max_len_frac':0},
                      'pref':{'lennorm_logp_acc':0.5,'implicit_reward_acc':0.5},
                      'minutes':0,'gpu':'SYNTHETIC'}
            (path/'result.json').write_text(json.dumps(result))
            (path/'generations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        return subprocess.run([sys.executable,str(ROOT/'aggregate.py')],cwd=folder,
                              capture_output=True,text=True)

    def test_reordered_fixture_produces_correct_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            completed = self.run_fixture(folder)
            self.assertEqual(completed.returncode,0,completed.stderr)
            rows = json.loads((Path(folder)/'summary.json').read_text())['rows']
            candidate = next(r for r in rows if r['beta']>0)
            self.assertEqual(candidate['rm_winrate_vs_base'],1.0)
            self.assertTrue((Path(folder)/'dpo_audit.png').is_file())

    def test_configured_baseline_need_not_have_beta0_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            completed = self.run_fixture(folder,baseline_mode='renamed')
            self.assertEqual(completed.returncode,0,completed.stderr)
            rows = json.loads((Path(folder)/'summary.json').read_text())['rows']
            self.assertEqual(next(r for r in rows if r['beta']>0)['rm_winrate_vs_base'],1)

    def test_missing_or_multiple_controls_preserve_summary(self):
        for mode in ['missing','multiple']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                summary = Path(folder)/'summary.json'
                summary.write_text('PRESERVE EXISTING SUMMARY')
                completed = self.run_fixture(folder,baseline_mode=mode)
                self.assertNotEqual(completed.returncode,0)
                self.assertIn('require exactly one beta-zero baseline',completed.stderr)
                self.assertEqual(summary.read_text(),'PRESERVE EXISTING SUMMARY')
                self.assertFalse((Path(folder)/'dpo_audit.png').exists())

    def test_incomplete_fixture_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = Path(folder)/'summary.json'
            summary.write_text('PRESERVE EXISTING SUMMARY')
            completed = self.run_fixture(folder,drop=True)
            self.assertNotEqual(completed.returncode,0)
            self.assertIn('prompt sets differ',completed.stderr)
            self.assertEqual(summary.read_text(),'PRESERVE EXISTING SUMMARY')
            self.assertFalse((Path(folder)/'dpo_audit.png').exists())


if __name__ == '__main__':
    unittest.main()
