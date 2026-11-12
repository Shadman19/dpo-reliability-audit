"""Numeric boundary checks using the actual aggregator as a subprocess."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AggregateNumericTests(unittest.TestCase):
    def run_case(self, folder, mode, value=None):
        for name, beta, scores in [('control',0,[0,2,4]),('candidate',0.1,[1,3,5])]:
            path=Path(folder)/'runs'/name
            path.mkdir(parents=True)
            records=[{'prompt':p,'rm':s,'len':i} for i,(p,s) in enumerate(zip(['a','b','c'],scores))]
            if mode=='constant-length':
                for row in records: row['len']=3
            if mode=='constant-reward':
                for row in records: row['rm']=2 if value is None else value
            if mode=='invalid-length' and name=='candidate':
                if value is None: del records[0]['len']
                else: records[0]['len']=value
            if mode=='overflow' and name=='candidate':
                for row in records: row['rm']=1e308
            result={'config':{'beta':beta,'seed':0},
                    'gen':{'kl_mean':0,'rm_mean':2,'len_mean':1,'hit_max_len_frac':0},
                    'pref':{'lennorm_logp_acc':0.5,'implicit_reward_acc':0.5},
                    'minutes':0,'gpu':'SYNTHETIC CPU CHECK'}
            if mode=='nonfinite-summary' and name=='candidate': result['gen']['rm_mean']=value
            (path/'result.json').write_text(json.dumps(result))
            (path/'generations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        for name in ['summary.json','dpo_audit.png']:
            (Path(folder)/name).write_bytes(b'PRESERVE EXISTING EVIDENCE')
        return subprocess.run([sys.executable,str(ROOT/'aggregate.py')],cwd=folder,capture_output=True,text=True)

    def assert_preserved(self, folder, completed):
        self.assertNotEqual(completed.returncode,0)
        for name in ['summary.json','dpo_audit.png']:
            self.assertEqual((Path(folder)/name).read_bytes(),b'PRESERVE EXISTING EVIDENCE')

    def test_invalid_lengths_preserve_both_outputs(self):
        for value in [None,True,'3',3.0,-1,float('nan'),float('inf'),-float('inf'),10**400]:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as folder:
                self.assert_preserved(folder,self.run_case(folder,'invalid-length',value))

    def test_constant_lengths_fail_before_replacing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            completed=self.run_case(folder,'constant-length')
            self.assert_preserved(folder,completed)
            self.assertIn('at least two distinct lengths',completed.stderr)

    def test_constant_reward_has_zero_adjustment_and_explicit_null_correlation(self):
        for reward in [2,1e308]:
            with self.subTest(reward=reward),tempfile.TemporaryDirectory() as folder:
                completed=self.run_case(folder,'constant-reward',reward)
                self.assertEqual(completed.returncode,0,completed.stderr)
                result=json.loads((Path(folder)/'summary.json').read_text(),parse_constant=lambda x:self.fail(x))
                self.assertIsNone(result['rm_len_corr'])
                self.assertEqual(result['rm_vs_len_slope_per_100tok'],0)
                self.assertTrue(all(row['rm_len_adjusted']==0 for row in result['rows']))
                self.assertTrue((Path(folder)/'dpo_audit.png').read_bytes().startswith(b'\x89PNG'))

    def test_nonfinite_result_summary_preserves_both_outputs(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as folder:
                completed=self.run_case(folder,'nonfinite-summary',value)
                self.assert_preserved(folder,completed)
                self.assertIn('Out of range float values',completed.stderr)

    def test_finite_inputs_with_nonfinite_regression_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            completed=self.run_case(folder,'overflow')
            self.assert_preserved(folder,completed)
            self.assertIn('nonfinite regression statistics',completed.stderr)

    def test_zero_lengths_and_varying_lengths_produce_strict_json(self):
        with tempfile.TemporaryDirectory() as folder:
            completed=self.run_case(folder,'valid')
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads((Path(folder)/'summary.json').read_text(),parse_constant=lambda x:self.fail(x))
            self.assertAlmostEqual(result['rm_vs_len_slope_per_100tok'],200)
            self.assertEqual(next(row for row in result['rows'] if row['beta']>0)['rm_winrate_vs_base'],1)


if __name__=='__main__': unittest.main()
