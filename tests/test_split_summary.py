"""The recorded summary must agree with score-derived results before success."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_split_sensitivity as audit


class RecordedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.actual = json.loads(audit.RECORDED.read_text())

    def check(self, expected):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            path.write_text(json.dumps(expected))
            audit.verify_recorded_summary(self.actual, path)

    def test_archive_and_reordered_seed_rows_match(self):
        self.check(self.actual)
        expected = copy.deepcopy(self.actual)
        expected['results'].reverse()
        self.check(expected)

    def test_float_tolerance_and_real_metric_change(self):
        expected = copy.deepcopy(self.actual)
        expected['results'][0]['all_ece'] += 5e-11
        self.check(expected)
        expected['results'][0]['all_ece'] += 1e-6
        with self.assertRaisesRegex(ValueError, 'recorded metric differs'):
            self.check(expected)

    def test_counts_are_exact_and_booleans_are_not_counts(self):
        for value in (92, 91.0, True):
            with self.subTest(value=value):
                expected = copy.deepcopy(self.actual)
                expected['results'][0]['jailbreak_n'] = value
                with self.assertRaises(ValueError):
                    self.check(expected)

    def test_missing_extra_duplicate_and_boolean_seeds_fail(self):
        for action in ('missing', 'extra', 'duplicate', 'boolean'):
            with self.subTest(action=action):
                expected = copy.deepcopy(self.actual)
                if action == 'missing': expected['results'].pop()
                if action == 'extra': expected['results'].append({'seed': 4})
                if action == 'duplicate': expected['results'][1]['seed'] = 0
                if action == 'boolean': expected['results'][0]['seed'] = False
                with self.assertRaises(ValueError): self.check(expected)

    def test_missing_metric_and_changed_design_fail(self):
        expected = copy.deepcopy(self.actual)
        del expected['results'][0]['all_ece']
        with self.assertRaises(ValueError): self.check(expected)
        expected = copy.deepcopy(self.actual)
        expected['existing_comparator'] = 2
        with self.assertRaises(ValueError): self.check(expected)
        expected = copy.deepcopy(self.actual)
        expected['planned_new_seeds'] = [True, 2, 3]
        with self.assertRaises(ValueError): self.check(expected)

    def test_ambiguous_and_nonfinite_json_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            for payload in ('{"a":1,"a":2}', '{"a":1,"\\u0061":2}', '{"a":NaN}'):
                path.write_text(payload)
                with self.assertRaises(ValueError): audit.verify_recorded_summary(self.actual, path)
        expected = copy.deepcopy(self.actual)
        expected['results'][0]['all_ece'] = 1e309
        with self.assertRaises(ValueError): self.check(expected)
        expected['results'][0]['all_ece'] = 10**1000
        with self.assertRaises(ValueError): self.check(expected)

    def test_cli_mismatch_has_no_success_stdout(self):
        with patch.object(audit, 'summarize', return_value=self.actual), \
             patch.object(audit, 'verify_recorded_summary', side_effect=ValueError('mismatch')), \
             patch('sys.argv', ['verify_split_sensitivity.py']), \
             patch('sys.stdout') as stdout, patch('sys.stderr'):
            with self.assertRaises(SystemExit) as result:
                audit.main()
            self.assertEqual(result.exception.code, 2)
            stdout.write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
