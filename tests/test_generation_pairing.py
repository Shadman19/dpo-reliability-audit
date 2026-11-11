import json
from pathlib import Path
import tempfile
import unittest

from generation_pairing import load_generations, select_baseline, winrate_vs_base


class GenerationPairingTests(unittest.TestCase):
    def test_baseline_selected_from_configuration_not_filename(self):
        runs = {'beta0_misleading': {'config': {'beta': 0.1}},
                'arbitrary-control-name': {'config': {'beta': 0}}}
        self.assertEqual(select_baseline(runs), 'arbitrary-control-name')

    def test_missing_multiple_or_invalid_baseline_design_fails(self):
        for runs in [{}, {'tuned':{'config':{'beta':0.1}}},
                     {'a':{'config':{'beta':0}},'b':{'config':{'beta':0}}},
                     {'a':{'config':{'beta':False}}},
                     {'a':{'config':{'beta':'0'}}},
                     {'a':{'config':{'beta':float('nan')}}},
                     {'a':{'config':{'beta':-0.1}}}]:
            with self.subTest(runs=runs):
                with self.assertRaises(ValueError):
                    select_baseline(runs)

    def load(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'generations.jsonl'
            path.write_text(text, encoding='utf-8')
            return load_generations(path)

    def test_row_reordering_does_not_change_pairs(self):
        base = [{'prompt': p, 'rm': n} for p, n in [('a', 0), ('b', 2), ('c', 4)]]
        mine = [{'prompt': p, 'rm': n} for p, n in [('c', 5), ('a', 1), ('b', 3)]]
        self.assertEqual(winrate_vs_base(mine, base), 1)
        self.assertAlmostEqual(sum(m['rm'] > b['rm'] for m, b in zip(mine, base))/3, 1/3)

    def test_missing_or_extra_prompts_fail_instead_of_truncating(self):
        a = [{'prompt': 'a', 'rm': 1}]
        b = a + [{'prompt': 'b', 'rm': 2}]
        for mine, base in [(a, b), (b, a)]:
            with self.subTest(mine=mine):
                with self.assertRaisesRegex(ValueError, 'prompt sets differ'):
                    winrate_vs_base(mine, base)

    def test_ties_are_not_wins(self):
        self.assertEqual(winrate_vs_base([{'prompt': 'a', 'rm': 1}],
                                        [{'prompt': 'a', 'rm': 1}]), 0)

    def test_exact_prompt_text_not_normalized(self):
        with self.assertRaisesRegex(ValueError, 'prompt sets differ'):
            winrate_vs_base([{'prompt': 'A ', 'rm': 1}], [{'prompt': 'a', 'rm': 0}])

    def test_valid_writer_schema_and_unicode(self):
        record = {'prompt': 'প্রশ্ন', 'rm': -0.25, 'text': 'x', 'len': 3}
        self.assertEqual(self.load(json.dumps(record)+'\n'), [record])

    def test_invalid_jsonl_is_rejected(self):
        invalid = ['', '\n', '[]\n', '{}\n', '{"prompt":" ","rm":1}\n',
                   '{"prompt":"x","rm":true}\n', '{"prompt":"x","rm":"1"}\n',
                   '{"prompt":"x","rm":NaN}\n', '{"prompt":"x","rm":1e999}\n',
                   '{"prompt":"x","rm":1,"rm":2}\n',
                   '{"prompt":"x","rm":1}\n\n',
                   '{"prompt":"x","rm":1}\n{"prompt":"x","rm":2}\n']
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.load(text)

    def test_direct_pairing_rejects_empty_duplicates_and_bad_scores(self):
        good = [{'prompt': 'a', 'rm': 0}]
        for bad in [[], good + good, [{'prompt':'a','rm':True}],
                    [{'prompt':'a','rm':float('nan')}]]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    winrate_vs_base(bad, good)


if __name__ == '__main__':
    unittest.main()
