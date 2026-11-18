"""Verify the four archived CPU fits without pooling dependent test rows."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from verify_safety_evidence import verify

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'runs/sensitivity'
RECORDED = ROOT / 'results/split-sensitivity-cpu-20261114/summary.json'


def verify_recorded_summary(actual, path):
    """Reconcile recorded scalar evidence, not free-text dates or conclusions."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate summary field: {key}')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f'nonfinite summary constant: {value}')

    expected = json.loads(Path(path).read_text(), object_pairs_hook=unique_fields,
                          parse_constant=reject_constant)

    def compare(a, b, field):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError(f'{field}: summary fields differ')
            for key in a:
                compare(a[key], b[key], f'{field}.{key}')
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise ValueError(f'{field}: summary list differs')
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f'{field}[{i}]')
        elif type(a) is float:
            try:
                valid = type(b) in (int, float) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f'{field}: recorded metric differs')
        elif type(a) is not type(b) or a != b:
            raise ValueError(f'{field}: recorded value differs')

    for key in ('planned_new_seeds', 'existing_comparator'):
        compare(actual[key], expected[key], key)

    def by_seed(rows):
        if not isinstance(rows, list):
            raise ValueError('summary results must be a list')
        indexed = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get('seed')) is not int or row['seed'] in indexed:
                raise ValueError('invalid or duplicate summary seed')
            indexed[row['seed']] = row
        if set(indexed) != {0, 1, 2, 3}:
            raise ValueError('summary must contain seeds 0, 1, 2, 3 exactly once')
        return indexed

    left, right = by_seed(actual['results']), by_seed(expected['results'])
    for seed in left:
        compare(left[seed], right[seed], f'seed-{seed}')


def summarize(archive=ARCHIVE, comparator=ROOT / 'results/reproduction-cpu-20261111'):
    paths = {0: Path(comparator), **{s: Path(archive) / f'seed-{s}' for s in (1, 2, 3)}}
    reference_ids = None
    reference_design = None
    results = []
    for seed, path in paths.items():
        checks = verify(path)
        report = json.loads((path / 'report.json').read_text())
        if report['split']['seed'] != seed:
            raise ValueError(f'seed-{seed}: seed metadata mismatch')
        design = {k: report[k] for k in ('dataset', 'configuration', 'source_sha256')}
        if reference_design is None:
            reference_design = design
        if design != reference_design:
            raise ValueError(f'seed-{seed}: experimental configuration differs')
        judge = report['judges']['tfidf_lr']
        rows = [json.loads(x) for x in (path / judge['predictions']['file']).read_text().splitlines()]
        test = [r for r in rows if r['split'] == 'test']
        identities = [(r['id'], r['y'], r['jailbreak']) for r in test]
        if reference_ids is None:
            reference_ids = identities
        if identities != reference_ids:
            raise ValueError(f'seed-{seed}: test identities or labels differ')
        p = np.array([r['probabilities']['platt'] for r in test])
        y = np.array([r['y'] for r in test])
        metrics = judge['variants']['platt']
        # Independently check scalar summaries using sklearn and a direct
        # fixed-bin sum. ECE uses the archive's 15 right-closed bins.
        bins = np.searchsorted(np.linspace(0, 1, 16)[1:], p, side='left')
        ece = sum(abs(float(np.sum(p[bins == i] - y[bins == i]))) / len(y) for i in range(15))
        independent = {'ece': ece, 'auroc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p)}
        for key, value in independent.items():
            if not np.isclose(value, metrics['all'][key], rtol=0, atol=1e-12):
                raise ValueError(f'seed-{seed}: independent {key} differs')
        results.append({
            'seed': seed, 'verified': checks,
            'all_ece': metrics['all']['ece'], 'all_auroc': metrics['all']['auroc'],
            'all_brier': metrics['all']['brier'],
            'test_fpr': metrics['all']['fpr_at_cal_thr'],
            'jailbreak_ece': metrics['jailbreak']['ece'],
            'jailbreak_missed': metrics['jailbreak']['false_negatives'],
            'jailbreak_n': metrics['jailbreak']['n'],
            'no_overlap_ece': metrics['test_without_train_text_overlap']['ece'],
            'no_overlap_jailbreak_ece': metrics['jailbreak_without_train_text_overlap']['ece'],
            'threshold': metrics['threshold'],
            'calibration_fpr': metrics['calibration_empirical_fpr'],
        })
    return {
        'new_run_date': '2026-11-14', 'planned_new_seeds': [1, 2, 3],
        'existing_comparator': 0, 'results': results,
        'limits': 'Four fits on one inspected test dataset; ranges are descriptive, not independent replications. No best-seed selection.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--comparator', type=Path, default=ROOT / 'results/reproduction-cpu-20261111')
    parser.add_argument('--expected-summary', type=Path, default=RECORDED)
    args = parser.parse_args()
    try:
        result = summarize(args.archive, args.comparator)
        verify_recorded_summary(result, args.expected_summary)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'split sensitivity verification: {exc}\n')


if __name__ == '__main__':
    main()
