"""Recompute a CPU safety report from its saved labels/scores, without raw text."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from safety_calib import ids_digest, summary_metrics
from safety_metrics import choose_fpr_threshold


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(expected, actual, path='report'):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError(f'{path}: fields differ')
        for key in expected:
            close(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f'{path}: lengths differ')
        for i,(left,right) in enumerate(zip(expected,actual)):
            close(left,right,f'{path}[{i}]')
    elif isinstance(expected,float):
        if not np.isclose(expected,actual,rtol=1e-10,atol=1e-12):
            raise ValueError(f'{path}: values differ')
    elif expected != actual:
        raise ValueError(f'{path}: values differ')


def verify(run_dir):
    run_dir=Path(run_dir)
    report=json.loads((run_dir/'report.json').read_text())
    for name,digest in report['source_sha256'].items():
        if file_hash(Path(__file__).parent/name)!=digest:
            raise ValueError(f'{name}: executable source differs from measured snapshot')
    membership_path=run_dir/'split_membership.json'
    if file_hash(membership_path)!=report['split']['membership_file_sha256']:
        raise ValueError('membership SHA-256 differs')
    membership=json.loads(membership_path.read_text())
    for key,ids in membership.items():
        if len(ids)!=len(set(ids)) or len(ids)!=report['split']['counts'][key] or ids_digest(ids)!=report['split']['id_set_sha256'][key]:
            raise ValueError(f'{key}: membership mismatch')
    if set(membership['fit']) & set(membership['calibration']):
        raise ValueError('fit and calibration IDs overlap')
    if (set(membership['fit'])|set(membership['calibration'])) & set(membership['test']):
        raise ValueError('train and test IDs overlap')
    if not set(membership['test_without_train_text_overlap']).issubset(membership['test']):
        raise ValueError('sensitivity IDs are not a subset of test IDs')
    checks=0
    for name,judge in report['judges'].items():
        path=run_dir/judge['predictions']['file']
        if file_hash(path)!=judge['predictions']['sha256']:
            raise ValueError('predictions SHA-256 differs')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=judge['predictions']['rows']:
            raise ValueError('prediction count differs')
        if any(set(row)!={'id','split','y','jailbreak','probabilities'} for row in rows):
            raise ValueError('unexpected prediction fields')
        by_split={split:[row for row in rows if row['split']==split] for split in ('calibration','test')}
        for split,items in by_split.items():
            ids=[row['id'] for row in items]
            if ids!=membership[split]:
                raise ValueError(f'{split}: exported row order/IDs differ from membership')
        cal,test=by_split['calibration'],by_split['test']
        yc=np.array([row['y'] for row in cal])
        yt=np.array([row['y'] for row in test])
        jt=np.array([row['jailbreak'] for row in test])
        clean_ids=set(membership['test_without_train_text_overlap'])
        clean=np.array([row['id'] in clean_ids for row in test])
        masks={'all':np.ones(len(test),dtype=bool),'ordinary':jt==0,'jailbreak':jt==1,
               'test_without_train_text_overlap':clean,'jailbreak_without_train_text_overlap':clean&(jt==1)}
        for variant,metrics in judge['variants'].items():
            pc=np.array([row['probabilities'][variant] for row in cal])
            pt=np.array([row['probabilities'][variant] for row in test])
            threshold=choose_fpr_threshold(pc[yc==0],report['configuration']['target_fpr'])
            close(metrics['threshold'],threshold,'threshold')
            close(metrics['calibration_negative_count'],int((yc==0).sum()),'calibration_negative_count')
            close(metrics['calibration_false_positives'],int((pc[yc==0]>=threshold).sum()),'calibration_false_positives')
            close(metrics['calibration_empirical_fpr'],float((pc[yc==0]>=threshold).mean()),'calibration_empirical_fpr')
            for group,mask in masks.items():
                actual=summary_metrics(pt[mask],yt[mask],threshold,
                    bootstraps=report['configuration']['bootstraps'],seed=report['split']['seed'])
                close(metrics[group],actual,f'{name}.{variant}.{group}')
                checks+=1
    return {'verified_judges':len(report['judges']),'recomputed_metric_groups':checks,
            'score_rows':sum(judge['predictions']['rows'] for judge in report['judges'].values()),
            'raw_text_required':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(verify(args.run_dir)))
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(2,f'evidence verification: {exc}\n')


if __name__=='__main__':
    main()
