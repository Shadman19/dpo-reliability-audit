"""Explore threshold-selection subsets using archived CPU safety scores.
Requires NumPy plus the repository CPU dependencies. No network or model fitting.
Dataset-derived numerical output retains ToxicChat CC BY-NC 4.0 attribution.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(repo, out, check=None):
    sys.path.insert(0,str(repo.resolve()))
    from safety_metrics import choose_fpr_threshold
    root=repo/'results/reproduction-cpu-20261111'
    report=json.loads((root/'report.json').read_text())
    path=root/'tfidf_lr_predictions.jsonl'
    expected=report['judges']['tfidf_lr']['predictions']['sha256']
    if digest(path)!=expected: raise ValueError('Prediction archive does not match recorded report')
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    require(len({r['id'] for r in rows})==len(rows), 'Duplicate evidence row IDs')
    cal=[r for r in rows if r['split']=='calibration' and r['y']==0]
    test=[r for r in rows if r['split']=='test'];neg=[r for r in test if r['y']==0];jb=[r for r in test if r['jailbreak']==1]
    require(all(r['y']==1 for r in jb) and len(jb)==91, 'Expected 91 all-positive jailbreak rows')
    require(len(cal)>=1000 and bool(neg), 'Insufficient calibration or test negatives')
    score=lambda r:r['probabilities']['platt']
    cp=np.array([score(r) for r in cal]);np_=np.array([score(r) for r in neg]);jp=np.array([score(r) for r in jb])
    def evaluate(indices, seed, budget):
        subset=cp[indices];threshold=choose_fpr_threshold(subset,0.05)
        # Separate pure-Python arithmetic: all score ties on boundary are excluded.
        boundary=sorted([score(cal[int(i)]) for i in indices],reverse=True)[math.floor(0.05*len(indices))]
        require(all(bool(v>=threshold)==bool(v>boundary) for v in cp), 'Threshold boundary decisions disagree')
        cfp=int((subset>=threshold).sum());fp=int((np_>=threshold).sum());miss=int((jp<threshold).sum())
        require(cfp<=math.floor(.05*len(indices)), 'Subset calibration FPR cap violated')
        require(fp==sum(score(r)>=threshold for r in neg), 'Independent false-positive count disagrees')
        require(miss==sum(score(r)<threshold for r in jb), 'Independent jailbreak-miss count disagrees')
        return dict(seed=seed,budget=budget,threshold=float(threshold),subset_false_positives=cfp,subset_empirical_fpr=cfp/len(indices),test_false_positives=fp,test_negatives=len(neg),test_fpr=fp/len(neg),jailbreak_misses=miss,jailbreak_positives=len(jb),jailbreak_fnr=miss/len(jb),selected_ids_sha256=hashlib.sha256('\n'.join(cal[int(i)]['id'] for i in indices).encode()).hexdigest())
    runs=[]
    for seed in range(40):
        perm=np.random.Generator(np.random.PCG64(seed)).permutation(len(cal))
        for budget in [100,500,1000]:runs.append(evaluate(perm[:budget],seed,budget))
    reference=evaluate(np.arange(len(cal)),None,len(cal))
    summary=[]
    for budget in [100,500,1000]:
        group=[r for r in runs if r['budget']==budget]; item={'budget':budget,'runs':len(group)}
        for key in ['threshold','test_fpr','jailbreak_misses']:
            vals=[r[key] for r in group];item[key]={'min':min(vals),'median':float(np.median(vals)),'max':max(vals)}
        item['test_fpr_above_005_runs']=sum(r['test_fpr']>.05 for r in group);summary.append(item)
    output={'source_predictions_sha256':digest(path),'source_report_sha256':digest(root/'report.json'),'threshold_implementation_sha256':digest(repo/'safety_metrics.py'),'numpy_version':np.__version__,'calibration_negatives_available':len(cal),'test_rows':len(test),'test_negatives':len(neg),'test_jailbreak_positives':len(jb),'reference_all_negatives':reference,'summary':summary,'runs':runs,'checks':{'all_120_subset_fpr_bounds':True,'all_121_independent_count_checks':True,'all_121_boundary_decision_checks':True,'unique_row_ids':True},'scope':'Fixed saved scores; exploratory threshold-subset reanalysis of an already inspected, reused test set. No fitting, new model, independent replications or population FPR guarantee. Original Platt calibration uses all calibration rows. Nested subsets and known text overlap limit interpretation.'}
    if check is not None:
        expected = json.loads(check.read_text())
        # Exact canonical JSON comparison is for the pinned environment, not authenticity.
        if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(output, sort_keys=True, allow_nan=False):
            raise ValueError('Recorded threshold-subset result differs')
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['runs']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--check',type=Path);a=p.parse_args();run(a.repo,a.out,a.check)
