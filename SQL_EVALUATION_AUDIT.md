# SQL evaluation evidence audit

This extension loads the archived CPU safety evaluation into a relational SQLite
schema and independently reconstructs its point metrics in SQL. It demonstrates
data modeling, keyed joins, common-table expressions, window functions, cohort
analysis and evidence validation on an actual evaluation artifact. It does not
create a new model result or establish years of professional database experience.

## Run it

Python 3.10+ and SQLite 3.25+ are sufficient; no Python package installation, raw
prompts, network calls or GPU are needed for this extension.

```bash
python sql_eval_audit.py results/reproduction-cpu-20261111
python sql_eval_audit.py results/reproduction-cpu-20261111 --database /tmp/safety-evidence.sqlite
python -m unittest discover -s tests -p 'test_sql_audit.py' -v
```

The second command creates a new database and refuses to overwrite an existing
path. A failed import removes only the new database created by that invocation.
For example, inspect the saved database with the SQLite CLI, if installed:

```sql
SELECT variant, cohort, n, ece, auroc, false_negatives,
       fpr_at_cal_thr, fnr_at_cal_thr
FROM evaluation_metrics
WHERE split = 'test' AND variant = 'platt'
ORDER BY cohort;

SELECT bin_index, n, mean_probability, positive_rate
FROM bin_statistics
WHERE split = 'test' AND variant = 'platt' AND cohort = 'all'
ORDER BY bin_index;

SELECT * FROM sensitivity_membership_counts;
```

## What SQL computes

[schema.sql](sql/schema.sql) separates runs, artifact hashes, partition membership,
items, score variants, scores and bin boundaries. Foreign keys and uniqueness
constraints reject orphan or duplicated evidence rows. Python ingestion checks
saved artifact hashes, ID-set hashes, disjoint primary partitions, exact exported
row membership/order, labels and score ranges. Inserts use bound parameters and
run in one transaction.

[analysis.sql](sql/analysis.sql) performs the calculations, without Python metric
functions or metric user-defined functions:

* Join labels and scores to their saved, frozen calibration thresholds.
* Construct aggregate, ordinary, jailbreak and overlap-exclusion cohorts.
* Compute confusion counts, class-specific rates and Brier scores with explicit
  denominators. A rate with no denominator is `NULL`.
* Aggregate fixed-width calibration bins and ECE using inclusive upper edges:
  `[0, 1/B]`, then `(1/B, 2/B]`, and so on. Explicit floating-point edges match
  the archived Python computation; score zero and one are included.
* Compute AUROC by grouping exact-score ties and using a window sum of negatives
  at lower scores. Positive-negative ties receive half credit. AUROC is `NULL`
  for a single-class cohort.
* Anti-join test membership against the published sensitivity membership to count
  excluded and retained **rows**.

The CLI checks **174 scalar metric fields** across three score variants and
18 split/cohort rows against the existing, independently computed Python report.
Its output also contains calibration-set metrics, whose ECE and AUROC are
descriptive fitted-calibration-set quantities, not held-out performance.
The verifier deliberately targets this archive's six nonempty split/cohort
combinations per variant; it is not a general importer for arbitrary CSV datasets
or archives with missing/empty cohorts.

## Measured archived results

For Platt-calibrated TF-IDF logistic regression, SQL reconstructs:

| Test cohort | Rows | ECE | AUROC | Misses / positive rows |
|---|---:|---:|---:|---:|
| All | 5,083 | 0.009935944367 | 0.908914969087 | 96 / 362 |
| Jailbreak | 91 | 0.325440796791 | Undefined | 8 / 91 |
| Excluding published normalized train-text matches | 4,833 | 0.010784364153 | 0.899658594589 | 96 / 329 |
| Jailbreak within that sensitivity slice | 75 | 0.366362203359 | Undefined | 8 / 75 |

The saved Platt threshold is `0.17652765054588185`; SQL reconstructs 92 false
positives among 1,840 calibration negatives (5.0%), and 245 among 4,721 test
negatives (about 5.19%). The workflow imports the frozen threshold; it does not
refit or select a new threshold on the test set.

## Provenance and limits

The evidence comes from the pinned public ToxicChat reproduction described in
[CPU_REPRODUCTION.md](CPU_REPRODUCTION.md). Dataset-derived artifacts retain the
source dataset's attribution and **CC BY-NC 4.0** restrictions; this extension
does not relicense them. Pseudonymous item IDs are linkable to the public source
and are not an anonymity guarantee. The database contains labels, scores and
membership, but no raw prompts.

SQL reconstructs **250 excluded test rows** from the published membership lists.
It cannot independently rediscover the source report's **230 distinct normalized
train/test text matches** or **52 fit/calibration matches**, because normalized
text hashes and raw text are absent from the archive. The membership exclusions
and source report still depend on the original diagnostic. The sensitivity slice
is not an untouched generalization test, and semantic/conversation dependencies
may remain.

The manifest hashes detect disagreements between the provided manifest and score
files; they do not authenticate a manifest or prevent a coordinated rewrite of
all evidence. This SQL path does not reconstruct confidence intervals, AUPRC,
model fitting, calibration fitting, raw-data normalization or GPU results. The
existing Python verifier remains the full archived-metric check. SQLite query
results are compared with numeric tolerances for floating-point summation order.
