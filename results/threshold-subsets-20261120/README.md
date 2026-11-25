# Threshold-selection subset sensitivity (exploratory, November 20, 2026)

This publishes a real reanalysis of the existing seed-0 TF-IDF CPU evidence. The
original local plan was recorded before computing these subsets but after the
benchmark was inspected; `PLAN.txt` is not a preregistration. No model fitting,
Platt refitting, new predictions, DPO generations or GPU run is involved.
The public packaging was prepared November 25 with AI-assisted implementation
and review; no independent human review or interview-readiness score is claimed.

## Reproduce and check every recorded field

Use the repository's pinned `requirements-cpu.txt` environment (Python 3.12;
NumPy 2.3.5), then run from the repository root:

```sh
python threshold_selection_sensitivity.py --repo . --out /tmp/threshold-subsets-check \
  --check results/threshold-subsets-20261120/results.json
```

The implementation checks the saved prediction bytes against their report hash,
unique IDs, subgroup denominators, calibration cap, boundary decisions and
independent count arithmetic. Explicit runtime checks also operate under
`python -O`. `--check` compares all generated result fields with the recorded
artifact before output replacement. A mismatch fails without success stdout.
Canonical JSON comparison is exact in this pinned environment, not a rerun
tolerance or authenticity guarantee. Source/code hashes and NumPy version are
recorded. The output is a single ordinary filesystem write, not an atomic
transaction; protection against filesystem I/O failures is outside scope.

## Design and all results

There are 1,840 calibration negatives. For seeds 0–39, take nested prefixes of a
PCG64 permutation without replacement: 100, 500 and 1,000 negatives select a
threshold targeting at most 5% empirical calibration FPR. Keep the saved Platt
mapping and the same 5,083 test rows fixed. `results.json` contains all 120 rows,
selection-ID digests, descriptive summaries and the all-negative reference.
No threshold is selected on test performance.

| Threshold-selection negatives | Test FPR min / median / max | Jailbreak misses min / median / max |
| --- | --- | --- |
| 100 | 2.25% / 6.44% / 12.86% | 5 / 8 / 18 of 91 |
| 500 | 4.02% / 5.20% / 6.95% | 7 / 8 / 14 of 91 |
| 1,000 | 4.66% / 5.21% / 6.63% | 8 / 8 / 13 of 91 |
| All 1,840 (reference) | 5.19% (245/4,721) | 8 of 91 |

Each subset meets the empirical calibration cap. Test FPR exceeds 5% in 29/40,
21/40 and 28/40 rows respectively. These are descriptive counts, not population
probabilities. Larger subsets have a narrower observed range here; monotonic
improvement or deployment-level FPR control is not established.

## Limits that travel with the table

- Platt fitting used the original full calibration partition. This varies only
  the **negative-score subset used to choose a threshold**, not end-to-end
  calibration sample size.
- Subsets are nested and overlap; test data are inspected and reused. These are
  not 120 independent experiments, new test datasets or confidence intervals.
- Known normalized-text train/test overlap remains in the underlying benchmark.
- The 91 jailbreak cases are all positive, so subgroup FPR/AUROC are undefined;
  report positive misses, not invented negative-class metrics.
- The checks establish internal agreement, not authenticity against coordinated
  rewriting of artifacts and manifests. This is not new model-performance,
  clinical, production-ownership or DPO/GPU evidence.

Numerical evidence is derived from **LMSYS ToxicChat**, dataset revision and
hashes recorded in `../reproduction-cpu-20261111/report.json`.
Dataset: <https://huggingface.co/datasets/lmsys/toxic-chat>, CC BY-NC 4.0;
see that dataset's attribution and citation. Dataset-derived numerical artifacts
retain those terms, separate from repository MIT code. No raw prompts are added.
