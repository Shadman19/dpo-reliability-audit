# Reproduce the safety baseline on CPU

This addition reproduces the existing **TF-IDF + logistic regression** ToxicChat
study and makes its scores, split membership and uncertainty auditable. It does
not reproduce the historical Qwen judges, train a new LLM, or establish deployment
safety. Historical tables under `results/` are unchanged.

## Run

Python 3.12.14 was used for the committed run. No GPU, torch, transformers,
Hugging Face account, API key or paid service is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python safety_calib.py \
  --baseline-only --out-dir runs/my-reproduction
python verify_safety_evidence.py runs/my-reproduction
```

The runner downloads the official public `lmsys/toxic-chat`, `toxicchat0124`
train/test CSVs at revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`.
Pinned byte hashes are verified before use. Raw CSVs stay under ignored `.cache/`;
`--data-dir` may point to an existing cache with the same exact bytes. The default
seed-0 split uses 3,082 training-fit rows and 2,000 calibration rows; evaluation
uses the official 5,083 test rows. Neither calibrators nor operating thresholds
use test labels. Existing output directories are rejected to preserve evidence.

The data are released by LMSYS under **CC BY-NC 4.0**. See the
[dataset card](https://huggingface.co/datasets/lmsys/toxic-chat) for attribution,
license and content details. Raw prompts and model outputs are not included here.
The labels and score artifacts here are dataset-derived research materials;
the repository's MIT code license does not relicense the source dataset or its
derived materials. Preserve the dataset's attribution and noncommercial terms.
Hashed dataset IDs remain linkable to this public dataset; they are pseudonymous
references, not a privacy or anonymization guarantee.

## Measured evidence: November 11, 2026

The committed [report](results/reproduction-cpu-20261111/report.json),
[score rows](results/reproduction-cpu-20261111/tfidf_lr_predictions.jsonl) and
[split membership](results/reproduction-cpu-20261111/split_membership.json)
contain executable/data hashes, source revision, environment and exact counts.
The verifier checks hashes and recomputes all 15 variant-by-group metric sets
from labels and probabilities without loading raw text.
Hashes identify the archived run exactly. Floating-point scores can differ at
approximately 1e-14 across fresh processes/BLAS configurations, so a rerun need
not produce identical score-file bytes or hashes; numerical comparisons use a
tolerance. The verifier recomputes the archived run with a tolerance as well.

For the Platt-calibrated TF-IDF baseline:

| Evaluation subset | Rows | AUROC | ECE | Missed toxic / toxic rows |
| --- | ---: | ---: | ---: | ---: |
| Full official test | 5,083 | 0.908915 | 0.009936 | 96 / 362 |
| Full test jailbreak subset | 91 | Undefined: one class | 0.325441 | 8 / 91 |
| Test rows without normalized train-text match | 4,833 | 0.899659 | 0.010784 | 96 / 329 |
| Jailbreak rows without normalized train-text match | 75 | Undefined: one class | 0.366362 | 8 / 75 |

The full-test false-positive rate is 245/4,721 = **0.051896**. The full jailbreak
miss rate is **0.087912**, Wilson 95% interval **[0.045220, 0.163986]**.
Its ECE percentile-bootstrap interval is **[0.268347, 0.388876]**. Small aggregate
ECE therefore continues to hide poor calibration within the jailbreak subset.
The 75-row sensitivity subset has miss-rate interval **[0.055049, 0.196614]**.

## Text-overlap diagnostic

Conversation IDs are disjoint across official train/test, but after collapsing
whitespace and applying Unicode casefold there are **230 distinct normalized
strings in common, covering 250 test rows**. Fit/calibration have **52 distinct
normalized matches**; fit/test have 152 and calibration/test 83. These are potential
contamination signals, not semantic duplicate adjudications.

The additional sensitivity slice removes exact normalized train-text matches
from test evaluation without refitting the model, calibrators or threshold. It
does not remove semantic similarity, within-test repetition or fit/calibration
overlap. This is a sensitivity analysis on an already inspected public benchmark,
not a new independent or untouched generalization test.

## Changes to measurement

- A tie-aware threshold allows at most `floor(0.05 * n_negative)` calibration
  false positives under the `p >= threshold` decision rule. It is just above the
  first disallowed score. Isotonic ties can make the observed rate lower than 5%.
  This controls empirical calibration-set exceedances only; test/population FPR
  can be higher. It differs from the old interpolated-quantile rule.
- ECE bootstrap retains single-class resamples, so jailbreak intervals are
  defined. ECE uses the historical 15 fixed, right-closed bins.
- FPR/FNR have explicit denominators and Wilson intervals. AUROC and FPR are
  undefined for the all-positive jailbreak subgroup and are reported as null.
- Intervals condition on the fitted model/calibrators/threshold and assume
  independent evaluation rows. They omit training and calibration uncertainty;
  text/user dependence can invalidate nominal interval coverage.
- The optional CUDA path imports heavy libraries lazily and deduplicates Yes/No
  first-token IDs before aggregation. This differs from the historical scoring
  implementation; no updated CUDA result is claimed here. It is still only a
  first-token Yes/No score, not whole-label sequence likelihood.

All reported measurements come from the committed run. The tests use synthetic
fixtures to check code behavior; their scores are not model-evaluation results.
