# CPU calibration split sensitivity

Three additional fits were run on November 14, 2026, after recording a local
[analysis plan](results/split-sensitivity-cpu-20261114/PLAN.txt). Seed 0 is the
existing, previously inspected comparator. Every planned new seed (1, 2, 3) is
reported, without best-seed selection. This is AI-assisted collaborative
research engineering, not evidence of independent interview performance.

## Evidence and verification

[summary.json](results/split-sensitivity-cpu-20261114/summary.json) records all
four measured fits. Full score/membership archives are retained with the private
career-audit supporting records; they are not included in this PR. Public
readers can regenerate the three new runs using the commands below, then verify
their saved evidence. The existing seed-0 archive remains in this repository.

The verification command checks each run's source/artifact hashes and recomputes
all 60 metric groups over 28,332 score rows. It also checks common test
identities/labels, matching configuration, seed metadata, and independent
sklearn AUROC/Brier and direct-bin ECE calculations. It prints JSON only after
all four fits pass and does not overwrite evidence. Hashes establish integrity,
not authenticity of a coordinated rewrite. Existing verifier and JSON parsing
limitations remain. No private account or career records are needed to refit.

## Observed Platt-calibrated results

| Seed | Overall ECE | Jailbreak ECE | Missed / 91 jailbreaks | Test FPR |
| --- | ---: | ---: | ---: | ---: |
| 0 (existing) | 0.009936 | 0.325441 | 8 | 5.190% |
| 1 | 0.006263 | 0.312877 | 10 | 5.105% |
| 2 | 0.008471 | 0.303003 | 11 | 5.740% |
| 3 | 0.010404 | 0.310881 | 9 | 4.152% |

Low overall ECE coexists with larger jailbreak ECE in all four fits. After
removing exact normalized train-text matches, jailbreak ECE ranges from
0.3360 to 0.3664. The 5% calibration-negative target does not guarantee test or
population FPR. Raw and isotonic metrics are retained in every report too.

## Refit the planned new seeds

From the repository root, with new output directories:

```bash
for seed in 1 2 3; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python safety_calib.py \
    --baseline-only --seed "$seed" --out-dir "runs/sensitivity/seed-$seed"
done
python verify_split_sensitivity.py --archive runs/sensitivity
```

The runner downloads the pinned public ToxicChat files and verifies their byte
hashes. It refuses existing output directories. Fresh processes/environments can
produce numerically equivalent scores with different file hashes; the verifier
checks each new run against its own manifest, not against archived score bytes.
The Python and dependency versions are recorded in each report. See
[CPU_REPRODUCTION.md](CPU_REPRODUCTION.md) for requirements and data provenance.

## Scope, dependence and licensing

Each fit uses 3,082 fit, 2,000 calibration and the same 5,083 inspected test rows.
Changing the seed jointly changes fit/calibration allocation, fitted TF-IDF/LR,
calibrators and bootstrap RNG. This does not isolate one variance component.
Ranges are descriptive; four fits are not four independent test datasets.
Item-level intervals condition on fitted components and omit fitting/calibration
uncertainty. Text overlap, repeated items and other dependence may remain.
The all-positive jailbreak subgroup has undefined AUROC/FPR. This is not new
untouched generalization, a new Qwen/DPO/GPU result, clinical validation or a
production serving benchmark. Historical result tables remain unchanged.

These score/label artifacts derive from LMSYS ToxicChat, revision
`29df8e4dba60e1f4af4b4075c0705c5b313548a8`, under **CC BY-NC 4.0**;
see the [dataset card](https://huggingface.co/datasets/lmsys/toxic-chat).
Retain that attribution and noncommercial restriction. The MIT code license does
not relicense these materials. No raw prompts are included. Hashed IDs remain
linkable to the public source; they do not establish anonymity.
