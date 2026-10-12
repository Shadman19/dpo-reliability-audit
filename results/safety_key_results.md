# Safety judge calibration under jailbreak shift (ToxicChat 0124), measured on ODU Wahab V100

Data: train 5,082 / test 5,083 user prompts; test has 362 toxic, 91 jailbreak prompts (all 91 labeled toxic).
Calibrators (Platt, isotonic) and a 5%-FPR threshold were fit on 2,000 held-out TRAIN prompts only, then frozen.

| Judge | Test AUROC (all) | ECE all (Platt) | ECE ordinary (Platt) | ECE jailbreak (raw / Platt) | Realized FPR | Miss rate ordinary toxic | Miss rate jailbreak |
|---|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.909 | 0.010 | 0.011 | 0.222 / 0.325 | 0.052 | 0.325 | 0.088 |
| Qwen2.5-1.5B-Instruct (prompted, P(Yes)) | 0.826 | 0.012 | 0.022 | 0.638 / 0.710 | 0.050 | 0.664 | 0.473 |
| Qwen2.5-0.5B-Instruct (prompted, P(Yes)) | 0.453 (raw) | 0.012 | 0.027 | 0.323 / 0.925 | 0.051 | 0.753 | 0.978 |

Findings
1. Aggregate calibration hides the failure: after Platt scaling, test ECE is about 0.01 overall, yet 0.33 (TF-IDF) and 0.71 (Qwen-1.5B) on jailbreak prompts, because jailbreaks are under 2% of the test set.
2. Post-hoc calibration fit on ordinary traffic made jailbreak calibration worse for every judge (raw to Platt: 0.22 to 0.33, 0.64 to 0.71, 0.32 to 0.92).
3. The 5%-FPR threshold transferred on false positives (realized 0.050 to 0.052) but the prompted 1.5B judge still missed 47% of jailbreak prompts.
4. A lexical TF-IDF baseline outranked the prompted 1.5B judge (AUROC 0.909 vs 0.826); the 0.5B judge was below chance before calibration and is not usable.
Bootstrap 95% CI for Platt ECE overall: TF-IDF [0.009, 0.017], Qwen-1.5B [0.009, 0.020].
