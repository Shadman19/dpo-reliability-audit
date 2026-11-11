# Archived CPU baseline reproduction

Produced November 11, 2026 from the public **LMSYS ToxicChat** `toxicchat0124`
release at revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`.
Source: https://huggingface.co/datasets/lmsys/toxic-chat

The source dataset uses **CC BY-NC 4.0**. These labels and model-score files are
dataset-derived research artifacts; the code repository's MIT license does not
relicense the source data or derived materials. Preserve source attribution and
noncommercial terms. No raw prompts, model outputs or original conversation IDs
are included. SHA-256 IDs are linkable to the public source and do not provide
anonymity.

Run `python verify_safety_evidence.py results/reproduction-cpu-20261111` from the
repository root to verify archived hashes and recompute metrics without raw text.
See [CPU_REPRODUCTION.md](../../CPU_REPRODUCTION.md) for methodology, normalized
text overlap, conditional uncertainty and limitations. Fresh runs can exhibit
small floating-point differences and need not reproduce archived hashes exactly.
