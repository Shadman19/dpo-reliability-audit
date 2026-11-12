# Strict prompt pairing for the historical DPO aggregator

Numeric length and regression boundary behavior is documented in
[`AGGREGATE_NUMERIC_POLICY.md`](AGGREGATE_NUMERIC_POLICY.md).

The historical `aggregate.py` compared generation rows with positional `zip`.
Reordered files can compare different prompts, and unequal files silently use
the shorter denominator. The proposed aggregator validates every generation
file and every baseline comparison before it writes summaries or figures.

The existing `dpo_lora.py` writer records prompt text but no stable sample ID.
Pairing therefore uses **exact prompt strings**, without whitespace/case
normalization. Unique reordered prompts are aligned; missing/extra prompts,
duplicates, empty files, blank lines, duplicate JSON keys and nonfinite or
nonnumeric reward scores are rejected. Ties remain non-wins, preserving the
historical strict `>` convention. Unique prompt text is not proof that samples
were drawn independently or generation settings/randomness were matched.

If repeated prompts are intentional, add stable sample IDs at data creation
and preserve their provenance across runs. Do not infer IDs from row positions
or collapse repeated prompts after the fact. This check does not establish
training/evaluation isolation, dataset validity, reward-model quality or
statistical significance. The pooled length adjustment remains descriptive.

Run the dependency-free checks with:

```bash
python -m unittest discover -s tests -p test_generation_pairing.py -v
```

The two integration checks require the aggregator's NumPy and Matplotlib
dependencies. The extension manifest preserves the exact recorded CPU manifest:

```bash
pip install -r requirements-aggregate.txt
python -m unittest discover -s tests -p test_aggregate_pairing.py -v
```

All fixtures are synthetic. Integration verifies that a reordered three-prompt
comparison reports 100% strict wins and that a truncated comparison fails
before overwriting an existing summary or writing a figure. No historical raw
DPO generations were recovered or rerun in this change. Committed historical
metrics/figures and the separate CPU safety reproduction remain unchanged.
This is AI-assisted collaborative software work, not a new model experiment,
accepted paper, independent audit certification or candidate assessment.

## Baseline identity

The aggregator now requires exactly one run whose configuration has numeric
`beta == 0`. Filenames do not establish baseline identity. Missing or multiple
controls fail before summary/figure output; multiple seeds need an explicit
comparison design rather than whichever file sorts first. Nonfinite, boolean,
string or negative beta values are rejected. This does not prove that the
control weights, generation settings or sampling randomness actually match.
Synthetic integration checks cover renamed controls and preservation of existing
evidence when controls are missing or ambiguous. No historical results rerun.
