# Numeric boundaries in historical DPO aggregation

Generation lengths must be JSON integers (booleans, floats and strings are not
accepted), nonnegative, and representable as finite numeric values. Zero is valid.
These checks cover generation length and reward fields, not the full provenance
or semantic consistency of run metadata.

The descriptive pooled length adjustment requires at least two distinct lengths.
If every pooled length is identical, the slope cannot be identified, so aggregation
fails before replacing summary or figure outputs. A constant reward with varied
lengths has slope zero and intercept equal to that reward; adjusted scores are
zero and undefined Pearson correlation is represented by JSON `null`.
Other nonfinite regression statistics fail. This policy does not make pooled
length adjustment a causal estimate or establish statistical significance.

The complete summary is serialized with `allow_nan=False`, and the figure is
rendered in memory before either output is replaced. Invalid inputs or rendering
failure therefore preserve both previous outputs. Disk/I/O failures can still
interrupt the two subsequent writes; they are not an atomic pair transaction.

The tests use synthetic boundary cases. Historical raw DPO generations are still
unavailable, historical results are unchanged, and no model training or generation
was performed. Existing separate CPU safety evidence remains separate evidence.
