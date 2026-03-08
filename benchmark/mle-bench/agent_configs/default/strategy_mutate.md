# Mutate Strategy

When mutating a solution via genetic mutation:

> **Important**: `mutation_aspect` is a sub-aspect **within** the target gene block (e.g., `optimizer` within `MODEL`), not an independent SECTION marker. Only modify the specified aspect; keep all other parts of the gene block unchanged.

## 1. Ensure Compatibility

### Check Dependencies
- If mutating `MODEL`, ensure loss_function aspect is still compatible
- If mutating `DATA`, ensure `MODEL` can accept the new input shape
- If mutating `TRAIN`, ensure training strategy aligns with model type

## 2. Documentation

Add a comment explaining the mutation rationale (bottleneck diagnosed → mutation applied → expected improvement).
