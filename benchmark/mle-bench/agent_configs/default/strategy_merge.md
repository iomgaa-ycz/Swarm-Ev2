# Merge Strategy

When merging two parent solutions via genetic crossover:

## 1. Analyze Parent Solutions

### Extract Gene Blocks
- Parse each parent's code to identify the 4 gene blocks: `DATA`, `MODEL`, `TRAIN`, `POSTPROCESS`
- Note the approach used in each block (e.g., "Parent A uses XGBoost, Parent B uses CNN")

### Assess Fitness
- Higher-fitness parent likely has better-performing blocks
- If gene plan selects a low-fitness parent's block, consider whether adaptation is needed

## 2. Ensure Coherence

### Compatibility Checks
- **DATA ↔ MODEL**: Model input shape matches data output shape
- **loss_function ↔ MODEL**: Loss function compatible with model output (logits vs. probabilities)
- **MODEL ↔ TRAIN**: Training strategy compatible with model type (sklearn vs DL)

### Syntax and Execution
- Ensure all imports are included
- No undefined variables or missing dependencies
- Code is syntactically correct

## 3. Documentation

Document which blocks came from which parent and how conflicts were resolved.
