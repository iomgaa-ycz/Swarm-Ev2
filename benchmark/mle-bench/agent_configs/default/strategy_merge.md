# Merge Strategy

When merging two parent solutions via genetic crossover:

## 1. Analyze Parent Solutions

### Extract Gene Blocks
- Parse each parent's code to identify the 4 gene blocks: `DATA`, `MODEL`, `TRAIN`, `POSTPROCESS`
- Note the approach used in each block (e.g., "Parent A uses XGBoost, Parent B uses CNN")

### Assess Fitness
- Higher-fitness parent likely has better-performing blocks
- If gene plan selects a low-fitness parent's block, consider whether adaptation is needed

## 2. Documentation

Document which blocks came from which parent and how conflicts were resolved.
