# Merge Guide

When merging two parent solutions, the goal is to **combine their strengths** while preserving functional coherence.

## Crossover Methods

| Method | Strategy | Best For |
|--------|----------|----------|
| **Uniform Crossover** (Recommended) | Select each gene block from parent A or B per gene plan | Diverse combinations |
| **Single-Point Crossover** | Split at one point: blocks before from A, after from B | Strong block dependencies |
| **Best-of-Both Selection** | Analyze each block's fitness contribution, pick better one | Maximizing theoretical fitness (risk: breaks synergies) |

## Key Principles

- **Preserve working components**: Retain high-performing gene blocks unchanged when possible
- **Respect the gene plan**: It's designed to balance exploration; don't arbitrarily deviate
- **Don't modify high-fitness parent blocks** unless conflicts require adaptation

## Handle Dependencies

- **DATA ↔ MODEL**: Model architecture must match data format
- **MODEL ↔ TRAIN**: Training strategy (CV, callbacks) must align with model type (sklearn vs DL)
- **TRAIN ↔ POSTPROCESS**: Inference must match training paradigm (e.g., fold models for CV ensemble)
- **loss_function ↔ MODEL**: Loss function must be compatible with model output (logits vs probabilities)

## Conflict Types and Resolution

### 1. Data Format Mismatch
**Symptom**: Parent A uses one-hot encoding, Parent B uses label encoding

**Resolution**: Transform to match the selected MODEL block's expectations:
```python
# If using Parent B's model that expects label encoding with Parent A's one-hot data
# → Convert back to labels
y_labels = np.argmax(y_onehot, axis=1)
```

### 2. Model Output Shape Incompatibility
**Symptom**: Model outputs logits but loss expects probabilities

**Resolution**: Adjust activation or loss configuration:
```python
# Option A: Add activation
probabilities = F.softmax(logits, dim=-1)
# Option B: Use CrossEntropyLoss (expects raw logits)
loss_fn = nn.CrossEntropyLoss()  # expects raw logits, NOT probabilities
```

### 3. Hyperparameter Scale Mismatch
**Symptom**: Parent A uses lr=0.001 (small model), Parent B uses lr=0.1 (large model)

**Resolution**: Favor higher-fitness parent's hyperparameters, or scale based on model complexity:
```python
num_params = sum(p.numel() for p in model.parameters())
lr = 0.01 if num_params < 1e6 else 0.001
```

### 4. Regularization Overlap
**Symptom**: Parent A uses L2, Parent B uses Dropout

**Resolution**: Combine both (often complementary) or follow gene plan's selection.

## Resolution Decision Tree

```
Conflict Detected
    ├─ Data format issue?
    │   └─ Transform data to match model requirements
    │
    ├─ Model-loss mismatch?
    │   └─ Adjust activation or loss configuration
    │
    ├─ Hyperparameter conflict?
    │   └─ Favor higher-fitness parent or average
    │
    └─ Unresolvable?
        └─ Fallback to higher-fitness parent's block
```

## Fallback Strategy

When a conflict **cannot be resolved** without major refactoring:
1. Check fitness scores: `parent_a.metric_value` vs `parent_b.metric_value`
2. Replace the problematic block entirely with the higher-fitness parent's version
3. Document the decision in a comment

## Anti-Patterns

- **Ignoring the gene plan**: Don't arbitrarily replace blocks without following the plan
- **Over-engineering**: Don't add unnecessary complexity (e.g., ensemble when single model suffices)
- **Silent failures**: Don't suppress errors; fix the root cause
- **Hardcoding**: Avoid hardcoded values specific to one parent's dataset
