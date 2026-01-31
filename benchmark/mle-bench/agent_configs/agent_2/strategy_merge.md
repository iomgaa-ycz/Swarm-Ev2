# Merge Strategy

When merging two parent solutions via genetic crossover:

## 1. Understand the Task

You are performing **genetic crossover**, combining gene blocks from two parent solutions to create a child solution.

### Inputs Provided
- **Parent A**: Code, fitness score, and gene blocks
- **Parent B**: Code, fitness score, and gene blocks
- **Gene Plan**: Dictionary specifying which parent to use for each gene block
  - Example: `{"DATA": "A", "MODEL": "B", "LOSS": "A", ...}`

### Goal
Create a **coherent, executable child solution** that:
- Respects the gene plan (use the specified parent for each block)
- Resolves any conflicts between incompatible blocks
- Preserves the strengths of both parents

## 2. Analyze Parent Solutions

### Extract Gene Blocks
- Parse each parent's code to identify the 7 gene blocks:
  - `DATA`, `MODEL`, `LOSS`, `OPTIMIZER`, `REGULARIZATION`, `INITIALIZATION`, `TRAINING_TRICKS`
- Note the approach used in each block (e.g., "Parent A uses XGBoost, Parent B uses CNN")

### Assess Fitness
- Higher-fitness parent likely has better-performing blocks
- If gene plan selects a low-fitness parent's block, consider whether adaptation is needed

## 3. Follow the Gene Plan

### Strict Adherence
- For each gene block, use the parent specified in the gene plan
- **Do not** arbitrarily replace blocks or ignore the plan

### Example
```python
# Gene Plan: {"DATA": "A", "MODEL": "B", "LOSS": "B", ...}

# [SECTION: DATA]
# From Parent A
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1).values
y = train['target'].values

# [SECTION: MODEL]
# From Parent B
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

# [SECTION: LOSS]
# From Parent B
loss = CategoricalCrossentropy()
```

## 4. Resolve Conflicts

### Detect Incompatibilities
Common conflict types:
- **Data format mismatch**: Parent A uses tabular data, Parent B's model expects images
- **Shape incompatibility**: Parent A's model outputs logits, Parent B's loss expects probabilities
- **Hyperparameter mismatch**: Parent A uses lr=0.001, Parent B uses lr=0.1

### Resolution Strategies

#### Adaptive Adjustment
Modify one block to match the other's requirements:
```python
# Conflict: Parent B's CNN expects images, Parent A provides tabular data
# Resolution: Reshape tabular data into pseudo-images
X_image = X.reshape(-1, height, width, channels)
```

#### Hybrid Approach
Combine elements from both parents:
```python
# Conflict: Parent A uses L2 regularization, Parent B uses Dropout
# Resolution: Use both (often complementary)
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # From A
    Dropout(0.3),  # From B
])
```

#### Fallback to Higher-Fitness Parent
If unresolvable, replace the problematic block with the higher-fitness parent's version:
```python
# [SECTION: MODEL]
# Conflict: Incompatible with DATA block
# Resolution: Using Parent A's model (higher fitness: 0.92 vs. 0.85)
model = parent_a_model
```

## 5. Ensure Coherence

### Compatibility Checks
- **DATA ↔ MODEL**: Model input shape matches data output shape
- **MODEL ↔ LOSS**: Loss function compatible with model output (logits vs. probabilities)
- **OPTIMIZER ↔ MODEL**: Learning rate appropriate for model complexity

### Syntax and Execution
- Ensure all imports are included
- No undefined variables or missing dependencies
- Code is syntactically correct

## 6. Implementation Checklist

Before finalizing the merged solution:
- [ ] All 7 gene blocks are present
- [ ] Gene plan is followed strictly (unless conflicts require adaptation)
- [ ] Conflicts are resolved (documented in comments)
- [ ] Code is syntactically correct and executable
- [ ] Model input/output shapes are compatible
- [ ] Validation metric is printed
- [ ] Predictions are saved to `./submission/submission.csv`

## 7. Documentation

Add comments to explain:
- Which blocks came from which parent
- How conflicts were resolved
- Any adaptations made to ensure compatibility

Example:
```python
# [SECTION: MODEL]
# From Parent B (fitness: 0.91)
# Adapted to accept tabular input from Parent A's DATA block
# by adding an embedding layer to convert tabular features to dense representation
```
