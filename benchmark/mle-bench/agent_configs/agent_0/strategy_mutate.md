# Mutate Strategy

When mutating a solution via genetic mutation:

## 1. Understand the Task

You are performing **genetic mutation**, modifying a specific gene block to explore variations and potentially improve fitness.

### Inputs Provided
- **Parent Solution**: Code, fitness score, execution results
- **Target Gene Block**: The specific block to mutate (e.g., `MODEL`, `OPTIMIZER`, `REGULARIZATION`)

### Goal
Create a **modified solution** that:
- Changes only the target gene block (leave other blocks unchanged)
- Introduces a meaningful variation (not trivial or arbitrary)
- Maintains compatibility with other blocks
- Aims to improve fitness

## 2. Analyze the Parent Solution

### Identify the Bottleneck
Before mutating, diagnose what might be limiting performance:

- **Validation metric is poor**:
  - Training metric also poor → **Underfitting** (increase model capacity, add features)
  - Training metric good → **Overfitting** (add regularization, reduce complexity)

- **Loss plateau**:
  - Optimizer stuck in local minimum → Adjust learning rate, switch optimizer

- **Slow convergence**:
  - Inefficient optimization → Use adaptive optimizer, better initialization

- **Unstable training**:
  - High learning rate → Reduce lr, add gradient clipping

### Target the Right Gene Block
Based on the bottleneck, choose an appropriate mutation:

| Bottleneck | Target Gene | Mutation Type |
|------------|-------------|---------------|
| Underfitting | MODEL | Increase capacity (add layers, widen layers) |
| Overfitting | REGULARIZATION | Increase dropout, add L2, early stopping |
| Loss plateau | OPTIMIZER | Adjust learning rate, switch optimizer |
| Slow training | OPTIMIZER, INITIALIZATION | Use Adam, He initialization |
| Unstable metrics | OPTIMIZER | Reduce learning rate, gradient clipping |
| Poor features | DATA | Add feature engineering, augmentation |

## 3. Choose a Mutation Strategy

### Small Perturbation (Low Risk, Low Reward)
Fine-tune hyperparameters without changing the core approach:
```python
# [SECTION: OPTIMIZER]
# Original: lr=0.001
# Mutation: Multiply by random factor [0.5, 2.0]
lr = 0.001 * 1.5  # New lr: 0.0015
optimizer = Adam(learning_rate=lr)
```

### Component Swap (Medium Risk, Medium Reward)
Replace a module with an alternative:
```python
# [SECTION: MODEL]
# Original: ReLU activation
# Mutation: Swap to LeakyReLU
model = Sequential([
    Dense(128),
    LeakyReLU(alpha=0.2),  # ← Changed from ReLU
    Dense(num_classes, activation='softmax'),
])
```

### Structural Change (High Risk, High Reward)
Modify architecture by adding/removing components:
```python
# [SECTION: MODEL]
# Original: 2-layer MLP
# Mutation: Add a hidden layer
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),   # ← Added layer
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
```

## 4. Apply the Mutation

### Gene-Specific Mutation Examples

#### DATA
```python
# Original: Basic preprocessing
X = (X - X.mean()) / X.std()

# Mutation: Add polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

#### MODEL
```python
# Original: Simple architecture
model = Sequential([Dense(64), Dense(num_classes)])

# Mutation: Add BatchNorm and Dropout
model = Sequential([
    Dense(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes)
])
```

#### LOSS
```python
# Original: Standard cross-entropy
loss = CategoricalCrossentropy()

# Mutation: Focal loss for imbalanced data
loss = FocalLoss(gamma=2.0)
```

#### OPTIMIZER
```python
# Original: Adam
optimizer = Adam(lr=0.001)

# Mutation: AdamW with weight decay
optimizer = AdamW(lr=0.001, weight_decay=0.01)
```

#### REGULARIZATION
```python
# Original: L2 penalty
regularizer = l2(0.01)

# Mutation: Increase L2 + add early stopping
regularizer = l2(0.05)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
```

#### INITIALIZATION
```python
# Original: Default initialization
model = Sequential([Dense(64)])

# Mutation: He initialization
model = Sequential([
    Dense(64, kernel_initializer='he_normal')
])
```

#### TRAINING_TRICKS
```python
# Original: Fixed learning rate
model.fit(X, y, epochs=50)

# Mutation: Learning rate schedule
lr_schedule = ReduceLROnPlateau(factor=0.5, patience=5)
model.fit(X, y, callbacks=[lr_schedule])
```

## 5. Ensure Compatibility

### Check Dependencies
- If mutating `MODEL`, ensure `LOSS` is still compatible
- If mutating `DATA`, ensure `MODEL` can accept the new input shape
- If mutating `OPTIMIZER`, ensure learning rate is appropriate for model

### Syntax and Execution
- Verify all imports are included
- Ensure no undefined variables
- Test that the code is syntactically correct

## 6. Implementation Checklist

Before finalizing the mutated solution:
- [ ] Only the target gene block is modified (other blocks unchanged)
- [ ] Mutation is meaningful (not arbitrary or trivial)
- [ ] Mutation addresses a suspected bottleneck
- [ ] Code is syntactically correct and executable
- [ ] Compatibility with other blocks is maintained
- [ ] Validation metric is printed
- [ ] Predictions are saved to `./submission/submission.csv`

## 7. Documentation

Add a comment explaining the mutation rationale:
```python
# [SECTION: OPTIMIZER]
# Mutation: Changed from Adam to AdamW with weight decay
# Rationale: Parent solution showed signs of overfitting (train=0.95, val=0.82)
# Weight decay should improve generalization
optimizer = AdamW(lr=0.001, weight_decay=0.01)
```
