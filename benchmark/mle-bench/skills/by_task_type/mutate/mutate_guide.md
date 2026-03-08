# Mutation Guide

Mutation introduces **controlled variation** to a solution, enabling exploration of the fitness landscape while avoiding catastrophic changes.

## Mutation Types

### 1. Small Perturbation (Exploitation)
**Goal**: Fine-tune hyperparameters without changing the core approach

Examples: adjust learning rate (`lr *= uniform(0.5, 2.0)`), modify regularization strength, change dropout rate.

```python
# Aspect: optimizer (within MODEL block)
lr = 0.001 * random.uniform(0.5, 2.0)
optimizer = Adam(learning_rate=lr)
```

### 2. Component Swap (Moderate Exploration)
**Goal**: Replace a module with an alternative while preserving structure

Examples: ReLU → LeakyReLU/GELU, Adam → AdamW/SGD, MSE → Huber loss, Dense → BatchNorm+Dense+Dropout.

```python
# Aspect: architecture (within MODEL block)
model = Sequential([
    Dense(128), LeakyReLU(alpha=0.2),  # Swapped from ReLU
    Dense(num_classes, activation='softmax'),
])
```

### 3. Structural Change (High Exploration)
**Goal**: Modify architecture by adding/removing layers or components

Examples: add residual connection, insert dropout layer, increase/decrease model depth.

```python
# Aspect: architecture (within MODEL block)
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),   # Added layer
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
```

## Mutation Intensity

| Mutation Type | Intensity | Fitness Change Risk | Use Case |
|---------------|-----------|---------------------|----------|
| Small Perturbation | Low (5-20%) | Low | Near-optimal solutions |
| Component Swap | Medium (30-50%) | Medium | Suspected bottlenecks |
| Structural Change | High (50%+) | High | Fitness plateau, exploration phase |

## Bottleneck Diagnosis

Before mutating, diagnose the bottleneck using the parent's metrics:

```
Parent Solution Analysis
    ├─ Validation metric is poor (< baseline)?
    │   ├─ Training metric also poor?
    │   │   └─ UNDERFITTING
    │   │       ├─ Mutate: MODEL [architecture] (increase capacity)
    │   │       ├─ Mutate: DATA [feature_engineering] (add features)
    │   │       └─ Mutate: MODEL [regularization] (remove excessive)
    │   └─ Training metric good?
    │       └─ OVERFITTING
    │           ├─ Mutate: MODEL [regularization] (add/increase dropout, L2)
    │           ├─ Mutate: DATA [augmentation] (add augmentation)
    │           └─ Mutate: TRAIN [early_stopping]
    │
    ├─ Loss not decreasing?
    │   └─ Mutate: MODEL [optimizer] (change optimizer, adjust lr)
    │       └─ Mutate: TRAIN [lr_schedule]
    │
    ├─ Training too slow?
    │   └─ Mutate: MODEL [optimizer] (faster optimizer, larger batch)
    │       └─ Mutate: MODEL [architecture] (reduce complexity)
    │
    └─ Metrics oscillating/unstable?
        └─ Mutate: MODEL [optimizer] (reduce lr)
            └─ Mutate: TRAIN [lr_schedule] (gradient clipping)
```

## Bottleneck-Specific Mutations

### 1. Underfitting (Both train and val metrics low)

**Root Cause**: Model capacity too low, or insufficient training
**Target Genes**: `MODEL` [architecture], `DATA` [feature_engineering], `TRAIN` [epochs]

```python
# MODEL [architecture] — increase depth
model = Sequential([
    Dense(256, activation='relu'),  # Increased from 128
    Dense(128, activation='relu'),  # Added layer
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
```

### 2. Overfitting (Train metric >> Val metric)

**Root Cause**: Model memorizing training data, poor generalization
**Target Genes**: `MODEL` [regularization], `DATA` [augmentation], `TRAIN` [early_stopping]

```python
# MODEL [regularization] — increase dropout + L2
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),  # Increased from 0.3
    Dense(64, kernel_regularizer=l2(0.01)),
    Dense(num_classes, activation='softmax'),
])
```

### 3. Loss Plateau (Validation loss stops decreasing)

**Root Cause**: Learning rate too high/low, or optimizer stuck
**Target Genes**: `MODEL` [optimizer], `TRAIN` [lr_schedule]

```python
# MODEL [optimizer] — switch optimizer
optimizer = AdamW(lr=0.001, weight_decay=0.01)

# TRAIN [lr_schedule] — add schedule
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
```

### 4. Slow Convergence (Training takes too long)

**Root Cause**: Inefficient optimizer, poor initialization, or large model
**Target Genes**: `MODEL` [optimizer], `MODEL` [architecture]

```python
# MODEL [optimizer] — better init + optimizer
optimizer = AdamW(lr=0.001)
model = Sequential([
    Dense(128, kernel_initializer=HeNormal()),
    Dense(64, kernel_initializer=HeNormal()),
])
```

### 5. Unstable Training (Metrics oscillate)

**Root Cause**: Learning rate too high, exploding gradients
**Target Genes**: `MODEL` [optimizer], `TRAIN` [lr_schedule]

```python
# MODEL [optimizer] — reduce lr + gradient clipping + BatchNorm
optimizer = Adam(lr=0.0001, clipnorm=1.0)
model = Sequential([
    Dense(128), BatchNormalization(), Activation('relu'),
    Dense(num_classes, activation='softmax'),
])
```

## Heuristic Table

| Symptom | Likely Bottleneck | Target Gene [Aspect] | Mutation Example |
|---------|-------------------|---------------------|------------------|
| Train ↓ Val ↓ | Underfitting | MODEL [architecture] | Add layers, increase width |
| Train ↑ Val ↓ | Overfitting | MODEL [regularization] | Increase dropout, add L2 |
| Loss plateau | Optimizer stuck | MODEL [optimizer] | Reduce lr, switch optimizer |
| Slow training | Inefficient optimization | MODEL [optimizer] | Use Adam, He init |
| Unstable metrics | High learning rate | MODEL [optimizer] | Reduce lr, gradient clipping |
| Poor class predictions | Data imbalance | DATA [encoding], MODEL [loss_function] | SMOTE, focal loss |

## Anti-Patterns

- **Random mutation without diagnosis**: Don't mutate blindly; analyze the bottleneck first
- **Multiple large mutations simultaneously**: Hard to attribute improvements; mutate one aspect at a time
- **Ignoring data quality**: No amount of model tuning can fix bad data
- **Premature complexity**: Don't add advanced techniques before exhausting simple fixes
- **Breaking DATA loading**: Don't mutate critical blocks randomly
- **Ignoring block compatibility**: MODEL and loss_function must align; DATA output must match MODEL input
