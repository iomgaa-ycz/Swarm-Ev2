# Local Optimization Techniques

When mutating a solution, **target the bottleneck** rather than applying random changes. This guide provides diagnostic heuristics to identify which gene block to mutate.

## Diagnostic Framework

### Step 1: Analyze Current Performance

Extract key metrics from the parent solution:
- **Validation metric** (e.g., accuracy, RMSE)
- **Training metric** (if available)
- **Training time** (convergence speed)
- **Error patterns** (e.g., confusion matrix for classification)

### Step 2: Identify Bottleneck

Use the following decision tree:

```
Parent Solution Analysis
    ├─ Validation metric is poor (< baseline)?
    │   ├─ Training metric also poor?
    │   │   └─ Yes → UNDERFITTING
    │   │       ├─ Mutate: MODEL (increase capacity)
    │   │       ├─ Mutate: DATA (add features, augmentation)
    │   │       └─ Mutate: INITIALIZATION (better init)
    │   └─ Training metric good?
    │       └─ Yes → OVERFITTING
    │           ├─ Mutate: REGULARIZATION (add/increase dropout, L2)
    │           ├─ Mutate: DATA (reduce noise, remove outliers)
    │           └─ Mutate: TRAINING_TRICKS (early stopping)
    │
    ├─ Loss not decreasing?
    │   └─ Mutate: OPTIMIZER (change optimizer, adjust learning rate)
    │       └─ Mutate: INITIALIZATION (better weight init)
    │
    ├─ Training too slow?
    │   └─ Mutate: OPTIMIZER (use faster optimizer, increase batch size)
    │       └─ Mutate: MODEL (reduce complexity)
    │
    └─ Metrics oscillating/unstable?
        └─ Mutate: OPTIMIZER (reduce learning rate)
            └─ Mutate: TRAINING_TRICKS (add gradient clipping)
```

## Bottleneck-Specific Mutations

### 1. Underfitting (Both train and val metrics low)

**Root Cause**: Model capacity too low, or insufficient training

**Target Genes**: `MODEL`, `DATA`, `TRAINING_TRICKS`

**Mutations**:
```python
# [SECTION: MODEL]
# Increase model depth (add layers)
model = Sequential([
    Dense(256, activation='relu'),  # Increased from 128
    Dense(128, activation='relu'),  # Added layer
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

# [SECTION: DATA]
# Add polynomial features to capture non-linearity
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# [SECTION: TRAINING_TRICKS]
# Train longer (increase epochs)
model.fit(X, y, epochs=100)  # Increased from 50
```

### 2. Overfitting (Train metric >> Val metric)

**Root Cause**: Model memorizing training data, poor generalization

**Target Genes**: `REGULARIZATION`, `DATA`, `TRAINING_TRICKS`

**Mutations**:
```python
# [SECTION: REGULARIZATION]
# Increase dropout rate
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),  # Increased from 0.3
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

# Add L2 regularization
from tensorflow.keras.regularizers import l2
model.add(Dense(64, kernel_regularizer=l2(0.01)))

# [SECTION: DATA]
# Add data augmentation (for images)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# [SECTION: TRAINING_TRICKS]
# Early stopping based on validation loss
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, validation_split=0.2, callbacks=[early_stop])
```

### 3. Loss Plateau (Validation loss stops decreasing)

**Root Cause**: Learning rate too high/low, or optimizer stuck in local minimum

**Target Genes**: `OPTIMIZER`, `TRAINING_TRICKS`

**Mutations**:
```python
# [SECTION: OPTIMIZER]
# Switch optimizer
optimizer = AdamW(lr=0.001, weight_decay=0.01)  # Changed from Adam

# Adjust learning rate
optimizer = Adam(lr=0.0005)  # Reduced from 0.001

# [SECTION: TRAINING_TRICKS]
# Add learning rate schedule
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
model.fit(X, y, callbacks=[lr_schedule])

# Or use learning rate warmup + decay
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001 * (epoch + 1) / 10  # Warmup
    else:
        return 0.001 * 0.95 ** (epoch - 10)  # Decay
```

### 4. Slow Convergence (Training takes too long)

**Root Cause**: Inefficient optimizer, poor initialization, or large model

**Target Genes**: `OPTIMIZER`, `INITIALIZATION`, `MODEL`

**Mutations**:
```python
# [SECTION: OPTIMIZER]
# Use faster optimizer
optimizer = AdamW(lr=0.001)  # AdamW often converges faster than SGD

# Increase batch size (if memory allows)
model.fit(X, y, batch_size=128)  # Increased from 32

# [SECTION: INITIALIZATION]
# Use better initialization
from tensorflow.keras.initializers import HeNormal
model = Sequential([
    Dense(128, kernel_initializer=HeNormal()),
    Dense(64, kernel_initializer=HeNormal()),
])

# [SECTION: MODEL]
# Reduce model complexity if overly large
model = Sequential([
    Dense(64, activation='relu'),  # Reduced from 256
    Dense(num_classes, activation='softmax'),
])
```

### 5. Unstable Training (Metrics oscillate)

**Root Cause**: Learning rate too high, exploding gradients

**Target Genes**: `OPTIMIZER`, `TRAINING_TRICKS`

**Mutations**:
```python
# [SECTION: OPTIMIZER]
# Reduce learning rate
optimizer = Adam(lr=0.0001)  # Reduced from 0.001

# [SECTION: TRAINING_TRICKS]
# Add gradient clipping
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001, clipnorm=1.0)

# Add batch normalization to stabilize training
from tensorflow.keras.layers import BatchNormalization
model = Sequential([
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dense(num_classes, activation='softmax'),
])
```

## Heuristic Table

| Symptom | Likely Bottleneck | Target Gene | Mutation Example |
|---------|-------------------|-------------|------------------|
| Train ↓ Val ↓ | Underfitting | MODEL | Add layers, increase width |
| Train ↑ Val ↓ | Overfitting | REGULARIZATION | Increase dropout, add L2 |
| Loss plateau | Optimizer stuck | OPTIMIZER | Reduce lr, switch optimizer |
| Slow training | Inefficient optimization | OPTIMIZER, INITIALIZATION | Use Adam, He init |
| Unstable metrics | High learning rate | OPTIMIZER | Reduce lr, gradient clipping |
| Poor predictions on specific classes | Data imbalance | DATA, LOSS | SMOTE, focal loss |

## Validation Strategy

After applying a local optimization mutation:
1. **Quick sanity check**: Run 1-2 epochs to ensure no errors
2. **Compare validation metric**: Check if mutation improved performance
3. **Monitor training curve**: Look for faster convergence or better generalization
4. **A/B test**: If uncertain, keep both parent and mutant for comparison

## Anti-Patterns

❌ **Random mutation without diagnosis**: Don't mutate blindly; analyze first
❌ **Over-optimization**: Don't apply multiple optimizations simultaneously (hard to attribute improvements)
❌ **Ignoring data quality**: No amount of model tuning can fix bad data
❌ **Premature complexity**: Don't add advanced techniques (e.g., attention mechanisms) before exhausting simple fixes
