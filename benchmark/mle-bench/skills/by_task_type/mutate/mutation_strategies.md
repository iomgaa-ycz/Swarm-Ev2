# Genetic Mutation Strategies

Mutation introduces **controlled variation** to a solution, enabling exploration of the fitness landscape while avoiding catastrophic changes.

## Mutation Types

### 1. Small Perturbation (Exploitation)
**Goal**: Fine-tune hyperparameters without changing the core approach

**Examples**:
- Adjust learning rate: `lr *= random.uniform(0.5, 2.0)`
- Modify regularization strength: `lambda_l2 += random.gauss(0, 0.01)`
- Change dropout rate: `dropout = clip(dropout + random.gauss(0, 0.1), 0.0, 0.5)`

**Use Case**: When current solution is near-optimal, minor tweaks may yield improvements

**Example Code**:
```python
# [SECTION: OPTIMIZER]
# Original: lr=0.001
# Mutation: Multiply by random factor [0.5, 2.0]
import random
lr = 0.001 * random.uniform(0.5, 2.0)  # Range: [0.0005, 0.002]
optimizer = Adam(learning_rate=lr)
```

### 2. Component Swap (Moderate Exploration)
**Goal**: Replace a module with an alternative while preserving structure

**Examples**:
- Activation: ReLU → LeakyReLU / GELU / Swish
- Optimizer: Adam → AdamW / SGD with momentum
- Loss function: MSE → Huber loss
- Model layer: Dense → BatchNorm + Dense + Dropout

**Use Case**: When specific components are suspected to be suboptimal

**Example Code**:
```python
# [SECTION: MODEL]
# Original: ReLU activation
# Mutation: Swap to LeakyReLU
model = Sequential([
    Dense(128, activation='relu'),  # Original
    # Mutated:
    Dense(128),
    LeakyReLU(alpha=0.2),
    Dense(64, activation='relu'),
])
```

### 3. Structural Change (High Exploration)
**Goal**: Modify architecture by adding/removing layers or components

**Examples**:
- Add a residual connection
- Insert a dropout layer after activations
- Remove redundant layers
- Increase/decrease model depth

**Use Case**: When fitness plateau suggests architectural limitations

**Example Code**:
```python
# [SECTION: MODEL]
# Original: 2-layer MLP
# Mutation: Add a hidden layer (increase depth)
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),   # ← Added layer
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax'),
])
```

## Mutation Intensity

Balance **exploration** (large changes) with **exploitation** (fine-tuning):

| Mutation Type | Intensity | Fitness Change Risk | Use Case |
|---------------|-----------|---------------------|----------|
| Small Perturbation | Low (5-20%) | Low | Near-optimal solutions |
| Component Swap | Medium (30-50%) | Medium | Suspected bottlenecks |
| Structural Change | High (50%+) | High | Fitness plateau, exploration phase |

## Target Gene Selection

Choose which gene block to mutate based on **performance bottleneck analysis**:

### Heuristics

1. **Loss Plateau** → Mutate `MODEL` (optimizer aspect) or `TRAIN` (lr_schedule aspect)
   - If validation loss stops decreasing, adjust optimization strategy

2. **Overfitting** (val << train) → Mutate `MODEL` (regularization aspect) or `TRAIN` (early_stopping aspect)
   - Add/increase dropout, L2 penalty, or data augmentation

3. **Underfitting** (both metrics low) → Mutate `MODEL` (architecture aspect)
   - Increase model capacity (more layers, wider layers)

4. **Slow Convergence** → Mutate `MODEL` (optimizer aspect) or `TRAIN` (lr_schedule aspect)
   - Try better weight initialization (Xavier, He) or adaptive optimizers

5. **Poor Data Representation** → Mutate `DATA` (feature_engineering aspect)
   - Add feature engineering, data augmentation, or normalization

## Adaptive Mutation Rate

Adjust mutation magnitude based on fitness:

```python
if current_fitness > parent_fitness:
    # Solution is improving → small perturbations
    mutation_scale = 0.1
else:
    # Fitness stagnant → larger exploration
    mutation_scale = 0.5
```

## Example Mutations by Gene Block

### DATA
```python
# Aspect: feature_engineering
# Original: Standard normalization
X = (X - X.mean()) / X.std()

# Mutation: Add feature interactions
X['feature_A_B'] = X['feature_A'] * X['feature_B']
X['feature_C_squared'] = X['feature_C'] ** 2
```

### MODEL
```python
# Aspect: architecture
# Original: Simple MLP
model = Sequential([Dense(64), Dense(num_classes)])

# Mutation: Add BatchNorm and Dropout
model = Sequential([
    Dense(64), BatchNormalization(), Dropout(0.3),
    Dense(num_classes)
])

# Aspect: loss_function (within MODEL block)
# Original: Categorical cross-entropy
criterion = CategoricalCrossentropy()
# Mutation: Focal loss (for imbalanced classes)
criterion = FocalLoss(gamma=2.0)

# Aspect: optimizer (within MODEL block)
# Original: Adam with default lr
optimizer = Adam(lr=0.001)
# Mutation: AdamW with weight decay
optimizer = AdamW(lr=0.001, weight_decay=0.01)

# Aspect: regularization (within MODEL block)
# Original: No regularization
# Mutation: Add L2 + Dropout
model = Sequential([
    Dense(64, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(num_classes)
])
```

### TRAIN
```python
# Aspect: cv_strategy
# Original: Simple train/test split
model.fit(X_train, y_train)

# Mutation: 5-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# Aspect: early_stopping
# Original: Fixed epochs
model.fit(X, y, epochs=50)

# Mutation: Early stopping + LR schedule
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(factor=0.5, patience=5)
model.fit(X, y, epochs=100, callbacks=[early_stop, lr_schedule])
```

### POSTPROCESS
```python
# Aspect: ensemble
# Original: Single model prediction
predictions = model.predict(X_test)

# Mutation: Average ensemble from CV folds
predictions = np.mean([m.predict(X_test) for m in fold_models], axis=0)

# Aspect: threshold
# Original: Default 0.5 threshold
preds = (proba > 0.5).astype(int)

# Mutation: Optimized threshold search
from sklearn.metrics import f1_score
best_thresh = max(np.arange(0.3, 0.7, 0.01), key=lambda t: f1_score(y_val, proba_val > t))
preds = (proba > best_thresh).astype(int)
```

## Validation After Mutation

Always verify the mutated solution:
1. **Syntax check**: Code runs without errors
2. **Shape compatibility**: Ensure dimensions match
3. **Metric extraction**: Confirm validation metric is printed
4. **Fitness comparison**: Evaluate if mutation improved performance

## Mutation Guidelines

✅ **DO**:
- Mutate one gene block at a time (isolate effects)
- Preserve working components from high-fitness parents
- Document mutation rationale in comments
- Test mutations on a small dataset first (sanity check)

❌ **DON'T**:
- Apply multiple large mutations simultaneously (hard to debug)
- Mutate critical blocks randomly (e.g., don't break DATA loading)
- Ignore compatibility between blocks (e.g., MODEL and LOSS must align)
- Introduce syntax errors or undefined variables
