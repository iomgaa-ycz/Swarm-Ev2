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

1. **Loss Plateau** → Mutate `OPTIMIZER` or `LEARNING_RATE`
   - If validation loss stops decreasing, adjust optimization strategy

2. **Overfitting** (val << train) → Mutate `REGULARIZATION`
   - Add/increase dropout, L2 penalty, or data augmentation

3. **Underfitting** (both metrics low) → Mutate `MODEL`
   - Increase model capacity (more layers, wider layers)

4. **Slow Convergence** → Mutate `INITIALIZATION` or `OPTIMIZER`
   - Try better weight initialization (Xavier, He) or adaptive optimizers

5. **Poor Data Representation** → Mutate `DATA`
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
# Original: Standard normalization
X = (X - X.mean()) / X.std()

# Mutation: Add feature interactions
X['feature_A_B'] = X['feature_A'] * X['feature_B']
X['feature_C_squared'] = X['feature_C'] ** 2
```

### MODEL
```python
# Original: Simple MLP
model = Sequential([Dense(64), Dense(num_classes)])

# Mutation: Add BatchNorm and Dropout
model = Sequential([
    Dense(64), BatchNormalization(), Dropout(0.3),
    Dense(num_classes)
])
```

### LOSS
```python
# Original: Categorical cross-entropy
loss = CategoricalCrossentropy()

# Mutation: Focal loss (for imbalanced classes)
loss = FocalLoss(gamma=2.0)
```

### OPTIMIZER
```python
# Original: Adam with default lr
optimizer = Adam(lr=0.001)

# Mutation: AdamW with weight decay
optimizer = AdamW(lr=0.001, weight_decay=0.01)
```

### REGULARIZATION
```python
# Original: L2 penalty
regularizer = l2(0.01)

# Mutation: Increase L2 strength + add early stopping
regularizer = l2(0.05)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
```

### INITIALIZATION
```python
# Original: Default initialization
model = Sequential([Dense(64)])

# Mutation: He initialization
model = Sequential([
    Dense(64, kernel_initializer='he_normal')
])
```

### TRAINING_TRICKS
```python
# Original: Fixed learning rate
model.fit(X, y, epochs=50, lr=0.001)

# Mutation: Learning rate schedule
lr_schedule = ReduceLROnPlateau(factor=0.5, patience=5)
model.fit(X, y, epochs=50, callbacks=[lr_schedule])
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
