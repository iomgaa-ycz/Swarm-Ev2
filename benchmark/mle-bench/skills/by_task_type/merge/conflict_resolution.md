# Conflict Resolution in Gene Merging

When combining gene blocks from two parents, conflicts often arise due to **incompatible assumptions** or **mismatched interfaces**. This guide provides systematic strategies for resolving such conflicts.

## Common Conflict Types

### 1. Data Format Mismatch
**Symptom**: Parent A expects tabular data (DataFrame), Parent B expects image tensors

**Resolution Strategies**:
- **Adaptive Preprocessing**: Transform data to match the dominant format
  ```python
  # If using Parent B's CNN model with Parent A's tabular data
  # → Convert tabular to pseudo-images (reshape + padding)
  X_image = X_tabular.values.reshape(-1, height, width, channels)
  ```

- **Hybrid Input**: Create a model that accepts both formats
  ```python
  # Dual-input model
  tabular_input = Input(shape=(num_features,))
  image_input = Input(shape=(height, width, channels))
  merged = concatenate([Dense(64)(tabular_input), Flatten()(Conv2D(...)(image_input))])
  ```

### 2. Model Output Shape Incompatibility
**Symptom**: Parent A's model outputs logits (shape: [batch, num_classes]), Parent B's loss expects probabilities

**Resolution Strategies**:
- **Add Activation Layer**: Insert softmax/sigmoid between model and loss
  ```python
  logits = model(inputs)
  probabilities = tf.nn.softmax(logits)  # Convert to probabilities
  loss = categorical_crossentropy(y_true, probabilities)
  ```

- **Adjust Loss Function**: Use `from_logits=True` parameter
  ```python
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  ```

### 3. Hyperparameter Scale Mismatch
**Symptom**: Parent A uses learning_rate=0.001 (small model), Parent B uses 0.1 (large model)

**Resolution Strategies**:
- **Favor Higher-Fitness Parent**: Use hyperparameters from the parent with better fitness
- **Average**: `lr = (lr_a + lr_b) / 2`
- **Adaptive**: Scale based on model complexity:
  ```python
  num_params = sum(p.numel() for p in model.parameters())
  lr = 0.01 if num_params < 1e6 else 0.001
  ```

### 4. Regularization Overlap
**Symptom**: Parent A uses L2 regularization, Parent B uses Dropout

**Resolution Strategies**:
- **Combine Both**: Apply L2 to weights and Dropout to activations (often complementary)
- **Choose One**: If gene plan selects one, respect it (don't add extra regularization)
- **Hybrid**: Use L2 for lower layers, Dropout for higher layers

## Resolution Decision Tree

```
Conflict Detected
    ├─ Is it a data format issue?
    │   └─ Yes → Transform data to match model requirements
    │
    ├─ Is it a model-loss mismatch?
    │   └─ Yes → Adjust activation or loss configuration
    │
    ├─ Is it a hyperparameter conflict?
    │   └─ Yes → Favor higher-fitness parent or average
    │
    └─ Is it unresolvable?
        └─ Yes → Fallback to higher-fitness parent's block
```

## Fallback Strategy

When a conflict **cannot be resolved** without major refactoring:
1. **Check fitness scores**: `parent_a.metric_value` vs. `parent_b.metric_value`
2. **Favor higher-fitness parent**: Replace the problematic block entirely
3. **Document decision**: Add a comment explaining the fallback

**Example**:
```python
# [SECTION: MODEL]
# Conflict: Parent A's CNN incompatible with Parent B's tabular data
# Resolution: Using Parent A's model (higher fitness: 0.92 vs. 0.85)
# and adapting data preprocessing to generate image-compatible input
model = parent_a.model  # Higher fitness
```

## Validation Steps

After resolving conflicts:
1. **Syntax Check**: Ensure code runs without import/syntax errors
2. **Shape Validation**: Print intermediate tensor shapes to verify compatibility
3. **Sanity Test**: Run a forward pass on dummy data
4. **Metric Extraction**: Confirm the code prints a validation metric

## Anti-Patterns to Avoid

❌ **Ignoring the gene plan**: Don't arbitrarily replace blocks without following the plan
❌ **Over-engineering**: Don't add unnecessary complexity (e.g., ensemble when a single model suffices)
❌ **Silent failures**: Don't suppress errors; fix the root cause
❌ **Hardcoding**: Avoid hardcoded values specific to one parent's dataset

## Example: Complete Conflict Resolution

**Setup**:
- Parent A: XGBoost on tabular data (fitness: 0.88)
- Parent B: ResNet-18 on images (fitness: 0.91)
- Gene Plan: `{"DATA": "A", "MODEL": "B", "TRAIN": "B", "POSTPROCESS": "A"}`

**Conflict**: ResNet-18 expects image input, but DATA comes from Parent A (tabular)

**Resolution**:
```python
# [SECTION: DATA]
# Parent A's tabular preprocessing
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1).values  # Shape: (N, num_features)

# Adapt to image format for Parent B's ResNet-18
# Reshape to pseudo-image: (N, 1, height, width)
height = int(np.sqrt(X.shape[1]))
width = X.shape[1] // height
X_image = X[:, :height*width].reshape(-1, 1, height, width)
X_image = np.repeat(X_image, 3, axis=1)  # Convert to 3 channels (RGB)

# [SECTION: MODEL]
# Parent B's ResNet-18 (expects 3-channel images)
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**Result**: Conflict resolved by transforming tabular data into pseudo-images compatible with CNN.
