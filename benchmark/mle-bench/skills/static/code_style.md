# Code Style Guidelines

## Naming Conventions

### Variables
- Use **descriptive names** that convey meaning
- ✅ `train_features`, `validation_scores`, `model_predictions`
- ❌ `x`, `temp`, `data1`, `arr`

### Functions and Classes
- Functions: `snake_case` (e.g., `preprocess_data`, `train_model`)
- Classes: `PascalCase` (e.g., `CustomTransformer`, `ModelEnsemble`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITER`, `RANDOM_SEED`)

## Code Organization

### Section Markers
Organize code into logical sections using **`# [SECTION: NAME]`** markers:

```python
# [SECTION: DATA]
# Data loading and preprocessing

# [SECTION: MODEL]
# Model definition and configuration

# [SECTION: LOSS]
# Loss function definition

# [SECTION: OPTIMIZER]
# Optimizer configuration

# [SECTION: REGULARIZATION]
# Regularization techniques (dropout, L2, etc.)

# [SECTION: INITIALIZATION]
# Weight initialization

# [SECTION: TRAINING_TRICKS]
# Training loop, early stopping, learning rate scheduling
```

### Function Length
- Keep functions **short and focused** (< 50 lines)
- Extract complex logic into helper functions
- Use meaningful function names

## Comments and Documentation

### When to Comment
- ✅ **DO** comment complex logic or non-obvious decisions
- ✅ **DO** explain hyperparameter choices
- ✅ **DO** document assumptions and constraints
- ❌ **DON'T** comment obvious code (e.g., `x = 0  # set x to zero`)

### Example
```python
# Use stratified split to preserve class distribution (imbalanced dataset)
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## Data Handling

### Edge Cases
- **Always handle missing values** (imputation, removal, or indicator variables)
- **Check data types** (convert if necessary)
- **Validate inputs** (e.g., ensure no NaNs in critical columns)

### Example
```python
# Handle missing values: fill numerical with median, categorical with mode
for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)

for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
```

## Efficiency

### Data Structures
- Use **appropriate data types** to minimize memory:
  ```python
  # Downcast integers and floats
  df['int_col'] = df['int_col'].astype('int32')
  df['float_col'] = df['float_col'].astype('float32')
  ```

### Avoid Unnecessary Loops
- Prefer **vectorized operations** (NumPy, Pandas) over explicit loops
- ✅ `df['new_col'] = df['col1'] * df['col2']`
- ❌ `for i in range(len(df)): df.loc[i, 'new_col'] = df.loc[i, 'col1'] * df.loc[i, 'col2']`

## Error Handling

### Graceful Degradation
- Use `try-except` for operations that might fail:
  ```python
  try:
      model.fit(X_train, y_train)
  except Exception as e:
      print(f"Training failed: {e}")
      # Fallback to simpler model or default behavior
  ```

### Informative Error Messages
- Provide context when errors occur:
  ```python
  assert len(train) > 0, "Training data is empty!"
  assert 'target' in train.columns, "Missing 'target' column in training data"
  ```

## Reproducibility

### Random Seeds
Always set random seeds at the beginning:
```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# For PyTorch
import torch
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# For TensorFlow
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
```

## Output

### Print Statements
- Use **informative print statements** for debugging:
  ```python
  print(f"Training data shape: {X_train.shape}")
  print(f"Validation metric: {score:.6f}")
  ```

- Avoid excessive output (only essential information)

### Submission File
- **Always verify submission format** before saving:
  ```python
  submission = pd.DataFrame({'id': test_ids, 'target': predictions})
  assert submission.shape[0] == len(test_ids), "Submission row count mismatch!"
  assert not submission.isnull().any().any(), "Submission contains NaN values!"
  submission.to_csv('./submission/submission.csv', index=False)
  ```
