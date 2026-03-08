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
Organize code into 4 logical sections using **`# [SECTION: NAME]`** markers:

```python
# [SECTION: DATA]
# Data loading, cleaning, feature engineering, augmentation
# sklearn: read_csv, missing values, encoding, feature construction, train_test_split
# GBDT: same as sklearn + categorical feature declaration
# DL: same + Dataset/DataLoader, image augmentation, Tokenizer

# [SECTION: MODEL]
# Model definition + all configuration (loss, optimizer, regularization, initialization)
# sklearn: RandomForestClassifier(n_estimators=100, max_depth=10)
# GBDT: lgb.LGBMRegressor(objective='regression', learning_rate=0.05, reg_lambda=0.1)
# DL: nn.Module definition + criterion = ... + optimizer = Adam(lr=...) + weight init

# [SECTION: TRAIN]
# Training/fitting execution + CV strategy + early stopping + LR scheduling
# sklearn: cross_val_score(model, X, y, cv=5)
# GBDT: lgb.train(params, dtrain, callbacks=[...])
# DL: training loop + ReduceLROnPlateau + gradient clipping

# [SECTION: POSTPROCESS]
# Inference + post-processing + ensemble + submission generation
# sklearn: model.predict(X_test), threshold tuning, to_csv()
# GBDT: same as sklearn
# DL: model.eval(), TTA, ensemble, submission generation
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
```

## Output

### Print Statements

Your solution's stdout is parsed by an automated review system. Excessive output (progress bars, per-epoch logs) can cause the review system to miss your validation metric. Follow these rules strictly:

**MUST**:
- Print data shape after loading: `print(f"Train shape: {train.shape}, Test shape: {test.shape}")`
- Print per-fold summary (1 line per fold): `print(f"Fold {i+1}: {metric_name}={value:.6f}")`
- Print final metric as the LAST informational line (see workspace_rules.md for exact format)

**MUST NOT**:
- Print per-epoch training logs. Only print final epoch or every 10th epoch at most
- Use tqdm progress bars on stdout. Either disable them or redirect to stderr:
  ```python
  # Option 1: disable
  for batch in tqdm(loader, disable=True):
  # Option 2: redirect to stderr (won't pollute stdout)
  import sys
  for batch in tqdm(loader, file=sys.stderr):
  ```
- Use `verbose=1` or higher in Keras `model.fit()`. Use `verbose=0`
- Print large arrays, dataframes, or model summaries

**Target**: Total stdout should be under 5000 characters. The shorter, the better.

### Submission File
- **Always verify submission format** before saving:
  ```python
  submission = pd.DataFrame({'id': test_ids, 'target': predictions})
  assert submission.shape[0] == len(test_ids), "Submission row count mismatch!"
  assert not submission.isnull().any().any(), "Submission contains NaN values!"
  submission.to_csv('./submission/submission.csv', index=False)  # See workspace_rules.md for path
  ```
