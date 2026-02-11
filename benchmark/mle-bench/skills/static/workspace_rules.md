# Workspace Rules

## Directory Structure

- 📁 **`./input/`** - All input data files (read-only)
- 📁 **`./working/`** - Temporary files and intermediate outputs
- 📁 **`./submission/`** - Final predictions (write `submission.csv` here)

## Code Requirements

### File Organization
- The code should be a **single-file Python program**
- Use relative paths for all file operations
- Ensure the code is self-contained and executable as-is

### Execution Behavior
- **MUST print the validation metric value** to stdout
  - Format: `Validation metric: {value}`
  - This is how the system extracts performance metrics
- **MUST save predictions to `./submission/submission.csv`**
  - Follow the required submission format
  - Ensure no missing values or formatting errors

### Available Resources
- **CRITICAL**: ONLY use packages listed in the "Installed Packages" section of System Environment
- Do NOT assume any package is available unless explicitly listed
- If a desired package is not listed, use an alternative from the installed packages
- For deep learning, prefer using GPU when available, and gracefully fall back to CPU if not. (GPU > Multi-core CPU > Single-core CPU)

### Validation Requirements (MANDATORY)

> **CRITICAL**: The validation strategy MUST follow these rules:

1. **For sklearn/tabular models**: Use `StratifiedKFold(n_splits=5)` (classification) or `KFold(n_splits=5)` (regression). Report the **mean** of all folds.
2. **For deep learning models**: Still use K-Fold (k=5), but you may **only run a subset of folds** if training is slow. The number of folds actually executed **MUST be > k/2** (e.g., at least 3 out of 5). Report the mean of the executed folds.
3. **MUST use the competition's evaluation metric** for validation (NOT training loss):
   - If competition uses `log_loss`, validate with `sklearn.metrics.log_loss`
   - If competition uses `AUC`, validate with `sklearn.metrics.roc_auc_score`
   - If competition uses `RMSE`, validate with `sqrt(mean_squared_error)`
   - **NEVER** report training loss (e.g., Focal Loss, BCELoss) as validation metric
4. Print the validation metric: `print(f"Validation metric: {metric_value:.6f}")`

### Best Practices
- Set random seeds for reproducibility
- Handle edge cases (missing values, data type mismatches)
- Use appropriate data types to minimize memory usage
- Add informative print statements for debugging (but keep output concise)

## Example Code Structure

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
# ... other imports

# [SECTION: DATA]
# CRITICAL: Check "Data Overview" section above for exact file names!
# File sizes help identify purpose: large file = training, small file = test
train = pd.read_csv('./input/???')  # Replace ??? with training file from Data Overview
test = pd.read_csv('./input/???')   # Replace ??? with test file from Data Overview
# ... data preprocessing

# [SECTION: MODEL]
model = ...  # model definition

# [SECTION: TRAINING_TRICKS]
# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='...')
print(f"Validation metric: {scores.mean():.6f}")

# Final training and prediction
model.fit(X, y)
predictions = model.predict(X_test)

# [SECTION: OUTPUT]
pd.DataFrame({'id': test_ids, 'target': predictions}).to_csv(
    './submission/submission.csv', index=False
)
```
