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
- Your solution can use any relevant machine learning packages
- Common packages available: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `pytorch`, `tensorflow`, `pandas`, `numpy`, etc.
- If using deep learning, ensure GPU availability is handled gracefully

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
