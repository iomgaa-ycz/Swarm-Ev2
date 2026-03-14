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
- **Minimize stdout** — disable tqdm, suppress per-epoch logs. Total stdout < 5000 chars. See Code Style Guidelines for detailed rules.

### Available Resources
- **CRITICAL**: ONLY use packages listed in the "Installed Packages" section of System Environment
- Do NOT assume any package is available unless explicitly listed
- If a desired package is not listed, use an alternative from the installed packages
- For deep learning, prefer using GPU when available, and gracefully fall back to CPU if not. (GPU > Multi-core CPU > Single-core CPU)

### Validation Requirements (MANDATORY)

> **CRITICAL**: The validation strategy MUST follow these rules:

1. **Default (ML)**: `StratifiedKFold(n_splits=5)` (classification) or `KFold(n_splits=5)` (regression). Run all 5 folds. Report **mean** of all folds.
2. **Default (DL)**: For deep learning (PyTorch/TensorFlow), **MUST use `n_splits=5` but only train folds 0 and 1** (2 out of 5). This gives 80% training data per fold while saving 60% training time. Report **mean** of the 2 folds. Save both fold checkpoints for ensemble inference.
3. **Exception — grouped data**: If training files share date/session/patient prefixes (few unique groups), use `GroupKFold(groups=...)` instead. Random KFold on grouped data causes severe leakage (CV ~0.99 but test ~0.5).
4. **Metric**: MUST use competition evaluation metric (NOT training loss). Print: `print(f"Validation metric: {metric_value:.6f}")`
5. **Deep Learning — Save & Ensemble**: For neural network models, **MUST save each fold's best checkpoint** (e.g., `torch.save(model.state_dict(), f"working/model_fold{i}.pt")`). At inference time, load ALL fold models and **average predictions** (ensemble). **NEVER** discard fold models and retrain a single model on full data — this wastes CV compute and loses ensemble benefit.

### DataLoader Workers (CRITICAL for Image/Audio Tasks)
- ALWAYS set `num_workers` to use multi-core CPU; `num_workers=0` serializes all I/O and causes GPU starvation
- For large datasets (>10k samples), use 8-16 workers for optimal GPU utilization
- CORRECT:
  ```python
  import os
  num_workers = min(8, os.cpu_count() or 1)
  DataLoader(dataset, batch_size=32, num_workers=num_workers, pin_memory=True)
  ```
- `num_workers=0` is ONLY acceptable when dataset size < 1000 samples

### Large Dataset Sampling
- For datasets > 1M rows, use **random** sampling — NOT sequential truncation
- WRONG: `pd.read_csv(path, nrows=5_000_000)` — reads only early rows, causes severe distribution bias for time-series data
- CORRECT: load with chunking or use `skiprows` with a random mask to get a representative sample

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

# [SECTION: TRAIN]
# Cross-validation + training
scores = cross_val_score(model, X, y, cv=5, scoring='...')
print(f"Validation metric: {scores.mean():.6f}")

# Final training
model.fit(X, y)

# [SECTION: POSTPROCESS]
# Inference + output
predictions = model.predict(X_test)
pd.DataFrame({'id': test_ids, 'target': predictions}).to_csv(
    './submission/submission.csv', index=False
)
```
