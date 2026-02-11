# Explore Strategy

When creating a new solution or improving an existing one:

## 1. Analyze the Context

### First Draft (No Previous Attempt)
- Read the task description carefully to understand the goal and evaluation metric
- Analyze the data preview (if provided) to understand data characteristics
- Choose an appropriate baseline model based on the problem type:
  - Tabular data → Gradient boosting (XGBoost, LightGBM, CatBoost)
  - Image data → CNNs (ResNet, EfficientNet)
  - Text data → Transformers (BERT, RoBERTa)
  - Time series → ARIMA, LSTM, or Prophet

### Improving Previous Attempt
- Review the previous solution's code and execution results
- Identify what worked well (preserve successful components)
- Analyze the validation metric to understand performance gaps
- Check for common issues: overfitting, underfitting, data leakage, inefficient preprocessing

### Debugging Buggy Code
- Examine the error message or exception carefully
- Identify the root cause (syntax error, shape mismatch, missing import, etc.)
- Fix the bug with minimal changes (avoid unnecessary refactoring)
- Ensure the fix doesn't introduce new issues

## 2. Design the Solution

### Data Preprocessing
- Handle missing values appropriately (imputation, removal, or indicator variables)
- Encode categorical variables (one-hot, label encoding, or target encoding)
- Normalize/standardize numerical features when necessary
- Create informative features based on domain knowledge
- Split data into train/validation sets (use stratification for classification)

### Model Selection
- Choose a model appropriate for the task and data type
- Start simple (e.g., logistic regression, decision tree) for baselines
- Use proven architectures for complex tasks (e.g., ResNet for images, XGBoost for tabular)
- Consider ensemble methods if single models plateau

### Validation Strategy
- **MANDATORY**: Use K-fold cross-validation (k=5) for ALL models
- For deep learning: You may run only a subset of folds if training is slow, but **must run > k/2 folds** (at least 3 out of 5). Report the mean metric of executed folds.
- **CRITICAL**: Validation metric MUST match the competition's evaluation metric exactly
  - WRONG: Using Focal Loss value as "validation metric" for a log_loss competition
  - CORRECT: Using `sklearn.metrics.log_loss()` for a log_loss competition
- Monitor both training and validation metrics to detect overfitting

### Hyperparameter Configuration
- Start with default hyperparameters or proven configurations
- Use learning rate scheduling if training neural networks
- Apply regularization (dropout, L2) to prevent overfitting
- Set random seeds for reproducibility

## 3. Implementation Guidelines

### Code Structure
- Organize code into clear sections using `# [SECTION: NAME]` markers:
  - `DATA`: Data loading and preprocessing
  - `MODEL`: Model definition
  - `LOSS`: Loss function (if applicable)
  - `OPTIMIZER`: Optimizer configuration
  - `REGULARIZATION`: Regularization techniques
  - `INITIALIZATION`: Weight initialization
  - `TRAINING_TRICKS`: Training loop, early stopping, LR scheduling
- Keep code concise and readable
- Add comments for complex or non-obvious logic

### Output Requirements
- Print the validation metric value: `print(f"Validation metric: {metric_value:.6f}")`
- Save predictions to `./submission/submission.csv` in the required format
- Include informative print statements for debugging (but keep output concise)

## 4. Iterate Based on Feedback

### If Validation Metric is Poor
- Check for data leakage or preprocessing errors
- Try a different model or increase model capacity
- Add feature engineering or data augmentation
- Tune hyperparameters (learning rate, regularization strength)

### If Overfitting (Train >> Val)
- Increase regularization (dropout, L2, early stopping)
- Reduce model complexity
- Add data augmentation
- Increase training data (if possible)

### If Underfitting (Both Train and Val are Low)
- Increase model capacity (more layers, more units)
- Add feature engineering
- Reduce regularization
- Train for more epochs

## 5. Quality Checklist

Before finalizing the solution:
- [ ] Code is syntactically correct and executable
- [ ] All imports are included
- [ ] Random seeds are set for reproducibility
- [ ] Validation metric is printed
- [ ] Predictions are saved to `./submission/submission.csv`
- [ ] Code follows the required structure (single-file, section markers)
- [ ] No hardcoded paths or dataset-specific assumptions
