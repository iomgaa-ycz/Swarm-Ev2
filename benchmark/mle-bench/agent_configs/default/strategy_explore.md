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

### Hyperparameter Configuration
- Start with default hyperparameters or proven configurations
- Use learning rate scheduling if training neural networks
- Apply regularization (dropout, L2) to prevent overfitting
- Set random seeds for reproducibility

## 3. Iterate Based on Feedback

| Symptom | Action |
|---------|--------|
| Poor validation metric | Check data leakage, try different model, add feature engineering |
| Overfitting (train >> val) | Increase regularization, reduce complexity, add augmentation |
| Underfitting (both low) | Increase capacity, add features, reduce regularization |
