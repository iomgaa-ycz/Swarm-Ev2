# Machine Learning Best Practices

## Validation Strategy

### Cross-Validation
- **Always use cross-validation** to estimate generalization performance
- K-Fold (k=5 or k=10) for most tasks
- Stratified K-Fold for imbalanced classification
- Time-series split for temporal data

### Train-Validation Split
- Reserve a held-out validation set when cross-validation is too expensive
- Use stratification for classification tasks
- Ensure data leakage prevention (no future information in past predictions)

## Feature Engineering

### Data-Driven Approach
- Analyze feature distributions and correlations
- Create interaction features based on domain knowledge
- Use polynomial features cautiously (risk of overfitting)
- Handle missing values appropriately (imputation vs. indicator variables)

### Feature Selection
- Remove low-variance features
- Use feature importance from tree-based models
- Consider dimensionality reduction (PCA, t-SNE) for high-dimensional data

## Model Selection

### Start Simple, Then Increase Complexity
1. **Baseline**: Simple models (Logistic Regression, Decision Tree)
2. **Intermediate**: Ensemble methods (Random Forest, Gradient Boosting)
3. **Advanced**: Deep learning, stacking, complex architectures

### Model Choice Guidelines
- **Tabular data**: XGBoost, LightGBM, CatBoost
- **Image data**: CNNs (ResNet, EfficientNet)
- **Text data**: Transformers (BERT, RoBERTa)
- **Time series**: ARIMA, LSTM, Temporal Fusion Transformer

## Hyperparameter Tuning

### Systematic Approach
- Start with default hyperparameters
- Use grid search or random search for small spaces
- Use Bayesian optimization (Optuna, Hyperopt) for large spaces
- Monitor validation performance during tuning (avoid overfitting to validation set)

### Key Hyperparameters
- **Learning rate**: Most critical for gradient-based models
- **Regularization**: L1/L2 penalties, dropout, early stopping
- **Model capacity**: Number of layers, hidden units, tree depth

## Overfitting Prevention

### Regularization Techniques
- L1/L2 regularization for linear models
- Dropout for neural networks
- Max depth / min samples split for tree-based models
- Early stopping based on validation performance

### Data Augmentation
- Image: rotation, flipping, color jittering
- Text: back-translation, synonym replacement
- Tabular: SMOTE for imbalanced data

## Reproducibility

### Random Seeds
- Set seeds for all random operations:
  ```python
  import random
  import numpy as np
  random.seed(42)
  np.random.seed(42)
  # For PyTorch/TensorFlow: set their respective seeds
  ```

### Deterministic Behavior
- Disable non-deterministic operations when possible
- Document any sources of randomness

## Evaluation Metrics

### Choose Appropriate Metrics
- **Classification**: Accuracy, F1, ROC-AUC, Precision, Recall
- **Regression**: RMSE, MAE, R²
- **Ranking**: NDCG, MAP
- **Always align with the task objective** (e.g., minimize false positives vs. false negatives)

## Debugging and Monitoring

### Sanity Checks
- Verify data loading (check shapes, types, distributions)
- Overfit on a small subset first (ensure model can learn)
- Monitor training loss and validation metric over epochs

### Common Pitfalls
- Data leakage (using test information during training)
- Label leakage (features that directly reveal the target)
- Incorrect train-test splits (e.g., shuffling time-series data)
- Ignoring class imbalance
