# Common Pitfalls (MUST READ)

## PyTorch Image Processing
- `transforms.ToTensor()` already converts [0,255] uint8 → [0.0,1.0] float32
- Do NOT add `/ 255.0` after ToTensor() — this causes double normalization
- `transforms.Normalize(mean, std)` should be applied AFTER ToTensor()

## Validation Metric
- The validation metric MUST match the competition's evaluation metric exactly
- WRONG: Using training loss (e.g., Focal Loss value) as "Validation metric"
- CORRECT: Using `sklearn.metrics.{actual_metric}()` on validation predictions
- Example: If competition uses `log_loss`, use `sklearn.metrics.log_loss(y_val, y_pred_proba)`

## Submission File
- ALWAYS verify row count: `assert len(submission) == len(test_data)`
- ALWAYS verify no NaN: `assert not submission.isnull().any().any()`
- Check column names match sample_submission.csv exactly

## Data Loading
- Read sample_submission.csv FIRST to understand expected output format
- For image tasks: check image extensions (.jpg/.png/.tif may differ)
- For special formats (.txt/.tsv): check delimiter and header presence
