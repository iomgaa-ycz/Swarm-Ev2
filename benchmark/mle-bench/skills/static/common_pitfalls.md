# Common Pitfalls

## PyTorch Image Processing
- `transforms.ToTensor()` already converts [0,255] → [0.0,1.0]; do NOT add `/ 255.0` (double normalization)
- Apply `Normalize(mean, std)` AFTER `ToTensor()`

## Data Loading
- Read `sample_submission.csv` FIRST to understand the required output format
- For image tasks: verify actual file extensions (`.jpg`/`.png`/`.tif` may differ from what you expect)
