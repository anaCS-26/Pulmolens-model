# Results

This directory contains evaluation results and performance metrics.

## Files

- `eval_results*.txt` - Classification reports and AUC scores
- `optimal_thresholds.json` - Per-class optimal thresholds
- `training_logs/` - Training history and metrics

## Generating Results

Run evaluation on a trained model:

```bash
python scripts/evaluate_enhanced.py \
    --model_path models/your_model.pth \
    --optimize_threshold
```

This will generate:
- Per-class performance metrics (AUC, PR-AUC, F1)
- Classification report with optimized thresholds
- Optimal thresholds saved to JSON
