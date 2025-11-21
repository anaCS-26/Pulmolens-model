# Visualizations

This directory contains Grad-CAM and Grad-CAM++ visualizations showing disease localization.

## Contents

- `gradcam_result_*.png` - Original Grad-CAM visualizations from baseline model
- `viz_*.png` - New Grad-CAM++ visualizations from enhanced models

## Generating Visualizations

Use the enhanced evaluation script:

```bash
python scripts/evaluate_enhanced.py \
    --model_path models/your_model.pth \
    --viz_method gradcam++ \
    --num_viz_samples 20
```

This will create visualizations showing model attention for predicted disease classes.
