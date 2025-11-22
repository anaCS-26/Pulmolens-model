# Pulmolens - Lung Disease Classification

A state-of-the-art deep learning framework for classifying lung diseases from chest X-ray images using the NIH ChestX-ray14 dataset.

## 🎯 Features

- **Advanced Architectures**: DenseNet121 with custom classifier
- **Smart Loss Functions**: Focal Loss, Asymmetric Loss for handling class imbalance
- **Medical Imaging Optimized**: CLAHE preprocessing, specialized augmentation
- **Interpretability**: Grad-CAM++ visualization
- **Performance Optimization**: Threshold optimization for maximum F1-score

## 📁 Project Structure

```
pulmolens/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   └── densenet.py          # DenseNet121 wrapper
│   ├── data/                     # Data loading and augmentation  
│   │   └── dataset.py           # NIH ChestX-ray dataset loader
│   ├── training/                 # Training utilities
│   │   └── losses.py            # Focal, ASL losses
│   ├── evaluation/               # Evaluation tools
│   │   ├── gradcam.py           # Grad-CAM++ visualization
│   │   └── optimizer.py         # Per-class threshold tuning
│   └── config.py                # Configuration
│
├── train.py                      # Main training script
├── models/                       # Trained model checkpoints
├── results/                      # Evaluation results & visualizations
├── data/                         # Dataset
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Train with Attention and Asymmetric Loss (Recommended)
python train.py --model attention_densenet --loss asl
```

### 3. Optimize Thresholds

After training, find the best decision thresholds for each class:

```bash
python -m src.evaluation.optimizer --model_path models/best_model.pth
```

### 4. Evaluate & Visualize

Generate Grad-CAM++ heatmaps (saved to `results/`):

```bash
    --image_path data/images_001/images/00000001_000.png \
    --model_path models/best_model.pth
```

### 5. Deploy (ONNX)

Convert the trained model to ONNX format for deployment:

```bash
python deployment/convert_to_onnx.py \
    --model_path models/best_model.pth \
    --output_path models/pulmolens.onnx
```

## 📊 Performance

| Model | Mean AUC | Mean Recall |
|-------|----------|-------------|
| Baseline DenseNet | 0.8886 | 0.19 |
| **Attention DenseNet + ASL** | **>0.90** | **>0.40** |

## 🔧 Model Options

- `densenet` - DenseNet121 (Standard)
- `attention_densenet` - DenseNet121 + CBAM Attention (Recommended)

## 📖 Documentation

- **[Walkthrough](walkthrough.md)** - Detailed implementation steps, verification results, and Grad-CAM visualizations.

## 📝 License

This project is for educational and research purposes.
