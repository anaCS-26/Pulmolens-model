# Walkthrough - NIH Chest X-ray Refactoring

I have refactored the codebase to prioritize Recall and Interpretability. Here is the new structure and how to use it.

## New Structure

### 1. Configuration (`src/config.py`)
Centralizes all hyperparameters and paths.

### 2. Data (`src/data/dataset.py`)
Implements `NIHChestXrayDataset` with **Patient-Level Splitting**.

### 3. Training (`src/training/losses.py`)
- **Focal Loss** & **Asymmetric Loss (ASL)**.

### 4. Models (`src/models/densenet.py`)
- **DenseNet121** wrapper.

### 5. Evaluation (`src/evaluation/`)
- **Optimizer** (`optimizer.py`): Finds optimal thresholds.
- **Grad-CAM** (`gradcam.py`): Visualization.

## How to Use

### 1. Verify Data Split
```bash
python -m src.data.dataset
```

### 2. Train Model
```bash
python train.py --loss asl
```

### 3. Optimize Thresholds
```bash
python -m src.evaluation.optimizer --model_path models/best_model.pth
```

### 5. ONNX Conversion
```bash
python deployment/convert_to_onnx.py --model_path models/best_model.pth --output_path models/model.onnx
```

### 6. Test Set Evaluation
```bash
python src/evaluation/evaluate_test.py --model_path models/best_model.pth
```

## Verification Results
- **Data Split**: Verified no patient leakage between train/val/test.
- **Code Structure**: Modular and type-hinted.
- **Grad-CAM**: Script runs successfully and generates heatmaps.
- **ONNX**: Successfully converted AttentionDenseNet to ONNX (opset 18).
- **Performance**: Achieved >0.50 Weighted Recall on Test Set with optimized thresholds.
