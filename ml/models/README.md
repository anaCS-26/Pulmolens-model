# Model Checkpoints

This directory contains trained model checkpoints and deployment artifacts.

## Models

### 1. Attention DenseNet (Best Model)
- **Filename**: `attention_densenet_asl_20251121_213351_best.pth`
- **Architecture**: DenseNet121 + CBAM (Convolutional Block Attention Module)
- **Loss Function**: Asymmetric Loss (ASL)
- **Performance**:
    - **Weighted Recall**: 0.50
    - **Weighted F1**: 0.41
    - **Focus**: Optimized for high sensitivity (Recall) to minimize missed diagnoses.

### 2. ONNX Deployment Model
- **Filename**: `pulmolens_best.onnx`
- **Format**: ONNX (Open Neural Network Exchange) Opset 18
- **Input Shape**: `[1, 3, 512, 512]`
- **Use Case**: Fast inference in production environments (e.g., Azure, AWS, Edge devices).

## Usage

### Load PyTorch Model
```python
import torch
from src.models.densenet import AttentionDenseNet
from src import config

# Initialize model
model = AttentionDenseNet(num_classes=len(config.CLASS_NAMES))

# Load checkpoint
checkpoint = torch.load('models/attention_densenet_asl_20251121_213351_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Run ONNX Inference
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/pulmolens_best.onnx")
input_name = session.get_inputs()[0].name

# Dummy input
x = np.random.randn(1, 3, 512, 512).astype(np.float32)
outputs = session.run(None, {input_name: x})
```
