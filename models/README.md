# Model Checkpoints

This directory contains trained model checkpoints.

## Files

- `best_model.pth` - Baseline DenseNet121 model (Mean AUC: 0.8886)
- `attention_densenet_*.pth` - Attention-enhanced DenseNet models
- `ensemble_*.pth` - Ensemble models (DenseNet + ResNet + EfficientNet)

## Usage

Load a model checkpoint:

```python
import torch
from src.models import get_model

# Load checkpoint
checkpoint = torch.load('models/best_model.pth')

# Create model
model = get_model('densenet', num_classes=14)
model.load_state_dict(checkpoint)
```

For new enhanced models:

```python
# Load with config
checkpoint = torch.load('models/attention_densenet_asl_best.pth')
config = checkpoint['config']
model = get_model(config['model_type'], num_classes=14)
model.load_state_dict(checkpoint['model_state_dict'])
```
