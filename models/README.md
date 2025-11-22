# Model Checkpoints

This directory contains trained model checkpoints.

## Files

- `best_model.pth` - Best performing model (saved during training)

## Usage

Load a model checkpoint:

```python
import torch
from src.models.densenet import DenseNet121
from src import config

# Create model
model = DenseNet121(num_classes=len(config.CLASS_NAMES))

# Load checkpoint
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint)
```
