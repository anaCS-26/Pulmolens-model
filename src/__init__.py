"""
Pulmolens - Lung Disease Classification

A comprehensive deep learning framework for classifying lung diseases from chest X-rays.

Features:
- Advanced model architectures (Attention DenseNet, Multi-scale, Ensemble)
- State-of-the-art loss functions (Focal, Asymmetric)
- Medical imaging augmentation (CLAHE, Mixup, CutMix)
- Grad-CAM++ visualization
- Per-class threshold optimization
"""

__version__ = '2.0.0'

# Make key components easily importable
from .models import get_model
from .data import NIHChestXrayDataset
from .training import Config
from .evaluation import GradCAMPlusPlus, ScoreCAM

__all__ = [
    'get_model',
    'NIHChestXrayDataset',
    'Config',
    'GradCAMPlusPlus',
    'ScoreCAM',
]
