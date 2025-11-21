"""
Pulmolens - Lung Disease Classification Models

This package contains model architectures for lung disease classification from chest X-rays.
"""

from .model import (
    LungDiseaseModel,
    AttentionDenseNet,
    MultiScaleModel,
    EnsembleModel,
    get_model
)

from .attention_modules import (
    CBAM,
    SEBlock,
    CoordinateAttention,
    ECABlock
)

__all__ = [
    'LungDiseaseModel',
    'AttentionDenseNet',
    'MultiScaleModel',
    'EnsembleModel',
    'get_model',
    'CBAM',
    'SEBlock',
    'CoordinateAttention',
    'ECABlock',
]
