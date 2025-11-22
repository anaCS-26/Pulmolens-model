import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. EfficientNet models will not work.")

from .attention_modules import CBAM, SEBlock, CoordinateAttention


class LungDiseaseModel(nn.Module):
    """Original DenseNet121 model (baseline)"""
    def __init__(self, num_classes=14, pretrained=True):
        super(LungDiseaseModel, self).__init__()
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        
        # Replace the classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)


class AttentionDenseNet(nn.Module):
    """
    DenseNet121 with attention mechanisms.
    Adds CBAM or SE blocks after dense blocks for improved feature representation.
    """
    def __init__(self, num_classes=14, pretrained=True, attention_type='cbam'):
        super(AttentionDenseNet, self).__init__()
        
        # Load pretrained DenseNet121
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        
        # Extract feature extractor (without classifier)
        self.features = densenet.features
        
        # Add attention modules after each dense block
        # DenseNet121 has 4 dense blocks with channels: 256, 512, 1024, 1024
        attention_channels = [256, 512, 1024, 1024]
        
        self.attention_blocks = nn.ModuleList()
        for channels in attention_channels:
            if attention_type == 'cbam':
                self.attention_blocks.append(CBAM(channels))
            elif attention_type == 'se':
                self.attention_blocks.append(SEBlock(channels))
            elif attention_type == 'ca':
                self.attention_blocks.append(CoordinateAttention(channels))
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Get number of features
        num_features = densenet.classifier.in_features
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
        self.attention_type = attention_type
        
    def forward(self, x):
        # Forward through features with attention
        # DenseNet structure: conv0 -> denseblock1 -> transition1 -> ... -> norm5
        
        # Initial conv
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        # Dense block 1 + attention
        x = self.features.denseblock1(x)
        x = self.attention_blocks[0](x)
        x = self.features.transition1(x)
        
        # Dense block 2 + attention
        x = self.features.denseblock2(x)
        x = self.attention_blocks[1](x)
        x = self.features.transition2(x)
        
        # Dense block 3 + attention
        x = self.features.denseblock3(x)
        x = self.attention_blocks[2](x)
        x = self.features.transition3(x)
        
        # Dense block 4 + attention
        x = self.features.denseblock4(x)
        x = self.attention_blocks[3](x)
        
        # Final norm
        x = self.features.norm5(x)
        x = nn.functional.relu(x, inplace=True)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class MultiScaleModel(nn.Module):
    """
    Multi-scale feature extraction model.
    Processes input at different resolutions and combines features.
    """
    def __init__(self, num_classes=14, pretrained=True, backbone='densenet121'):
        super(MultiScaleModel, self).__init__()
        
        if backbone == 'densenet121':
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            num_features = base_model.classifier.in_features
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_features = base_model.fc.in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract features
        if backbone == 'densenet121':
            self.feature_extractor = base_model.features
        else:  # resnet
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        
        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        
        # Combine features
        combined_features = num_features * (1 + 4 + 16)  # 1x1 + 2x2 + 4x4
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Multi-scale pooling
        feat1 = self.pool1(features).flatten(1)
        feat2 = self.pool2(features).flatten(1)
        feat3 = self.pool3(features).flatten(1)
        
        # Concatenate
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for robust predictions.
    Combines DenseNet, ResNet, and optionally EfficientNet.
    """
    def __init__(self, num_classes=14, pretrained=True, use_efficientnet=True):
        super(EnsembleModel, self).__init__()
        
        # DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        densenet_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(densenet_features, num_classes)
        
        # ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(resnet_features, num_classes)
        
        # EfficientNet-B4 (if available)
        self.use_efficientnet = use_efficientnet and TIMM_AVAILABLE
        if self.use_efficientnet:
            self.efficientnet = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
        
        # Ensemble strategy: weighted average
        num_models = 3 if self.use_efficientnet else 2
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x):
        # Get predictions from each model
        out_densenet = self.densenet(x)
        out_resnet = self.resnet(x)
        
        # Normalize weights
        weights_norm = torch.softmax(self.weights, dim=0)
        
        if self.use_efficientnet:
            out_efficientnet = self.efficientnet(x)
            # Weighted combination
            output = (weights_norm[0] * out_densenet + 
                     weights_norm[1] * out_resnet + 
                     weights_norm[2] * out_efficientnet)
        else:
            # Weighted combination
            output = (weights_norm[0] * out_densenet + 
                     weights_norm[1] * out_resnet)
        
        return output


def get_model(model_type='densenet', num_classes=14, pretrained=True, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'densenet', 'attention_densenet', 'multiscale', 'ensemble'
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    if model_type == 'densenet':
        return LungDiseaseModel(num_classes=num_classes, pretrained=pretrained)
    
    elif model_type == 'attention_densenet':
        attention_type = kwargs.get('attention_type', 'cbam')
        return AttentionDenseNet(num_classes=num_classes, pretrained=pretrained, 
                                attention_type=attention_type)
    
    elif model_type == 'multiscale':
        backbone = kwargs.get('backbone', 'densenet121')
        return MultiScaleModel(num_classes=num_classes, pretrained=pretrained, 
                              backbone=backbone)
    
    elif model_type == 'ensemble':
        use_efficientnet = kwargs.get('use_efficientnet', True)
        return EnsembleModel(num_classes=num_classes, pretrained=pretrained,
                            use_efficientnet=use_efficientnet)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...")
    
    x = torch.randn(2, 3, 224, 224)
    
    models_to_test = [
        ('densenet', {}),
        ('attention_densenet', {'attention_type': 'cbam'}),
        ('attention_densenet', {'attention_type': 'se'}),
        ('multiscale', {'backbone': 'densenet121'}),
    ]
    
    for model_type, kwargs in models_to_test:
        model = get_model(model_type, num_classes=14, pretrained=False, **kwargs)
        output = model(x)
        print(f"{model_type} ({kwargs}): Input {x.shape} -> Output {output.shape}")
    
    # Test ensemble separately (slower)
    print("\nTesting ensemble model...")
    ensemble = get_model('ensemble', num_classes=14, pretrained=False, use_efficientnet=False)
    output = ensemble(x)
    print(f"Ensemble: Input {x.shape} -> Output {output.shape}")
    print(f"Ensemble weights: {torch.softmax(ensemble.weights, dim=0)}")
    
    print("\nAll models working correctly!")
