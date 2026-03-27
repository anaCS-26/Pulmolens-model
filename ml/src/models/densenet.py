import torch
import torch.nn as nn
import torchvision.models as models
from .attention import CBAM

class DenseNet121(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

class AttentionDenseNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(AttentionDenseNet, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        self.features = self.densenet.features
        
        # Attention Modules
        # DenseNet121 channels after blocks: 256, 512, 1024, 1024
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(1024)
        
        # Classifier
        num_features = self.densenet.classifier.in_features
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features.conv0(x)
        features = self.features.norm0(features)
        features = self.features.relu0(features)
        features = self.features.pool0(features)
        
        # Block 1
        features = self.features.denseblock1(features)
        features = self.cbam1(features)
        features = self.features.transition1(features)
        
        # Block 2
        features = self.features.denseblock2(features)
        features = self.cbam2(features)
        features = self.features.transition2(features)
        
        # Block 3
        features = self.features.denseblock3(features)
        features = self.cbam3(features)
        features = self.features.transition3(features)
        
        # Block 4
        features = self.features.denseblock4(features)
        features = self.cbam4(features)
        features = self.features.norm5(features)
        
        out = nn.functional.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = AttentionDenseNet(num_classes=14)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
