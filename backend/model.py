import torch
import torch.nn as nn
import torchvision.models as models

# --- ATTENTION MODULES ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        b, c, _, _ = x.size()
        max_pool_out = torch.max(x.view(b, c, -1), dim=2)[0].view(b, c, 1, 1)
        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# --- DENSENET MODELS ---
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
