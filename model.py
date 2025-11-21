import torch
import torch.nn as nn
import torchvision.models as models

class LungDiseaseModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(LungDiseaseModel, self).__init__()
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        
        # Replace the classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

if __name__ == "__main__":
    # Simple test
    model = LungDiseaseModel(num_classes=14)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
