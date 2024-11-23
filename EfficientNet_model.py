import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FruitFreshnessModel(nn.Module):
    def __init__(self, num_fruit_classes):
        super(FruitFreshnessModel, self).__init__()
        # Load a pre-trained EfficientNet-B0 model with default weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        # Remove the last classifier layer
        self.backbone.classifier = nn.Identity()
        # Define separate classifiers for fruit type and freshness
        self.fruit_classifier = nn.Linear(num_features, num_fruit_classes)
        self.freshness_classifier = nn.Linear(num_features, 1)  # Binary classification

    def forward(self, x):
        features = self.backbone(x)
        fruit_logits = self.fruit_classifier(features)
        freshness_logits = self.freshness_classifier(features)
        return fruit_logits, freshness_logits
