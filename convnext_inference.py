import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.metrics import f1_score, classification_report

from FoodFresh_dataProcess import num_fruit_classes, test_loader

# Lists of freshness and type
l_fresh = ["fresh", "rotten"]
l_type = ["apple", "banana", "bittergroud", "capsicum", "cucumber", "okra", "oranges", "potato", "tomato"]

# Load the pretrained ConvNeXt model
class custom_ConvNeXt(nn.Module):
    def __init__(self, num_fruit_classes):
        super(custom_ConvNeXt, self).__init__()
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.classify_fresh = nn.Linear(in_features, 1)
        self.classify_type = nn.Linear(in_features, num_fruit_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        output_fresh = self.classify_fresh(x)
        output_type = self.classify_type(x)
        return output_fresh, output_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_ConvNeXt(num_fruit_classes)
state_dict = torch.load("convnext.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded")

# Inference code
total_samples = 0
correct_fresh = 0
correct_type = 0
truth_fresh = []
pred_fresh = []
truth_type = []
pred_type = []

transform = transforms.ToPILImage()  # Convert tensor to PIL image

with torch.no_grad():
    for i, (images, label_type, label_fresh) in enumerate(test_loader):
        images = images.to(device)
        label_fresh = label_fresh.to(device)  # Freshness labels
        label_type = label_type.to(device)  # Type labels

        batch_size = images.size(0)

        # Freshness classification inference
        output_fresh, output_type = model(images)
        output_fresh = output_fresh.squeeze(1)
        predicted_fresh = (torch.sigmoid(output_fresh) > 0.5).long()
        correct_fresh += (predicted_fresh == label_fresh).sum().item()
        _, predicted_type = torch.max(output_type, 1)
        correct_type += (predicted_type == label_type).sum().item()

        # Store ground truth and predictions for F1 score calculation
        truth_fresh.extend(label_fresh.cpu().tolist())
        pred_fresh.extend(predicted_fresh.cpu().tolist())
        truth_type.extend(label_type.cpu().tolist())
        pred_type.extend(predicted_type.cpu().tolist())

        # Save images for every 100 samples
        for j in range(batch_size):
            if (total_samples + j) % 100 == 0:
                # Convert image tensor to PIL image
                img = transform(images[j].cpu())

                # Write ground truth and predicted labels on the image
                draw = ImageDraw.Draw(img)
                # text = (
                #     f"Freshness: True={label_fresh[j].item()}, Pred={predicted_fresh[j].item()} \n"
                #     f"Type: True={label_type[j].item()}, Pred={predicted_type[j].item()}"
                # )
                text = (
                    f"Prediction: {l_fresh[predicted_fresh[j].item()]} {l_type[predicted_type[j].item()]}\n"
                    f"Truth: {l_fresh[label_fresh[j].item()]} {l_type[label_type[j].item()]}"
                )
                draw.text((10, 10), text, fill="red")

                # Save the image
                img.save(f"convnext_testing/sample_{total_samples + j}.png")
        
        total_samples += batch_size

# Calculate and print accuracies/F1
accuracy_fresh = correct_fresh / total_samples * 100
accuracy_type = correct_type / total_samples * 100
f1_fresh = f1_score(truth_fresh, pred_fresh, average="binary")
f1_type = f1_score(truth_type, pred_type, average="weighted")
report_fresh = classification_report(truth_fresh, pred_fresh, target_names=l_fresh, zero_division=0)
report_type = classification_report(truth_type, pred_type, target_names=l_type, zero_division=0)
print(f"Freshness Classification: Accuracy={accuracy_fresh:.2f}%, F1={f1_fresh:.4f}")
print("Classification Report for Freshness Classification:")
print(report_fresh)
print(f"Type Classification: Accuracy={accuracy_type:.2f}%, F1={f1_type:.4f}")
print("Classification Report for Type Classification:")
print(report_type)
