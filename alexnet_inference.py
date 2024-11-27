import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from FoodFresh_dataProcess import num_fruit_classes, test_loader

# Load the pretrained AlexNet models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freshness classification model
model_fresh = models.alexnet()
model_fresh.classifier[6] = nn.Linear(4096, 1)  # Single output for Freshness classification
state_dict = torch.load("alexnet_fresh.pth", map_location=device, weights_only=True)
new_state_dict = {name.replace("base_model.", ""): content for name, content in state_dict.items()}
model_fresh.load_state_dict(new_state_dict)
model_fresh.to(device)
model_fresh.eval()
print("Freshness classification model loaded")

# Type classification model
model_type = models.alexnet()
model_type.classifier[6] = nn.Linear(4096, num_fruit_classes)  # Type classification
state_dict = torch.load("alexnet_type.pth", map_location=device, weights_only=True)
new_state_dict = {name.replace("base_model.", ""): content for name, content in state_dict.items()}
model_type.load_state_dict(new_state_dict)
model_type.to(device)
model_type.eval()
print("Type classification model loaded")

# Inference code
total_samples = 0
correct_fresh = 0
correct_Type = 0

transform = transforms.ToPILImage()  # Convert tensor to PIL image

with torch.no_grad():
    for i, (images, label_type, label_fresh) in enumerate(test_loader):
        images = images.to(device)
        label_fresh = label_fresh.to(device)  # Freshness labels
        label_type = label_type.to(device)  # Type labels

        batch_size = images.size(0)

        # Freshness classification inference
        output_fresh = model_fresh(images).squeeze(1)
        predicted_fresh = (torch.sigmoid(output_fresh) > 0.5).long()
        correct_fresh += (predicted_fresh == label_fresh).sum().item()

        # Type classification inference
        output_type = model_type(images)
        _, predicted_type = torch.max(output_type, 1)
        correct_Type += (predicted_type == label_type).sum().item()

        # Save images for every 200 samples
        for j in range(batch_size):
            if (total_samples + j) % 200 == 0:
                # Convert image tensor to PIL image
                img = transform(images[j].cpu())

                # Write ground truth and predicted labels on the image
                draw = ImageDraw.Draw(img)
                text = (
                    f"Freshness: True={label_fresh[j].item()}, Pred={predicted_fresh[j].item()} \n"
                    f"Type: True={label_type[j].item()}, Pred={predicted_type[j].item()}"
                )
                draw.text((10, 10), text, fill="red")

                # Save the image
                img.save(f"alexnet_testing/sample_{total_samples - batch_size + j}.png")
        
        total_samples += batch_size

# Calculate and print accuracies
accuracy_fresh = correct_fresh / total_samples * 100
accuracy_type = correct_Type / total_samples * 100

print(f"Freshness Classification Accuracy: {accuracy_fresh:.2f}%")
print(f"Type Classification Accuracy: {accuracy_type:.2f}%")