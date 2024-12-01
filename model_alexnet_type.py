import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights

from FoodFresh_dataProcess import num_fruit_classes, train_loader, val_loader, test_loader


# Load pretrained AlexNet
class custom_AlexNet(nn.Module):
    def __init__(self, num_fruit_classes):
        super(custom_AlexNet, self).__init__()
        self.base_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        #self.base_model = models.alexnet()
        self.base_model.classifier[6] = nn.Linear(4096, num_fruit_classes)

    def forward(self, x):
        output = self.base_model(x)
        return output


# Prepare for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_AlexNet(num_fruit_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20


# Training loop
for epoch in range(num_epochs):
    ### Training Phase ###
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, labels_type, labels_fresh in train_loader:
        # Move data to the device
        inputs = inputs.to(device)
        labels = (labels_type).to(device).long()  # Multiclass labels (long for CrossEntropyLoss)       

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        running_loss += loss.item() * inputs.size(0)

        # Track accuracy for multiclass classification
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate average losses and accuracies
    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    ### Validation Phase ###
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total_samples = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for inputs, labels_type, labels_fresh in val_loader:
            # Move data to the device
            inputs = inputs.to(device)
            labels = (labels_type).to(device).long()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate losses
            val_loss += loss.item() * inputs.size(0)

            # Track accuracy for multiclass classification
            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total_samples += labels.size(0)

    # Calculate average validation losses and accuracies
    val_avg_loss = val_loss / val_total_samples
    val_accuracy = val_correct / val_total_samples

    print(f"Validation Loss: {val_avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

print("Training and validation complete.")
torch.save(model.state_dict(), "alexnet_type.pth")
print("Model saved to alexnet_type.pth")