import torch
import torch.nn as nn
import torchvision.models as models

from FoodFresh_dataProcess import num_fruit_classes, train_loader, val_loader, test_loader


# Load pretrained ConvNeXt
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


# Prepare for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_ConvNeXt(num_fruit_classes).to(device)
criterion_fresh = nn.BCEWithLogitsLoss()
criterion_type = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10


# Training loop
for epoch in range(num_epochs):
    ### Training Phase ###
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_fresh = 0
    correct_type = 0
    total_samples = 0

    for inputs, labels_type, labels_fresh in train_loader:
        # Move data to the device
        inputs = inputs.to(device)
        labels_fresh = (labels_fresh).to(device).float()
        labels_type = (labels_type).to(device).long()

        # Forward pass
        preds_fresh, preds_type = model(inputs)
        loss_fresh = criterion_fresh(preds_fresh.squeeze(), labels_fresh)
        loss_type = criterion_type(preds_type, labels_type)
        loss = loss_fresh + loss_type

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        running_loss += loss.item() * inputs.size(0)

        # Track accuracy
        preds_fresh = (torch.sigmoid(preds_fresh.squeeze()) > 0.5).long()
        correct_fresh += (preds_fresh == labels_fresh).sum().item()
        preds_type = torch.argmax(preds_type, dim=1)
        correct_type += (preds_type == labels_type).sum().item()
        total_samples += labels_fresh.size(0)

    # Calculate average losses and accuracies
    avg_loss = running_loss / total_samples
    accuracy_fresh = correct_fresh / total_samples
    accuracy_type = correct_type / total_samples

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_loss:.4f}, Freshness Train Accuracy: {accuracy_fresh:.4f}, Type Train Accuracy: {accuracy_type:.4f}")

    ### Validation Phase ###
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_fresh = 0
    val_correct_type = 0
    val_total_samples = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for inputs, labels_type, labels_fresh in val_loader:
            # Move data to the device
            inputs = inputs.to(device)
            labels_fresh = (labels_fresh).to(device).float()
            labels_type = (labels_type).to(device).long()

            # Forward pass
            preds_fresh, preds_type = model(inputs)
            loss_fresh = criterion_fresh(preds_fresh.squeeze(), labels_fresh)
            loss_type = criterion_type(preds_type, labels_type)
            loss = loss_fresh + loss_type

            # Accumulate losses
            val_loss += loss.item() * inputs.size(0)

            # Track accuracy for multiclass classification
            preds_fresh = (torch.sigmoid(preds_fresh.squeeze()) > 0.5).long()
            val_correct_fresh += (preds_fresh == labels_fresh).sum().item()
            preds_type = torch.argmax(preds_type, dim=1)
            val_correct_type += (preds_type == labels_type).sum().item()
            val_total_samples += labels_fresh.size(0)

    # Calculate average validation losses and accuracies
    val_avg_loss = val_loss / val_total_samples
    val_accuracy_fresh = val_correct_fresh / val_total_samples
    val_accuracy_type = val_correct_type / val_total_samples

    print(f"Validation Loss: {val_avg_loss:.4f}, Freshness Validation Accuracy: {val_accuracy_fresh:.4f}, Type Validation Accuracy: {val_accuracy_type:.4f}")

print("Training and validation complete.")
torch.save(model.state_dict(), "convnext.pth")
print("Model saved to convnext.pth")