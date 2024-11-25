import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
import copy
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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



model = FruitFreshnessModel(num_fruit_classes).to(device)

# Define loss functions
criterion_fruit = nn.CrossEntropyLoss()
criterion_freshness = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, dataloaders, criterion_fruit, criterion_freshness, optimizer, scheduler, num_epochs):
    since = time.time()  # Record the start time
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        epoch_since = time.time()  # Start time for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']

            running_loss = 0.0
            fruit_preds = []
            fruit_labels = []
            freshness_preds = []
            freshness_labels = []

            # Iterate over data
            for inputs, labels_fruit, labels_freshness in dataloader:
                inputs = inputs.to(device)
                labels_fruit = labels_fruit.to(device)
                labels_freshness = labels_freshness.to(device).float()

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_fruit, outputs_freshness = model(inputs)
                    loss_fruit = criterion_fruit(outputs_fruit, labels_fruit)
                    loss_freshness = criterion_freshness(outputs_freshness.squeeze(), labels_freshness)
                    loss = loss_fruit + loss_freshness  # Combined loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Collect predictions and labels for evaluation
                _, preds_fruit = torch.max(outputs_fruit, 1)
                preds_freshness = torch.sigmoid(outputs_freshness).squeeze() >= 0.5

                fruit_preds.extend(preds_fruit.cpu().numpy())
                fruit_labels.extend(labels_fruit.cpu().numpy())
                freshness_preds.extend(preds_freshness.cpu().numpy())
                freshness_labels.extend(labels_freshness.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)

            # Compute metrics
            fruit_report = classification_report(fruit_labels, fruit_preds, output_dict=True, zero_division=0)
            freshness_report = classification_report(freshness_labels, freshness_preds, output_dict=True, zero_division=0)

            idx_to_fruit = {idx: fruit for fruit, idx in fruit_to_idx.items()}
            fruit_preds_names = [idx_to_fruit[pred] for pred in fruit_preds]
            fruit_labels_names = [idx_to_fruit[label] for label in fruit_labels]

            freshness_mapping = {0: "fresh", 1: "rotten"}
            freshness_labels_names = [freshness_mapping[pred] for pred in freshness_preds]
            freshness_preds_names = [freshness_mapping[label] for label in freshness_labels]

            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f'{phase} Fruit Classification Report:')
            print(classification_report( fruit_labels_names , fruit_preds_names, zero_division=0))
            print(f'{phase} Freshness Classification Report:')
            print(classification_report(freshness_labels_names, freshness_preds_names, zero_division=0))

            # Deep copy the model if it has better F1-score
            avg_f1 = (fruit_report['weighted avg']['f1-score'] + freshness_report['weighted avg']['f1-score']) / 2
            if phase == 'val' and avg_f1 > best_f1:
                best_f1 = avg_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model weights
                torch.save(best_model_wts, 'best_model_weights.pth')
                print('Best model weights updated and saved.')

        # Calculate and display the time taken for the current epoch
        epoch_time_elapsed = time.time() - epoch_since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epoch_time_elapsed // 60, epoch_time_elapsed % 60))
        print()

    # Calculate and display the total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val F1: {:4f}'.format(best_f1))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

dataloaders = {
    'train': train_loader,
    'val': val_loader
}

num_epochs = 1 # You can adjust this as needed
model = train_model(model, dataloaders, criterion_fruit, criterion_freshness, optimizer, scheduler, num_epochs)

def evaluate_model(model, dataloader):
    model.eval()
    fruit_preds = []
    fruit_labels = []
    freshness_preds = []
    freshness_labels = []

    idx_to_fruit = {idx: fruit for fruit, idx in fruit_to_idx.items()}
    freshness_mapping = {0: "fresh", 1: "rotten"}

    images_to_show = []
    pred_fruit_labels = []
    true_fruit_labels = []
    pred_freshness_labels = []
    true_freshness_labels = []

    with torch.no_grad():
        for inputs, labels_fruit, labels_freshness in dataloader:
            inputs = inputs.to(device)
            labels_fruit = labels_fruit.to(device)
            labels_freshness = labels_freshness.to(device).float()

            outputs_fruit, outputs_freshness = model(inputs)
            _, preds_fruit = torch.max(outputs_fruit, 1)
            preds_freshness = torch.sigmoid(outputs_freshness).squeeze() >= 0.5

            fruit_preds.extend(preds_fruit.cpu().numpy())
            fruit_labels.extend(labels_fruit.cpu().numpy())
            freshness_preds.extend(preds_freshness.cpu().numpy())
            freshness_labels.extend(labels_freshness.cpu().numpy())

            # For visualization, collect images from the first batch
            if len(images_to_show) == 0:
                # Move inputs to CPU and unnormalize
                inputs_cpu = inputs.cpu()
                mean = torch.tensor(weights.transforms().mean).view(3,1,1)
                std = torch.tensor(weights.transforms().std).view(3,1,1)
                inputs_cpu = inputs_cpu * std + mean
                images_to_show.extend(inputs_cpu)
                pred_fruit_labels.extend(preds_fruit.cpu().numpy())
                true_fruit_labels.extend(labels_fruit.cpu().numpy())
                pred_freshness_labels.extend(preds_freshness.cpu().numpy())
                true_freshness_labels.extend(labels_freshness.cpu().numpy())

            # Break after processing one batch
            break

    # Map labels and predictions to names
    fruit_labels_names = [idx_to_fruit[label] for label in fruit_labels]
    fruit_preds_names = [idx_to_fruit[pred] for pred in fruit_preds]

    freshness_labels_names = [freshness_mapping[int(label)] for label in freshness_labels]
    freshness_preds_names = [freshness_mapping[int(pred)] for pred in freshness_preds]

    # Compute metrics
    fruit_report = classification_report(fruit_labels_names, fruit_preds_names, zero_division=0)
    freshness_report = classification_report(freshness_labels_names, freshness_preds_names, zero_division=0)

    print('Test Fruit Classification Report:')
    print(fruit_report)
    print('Test Freshness Classification Report:')
    print(freshness_report)

    # Display images with predicted labels
    num_images = min(5, len(images_to_show))  # Display up to 5 images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image = images_to_show[i].numpy().transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        pred_fruit = idx_to_fruit[pred_fruit_labels[i]]
        true_fruit = idx_to_fruit[true_fruit_labels[i]]
        pred_freshness = freshness_mapping[int(pred_freshness_labels[i])]
        true_freshness = freshness_mapping[int(true_freshness_labels[i])]

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'True: {true_freshness} {true_fruit}\nPred: {pred_freshness} {pred_fruit}')
    plt.show()


# test code, may need to change
model = FruitFreshnessModel(num_fruit_classes).to(device)

#load best weights
model.load_state_dict(torch.load('best_model_weights.pth'))


evaluate_model(model, val_loader)
