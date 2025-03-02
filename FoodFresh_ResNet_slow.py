import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision
import torchmetrics
import torch.nn as nn
from tqdm.notebook import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
import time  # Import time for recording training time
from torchvision.models import resnet50, ResNet50_Weights  # Using ResNet as per previous modification

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 1: Define the Custom Dataset (unchanged)
class FruitFreshnessDataset(Dataset):
    def __init__(self, root_dirs, transform=None, indices=None, fruit_to_idx=None):
        self.transform = transform
        self.image_paths = []
        self.fruit_labels = []
        self.fresh_labels = []
        self.fruit_to_idx = fruit_to_idx  # Mapping from fruit names to indices
        fruit_set = set()

        # Dictionary to correct misspellings
        fruit_name_corrections = {'tamto': 'tomato', 'patato': 'potato'}

        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        # Collect all images and labels
        for root_dir in root_dirs:
            classes = os.listdir(root_dir)
            for class_name in classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                # Determine freshness and fruit type
                if class_name.startswith('fresh'):
                    fresh_label = 0  # Label '0' for fresh
                    fruit_name = class_name[5:]  # Remove 'fresh' prefix
                elif class_name.startswith('rotten'):
                    fresh_label = 1  # Label '1' for rotten
                    fruit_name = class_name[6:]  # Remove 'rotten' prefix
                else:
                    continue

                # Correct any misspellings in fruit_name
                fruit_name = fruit_name_corrections.get(fruit_name, fruit_name)

                # Skip fruits not in fruit_to_idx when provided
                if self.fruit_to_idx is not None and fruit_name not in self.fruit_to_idx:
                    continue

                fruit_set.add(fruit_name)
                # Collect image paths
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):
                        if self.fruit_to_idx is not None and fruit_name not in self.fruit_to_idx:
                            continue  # Skip unknown fruits
                        self.image_paths.append(img_path)
                        self.fruit_labels.append(fruit_name)
                        self.fresh_labels.append(fresh_label)

        # Create fruit to index mapping if not provided
        if self.fruit_to_idx is None:
            fruit_list = sorted(fruit_set)
            self.fruit_to_idx = {fruit_name: idx for idx, fruit_name in enumerate(fruit_list)}
        else:
            # Ensure fruit_set is a subset of the keys in fruit_to_idx
            if not fruit_set.issubset(set(self.fruit_to_idx.keys())):
                missing_fruits = fruit_set - set(self.fruit_to_idx.keys())
                print(f"Warning: The following fruits are not in fruit_to_idx and will be skipped: {missing_fruits}")
                # Optionally, you can remove these fruits or raise an error

        # Convert fruit names to indices
        self.fruit_labels = [self.fruit_to_idx[name] for name in self.fruit_labels]

        # Filter by indices if provided
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.fruit_labels = [self.fruit_labels[i] for i in indices]
            self.fresh_labels = [self.fresh_labels[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        fruit_label = self.fruit_labels[idx]
        fresh_label = self.fresh_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, fruit_label, fresh_label

# Step 2: Define Data Transforms
# Load ResNet50 weights and get the required transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Data augmentation techniques
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])

# Step 3: Create Datasets and DataLoaders
# Define the base path for the dataset
BASE_PATH = os.getcwd()
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
TRAIN_PATH = os.path.join(DATASET_PATH, 'Train')
TEST_PATH = os.path.join(DATASET_PATH, 'Test')

# Create dataset using only TRAIN_PATH
train_full_dataset = FruitFreshnessDataset(root_dirs=TRAIN_PATH)

# Extract the fruit_to_idx mapping to ensure consistent labels across datasets
fruit_to_idx = train_full_dataset.fruit_to_idx
print("Fruit to Index Mapping:", fruit_to_idx)

# Split the training data into training and validation sets
train_indices, val_indices = train_test_split(
    list(range(len(train_full_dataset))),
    test_size=0.15,
    random_state=10,
    stratify=train_full_dataset.fruit_labels  # Stratify by fruit labels
)

# Create separate datasets for training and validation with transforms
train_dataset = FruitFreshnessDataset(
    root_dirs=TRAIN_PATH,
    transform=train_transform,
    indices=train_indices,
    fruit_to_idx=fruit_to_idx
)
val_dataset = FruitFreshnessDataset(
    root_dirs=TRAIN_PATH,
    transform=val_transform,
    indices=val_indices,
    fruit_to_idx=fruit_to_idx
)

# Create the test dataset using TEST_PATH
test_dataset = FruitFreshnessDataset(
    root_dirs=TEST_PATH,
    transform=val_transform,
    fruit_to_idx=fruit_to_idx  # Use the same fruit_to_idx mapping

)

# Create DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_fruit_classes = len(fruit_to_idx)
print("Number of fruit classes:", num_fruit_classes)

##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
import copy

# Model Definition Using ResNet (unchanged from previous modification)
class FruitFreshnessModel(nn.Module):
    def __init__(self, num_fruit_classes):
        super(FruitFreshnessModel, self).__init__()
        # Load a pre-trained ResNet50 model with default weights
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        # Remove the last fully connected layer
        self.backbone.fc = nn.Identity()
        # Define separate classifiers for fruit type and freshness
        self.fruit_classifier = nn.Linear(num_features, num_fruit_classes)
        self.freshness_classifier = nn.Linear(num_features, 1)  # Binary classification

    def forward(self, x):
        features = self.backbone(x)
        fruit_logits = self.fruit_classifier(features)
        freshness_logits = self.freshness_classifier(features)
        return fruit_logits, freshness_logits

# Instantiate the model
model = FruitFreshnessModel(num_fruit_classes).to(device)

# Define loss functions
criterion_fruit = nn.CrossEntropyLoss()
criterion_freshness = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Updated train_model function with time recording and model saving (unchanged)
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

            # Map labels and predictions to names
            idx_to_fruit = {idx: fruit for fruit, idx in fruit_to_idx.items()}
            fruit_preds_names = [idx_to_fruit[pred] for pred in fruit_preds]
            fruit_labels_names = [idx_to_fruit[label] for label in fruit_labels]

            freshness_mapping = {0: "fresh", 1: "rotten"}
            freshness_labels_names = [freshness_mapping[int(label)] for label in freshness_labels]
            freshness_preds_names = [freshness_mapping[int(pred)] for pred in freshness_preds]

            # Compute metrics
            fruit_report = classification_report(fruit_labels_names, fruit_preds_names, output_dict=True, zero_division=0)
            freshness_report = classification_report(freshness_labels_names, freshness_preds_names, output_dict=True, zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f'{phase} Fruit Classification Report:')
            print(classification_report(fruit_labels_names, fruit_preds_names, zero_division=0))
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

num_epochs = 5  # Adjust the number of epochs as needed
model = train_model(model, dataloaders, criterion_fruit, criterion_freshness, optimizer, scheduler, num_epochs)

# # Updated evaluate_model function to display fruit names and freshness labels (unchanged)
# def evaluate_model(model, dataloader):
#     model.eval()
#     fruit_preds = []
#     fruit_labels = []
#     freshness_preds = []
#     freshness_labels = []
#
#     idx_to_fruit = {idx: fruit for fruit, idx in fruit_to_idx.items()}
#     freshness_mapping = {0: "fresh", 1: "rotten"}
#
#     images_to_show = []
#     pred_fruit_labels = []
#     true_fruit_labels = []
#     pred_freshness_labels = []
#     true_freshness_labels = []
#
#     with torch.no_grad():
#         for inputs, labels_fruit, labels_freshness in dataloader:
#             inputs = inputs.to(device)
#             labels_fruit = labels_fruit.to(device)
#             labels_freshness = labels_freshness.to(device).float()
#
#             outputs_fruit, outputs_freshness = model(inputs)
#             _, preds_fruit = torch.max(outputs_fruit, 1)
#             preds_freshness = torch.sigmoid(outputs_freshness).squeeze() >= 0.5
#
#             fruit_preds.extend(preds_fruit.cpu().numpy())
#             fruit_labels.extend(labels_fruit.cpu().numpy())
#             freshness_preds.extend(preds_freshness.cpu().numpy())
#             freshness_labels.extend(labels_freshness.cpu().numpy())
#
#             # For visualization, collect images from the first batch
#             if len(images_to_show) == 0:
#                 # Move inputs to CPU and unnormalize
#                 inputs_cpu = inputs.cpu()
#                 mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
#                 std = torch.tensor(weights.transforms().std).view(3, 1, 1)
#                 inputs_cpu = inputs_cpu * std + mean
#                 images_to_show.extend(inputs_cpu)
#                 pred_fruit_labels.extend(preds_fruit.cpu().numpy())
#                 true_fruit_labels.extend(labels_fruit.cpu().numpy())
#                 pred_freshness_labels.extend(preds_freshness.cpu().numpy())
#                 true_freshness_labels.extend(labels_freshness.cpu().numpy())
#
#             # Remove the break statement to process the entire dataset
#             # break  # Comment out or remove this line to evaluate all data
#
#     # Map labels and predictions to names
#     fruit_labels_names = [idx_to_fruit[label] for label in fruit_labels]
#     fruit_preds_names = [idx_to_fruit[pred] for pred in fruit_preds]
#
#     freshness_labels_names = [freshness_mapping[int(label)] for label in freshness_labels]
#     freshness_preds_names = [freshness_mapping[int(pred)] for pred in freshness_preds]
#
#     # Compute metrics
#     fruit_report = classification_report(fruit_labels_names, fruit_preds_names, zero_division=0)
#     freshness_report = classification_report(freshness_labels_names, freshness_preds_names, zero_division=0)
#
#     print('Test Fruit Classification Report:')
#     print(fruit_report)
#     print('Test Freshness Classification Report:')
#     print(freshness_report)
#
#     # Display images with predicted labels
#     num_images = min(5, len(images_to_show))  # Display up to 5 images
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#     for i in range(num_images):
#         image = images_to_show[i].numpy().transpose((1, 2, 0))
#         image = np.clip(image, 0, 1)
#         pred_fruit = idx_to_fruit[pred_fruit_labels[i]]
#         true_fruit = idx_to_fruit[true_fruit_labels[i]]
#         pred_freshness = freshness_mapping[int(pred_freshness_labels[i])]
#         true_freshness = freshness_mapping[int(true_freshness_labels[i])]
#
#         axes[i].imshow(image)
#         axes[i].axis('off')
#         axes[i].set_title(f'True: {true_freshness} {true_fruit}\nPred: {pred_freshness} {pred_fruit}')
#     plt.show()
#
# # Evaluate the model on the test dataset
# print("Evaluating on Test Dataset:")
# evaluate_model(model, test_loader)

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

            # Collect all images and predictions for visualization
            inputs_cpu = inputs.cpu()
            mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
            std = torch.tensor(weights.transforms().std).view(3, 1, 1)
            inputs_cpu = inputs_cpu * std + mean
            images_to_show.extend(inputs_cpu)
            pred_fruit_labels.extend(preds_fruit.cpu().numpy())
            true_fruit_labels.extend(labels_fruit.cpu().numpy())
            pred_freshness_labels.extend(preds_freshness.cpu().numpy())
            true_freshness_labels.extend(labels_freshness.cpu().numpy())

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

    # Prepare images for display per class
    num_images_per_class = 4  # Number of images to show per class
    unique_classes = set(zip(true_freshness_labels, true_fruit_labels))

    # Create figure and axes for display
    fig, axes = plt.subplots(len(unique_classes), num_images_per_class, figsize=(5 * num_images_per_class, 5 * len(unique_classes)))

    # Ensure axes is a 2D array for consistent indexing
    if len(unique_classes) == 1:
        axes = [axes]
    if num_images_per_class == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (freshness_label, fruit_label) in enumerate(unique_classes):
        # Find indices for images of the current class
        class_indices = [i for i, (fl, fr) in enumerate(zip(true_freshness_labels, true_fruit_labels))
                         if (fl == freshness_label and fr == fruit_label)]

        # Limit images displayed to `num_images_per_class`
        num_images = min(num_images_per_class, len(class_indices))

        for col_idx in range(num_images):
            idx = class_indices[col_idx]
            image = images_to_show[idx].numpy().transpose((1, 2, 0))
            image = np.clip(image, 0, 1)
            pred_fruit = idx_to_fruit[pred_fruit_labels[idx]]
            true_fruit = idx_to_fruit[true_fruit_labels[idx]]
            pred_freshness = freshness_mapping[int(pred_freshness_labels[idx])]
            true_freshness = freshness_mapping[int(true_freshness_labels[idx])]

            axes[row_idx][col_idx].imshow(image)
            axes[row_idx][col_idx].axis('off')
            axes[row_idx][col_idx].set_title(
                f'True: {true_freshness} {true_fruit}\nPred: {pred_freshness} {pred_fruit}')

        # Hide unused subplots in the current row
        for col_idx in range(num_images, num_images_per_class):
            axes[row_idx][col_idx].axis('off')

    plt.tight_layout()
    plt.show()
# test code, may need to change
model = FruitFreshnessModel(num_fruit_classes).to(device)

#load best weights
model.load_state_dict(torch.load('best_model_weights.pth'))

# Evaluate the model on the test dataset
print("Evaluating on Test Dataset:")
evaluate_model(model, test_loader)


