# Import necessary libraries
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

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 1: Define the Custom Dataset
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

                fruit_set.add(fruit_name)
                # Collect image paths
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.fruit_labels.append(fruit_name)
                        self.fresh_labels.append(fresh_label)

        # Create fruit to index mapping if not provided
        if self.fruit_to_idx is None:
            fruit_list = sorted(fruit_set)
            self.fruit_to_idx = {fruit_name: idx for idx, fruit_name in enumerate(fruit_list)}
        else:
            # Ensure fruit_set matches the keys in fruit_to_idx
            assert fruit_set == set(self.fruit_to_idx.keys()), "Mismatch in fruit names and indices."

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
# Pretrained model normalization parameters

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

# Step 3: Create Datasets and DataLoaders
# Define the base path for the dataset
BASE_PATH = os.getcwd()
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
TRAIN_PATH = os.path.join(DATASET_PATH, 'Train')
TEST_PATH = os.path.join(DATASET_PATH, 'Test')

# Create combined dataset without transforms for now
full_dataset = FruitFreshnessDataset(root_dirs=[TRAIN_PATH, TEST_PATH])

# Extract the fruit_to_idx mapping to ensure consistent labels
fruit_to_idx = full_dataset.fruit_to_idx
print("Fruit to Index Mapping:", fruit_to_idx)

# Balance the dataset
fruit_to_indices = defaultdict(list)
for idx, fruit_label in enumerate(full_dataset.fruit_labels):
    fruit_to_indices[fruit_label].append(idx)

# Cap samples per fruit type
max_samples_per_class = 1500
balanced_indices = []
for indices in fruit_to_indices.values():
    if len(indices) > max_samples_per_class:
        indices = random.sample(indices, max_samples_per_class)
    balanced_indices.extend(indices)

# Shuffle the balanced indices
random.seed(10)
random.shuffle(balanced_indices)

# Get the labels for stratification
balanced_labels = [full_dataset.fruit_labels[i] for i in balanced_indices]

# Split balanced indices into training and validation sets
train_indices, val_indices = train_test_split(
    balanced_indices, test_size=0.15, random_state=10, stratify=balanced_labels
)

# Create separate datasets for training and validation with transforms
train_dataset = FruitFreshnessDataset(
    root_dirs=[TRAIN_PATH, TEST_PATH],
    transform=train_transform,
    indices=train_indices,
    fruit_to_idx=fruit_to_idx
)
val_dataset = FruitFreshnessDataset(
    root_dirs=[TRAIN_PATH, TEST_PATH],
    transform=val_transform,
    indices=val_indices,
    fruit_to_idx=fruit_to_idx
)

# Create DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

num_fruit_classes = len(fruit_to_idx)
print("Number of fruit classes:", num_fruit_classes)




