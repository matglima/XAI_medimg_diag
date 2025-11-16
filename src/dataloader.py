# -----------------------------------------------------------------
# File: dataloader.py
# -----------------------------------------------------------------
# Description:
# Contains the Dataset classes for both binary (expert) and
# multi-label (gate/calibration) training.
# -----------------------------------------------------------------

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# --- Define Global Transforms ---

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Dataset for Binary Experts ---

class RetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, target_label, transform=None, image_ids=None):
        """
        Args:
            dataframe: DataFrame containing all labels
            image_dir: Directory with fundus images
            target_label: Specific disease label to use for binary classification
            transform: Optional transform to apply
            image_ids: Specific images to include in this dataset
        """
        self.dataframe = dataframe.copy()
        self.image_dir = image_dir
        self.target_label = target_label
        self.transform = transform
        
        if image_ids is not None:
            self.dataframe = self.dataframe[self.dataframe['image_id'].isin(image_ids)]
        
        # Verify we have valid labels
        self.labels = self.dataframe[target_label].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            
        if not os.path.exists(image_path):
            # Handle potential file extension issues
            image_path = os.path.join(self.image_dir, str(image_id))
            if os.path.exists(f"{image_path}.png"):
                image_path = f"{image_path}.png"
            elif os.path.exists(f"{image_path}.jpg"):
                image_path = f"{image_path}.jpg"

        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Dataset for Multi-Label Gate & Calibration ---

class MultiLabelRetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, pathology_columns, transform=None, image_ids=None):
        """
        Args:
            dataframe: DataFrame containing all labels
            image_dir: Directory with fundus images
            pathology_columns: List of all target pathology column names
            transform: Optional transform to apply
            image_ids: Specific images to include in this dataset
        """
        self.dataframe = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.pathology_columns = pathology_columns

        if image_ids is not None:
            self.dataframe = self.dataframe[self.dataframe['image_id'].isin(image_ids)]

        self.labels = self.dataframe[self.pathology_columns].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            
        if not os.path.exists(image_path):
            # Handle potential file extension issues
            image_path = os.path.join(self.image_dir, str(image_id))
            if os.path.exists(f"{image_path}.png"):
                image_path = f"{image_path}.png"
            elif os.path.exists(f"{image_path}.jpg"):
                image_path = f"{image_path}.jpg"

        image = Image.open(image_path).convert('RGB')
        # The label is now a vector of 0s and 1s
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Data Splitting Functions ---

def get_stratified_splits(dataframe, target_label, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates stratified splits at image level (not patient level)
    Returns: {'train': [], 'val': [], 'test': []} image_ids
    """
    # First split into train+val and test
    train_val_ids, test_ids = train_test_split(
        dataframe['image_id'],
        test_size=test_size,
        stratify=dataframe[target_label],
        random_state=random_state
    )
    
    # Then split train_val into train and val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size/(1-test_size),  # Adjust for relative split
        stratify=dataframe.set_index('image_id').loc[train_val_ids][target_label],
        random_state=random_state
    )
    
    return {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }

def get_random_splits(dataframe, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates random splits for MULTI-LABEL classification.
    Multi-label stratification is complex, so we use a simple random split.
    """
    train_val_ids, test_ids = train_test_split(
        dataframe['image_id'],
        test_size=test_size,
        random_state=random_state
    )
    
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )
    
    return {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
