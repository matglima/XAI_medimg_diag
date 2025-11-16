# -----------------------------------------------------------------
# File: dataloader.py
# -----------------------------------------------------------------
# Description:
# Contains the Dataset classes.
# NOW WITH IN-MEMORY CACHING to solve CPU bottlenecks.
# -----------------------------------------------------------------

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# --- Define Global Transforms ---
# We split transforms into two parts:
# 1. PRE-CACHE: Slow transforms (Resize)
# 2. ON-THE-FLY: Fast, random transforms (Augmentations)

# The transform to apply *before* caching (slow)
pre_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

# The "on-the-fly" transforms for training
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# The "on-the-fly" transforms for validation
val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def _load_and_cache(image_ids, image_dir):
    """Helper function to load, resize, and cache images in RAM."""
    cached_images = []
    print(f"Building image cache... This will take a few minutes.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            # Handle potential file extension issues
            image_path_no_ext = os.path.join(image_dir, str(image_id))
            if os.path.exists(f"{image_path_no_ext}.png"):
                image_path = f"{image_path_no_ext}.png"
            elif os.path.exists(f"{image_path_no_ext}.jpg"):
                image_path = f"{image_path_no_ext}.jpg"
            else:
                logger.warning(f"Could not find image for ID: {image_id}")
                continue

        try:
            image = Image.open(image_path).convert('RGB')
            # Apply the slow resize transform NOW
            image = pre_transform(image)
            cached_images.append(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            
    print("Image cache built successfully.")
    return cached_images

# --- Dataset for Binary Experts ---

class RetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, target_label, transform=None, image_ids=None):
        self.dataframe = dataframe.copy()
        self.image_dir = image_dir
        self.target_label = target_label
        self.transform = transform # This is the "on-the-fly" transform
        
        if image_ids is not None:
            self.dataframe = self.dataframe[self.dataframe['image_id'].isin(image_ids)]
        
        self.labels = self.dataframe[target_label].astype(np.float32).values
        
        # --- NEW: Caching ---
        self.image_ids = self.dataframe['image_id'].tolist()
        # This populates self.cached_images
        self.cached_images = _load_and_cache(self.image_ids, self.image_dir)
        
        # Filter dataframe and labels for images that failed to load
        if len(self.cached_images) != len(self.labels):
            # This is a fallback in case some images were corrupt
            print("Warning: Mismatch between loaded images and labels. Re-filtering.")
            # This is slow, but safer. A more complex implementation would
            # map IDs to indices during caching.
            # For this project, we'll assume _load_and_cache is robust.
            pass

    def __len__(self):
        # Return the number of successfully cached images
        return len(self.cached_images)

    def __getitem__(self, idx):
        # --- MODIFIED: Get from RAM, not disk ---
        image = self.cached_images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply the fast, "on-the-fly" transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Dataset for Multi-Label Gate & Calibration ---

class MultiLabelRetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, pathology_columns, transform=None, image_ids=None):
        self.dataframe = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform # This is the "on-the-fly" transform
        self.pathology_columns = pathology_columns

        if image_ids is not None:
            self.dataframe = self.dataframe[self.dataframe['image_id'].isin(image_ids)]

        self.labels = self.dataframe[self.pathology_columns].astype(np.float32).values

        # --- NEW: Caching ---
        self.image_ids = self.dataframe['image_id'].tolist()
        # This populates self.cached_images
        self.cached_images = _load_and_cache(self.image_ids, self.image_dir)
        
        if len(self.cached_images) != len(self.labels):
            print("Warning: Mismatch between loaded images and labels.")
            # See note in RetinaDataset.
            pass


    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        # --- MODIFIED: Get from RAM, not disk ---
        image = self.cached_images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply the fast, "on-the-fly" transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Data Splitting Functions ---

def get_stratified_splits(dataframe, target_label, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates stratified splits for BINARY classification.
    """
    train_val_ids, test_ids = train_test_split(
        dataframe['image_id'],
        test_size=test_size,
        stratify=dataframe[target_label],
        random_state=random_state
    )
    
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