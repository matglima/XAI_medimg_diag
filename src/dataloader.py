# -----------------------------------------------------------------
# File: dataloader.py
# -----------------------------------------------------------------
# Description:
# Contains the Dataset classes.
# NOW loads a pre-built global cache from disk.
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
import concurrent.futures

logger = logging.getLogger(__name__)

# --- Define Global Transforms ---
# These are the "on-the-fly" transforms.
# The slow "pre_transform" (Resize) is now in 0_build_cache.py

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Global Cache ---
# This will hold the images in RAM, loaded from disk once.
_GLOBAL_IMAGE_CACHE = None

def get_global_cache():
    """Loads the cache from disk into a global var, once."""
    global _GLOBAL_IMAGE_CACHE
    if _GLOBAL_IMAGE_CACHE is None:
        cache_path = "image_cache.pth"
        if not os.path.exists(cache_path):
            logger.error(f"Cache file not found at {cache_path}.")
            logger.error("Please run 'python 0_build_cache.py' first!")
            raise FileNotFoundError(cache_path)
        
        print("Loading image cache from disk... This may take a moment.")
        _GLOBAL_IMAGE_CACHE = torch.load(cache_path)
        print(f"Image cache loaded. {len(_GLOBAL_IMAGE_CACHE)} images in RAM.")
    return _GLOBAL_IMAGE_CACHE

# --- Dataset for Binary Experts ---

class RetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, target_label, transform=None, image_ids=None):
        self.transform = transform
        global_cache = get_global_cache()
        
        # Filter the main dataframe by the specified image_ids
        if image_ids is not None:
            dataframe = dataframe[dataframe['image_id'].isin(image_ids)]
        
        # --- NEW: Robust Filtering ---
        # Filter the dataframe *again* to only include images that
        # were successfully loaded into the cache.
        cached_ids = set(global_cache.keys())
        self.dataframe = dataframe[dataframe['image_id'].isin(cached_ids)].reset_index(drop=True)
        
        # Now that dataframe is filtered, create the labels and the final image list IN ORDER
        self.labels = self.dataframe[target_label].astype(np.float32).values
        self.cached_images = [global_cache[id] for id in self.dataframe['image_id']]

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Dataset for Multi-Label Gate & Calibration ---

class MultiLabelRetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, pathology_columns, transform=None, image_ids=None):
        self.transform = transform
        self.pathology_columns = pathology_columns
        global_cache = get_global_cache()

        if image_ids is not None:
            dataframe = dataframe[dataframe['image_id'].isin(image_ids)]

        # --- NEW: Robust Filtering ---
        cached_ids = set(global_cache.keys())
        self.dataframe = dataframe[dataframe['image_id'].isin(cached_ids)].reset_index(drop=True)
        
        self.labels = self.dataframe[self.pathology_columns].astype(np.float32).values
        self.cached_images = [global_cache[id] for id in self.dataframe['image_id']]


    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Data Splitting Functions ---

def get_stratified_splits(dataframe, target_label, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates stratified splits for BINARY classification.
    """
    # Note: We stratify on the *original* dataframe, not the cached one
    train_val_ids, test_ids = train_test_split(
        dataframe['image_id'],
        test_size=test_size,
        stratify=dataframe[target_label],
        random_state=random_state
    )
    
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size/(1-test_size),
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