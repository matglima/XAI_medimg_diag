# -----------------------------------------------------------------
# File: dataloader.py
# -----------------------------------------------------------------
# Description:
# Contains the Dataset classes.
# NOW WITH MULTI-THREADED in-memory caching to solve CPU bottlenecks
# during the initial "pre-cache" step.
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
import concurrent.futures # <-- IMPORT FOR MULTI-THREADING

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

# --- START: NEW Multi-threaded Caching ---

def _find_image_path(image_dir, image_id):
    """Helper to find the correct image file path."""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        image_path_no_ext = os.path.join(image_dir, str(image_id))
        if os.path.exists(f"{image_path_no_ext}.png"):
            image_path = f"{image_path_no_ext}.png"
        elif os.path.exists(f"{image_path_no_ext}.jpg"):
            image_path = f"{image_path_no_ext}.jpg"
        else:
            return None # Image not found
    return image_path

def _load_and_resize_image(args):
    """
    Function to be run by each thread.
    Loads one image, resizes it, and returns it.
    """
    image_id, image_dir = args
    image_path = _find_image_path(image_dir, image_id)
    
    if image_path is None:
        logger.warning(f"Could not find image for ID: {image_id}")
        return image_id, None # Return None if image fails to load

    try:
        image = Image.open(image_path).convert('RGB')
        # Apply the slow resize transform NOW
        image = pre_transform(image)
        return image_id, image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return image_id, None

def _load_and_cache_parallel(image_ids, image_dir):
    """
    Helper function to load, resize, and cache images in RAM
    using all available CPU cores.
    """
    cached_images_dict = {}
    
    # Prepare arguments for the worker threads
    # We pass the image_dir with each ID
    tasks = [(image_id, image_dir) for image_id in image_ids]
    
    num_workers = os.cpu_count() or 4 # Get all CPU cores
    print(f"Building image cache using {num_workers} threads... This will take a few minutes.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to process images in parallel
        # Wrap with tqdm to get a progress bar
        results = list(tqdm(executor.map(_load_and_resize_image, tasks), total=len(tasks)))

    # Process results and filter out failed images
    for image_id, image in results:
        if image is not None:
            cached_images_dict[image_id] = image
            
    print(f"Image cache built. Successfully loaded {len(cached_images_dict)} / {len(image_ids)} images.")
    
    # Return a dictionary (ID -> Image) for robust filtering
    return cached_images_dict

# --- END: NEW Multi-threaded Caching ---


# --- Dataset for Binary Experts ---

class RetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, target_label, transform=None, image_ids=None):
        self.dataframe = dataframe.copy()
        self.image_dir = image_dir
        self.target_label = target_label
        self.transform = transform # This is the "on-the-fly" transform
        
        if image_ids is not None:
            self.dataframe = self.dataframe[self.dataframe['image_id'].isin(image_ids)]
        
        # --- NEW: Caching ---
        all_image_ids = self.dataframe['image_id'].tolist()
        # This populates self.cached_images
        cached_images_dict = _load_and_cache_parallel(all_image_ids, self.image_dir)
        
        # --- NEW: Robust Filtering ---
        # Filter the dataframe to ONLY include images that were successfully cached
        self.cached_image_ids = list(cached_images_dict.keys())
        self.dataframe = self.dataframe[self.dataframe['image_id'].isin(self.cached_image_ids)].reset_index(drop=True)
        
        # Now that dataframe is filtered, create the labels and the final image list IN ORDER
        self.labels = self.dataframe[target_label].astype(np.float32).values
        # Create the final list of images in the same order as the dataframe
        self.cached_images = [cached_images_dict[id] for id in self.dataframe['image_id']]

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

        # --- NEW: Caching ---
        all_image_ids = self.dataframe['image_id'].tolist()
        cached_images_dict = _load_and_cache_parallel(all_image_ids, self.image_dir)

        # --- NEW: Robust Filtering ---
        # Filter the dataframe to ONLY include images that were successfully cached
        self.cached_image_ids = list(cached_images_dict.keys())
        self.dataframe = self.dataframe[self.dataframe['image_id'].isin(self.cached_image_ids)].reset_index(drop=True)
        
        # Now that dataframe is filtered, create the labels and the final image list IN ORDER
        self.labels = self.dataframe[self.pathology_columns].astype(np.float32).values
        self.cached_images = [cached_images_dict[id] for id in self.dataframe['image_id']]


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