# -----------------------------------------------------------------
# File: 0_build_cache.py
# -----------------------------------------------------------------
# Description:
# This script runs ONCE. It loads all images from disk,
# resizes them in parallel, and saves them to a single
# cache file. This prevents all other scripts from having
# to do this work ever again.
# -----------------------------------------------------------------

import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import logging
import concurrent.futures
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The transform to apply *before* caching (slow)
pre_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

CACHE_FILE_PATH = "image_cache.pth"

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

def main(args):
    if os.path.exists(CACHE_FILE_PATH):
        logger.info(f"Cache file already exists at {CACHE_FILE_PATH}. Skipping build.")
        return

    logger.info(f"Loading labels from {args.labels_path} to find all images.")
    labels_df = pd.read_csv(args.labels_path)
    image_ids = labels_df['image_id'].astype(str).unique().tolist()
    logger.info(f"Found {len(image_ids)} unique images to cache.")
    
    cached_images_dict = {}
    tasks = [(image_id, args.image_dir) for image_id in image_ids]
    
    num_workers = os.cpu_count() or 4 # Get all CPU cores
    logger.info(f"Building image cache using {num_workers} threads...")

    # --- START FIX: Use submit and as_completed for a clean tqdm bar ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(_load_and_resize_image, task): task for task in tasks}
        
        # Wrap as_completed in tqdm to track progress as tasks finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Building Cache"):
            image_id, image = future.result()
            if image is not None:
                cached_images_dict[image_id] = image
    # --- END FIX ---
            
    logger.info(f"Successfully loaded {len(cached_images_dict)} / {len(image_ids)} images.")
    
    logger.info(f"Saving cache to {CACHE_FILE_PATH}...")
    torch.save(cached_images_dict, CACHE_FILE_PATH)
    logger.info("Cache build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Image Cache")
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")

    args, _ = parser.parse_known_args()
    if args.labels_path and args.image_dir:
        main(args)
    else:
        print("Note: This script is intended to be run from the pipeline. Skipping cache build.")