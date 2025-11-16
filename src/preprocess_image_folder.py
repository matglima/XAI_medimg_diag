# -----------------------------------------------------------------
# File: src/preprocess_image_folder.py
# -----------------------------------------------------------------
# Description:
# A utility script to convert a "folder-per-class" (ImageFolder)
# dataset structure into a single multi-label CSV file that
# our pipeline understands.
#
# Example usage:
# python src/preprocess_image_folder.py \
#    --dataset-dir /path/to/image_folder_dataset \
#    --output-csv /path/to/new_labels.csv
# -----------------------------------------------------------------

import os
import pandas as pd
from tqdm import tqdm
import argparse

def create_labels_csv_from_folders(dataset_dir: str, output_csv: str):
    """
    Walks a directory (ImageFolder format) and creates a multi-label CSV.
    
    Directory Structure:
    /dataset_dir/
        /class_A/
            img1.png
            img2.png
        /class_B/
            img3.png
    
    Output CSV:
    image_id,class_A,class_B
    img1.png,1,0
    img2.png,1,0
    img3.png,0,1
    """
    print(f"Scanning directory: {dataset_dir}")
    
    # 1. Find all class names (folder names)
    try:
        class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    except FileNotFoundError:
        print(f"Error: Directory not found at {dataset_dir}")
        return

    if not class_names:
        print(f"Error: No class subdirectories found in {dataset_dir}")
        return
        
    print(f"Found {len(class_names)} classes: {class_names}")
    
    records = []
    
    # 2. Walk through each class folder and list images
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        
        for filename in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                # Create a record (a dictionary) for this image
                record = {
                    'image_id': filename,
                }
                
                # Set the label for this class to 1 and all others to 0
                for c in class_names:
                    record[c] = 1 if c == class_name else 0
                
                records.append(record)

    # 3. Create and save the DataFrame
    if not records:
        print("Error: No images found in any class subdirectories.")
        return

    df = pd.DataFrame(records)
    
    # Re-order columns to put 'image_id' first
    df = df[['image_id'] + class_names]
    
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created labels CSV at: {output_csv}")
    print(f"Total images processed: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ImageFolder to Multi-Label CSV")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Path to the dataset (ImageFolder structure)")
    parser.add_argument('--output-csv', type=str, required=True, help="Path to save the output labels.csv file")
    
    args = parser.parse_args()
    create_labels_csv_from_folders(args.dataset_dir, args.output_csv)