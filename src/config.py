# -----------------------------------------------------------------
# File: src/config.py
# -----------------------------------------------------------------
# Description:
# Dynamically loads project configuration, like the pathology list,
# by reading the labels.csv file.
# -----------------------------------------------------------------

import pandas as pd

def get_pathology_list(labels_csv_path: str) -> list[str]:
    """
    Reads the header of the labels CSV file and returns all
    column names *except* 'image_id' as the list of pathologies.
    
    This makes the entire pipeline agnostic to the specific
    pathologies in the dataset.
    """
    try:
        df = pd.read_csv(labels_csv_path, nrows=0) # Read only the header row
        columns = df.columns.tolist()
        
        if 'image_id' not in columns:
            raise ValueError(f"CRITICAL: 'image_id' column not found in {labels_csv_path}.")
            
        # Remove 'image_id' and any other potential non-label columns
        # You can expand this list if your CSVs have other metadata
        # (e.g., 'patient_id', 'split')
        columns_to_exclude = {'image_id', 'patient_id', 'split'}
        
        pathologies = [col for col in columns if col not in columns_to_exclude]
        
        if not pathologies:
            raise ValueError(f"No pathology columns found in {labels_csv_path}.")
            
        print(f"Loaded {len(pathologies)} pathologies from CSV: {pathologies}")
        return pathologies
        
    except FileNotFoundError:
        print(f"CRITICAL: Labels CSV file not found at {labels_csv_path}")
        raise
    except Exception as e:
        print(f"CRITICAL: Error reading pathology list from {labels_csv_path}: {e}")
        raise