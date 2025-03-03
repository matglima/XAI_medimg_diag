import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Define augmentations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        
        # Handle possible image extensions
        for ext in ['.png', '.jpg']:
            image_path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                break
        
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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

## Usage example

# # Load labels dataframe
# labels_df = pd.read_csv("path/to/labels.csv")

# # Choose target label
# target = "diabetic_retinopathy"

# # Create splits
# splits = get_stratified_splits(labels_df, target_label=target)

# # Create datasets
# train_ds = RetinaDataset(labels_df, "path/to/images", target, 
#                         transform=train_transform, image_ids=splits['train'])
# val_ds = RetinaDataset(labels_df, "path/to/images", target,
#                       transform=test_transform, image_ids=splits['val'])
# test_ds = RetinaDataset(labels_df, "path/to/images", target,
#                        transform=test_transform, image_ids=splits['test'])

# # Create dataloaders
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)