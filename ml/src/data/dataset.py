import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from glob import glob
from src import config

class NIHChestXrayDataset(Dataset):
    def __init__(self, df, image_paths, transform=None):
        self.df = df
        self.image_paths = image_paths
        self.transform = transform
        self.labels = config.CLASS_NAMES
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        
        # Find image path
        if img_name in self.image_paths:
            img_path = self.image_paths[img_name]
        else:
            raise FileNotFoundError(f"Image {img_name} not found")
            
        # Load image using PIL to handle ICC profile warnings better
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default transform if none provided (resize + tensor)
            default_transform = A.Compose([
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            augmented = default_transform(image=image)
            image = augmented['image']
            
        # Get labels
        # Labels are one-hot encoded in the dataframe columns
        label_vec = row[self.labels].values.astype('float32')
        
        return image, torch.from_numpy(label_vec)

def get_transforms(split):
    if split == 'train':
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.CLAHE(p=0.2),
            A.OneOf([
                A.GaussNoise(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def prepare_data():
    print("Loading data...")
    df = pd.read_csv(config.CSV_FILE)
    
    # Pre-process labels
    # The 'Finding Labels' column contains labels separated by '|'
    for label in config.CLASS_NAMES:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
        
    # Map image names to paths
    # Images are in subdirectories images_001, images_002, etc.
    image_paths = {}
    # Search recursively for png files
    all_image_files = glob(os.path.join(config.IMAGES_DIR, '**', '*.png'), recursive=True)
    for p in all_image_files:
        image_paths[os.path.basename(p)] = p
        
    # Patient-level split
    patient_ids = df['Patient ID'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    train_df = df[df['Patient ID'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_ids)].reset_index(drop=True)
    test_df = df[df['Patient ID'].isin(test_ids)].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} images ({len(train_ids)} patients)")
    print(f"Val set: {len(val_df)} images ({len(val_ids)} patients)")
    print(f"Test set: {len(test_df)} images ({len(test_ids)} patients)")
    
    # Verify no leakage
    train_patients = set(train_df['Patient ID'])
    val_patients = set(val_df['Patient ID'])
    test_patients = set(test_df['Patient ID'])
    
    assert len(train_patients.intersection(val_patients)) == 0, "Leakage between train and val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Leakage between train and test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Leakage between val and test!"
    print("Data split verification passed: No patient leakage.")
    
    return train_df, val_df, test_df, image_paths

def get_data_loaders():
    train_df, val_df, test_df, image_paths = prepare_data()
    
    train_dataset = NIHChestXrayDataset(train_df, image_paths, transform=get_transforms('train'))
    val_dataset = NIHChestXrayDataset(val_df, image_paths, transform=get_transforms('val'))
    test_dataset = NIHChestXrayDataset(test_df, image_paths, transform=get_transforms('test'))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Verify split when run as script
    prepare_data()
