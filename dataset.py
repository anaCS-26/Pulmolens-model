import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class NIHChestXrayDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.split = split
        
        # Filter dataset if needed or create splits
        # For now, we'll just use the whole dataset or implement a simple split logic
        # In a real scenario, we should use the provided train_val_list.txt and test_list.txt if available
        # Or split the dataframe here.
        
        # Handle multiple image directories
        self.image_paths = {}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths[file] = os.path.join(root, file)
        
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # Process labels: One-hot encoding
        # The 'Finding Labels' column contains labels separated by '|'
        for label in self.all_labels:
            self.df[label] = self.df['Finding Labels'].apply(lambda x: 1.0 if label in x else 0.0)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx, 0]
        if img_name in self.image_paths:
            img_path = self.image_paths[img_name]
        else:
            # Fallback or error
            # For robustness, we might skip or return a placeholder, but for now let's error
            raise FileNotFoundError(f"Image {img_name} not found in {self.root_dir}")
            
        image = Image.open(img_path).convert('RGB')
        
        labels = self.df.iloc[idx][self.all_labels].values.astype('float32')
        labels = torch.from_numpy(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels
