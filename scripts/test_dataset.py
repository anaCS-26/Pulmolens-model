import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NIHChestXrayDataset
from torchvision import transforms
import torch
import os

def test_dataset():
    data_dir = './data'
    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    
    if not os.path.exists(csv_path):
        print("CSV not found yet.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = NIHChestXrayDataset(root_dir=data_dir, csv_file=csv_path, transform=transform)
    print(f"Dataset length: {len(dataset)}")
    
    # Try to load a few items
    # We might fail if the specific image for index 0 is not yet unzipped
    # So we iterate until we find one that exists
    
    found = 0
    for i in range(len(dataset)):
        try:
            img, label = dataset[i]
            print(f"Successfully loaded image at index {i}")
            print(f"Image shape: {img.shape}")
            print(f"Label: {label}")
            found += 1
            if found >= 3:
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading index {i}: {e}")
            break
            
    if found == 0:
        print("Could not load any images (maybe unzipping hasn't reached them yet).")

if __name__ == "__main__":
    test_dataset()
