import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import NIHChestXrayDataset
from model import LungDiseaseModel
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

def train_model(num_epochs=10, batch_size=32, learning_rate=1e-4, data_dir='./data'):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Data Transforms with Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    # Assuming images are in data_dir/images and csv is data_dir/Data_Entry_2017.csv
    # We might need to adjust paths after download completes and we see structure
    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    images_dir = data_dir # dataset.py will walk this directory to find images
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Waiting for download to complete...")
        return

    dataset = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply validation transform to val_dataset (requires a bit of a hack with random_split or custom subset)
    # Since random_split shares the underlying dataset, we can't just change transform on val_dataset directly easily without wrapper
    # Let's create a simple wrapper or just re-instantiate dataset for simplicity if we want clean separation
    # But for now, to avoid complexity, we can just use the same transform or create two datasets
    
    # Better approach: Create two datasets with different transforms
    train_dataset_full = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path, transform=transform)
    val_dataset_full = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path, transform=val_transform)
    
    # We need to ensure we split them the same way. 
    # We can use a fixed generator for random_split or just use indices.
    indices = torch.randperm(len(train_dataset_full)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = LungDiseaseModel(num_classes=14).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader)}")
        
        # Validation
        model.eval()
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                outputs = model(images)
                val_targets.append(labels.cpu().numpy())
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                
        val_targets = np.vstack(val_targets)
        val_preds = np.vstack(val_preds)
        
        # Calculate AUC for each class and average
        aucs = []
        for i in range(14):
            try:
                # Check if we have both positive and negative examples
                if len(np.unique(val_targets[:, i])) > 1:
                    auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
                    aucs.append(auc)
            except ValueError:
                pass
        
        if len(aucs) > 0:
            mean_auc = np.mean(aucs)
            print(f"Validation Mean AUC: {mean_auc}")
            
            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save(model.state_dict(), 'best_model.pth')
                print("Saved best model")
        else:
            print("Could not calculate AUC (no valid classes)")

if __name__ == "__main__":
    train_model()
