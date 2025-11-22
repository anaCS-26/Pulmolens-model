import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime

from src import config
from src.data.dataset import get_data_loaders
from src.models.densenet import DenseNet121, AttentionDenseNet
from src.training.losses import AsymmetricLoss, FocalLoss

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # Collect for AUC
            probs = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    # Calculate Mean AUC
    try:
        aucs = []
        for i in range(all_targets.shape[1]):
            # Handle case where a class might not be present in the batch
            if len(np.unique(all_targets[:, i])) > 1:
                aucs.append(roc_auc_score(all_targets[:, i], all_probs[:, i]))
        mean_auc = np.mean(aucs) if aucs else 0.0
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        mean_auc = 0.0
            
    return running_loss / len(loader), mean_auc

def main(args):
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, _ = get_data_loaders()
    
    # Model
    if args.model == 'attention_densenet':
        print("Initializing Attention-DenseNet121...")
        model = AttentionDenseNet(num_classes=len(config.CLASS_NAMES))
    else:
        print("Initializing Baseline DenseNet121...")
        model = DenseNet121(num_classes=len(config.CLASS_NAMES))
        
    model.to(device)
    
    # Loss
    if args.loss == 'asl':
        print("Using Asymmetric Loss")
        criterion = AsymmetricLoss()
    elif args.loss == 'focal':
        print("Using Focal Loss")
        criterion = FocalLoss()
    else:
        print("Using BCE Loss")
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop
    best_auc = 0.0
    start_time = time.time()
    
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_auc)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            save_name = f"{args.model}_{args.loss}_{timestamp}_best.pth"
            save_path = os.path.join(config.MODEL_SAVE_DIR, save_name)
            
            # Save metadata as well
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
                'config': {
                    'model_type': args.model,
                    'loss_type': args.loss
                }
            }, save_path)
            print(f"✓ Saved best model (AUC: {best_auc:.4f}) to {save_name}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")
    print(f"Best Validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='attention_densenet', choices=['densenet', 'attention_densenet'], help='Model architecture')
    parser.add_argument('--loss', type=str, default='asl', choices=['asl', 'focal', 'bce'], help='Loss function')
    args = parser.parse_args()
    
    main(args)
