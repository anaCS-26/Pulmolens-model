import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
from datetime import datetime

# Import# Custom modules - add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NIHChestXrayDataset
from src.models.model import get_model
from src.training.losses import get_loss_function
from src.training.config import Config, create_directories, CLASS_NAMES, DEFAULT_POS_WEIGHTS
from src.data.augmentation import get_training_augmentation, get_validation_augmentation, mixup_data, cutmix_data
from PIL import Image


class AlbumentationsAdapter(torch.utils.data.Dataset):
    """Adapter to use Albumentations with PyTorch dataset."""
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image path and label from base dataset
        img_name = self.base_dataset.df.iloc[idx, 0]
        img_path = self.base_dataset.image_paths.get(img_name)
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found")
        
        # Load image as numpy array
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Get labels
        labels = self.base_dataset.df.iloc[idx][self.base_dataset.all_labels].values.astype('float32')
        labels = torch.from_numpy(labels)
        
        # Apply albumentations transform
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, labels


def train_model(config=None, **kwargs):
    """
    Enhanced training function with advanced features.
    
    Args:
        config: Config object or None (will use defaults)
        **kwargs: Override config parameters
    """
    # Load or create config
    if config is None:
        config = Config()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if key in config.train:
            config.train[key] = value
    
    # Create directories
    create_directories(config)
    
    # Set random seed
    torch.manual_seed(config.train['random_seed'])
    np.random.seed(config.train['random_seed'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # === Data Loading ===
    print("\n=== Loading Data ===")
    
    # Create base dataset (we'll wrap it with AlbumentationsAdapter)
    csv_path = config.paths['csv_file']
    images_dir = config.paths['data_dir']
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return
    
    # Get augmentation pipelines
    train_aug = get_training_augmentation(
        image_size=config.train['image_size'],
        advanced=config.train['use_advanced_aug']
    )
    val_aug = get_validation_augmentation(image_size=config.train['image_size'])
    
    # Create base datasets
    base_train_dataset = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path)
    base_val_dataset = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path)
    
    # Split indices
    dataset_size = len(base_train_dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(config.train['train_split'] * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset_base = torch.utils.data.Subset(base_train_dataset, train_indices)
    val_dataset_base = torch.utils.data.Subset(base_val_dataset, val_indices)
    
    # Wrap with Albumentations adapter
    train_dataset = AlbumentationsAdapter(base_train_dataset, train_aug)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    val_dataset = AlbumentationsAdapter(base_val_dataset, val_aug)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train['batch_size'],
        shuffle=True,
        num_workers=config.train['num_workers'],
        pin_memory=config.train['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train['batch_size'],
        shuffle=False,
        num_workers=config.train['num_workers'],
        pin_memory=config.train['pin_memory']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # === Model ===
    print("\n=== Creating Model ===")
    model = get_model(
        model_type=config.train['model_type'],
        num_classes=len(config.class_names),
        pretrained=config.train['pretrained'],
        attention_type=config.train.get('attention_type', 'cbam')
    ).to(device)
    
    print(f"Model: {config.train['model_type']}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # === Loss Function ===
    print("\n=== Loss Function ===")
    pos_weights = torch.tensor(config.pos_weights).to(device)
    criterion = get_loss_function(
        loss_type=config.train['loss_type'],
        pos_weights=pos_weights,
        focal_alpha=config.train.get('focal_alpha', 0.25),
        focal_gamma=config.train.get('focal_gamma', 2.0),
        asl_gamma_neg=config.train.get('asl_gamma_neg', 4),
        asl_gamma_pos=config.train.get('asl_gamma_pos', 1),
        asl_clip=config.train.get('asl_clip', 0.05)
    )
    print(f"Loss: {config.train['loss_type']}")
    
    # === Optimizer ===
    if config.train['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train['learning_rate'],
            weight_decay=config.train['weight_decay']
        )
    elif config.train['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train['learning_rate'],
            weight_decay=config.train['weight_decay']
        )
    elif config.train['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train['learning_rate'],
            momentum=config.train['momentum'],
            weight_decay=config.train['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.train['optimizer']}")
    
    print(f"Optimizer: {config.train['optimizer']}")
    
    # === Scheduler ===
    if config.train['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.train['scheduler_t0'],
            T_mult=config.train['scheduler_t_mult'],
            eta_min=config.train['scheduler_eta_min']
        )
    elif config.train['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train['step_size'],
            gamma=config.train['gamma']
        )
    elif config.train['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:
        scheduler = None
    
    print(f"Scheduler: {config.train['scheduler']}")
    
    # === Mixed Precision ===
    scaler = GradScaler() if config.train['use_amp'] else None
    if config.train['use_amp']:
        print("Using Automatic Mixed Precision (AMP)")
    
    # === Training Loop ===
    print(f"\n=== Training for {config.train['num_epochs']} epochs ===\n")
    
    best_auc = 0.0
    patience_counter = 0
    
    # Experiment name
    exp_name = f"{config.train['model_type']}_{config.train['loss_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_save_path = os.path.join(config.paths['models_dir'], f'{exp_name}_best.pth')
    
    for epoch in range(config.train['num_epochs']):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{config.train['num_epochs']}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Mixup/CutMix
            if config.train['use_mixup'] and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, config.train['mixup_alpha'])
                
                optimizer.zero_grad()
                
                if config.train['use_amp']:
                    with autocast():
                        outputs = model(images)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    scaler.scale(loss).backward()
                    if config.train['clip_grad_norm'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    loss.backward()
                    if config.train['clip_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    optimizer.step()
                    
            elif config.train['use_cutmix'] and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, config.train['cutmix_alpha'])
                
                optimizer.zero_grad()
                
                if config.train['use_amp']:
                    with autocast():
                        outputs = model(images)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    scaler.scale(loss).backward()
                    if config.train['clip_grad_norm'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    loss.backward()
                    if config.train['clip_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    optimizer.step()
            else:
                # Normal training
                optimizer.zero_grad()
                
                if config.train['use_amp']:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    if config.train['clip_grad_norm'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    if config.train['clip_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train['clip_grad_norm'])
                    optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels_cpu = labels.numpy()
                labels = labels.to(device)
                
                if config.train['use_amp']:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_targets.append(labels_cpu)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_targets = np.vstack(val_targets)
        val_preds = np.vstack(val_preds)
        
        # Calculate AUC
        aucs = []
        for i in range(len(config.class_names)):
            if len(np.unique(val_targets[:, i])) > 1:
                try:
                    auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass
        
        mean_auc = np.mean(aucs) if aucs else 0.0
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Mean AUC: {mean_auc:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if config.train['scheduler'] == 'plateau':
                scheduler.step(mean_auc)
            else:
                scheduler.step()
        
        # Save best model
        if mean_auc > best_auc + config.train['min_delta']:
            best_auc = mean_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': config.train,
            }, model_save_path)
            print(f"✓ Saved best model (AUC: {best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if config.train['early_stopping'] and patience_counter >= config.train['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print(f"\n=== Training Complete ===")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Model saved to: {model_save_path}")
    
    return model, best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train lung disease classification model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['densenet', 'attention_densenet', 'multiscale', 'ensemble'])
    parser.add_argument('--loss_type', type=str, choices=['bce', 'focal', 'asl', 'combined'])
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line args
    kwargs = {}
    if args.model_type:
        kwargs['model_type'] = args.model_type
    if args.loss_type:
        kwargs['loss_type'] = args.loss_type
    if args.num_epochs:
        kwargs['num_epochs'] = args.num_epochs
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.learning_rate:
        kwargs['learning_rate'] = args.learning_rate
    if args.use_amp:
        kwargs['use_amp'] = True
    
    # Train
    train_model(config=config, **kwargs)
