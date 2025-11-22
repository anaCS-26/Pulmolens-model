import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report
import seaborn as sns

# Import custom modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NIHChestXrayDataset
from src.models.model import get_model
from src.evaluation.gradcam_plus_plus import GradCAMPlusPlus, ScoreCAM, show_cam_on_image
from src.evaluation.threshold_optimizer import optimize_thresholds, load_thresholds, visualize_threshold_curves
from src.data.augmentation import get_validation_augmentation, get_tta_augmentation
from src.training.config import Config, CLASS_NAMES
from torchvision import transforms


def denormalize_image(img_tensor):
    """Denormalize image tensor for visualization."""
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img_tensor.squeeze()).cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def evaluate_with_tta(model, image, device, n_augs=5):
    """
    Evaluate with Test-Time Augmentation.
    
    Args:
        model: Trained model
        image: PIL Image
        device: Device to run on
        n_augs: Number of augmentation variants
    
    Returns:
        Averaged predictions
    """
    tta_transforms = get_tta_augmentation(n_augs=n_augs)
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for transform in tta_transforms:
            img_array = np.array(image)
            augmented = transform(image=img_array)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            output = model(img_tensor)
            pred = torch.sigmoid(output).cpu().numpy()
            predictions.append(pred)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred


def evaluate_model(model_path, config=None, **kwargs):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to model checkpoint
        config: Config object or None
        **kwargs: Override config parameters
    """
    # Load config
    if config is None:
        config = Config()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if key in config.eval:
            config.eval[key] = value
        elif key in config.train:
            config.train[key] = value
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # === Load Model ===
    print("=== Loading Model ===")
    
    # Try to load checkpoint to get config
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # Load model with saved config
        saved_config = checkpoint['config']
        model = get_model(
            model_type=saved_config.get('model_type', 'densenet'),
            num_classes=len(config.class_names),
            pretrained=False,
            attention_type=saved_config.get('attention_type', 'cbam')
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"Model type: {saved_config.get('model_type', 'densenet')}")
        print(f"Best training AUC: {checkpoint.get('best_auc', 'unknown'):.4f}")
    else:
        # Old format - just state dict
        model_type = config.train.get('model_type', 'densenet')
        model = get_model(
            model_type=model_type,
            num_classes=len(config.class_names),
            pretrained=False
        ).to(device)
        model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) else checkpoint)
        print(f"Loaded model weights (assumed type: {model_type})")
    
    model.eval()
    
    # === Setup Visualization ===
    if config.eval['viz_method'] == 'gradcam++':
        # Get target layer (last normalization before classifier)
        if hasattr(model, 'features'): # DenseNet or AttentionDenseNet
            target_layer = model.features.norm5 if hasattr(model.features, 'norm5') else model.features[-1]
        elif hasattr(model, 'densenet'):  # LungDiseaseModel
            target_layer = model.densenet.features.norm5
        else:
            print("Warning: Could not find target layer for Grad-CAM++")
            target_layer = None
        
        if target_layer:
            viz = GradCAMPlusPlus(model, target_layer)
            print(f"Visualization: Grad-CAM++")
    elif config.eval['viz_method'] == 'scorecam':
        if hasattr(model, 'features'):
            target_layer = model.features.norm5 if hasattr(model.features, 'norm5') else model.features[-1]
        elif hasattr(model, 'densenet'):
            target_layer = model.densenet.features.norm5
        else:
            target_layer = None
        
        if target_layer:
            viz = ScoreCAM(model, target_layer)
            print(f"Visualization: Score-CAM")
    else:
        viz = None
    
    # === Load Dataset ===
    print("\n=== Loading Dataset ===")
    
    csv_path = config.paths['csv_file']
    images_dir = config.paths['data_dir']
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return
    
    # Use validation augmentation
    val_aug = get_validation_augmentation(image_size=config.train['image_size'])
    
    # Create dataset (we'll use a wrapper for albumentations)
    base_dataset = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path)
    
    # Use validation split (assuming same split as training)
    dataset_size = len(base_dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(config.train['train_split'] * dataset_size)
    val_indices = indices[train_size:]
    
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    
    print(f"Evaluation samples: {len(val_dataset)}")
    
    # === Full Dataset Evaluation ===
    print("\n=== Running Full Evaluation ===")
    
    all_preds = []
    all_targets = []
    
    # Create simple dataloader for evaluation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for idx in tqdm(val_indices, desc="Evaluating"):
        img_name = base_dataset.df.iloc[idx, 0]
        img_path = base_dataset.image_paths.get(img_name)
        
        if img_path is None:
            continue
        
        image = Image.open(img_path).convert('RGB')
        labels = base_dataset.df.iloc[idx][base_dataset.all_labels].values.astype('float32')
        
        if config.eval['use_tta']:
            # Use TTA
            pred = evaluate_with_tta(model, image, device, config.eval['tta_n_augs'])
        else:
            # Normal evaluation
            img_tensor = val_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.sigmoid(output).cpu().numpy()
        
        all_preds.append(pred[0])
        all_targets.append(labels)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print(f"\nEvaluated {len(all_preds)} samples")
    
    # === Calculate Metrics ===
    print("\n=== Model Performance Metrics ===")
    
    # AUC-ROC for each class
    aucs = []
    pr_aucs = []
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'AUC-ROC':<10} {'PR-AUC':<10} {'Prevalence':<12}")
    print("=" * 55)
    
    for i, class_name in enumerate(config.class_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        prevalence = y_true.mean()
        
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_pred)
                pr_auc = average_precision_score(y_true, y_pred)
                aucs.append(auc)
                pr_aucs.append(pr_auc)
                print(f"{class_name:<20} {auc:<10.4f} {pr_auc:<10.4f} {prevalence:<12.4f}")
            except ValueError:
                print(f"{class_name:<20} {'N/A':<10} {'N/A':<10} {prevalence:<12.4f}")
        else:
            print(f"{class_name:<20} {'N/A':<10} {'N/A':<10} {prevalence:<12.4f}")
    
    print("=" * 55)
    print(f"{'Mean':<20} {np.mean(aucs):<10.4f} {np.mean(pr_aucs):<10.4f}")
    
    # === Threshold Optimization ===
    if config.eval['optimize_threshold']:
        print(f"\n=== Optimizing Thresholds (metric: {config.eval['threshold_metric']}) ===")
        
        # Since we don't have a dataloader, create optimal thresholds directly
        from src.evaluation.threshold_optimizer import find_optimal_threshold_per_class
        
        optimal_thresholds = find_optimal_threshold_per_class(
            all_targets,
            all_preds,
            metric=config.eval['threshold_metric']
        )
        
        # Save thresholds
        threshold_dict = {
            config.class_names[i]: threshold
            for i, threshold in optimal_thresholds.items()
        }
        
        import json
        threshold_path = 'optimal_thresholds.json'
        with open(threshold_path, 'w') as f:
            json.dump({
                'thresholds': threshold_dict,
                'metric': config.eval['threshold_metric'],
                'default_threshold': 0.5
            }, f, indent=2)
        
        print(f"Optimal thresholds saved to {threshold_path}")
        
        # Apply optimized thresholds
        thresholds_array = np.array([optimal_thresholds[i] for i in range(len(config.class_names))])
        all_preds_binary = (all_preds > thresholds_array).astype(int)
    else:
        # Use default 0.5 threshold
        all_preds_binary = (all_preds > 0.5).astype(int)
    
    # === Classification Report ===
    print("\n=== Classification Report (Optimized Thresholds) ===")
    print(classification_report(all_targets, all_preds_binary, target_names=config.class_names, zero_division=0))
    
    # === Visualizations ===
    if config.eval['save_visualizations'] and viz is not None:
        print(f"\n=== Generating {config.eval['num_viz_samples']} Visualizations ===")
        
        os.makedirs(config.paths['viz_dir'], exist_ok=True)
        
        # Select random samples
        viz_indices = np.random.choice(len(val_indices), min(config.eval['num_viz_samples'], len(val_indices)), replace=False)
        
        for i, val_idx_pos in enumerate(viz_indices):
            idx = val_indices[val_idx_pos]
            
            img_name = base_dataset.df.iloc[idx, 0]
            img_path = base_dataset.image_paths.get(img_name)
            
            if img_path is None:
                continue
            
            image = Image.open(img_path).convert('RGB')
            labels = base_dataset.df.iloc[idx][base_dataset.all_labels].values.astype('float32')
            
            # Get prediction
            img_tensor = val_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred_probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # Get top predicted class
            top_class_idx = np.argmax(pred_probs)
            top_class_prob = pred_probs[top_class_idx]
            class_name = config.class_names[top_class_idx]
            
            # Generate Grad-CAM
            heatmap = viz(img_tensor, class_idx=top_class_idx)
            
            # Denormalize for visualization
            orig_img = denormalize_image(img_tensor)
            
            # Apply heatmap
            visualization = show_cam_on_image(orig_img, heatmap)
            
            # Get true labels
            true_labels_indices = np.where(labels == 1)[0]
            true_labels = [config.class_names[j] for j in true_labels_indices]
            true_label_str = ", ".join(true_labels) if true_labels else "No Finding"
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\nTrue: {true_label_str}", fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(visualization)
            axes[1].set_title(f"{config.eval['viz_method']}\nPred: {class_name} ({top_class_prob:.3f})", fontsize=10)
            axes[1].axis('off')
            
            plt.tight_layout()
            
            save_path = os.path.join(config.paths['viz_dir'], f"viz_{i:03d}_{img_name}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {config.paths['viz_dir']}/")
    
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate lung disease classification model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--viz_method', type=str, choices=['gradcam++', 'scorecam'], default='gradcam++')
    parser.add_argument('--num_viz_samples', type=int, default=10, help='Number of visualization samples')
    parser.add_argument('--optimize_threshold', action='store_true', help='Optimize classification thresholds')
    parser.add_argument('--threshold_metric', type=str, choices=['f1', 'youden', 'precision', 'recall'], default='f1')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line args
    kwargs = {}
    if args.use_tta:
        kwargs['use_tta'] = True
    if args.viz_method:
        kwargs['viz_method'] = args.viz_method
    if args.num_viz_samples:
        kwargs['num_viz_samples'] = args.num_viz_samples
    if args.optimize_threshold:
        kwargs['optimize_threshold'] = True
    if args.threshold_metric:
        kwargs['threshold_metric'] = args.threshold_metric
    
    # Evaluate
    evaluate_model(args.model_path, config=config, **kwargs)
