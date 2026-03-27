import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score
import argparse
import json
import os
from tqdm import tqdm
from src import config
from src.models.densenet import DenseNet121
from src.data.dataset import get_data_loaders

def optimize_thresholds(model_path):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine model type
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            model_type = checkpoint['config'].get('model_type', 'densenet')
        else:
            model_type = 'attention_densenet' if 'attention' in model_path else 'densenet'
            
        print(f"Initializing {model_type}...")
        from src.models.densenet import AttentionDenseNet, DenseNet121
        
        if model_type == 'attention_densenet':
            model = AttentionDenseNet(num_classes=len(config.CLASS_NAMES))
        else:
            model = DenseNet121(num_classes=len(config.CLASS_NAMES))
            
        # Handle both full checkpoint and state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model path {model_path} not found.")
        return
    
    model.to(device)
    model.eval()
    
    # Get validation loader
    _, val_loader, _ = get_data_loaders()
    
    # Get predictions
    all_probs = []
    all_targets = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Find optimal thresholds
    optimal_thresholds = {}
    class_metrics = {}
    
    print("\nOptimizing thresholds per class:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        precision, recall, thresholds = precision_recall_curve(all_targets[:, i], all_probs[:, i])
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        optimal_thresholds[class_name] = float(best_threshold)
        class_metrics[class_name] = {
            "best_threshold": float(best_threshold),
            "best_f1": float(best_f1)
        }
        
        print(f"{class_name:20s}: Threshold={best_threshold:.4f}, F1={best_f1:.4f}")
        
    # Save thresholds
    save_path = os.path.join(config.RESULTS_DIR, 'optimal_thresholds.json')
    with open(save_path, 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)
        
    print(f"\nOptimal thresholds saved to {save_path}")
    return optimal_thresholds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    optimize_thresholds(args.model_path)
