import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import argparse
import json
import os
from tqdm import tqdm
from src import config
from src.models.densenet import AttentionDenseNet, DenseNet121
from src.data.dataset import get_data_loaders

def evaluate_test(model_path, thresholds_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load thresholds
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {thresholds_path}")
    else:
        print("Thresholds not found, using 0.5 default")
        thresholds = {c: 0.5 for c in config.CLASS_NAMES}
        
    # Load model
    # Check config in checkpoint to determine model type
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        model_type = checkpoint['config'].get('model_type', 'attention_densenet')
    else:
        model_type = 'attention_densenet'
        
    print(f"Initializing {model_type}...")
    if model_type == 'attention_densenet':
        model = AttentionDenseNet(num_classes=len(config.CLASS_NAMES))
    else:
        model = DenseNet121(num_classes=len(config.CLASS_NAMES))
        
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # Get test loader
    _, _, test_loader = get_data_loaders()
    
    # Inference
    all_probs = []
    all_targets = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate Metrics
    print("\n=== Test Set Evaluation ===")
    
    # AUC
    aucs = []
    for i, class_name in enumerate(config.CLASS_NAMES):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            aucs.append(auc)
            print(f"AUC - {class_name:20s}: {auc:.4f}")
        except:
            pass
    print(f"\nMean AUC: {np.mean(aucs):.4f}")
    
    # Apply Thresholds
    all_preds = np.zeros_like(all_probs)
    for i, class_name in enumerate(config.CLASS_NAMES):
        thresh = thresholds.get(class_name, 0.5)
        all_preds[:, i] = (all_probs[:, i] >= thresh).astype(int)
        
    # Classification Report
    print("\n=== Classification Report (Optimized Thresholds) ===")
    print(classification_report(all_targets, all_preds, target_names=config.CLASS_NAMES, zero_division=0))
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, 'test_evaluation.txt')
    with open(results_path, 'w') as f:
        f.write(f"Mean AUC: {np.mean(aucs):.4f}\n\n")
        f.write(classification_report(all_targets, all_preds, target_names=config.CLASS_NAMES, zero_division=0))
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--thresholds_path', type=str, default='results/optimal_thresholds.json')
    args = parser.parse_args()
    
    evaluate_test(args.model_path, args.thresholds_path)
