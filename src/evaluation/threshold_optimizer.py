import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os


def find_optimal_threshold_per_class(y_true, y_pred, metric='f1'):
    """
    Find optimal threshold for each class.
    
    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        metric: Optimization metric ('f1', 'youden', 'precision', 'recall')
    
    Returns:
        Dictionary of {class_idx: optimal_threshold}
    """
    n_classes = y_true.shape[1]
    optimal_thresholds = {}
    
    for class_idx in range(n_classes):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]
        
        # Skip if only one class present
        if len(np.unique(y_true_class)) < 2:
            optimal_thresholds[class_idx] = 0.5
            continue
        
        if metric == 'f1':
            # Find threshold that maximizes F1 score
            precision, recall, thresholds = precision_recall_curve(y_true_class, y_pred_class)
            
            # Calculate F1 for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find best threshold
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                optimal_threshold = thresholds[best_idx]
            else:
                optimal_threshold = 0.5
                
        elif metric == 'youden':
            # Youden's J statistic (TPR - FPR)
            fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[best_idx]
            
        elif metric == 'precision':
            # Maximize precision at reasonable recall (>0.5)
            precision, recall, thresholds = precision_recall_curve(y_true_class, y_pred_class)
            valid_idx = recall >= 0.5
            if valid_idx.sum() > 0:
                best_idx = np.argmax(precision[valid_idx])
                optimal_threshold = thresholds[valid_idx][best_idx] if best_idx < len(thresholds[valid_idx]) else 0.5
            else:
                optimal_threshold = 0.5
                
        elif metric == 'recall':
            # Maximize recall at reasonable precision (>0.5)
            precision, recall, thresholds = precision_recall_curve(y_true_class, y_pred_class)
            valid_idx = precision >= 0.5
            if valid_idx.sum() > 0:
                best_idx = np.argmax(recall[valid_idx])
                optimal_threshold = thresholds[valid_idx][best_idx] if best_idx < len(thresholds[valid_idx]) else 0.5
            else:
                optimal_threshold = 0.5
        else:
            optimal_threshold = 0.5
        
        optimal_thresholds[class_idx] = float(optimal_threshold)
    
    return optimal_thresholds


def optimize_thresholds(model, dataloader, device, class_names, metric='f1', save_path='optimal_thresholds.json'):
    """
    Optimize thresholds for all classes using validation data.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Device to run on
        class_names: List of class names
        metric: Optimization metric
        save_path: Path to save thresholds
    
    Returns:
        Dictionary of optimal thresholds
    """
    print(f"Finding optimal thresholds using {metric} metric...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_threshold_per_class(all_targets, all_preds, metric=metric)
    
    # Map to class names
    threshold_dict = {
        class_names[i]: threshold 
        for i, threshold in optimal_thresholds.items()
    }
    
    # Add metadata
    result = {
        'thresholds': threshold_dict,
        'metric': metric,
        'default_threshold': 0.5
    }
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Optimal thresholds saved to {save_path}")
    print("\nThresholds:")
    for class_name, threshold in threshold_dict.items():
        print(f"  {class_name}: {threshold:.3f}")
    
    return threshold_dict


def load_thresholds(path='optimal_thresholds.json', class_names=None):
    """
    Load optimal thresholds from file.
    
    Args:
        path: Path to threshold file
        class_names: List of class names (for ordering)
    
    Returns:
        Numpy array of thresholds or dict
    """
    if not os.path.exists(path):
        print(f"Threshold file {path} not found. Using default 0.5")
        if class_names is not None:
            return np.array([0.5] * len(class_names))
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    threshold_dict = data['thresholds']
    
    if class_names is not None:
        # Convert to array in correct order
        thresholds = np.array([threshold_dict.get(name, 0.5) for name in class_names])
        return thresholds
    
    return threshold_dict


def visualize_threshold_curves(y_true, y_pred, class_names, save_dir='threshold_plots'):
    """
    Visualize precision, recall, F1 vs threshold for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        class_names: List of class names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = y_true.shape[1]
    
    for class_idx in range(n_classes):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]
        
        # Skip if only one class
        if len(np.unique(y_true_class)) < 2:
            continue
        
        # Compute metrics at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true_class, y_pred_class)
        
        # Calculate F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Extend threshold array for plotting
        thresholds_extended = np.concatenate([[0], thresholds])
        
        plt.plot(thresholds_extended, precision, label='Precision', linewidth=2)
        plt.plot(thresholds_extended, recall, label='Recall', linewidth=2)
        plt.plot(thresholds_extended, f1_scores, label='F1-Score', linewidth=2)
        
        # Mark optimal threshold
        plt.axvline(best_threshold, color='r', linestyle='--', 
                   label=f'Optimal (F1={f1_scores[best_idx]:.3f})')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'{class_names[class_idx]} - Metrics vs Threshold', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        save_path = os.path.join(save_dir, f'{class_names[class_idx]}_threshold_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Threshold curves saved to {save_dir}/")


if __name__ == "__main__":
    print("Threshold optimizer ready!")
    print("Usage:")
    print("  from threshold_optimizer import optimize_thresholds, load_thresholds")
    print("  ")
    print("  # Optimize")
    print("  thresholds = optimize_thresholds(model, val_loader, device, class_names)")
    print("  ")
    print("  # Load")
    print("  thresholds = load_thresholds('optimal_thresholds.json', class_names)")
