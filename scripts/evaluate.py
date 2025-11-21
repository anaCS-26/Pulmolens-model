import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import NIHChestXrayDataset
from model import LungDiseaseModel
from gradcam import GradCAM, show_cam_on_image
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def evaluate_model(model_path='best_model.pth', data_dir='./data'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = LungDiseaseModel(num_classes=14).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights.")
    else:
        print("Model weights not found. Using random weights for demo.")
    
    model.eval()
    
    # Target layer for Grad-CAM (last dense block's last layer usually)
    # For DenseNet121: model.densenet.features.denseblock4.denselayer16
    # But let's target the last BN layer of features: model.densenet.features.norm5
    target_layer = model.densenet.features.norm5
    grad_cam = GradCAM(model, target_layer)
    
    # Load a few samples
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Just grab a few images from the directory manually or use dataset
    # We'll use dataset to get labels too
    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    images_dir = data_dir
    
    if not os.path.exists(csv_path):
        print("Data not found.")
        return

    dataset = NIHChestXrayDataset(root_dir=images_dir, csv_file=csv_path, transform=transform)
    
    # --- Full Evaluation ---
    print("Running full evaluation on dataset...")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            
            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    from sklearn.metrics import roc_auc_score, classification_report
    
    print("\n--- Model Metrics ---")
    aucs = []
    for i, label in enumerate(dataset.all_labels):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            aucs.append(auc)
            print(f"{label}: AUC = {auc:.4f}")
        except ValueError:
            print(f"{label}: AUC = N/A (Only one class present)")
            
    print(f"\nMean AUC: {np.mean(aucs):.4f}")
    
    # F1 Score (using 0.5 threshold)
    print("\n--- Classification Report (Threshold 0.5) ---")
    print(classification_report(all_targets, (all_preds > 0.5).astype(int), target_names=dataset.all_labels, zero_division=0))
    
    # --- Visualization ---
    print("\nGenerating Grad-CAM visualizations...")
    # Visualize 5 samples
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    for idx in indices:
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        output = model(img_tensor)
        pred_probs = torch.sigmoid(output).detach().cpu().numpy()[0]
        
        # Get top prediction
        top_class_idx = np.argmax(pred_probs)
        top_class_prob = pred_probs[top_class_idx]
        class_name = dataset.all_labels[top_class_idx]
        
        print(f"Image {idx}: Predicted {class_name} ({top_class_prob:.2f})")
        
        # Generate Grad-CAM
        mask = grad_cam(img_tensor, class_idx=top_class_idx)
        
        # Original image for visualization
        # Denormalize
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        orig_img = inv_normalize(img_tensor.squeeze()).cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        visualization = show_cam_on_image(orig_img, mask)
        
        # Get true labels
        true_labels_indices = np.where(label.numpy() == 1)[0]
        true_labels = [dataset.all_labels[i] for i in true_labels_indices]
        true_label_str = ", ".join(true_labels) if true_labels else "No Finding"
        
        # Save
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title(f"Original\nTrue: {true_label_str}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM\nPred: {class_name} ({top_class_prob:.2f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"gradcam_result_{idx}.png")
        plt.close()
        print(f"Saved gradcam_result_{idx}.png")

if __name__ == "__main__":
    evaluate_model()
