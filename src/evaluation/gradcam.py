import torch
import cv2
import numpy as np
import argparse
import os
import json
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from src import config
from src.models.densenet import DenseNet121
from src.data.dataset import get_transforms

def load_thresholds(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {c: 0.5 for c in config.CLASS_NAMES}

def evaluate_and_visualize(model_path, image_path, thresholds_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DenseNet121(num_classes=len(config.CLASS_NAMES))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load thresholds
    thresholds = load_thresholds(thresholds_path) if thresholds_path else {c: 0.5 for c in config.CLASS_NAMES}
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img) / 255.0
    
    transform = get_transforms('val')
    input_tensor = transform(image=img)['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
        
    # Print predictions
    print(f"Predictions for {os.path.basename(image_path)}:")
    print("-" * 40)
    found_findings = False
    for i, class_name in enumerate(config.CLASS_NAMES):
        prob = probs[i]
        thresh = thresholds.get(class_name, 0.5)
        if prob >= thresh:
            print(f"[FOUND] {class_name:20s}: {prob:.4f} (Threshold: {thresh:.4f})")
            found_findings = True
        else:
            # Optional: print low confidence ones if needed
            pass
            
    if not found_findings:
        print("No findings detected.")
        
    # Grad-CAM++ Visualization
    # Target the last dense block
    target_layers = [model.densenet.features.denseblock4.denselayer16.conv2]
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # Generate CAM for the highest probability class or specific target
    # Here we visualize the top predicted class
    top_class_idx = np.argmax(probs)
    top_class_name = config.CLASS_NAMES[top_class_idx]
    
    targets = [ClassifierOutputTarget(top_class_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize original image to 512x512 if needed for overlay
    img_resized = cv2.resize(img_float, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
    
    # Save result
    save_name = f"gradcam_{top_class_name}_{os.path.basename(image_path)}"
    save_path = os.path.join(config.RESULTS_DIR, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"\nGrad-CAM++ for {top_class_name} saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    parser.add_argument('--image_path', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--thresholds_path', type=str, default='results/optimal_thresholds.json')
    args = parser.parse_args()
    
    evaluate_and_visualize(args.model_path, args.image_path, args.thresholds_path)
