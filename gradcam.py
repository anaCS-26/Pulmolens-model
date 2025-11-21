import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        self.target_layer.register_backward_hook(self.save_gradient)
        # Hook for activations
        self.target_layer.register_forward_hook(self.save_activation)
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap (Min-Max)
        heatmap_min = torch.min(heatmap)
        heatmap_max = torch.max(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # Thresholding to reduce noise (reduce "all red" effect)
        heatmap[heatmap < 0.2] = 0
        
        heatmap = heatmap.cpu().numpy()
        
        # Resize heatmap to 224x224
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
