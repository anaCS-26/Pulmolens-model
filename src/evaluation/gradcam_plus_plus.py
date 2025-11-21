import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for improved localization.
    Better handling of multiple instances and more precise heatmaps.
    
    Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_backward_hook(self.save_gradient)
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
        
        # Get gradients and activations
        gradients = self.gradients  # [batch, channels, h, w]
        activations = self.activations.detach()  # [batch, channels, h, w]
        
        b, k, u, v = gradients.size()
        
        # Calculate alpha (Grad-CAM++ weighting)
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        
        # Add small epsilon to avoid division by zero
        ag = activations * gradients.pow(3)
        denominator += ag.view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        denominator = torch.where(
            denominator != 0.0, 
            denominator, 
            torch.ones_like(denominator)
        )
        
        alpha = numerator / (denominator + 1e-8)
        
        # ReLU on gradients (positive influence)
        positive_gradients = F.relu(output[0][class_idx].exp() * gradients)
        
        # Weight the channels with alpha
        weights = (alpha * positive_gradients).view(b, k, u*v).sum(-1)
        
        # Weight activations by the weights
        weighted_activations = weights.view(b, k, 1, 1) * activations
        
        # Sum over channels
        heatmap = weighted_activations.sum(1).squeeze(0)
        
        # ReLU on heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap = heatmap.cpu().numpy()
        
        # Resize to input size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap


class ScoreCAM:
    """
    Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
    Gradient-free alternative to Grad-CAM that uses forward passes only.
    More stable and doesn't require backpropagation.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register forward hook only
        self.target_layer.register_forward_hook(self.save_activation)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def __call__(self, x, class_idx=None, batch_size=32):
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass to get activations
            output = self.model(x)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            activations = self.activations  # [1, channels, h, w]
            b, k, u, v = activations.size()
            
            # Upsample each activation map to input size
            scores = []
            
            # Process in batches to avoid memory issues
            for i in range(0, k, batch_size):
                batch_end = min(i + batch_size, k)
                batch_acts = activations[:, i:batch_end, :, :]
                
                # Normalize and upsample each activation map
                upsampled = F.interpolate(
                    batch_acts, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Normalize to [0, 1]
                normalized = torch.zeros_like(upsampled)
                for j in range(upsampled.size(1)):
                    act_map = upsampled[0, j, :, :]
                    if act_map.max() > act_map.min():
                        normalized[0, j, :, :] = (act_map - act_map.min()) / (act_map.max() - act_map.min())
                
                # Multiply with input image and get predictions
                batch_scores = []
                for j in range(normalized.size(1)):
                    masked_input = x * normalized[:, j:j+1, :, :].expand_as(x)
                    score = torch.sigmoid(self.model(masked_input))[0, class_idx].item()
                    batch_scores.append(score)
                
                scores.extend(batch_scores)
            
            scores = torch.FloatTensor(scores).to(activations.device)
            scores = F.softmax(scores, dim=0)
            
            # Weight activation maps by their scores
            weighted_activations = (activations.squeeze(0) * scores.view(-1, 1, 1)).sum(0)
            
            # Normalize
            heatmap = weighted_activations.cpu().numpy()
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Resize to input size
            heatmap = cv2.resize(heatmap, (224, 224))
            
            return heatmap


def show_cam_on_image(img, mask, use_rgb=True):
    """
    Overlay heatmap on image.
    
    Args:
        img: Original image, numpy array in range [0, 1]
        mask: Heatmap, numpy array in range [0, 1]
        use_rgb: If True, assume RGB input
    
    Returns:
        Visualization as uint8 numpy array
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255
    
    # Weighted combination
    cam = 0.4 * heatmap + 0.6 * np.float32(img)
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)


def compare_visualizations(img, masks_dict, titles=None):
    """
    Create side-by-side comparison of different visualization methods.
    
    Args:
        img: Original image
        masks_dict: Dictionary of {method_name: heatmap}
        titles: Optional list of titles
    
    Returns:
        Combined visualization
    """
    import matplotlib.pyplot as plt
    
    n_methods = len(masks_dict) + 1  # +1 for original image
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    # Show original
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show each method
    for idx, (method_name, mask) in enumerate(masks_dict.items(), 1):
        cam_img = show_cam_on_image(img, mask)
        axes[idx].imshow(cam_img)
        axes[idx].set_title(method_name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Grad-CAM++ and Score-CAM implementation ready!")
    print("Usage:")
    print("  gradcam_pp = GradCAMPlusPlus(model, target_layer)")
    print("  heatmap = gradcam_pp(input_tensor, class_idx=0)")
    print("  ")
    print("  score_cam = ScoreCAM(model, target_layer)")
    print("  heatmap = score_cam(input_tensor, class_idx=0)")
