import torch
import torch.onnx
import os
import argparse
from src.models.densenet import AttentionDenseNet, DenseNet121
from src import config

def convert_to_onnx(model_path, output_path):
    # Load checkpoint to check config
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Determine model type from checkpoint or default to AttentionDenseNet
    if 'config' in checkpoint:
        model_type = checkpoint['config'].get('model_type', 'attention_densenet')
    else:
        model_type = 'attention_densenet' # Default for this run
        
    print(f"Initializing {model_type}...")
    if model_type == 'attention_densenet':
        model = AttentionDenseNet(num_classes=len(config.CLASS_NAMES))
    else:
        model = DenseNet121(num_classes=len(config.CLASS_NAMES))
        
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Create dummy input (Batch Size 1, 3 Channels, 512x512)
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✓ Model converted successfully to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--output_path', type=str, default='pulmolens_model.onnx', help='Output ONNX file path')
    args = parser.parse_args()
    
    convert_to_onnx(args.model_path, args.output_path)
