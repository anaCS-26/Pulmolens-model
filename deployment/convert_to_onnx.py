import torch
import torch.onnx
from model import LungDiseaseModel
import os

def convert_to_onnx():
    # Initialize model
    model = LungDiseaseModel(num_classes=14)
    
    # Load weights if available
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded best_model.pth")
    else:
        print("Warning: best_model.pth not found, using random weights for demonstration")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        "pulmolens_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Model converted to pulmolens_model.onnx")

if __name__ == "__main__":
    convert_to_onnx()
