from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
from model import LungDiseaseModel

app = FastAPI(title="PulmoLens API", description="Lung Disease Identification API")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LungDiseaseModel(num_classes=14)
# In a real deployment, you would load the best weights here
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
        # Format results
        results = {label: float(prob) for label, prob in zip(LABELS, probs)}
        
        # Sort by probability
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        
        return JSONResponse(content={"predictions": sorted_results})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
