import io
import numpy as np
import torch
import cv2
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from PIL import Image
import uvicorn
import os
import uuid
import json
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import AttentionDenseNet
import logging

logger = logging.getLogger("pulmolens")
logging.basicConfig(level=logging.INFO)

API_VERSION = "1.0.0"

# --- MCP: Model Context Protocol Tool Interface ---
MCP_TOOL_DEFINITION = {
    "name": "analyze_chest_xray",
    "description": "Analyze a chest X-Ray for 14 pathologies (Pneumonia, etc) using a RAG-grounded PyTorch model.",
    "input_schema": {
        "type": "object",
        "properties": {
            "image_b64": {"type": "string", "description": "Base64 encoded CXR image string"}
        },
        "required": ["image_b64"]
    }
}

app = FastAPI(title="PulmoLens API", description="AI-Assisted Radiographic Guidance API", version=API_VERSION)

# Azure Configuration (Env Vars or Defaults for Dev)
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
STORAGE_CONN_STR = os.getenv("STORAGE_CONN_STR")

# Initialize Clients
cosmos_container = None
blob_container_client = None

try:
    if COSMOS_ENDPOINT and COSMOS_KEY:
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = cosmos_client.get_database_client("pulmolens-db")
        cosmos_container = database.get_container_client("feedback")
        logger.info("Cosmos DB client initialized")
    
    if STORAGE_CONN_STR:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        blob_container_client = blob_service_client.get_container_client("images")
        logger.info("Blob Storage client initialized")
except Exception as e:
    logger.error(f"Error initializing Azure clients: {e}")

# Initialize RAG Components
load_dotenv()
try:
    rag_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vector_store = PineconeVectorStore(index_name="pulmolens-guidelines", embedding=rag_embeddings)
    # Using the most recent Gemini Flash Lite model!
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    logger.info("RAG components initialized (Gemini Flash Lite + Pinecone)")
except Exception as e:
    logger.error(f"Error initializing RAG components: {e}")
    vector_store = None
    llm = None

# Add CORS middleware
origins = [
    "https://victorious-sky-0836ce10f.3.azurestaticapps.net",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

from fastapi import Form

@router.post("/feedback")
async def submit_feedback(
    file: UploadFile = File(...),
    rating: str = Form(...),
    details: Optional[str] = Form(None),
    predictions: Optional[str] = Form(None) # JSON string
):
    # Upload to Blob Storage
    blob_url = None
    if blob_container_client:
        try:
            # Reset file pointer if needed, though usually fresh from upload
            blob_name = f"{uuid.uuid4()}_{file.filename}"
            blob_container_client.upload_blob(blob_name, file.file)
            blob_url = blob_container_client.get_blob_client(blob_name).url
            logger.info(f"Image uploaded to blob: {blob_url}")
        except Exception as e:
            logger.error(f"Error uploading to blob: {e}")

    # Save to Cosmos DB
    if cosmos_container:
        try:
            item = {
                "id": str(uuid.uuid4()),
                "image_id": blob_url or "upload_failed", # Store blob URL as image_id
                "rating": rating,
                "details": details,
                "predictions": json.loads(predictions) if predictions else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            cosmos_container.create_item(body=item)
            logger.info("Feedback saved to Cosmos DB")
        except Exception as e:
            logger.error(f"Error saving to Cosmos DB: {e}")
            
    return {"status": "received", "message": "Thank you for your feedback!"}

# --- CLOUD DECOUPLING: Model Download Helper ---
def download_model(url, path):
    try:
        logger.info(f"Downloading model weights from Azure Blob Storage...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024 * 1024):
                file.write(data)
        logger.info(f"Model downloaded successfully to {path}")
    except Exception as e:
        logger.critical(f"Model download failed: {e}")
        raise e

# Load PyTorch model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to relative path for Local Dev, but allow env override for Cloud
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "models", "attention_densenet_asl_20251121_213351_best.pth")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
device = torch.device("cpu")

try:
    # 1. Check for Cloud Download (MODEL_SAS_URL)
    sas_url = os.environ.get("MODEL_SAS_URL")
    if not os.path.exists(MODEL_PATH) and sas_url:
        download_model(sas_url, MODEL_PATH)

    # 2. Initialize and Load
    model = AttentionDenseNet(num_classes=14)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Inference will be unavailable.")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Class names
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def preprocess_image(image_bytes):
    """
    Preprocess image for model inference and Grad-CAM
    Returns:
        input_tensor: (1, C, H, W) tensor for model
        img_float: (H, W, C) float32 numpy array for Grad-CAM overlay
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = image.resize((512, 512))
        
        # Convert to numpy and normalize
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Keep a copy for Grad-CAM overlay (H, W, C)
        img_float = img_array.copy()
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_array = (img_array - mean) / std
        
        # Transpose to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return input_tensor, img_float, image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@router.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded (PyTorch)", "version": API_VERSION}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Initialize response variables to avoid UnboundLocalError
    clinical_report = "Analysis complete."
    cited_sources = []
    
    contents = await file.read()
    input_tensor, img_float, image = preprocess_image(contents)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Format predictions
    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        results[class_name] = float(probs[i])
        
    # Generate Grad-CAM
    heatmap_b64 = None
    try:
        # Target the last norm layer of AttentionDenseNet
        target_layers = [model.features.norm5]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        
        # Target the highest probability class
        target_class = np.argmax(probs)
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on image
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Resize back to original size
        orig_w, orig_h = image.size
        visualization = cv2.resize(visualization, (orig_w, orig_h))
        
        # Convert to base64
        img = Image.fromarray(visualization)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        
    # Upload to Blob Storage REMOVED for privacy
    # Images are only stored if user submits feedback
    blob_url = None 
    
    # --- RAG PIPELINE: GENERATE REPORT ---
    top_findings = [cls_name for cls_name, prob in results.items() if prob > 0.4]
    clinical_report = "The AI vision model did not detect any major pathologies with high confidence."
    
    if top_findings and vector_store and llm:
        try:
            # 1. Retrieval
            search_query = " ".join(top_findings)
            retrieved_docs = vector_store.similarity_search(search_query, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # --- AGENTIC GATE: Confidence Check (Deterministic Flow) ---
            # If the top finding is below 50% confidence, we trigger the 'Uncertainty Agent'
            sorted_probs = sorted(probs, reverse=True)
            is_low_confidence = sorted_probs[0] < 0.5 if len(sorted_probs) > 0 else True
            
            if is_low_confidence:
                prompt_task = "The model is UNCERTAIN (confidence < 50%). Explain that the image quality or features are indeterminate and suggest a higher-fidelity scan or clinical correlation to confirm findings."
                guidance_tone = "Cautious, seeking more data."
            else:
                prompt_task = f"The core detection is {top_findings[0]}. Provide an expert radiographic signature and management plan based on the guidelines."
                guidance_tone = "Authoritative, expert consultant."

            # 2. Generation Prompt - Senior Expert Persona Implementation
            prompt = f"""
            SYSTEM ROLE: You are a Senior Radiographic Consultant AI.
            TASK: {prompt_task}
            TONE: {guidance_tone}
            
            VISION ANALYSIS: {', '.join(top_findings)}
            
            STRUCTURE YOUR RESPONSE AS FOLLOWS:
            **Radiographic Signature**: [Visual cues]
            **Clinical Management**: [Next steps from guidelines]
            **Patient Summary**: [Simple explanation]
            
            GUIDELINES:
            {context}
            """
            
            # 3. Call LLM
            response = llm.invoke(prompt)
            
            # --- FIX: Handle List-style response.content from newer Gemini models ---
            if isinstance(response.content, str):
                clinical_report = response.content
            elif isinstance(response.content, list):
                # Extract text parts if Gemini returns a list of blocks
                text_parts = [p.get('text', '') if isinstance(p, dict) else str(p) for p in response.content]
                clinical_report = "".join(text_parts)
            else:
                clinical_report = str(response.content)
            
            # --- EXTRACT SOURCES for transparency ---
            cited_sources = []
            for doc in retrieved_docs:
                src = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '?')
                # Clean up path to just filename
                src_filename = os.path.basename(src)
                cited_sources.append(f"{src_filename} (Page {page})")
            
            # Remove duplicates while preserving order
            cited_sources = list(dict.fromkeys(cited_sources))
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            clinical_report = f"Detected {', '.join(top_findings)}. (RAG explanation temporarily unavailable)."
            cited_sources = []

    return {
        "predictions": results,
        "heatmap": heatmap_b64,
        "version": API_VERSION,
        "imageId": blob_url or file.filename,
        "report": clinical_report,
        "sources": cited_sources
    }

# --- MCP ENDPOINTS: Enabling Claude/Desktop tool use ---
@app.get("/mcp/tools")
async def get_mcp_tools():
    """List available tools for Model Context Protocol hosts"""
    return {"tools": [MCP_TOOL_DEFINITION]}

@app.post("/mcp/analyze")
async def mcp_analyze(request: dict):
    """Execute analysis via MCP protocol"""
    try:
        image_b64 = request.get("image_b64")
        if not image_b64: return {"error": "No image_b64 found"}
        
        # Decode and wrap in mock UploadFile
        img_bytes = base64.b64decode(image_b64)
        from fastapi import UploadFile
        import io
        mock_file = UploadFile(filename="mcp_input.jpg", file=io.BytesIO(img_bytes))
        
        result = await analyze(mock_file)
        return {
            "content": [
                {"type": "text", "text": f"PulmoLens Analysis Result: {result['report']}"},
                {"type": "text", "text": f"Findings summary: {', '.join([p['label'] for p in result['predictions'] if p['prob'] > 0.5])}"}
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Include router at root and /api
app.include_router(router)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
