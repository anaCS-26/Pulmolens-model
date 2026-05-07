import io
import numpy as np
import torch
import cv2
import base64
import binascii
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Header
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

# --- INITIALIZATION: Load environment before any cloud clients ---
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
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
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").strip().lower() == "true"
RAG_MIN_TOP_PROB = float(os.getenv("RAG_MIN_TOP_PROB", "0.5"))
ALLOW_UNSAFE_MODEL_DESERIALIZATION = os.getenv("ALLOW_UNSAFE_MODEL_DESERIALIZATION", "false").strip().lower() == "true"
MCP_API_KEY = os.getenv("MCP_API_KEY")

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

# --- CLOUD CLIENTS: Azure Configuration ---
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
        logger.info("✅ Cosmos DB client initialized")
    
    if STORAGE_CONN_STR:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        blob_container_client = blob_service_client.get_container_client("images")
        logger.info("✅ Blob Storage client initialized")
except Exception as e:
    logger.error(f"⚠️ Azure client initialization warning: {e}")

# --- AI CLIENTS: RAG Components ---
vector_store = None
llm = None

try:
    # Use explicit model strings for new Gemini 2.5 releases
    # 3072 is the native dimensionality for gemini-embedding-2-preview
    rag_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        output_dimensionality=3072
    )
    vector_store = PineconeVectorStore(index_name="pulmolens-guidelines", embedding=rag_embeddings)
    
    # Updated to the new Gemma 4 31B Dense model for advanced reasoning
    # Gemma 4 is a state-of-the-art open-weight model with multimodal capabilities
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        system_instruction="You are a Senior Radiographic Consultant AI. Provide expert radiographic signatures and clinical management plans grounded in guidelines. Always verify your summary against the specific raw vision data provided. Use your advanced reasoning to explain the relationship between findings and guidelines.",
        thinking_level="high",
        temperature=1.0
    )
    logger.info("✅ RAG components initialized (Gemini 2.5 Flash Lite + Thinking Mode + Pinecone 3072)")
except Exception as e:
    logger.error(f"❌ RAG initialization failure: {e}")
    vector_store = None
    llm = None

# Add CORS middleware
origins = [
    "https://victorious-sky-0836ce10f.3.azurestaticapps.net",
    "https://ashy-field-00930a60f.azurestaticapps.net",
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
    if rating not in {"good", "bad"}:
        raise HTTPException(status_code=400, detail="rating must be 'good' or 'bad'")

    contents = await file.read()
    _read_and_validate_upload(contents, file.content_type)

    parsed_predictions = None
    if predictions:
        try:
            parsed_predictions = json.loads(predictions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="predictions must be valid JSON")

    # Upload to Blob Storage
    blob_url = None
    if blob_container_client:
        try:
            ext = _extension_from_content_type(file.content_type)
            blob_name = f"{uuid.uuid4()}.{ext}"
            blob_container_client.upload_blob(blob_name, contents, overwrite=False, content_type=file.content_type)
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
                "predictions": parsed_predictions,
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


def _read_and_validate_upload(contents: bytes, content_type: str):
    if not content_type or not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )


def _extension_from_content_type(content_type: str) -> str:
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp"
    }
    return mapping.get((content_type or "").lower(), "bin")

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

    # 2. Initialize and Load (disable default pretrained weights as we load a custom checkpoint)
    model = AttentionDenseNet(num_classes=14, pretrained=False)
    if os.path.exists(MODEL_PATH):
        if ALLOW_UNSAFE_MODEL_DESERIALIZATION:
            logger.warning("Unsafe model deserialization enabled via ALLOW_UNSAFE_MODEL_DESERIALIZATION=true")
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
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

@router.get("/warmup")
async def warmup():
    """Trigger container spin-up and model loading if needed"""
    if model is None:
        return {"status": "warming", "message": "Model is loading..."}
    return {"status": "ready", "message": "Model is ready"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Initialize response variables to avoid UnboundLocalError
    clinical_report = "Analysis complete."
    cited_sources = []
    
    contents = await file.read()
    _read_and_validate_upload(contents, file.content_type)
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
    # Trigger RAG even for low confidence but adjust the logic
    top_findings = [cls_name for cls_name, prob in results.items() if prob > 0.4]
    
    # We still want RAG/LLM to explain if results are low confidence
    sorted_probs = sorted(probs, reverse=True)
    max_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0

    if vector_store and llm and RAG_ENABLED and max_prob >= RAG_MIN_TOP_PROB:
        try:
            # 1. Retrieval
            search_query = " ".join(top_findings) if top_findings else "no findings indeterminate"
            retrieved_docs = vector_store.similarity_search(search_query, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # --- PREPARE DATA: Sort findings for the LLM to prioritize correctly ---
            sorted_all_findings = sorted(results.items(), key=lambda x: x[1], reverse=True)
            formatted_vision_data = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in sorted_all_findings if v > 0.05])

            # --- AGENTIC GATE: Confidence Check (Deterministic Flow) ---
            if len(top_findings) == 0:
                prompt_task = "Explain that no specific pathology was detected with high confidence and provide general preventative chest health advice."
                vision_summary = "No significant abnormalities detected above threshold."
            elif max_prob < 0.5:
                prompt_task = f"The model is UNCERTAIN (top finding '{sorted_all_findings[0][0]}' is only {max_prob*100:.1f}%). Explain indeterminate features and suggest follow-up."
                vision_summary = f"Marginal detection for {sorted_all_findings[0][0]}."
            else:
                prompt_task = f"The primary detection is {sorted_all_findings[0][0]} (confidence {max_prob*100:.1f}%). Provide expert radiographic signature and management plan."
                vision_summary = f"High confidence in {sorted_all_findings[0][0]}."

            # 2. Generation Prompt - Professional Persona & Internal Knowledge Fallback
            prompt = f"""
            [INTERNAL DATA - DO NOT REFERENCE BY NAME IN OUTPUT]
            FINDINGS: {formatted_vision_data}
            RELEVANT_GUIDELINES: {context}
            
            ### INSTRUCTIONS for Senior Radiographic Consultant
            You are providing a clinical interpretation of a chest X-ray. 
            
            1. PRIMARY FINDING: Focus your report on the finding with the highest confidence: {sorted_all_findings[0][0]} ({max_prob*100:.1f}%).
            2. GUIDELINE ADHERENCE: If 'RELEVANT_GUIDELINES' contains specific management for {sorted_all_findings[0][0]}, prioritize that information.
            3. VISUAL GROUNDING: Use the attached Grad-CAM heatmap to guide your signature. The heatmap highlights areas the classification model focused on. Confirm if these areas align with the expected pathology location.
            4. KNOWLEDGE FALLBACK: If the provided guidelines do not cover {sorted_all_findings[0][0]}, use your internal medical training to provide standard radiographic descriptors and general management steps. Do NOT state that the guidelines are missing or mention 'raw data'; remain professional and helpful.
            5. TONE: Maintain a formal, consultant-level tone. Use clear, medical language.
            
            STRUCTURE YOUR RESPONSE AS FOLLOWS:
            **Radiographic Signature**: [Visual cues for {sorted_all_findings[0][0]}]
            **Clinical Management**: [Expert next steps or diagnostic follow-up]
            **Patient Summary**: [A clear, empathetic explanation of the {sorted_all_findings[0][0]} finding]
            """

            # 3. Call LLM with Multimodal Input (Phase 2 Enabled)
            # We pass the prompt and the combined overlaid heatmap image
            logger.info(f"📤 Sending multimodal request to {llm.model} with heatmap image ({len(heatmap_b64)} chars)")
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": heatmap_b64}}
            ])
            response = llm.invoke([message])
            
            # --- FIX: Handle List-style response.content from newer Gemma/Gemini models ---
            if isinstance(response.content, str):
                clinical_report = response.content
            elif isinstance(response.content, list):
                # Extract text parts, but EXCLUDE internal thinking blocks for the patient report
                # Thinking blocks are logged for audit/transparency but not shown in clinical_report
                text_parts = []
                for p in response.content:
                    if isinstance(p, dict):
                        if p.get('thought') is True:
                            logger.info(f"🧠 Gemma 4 Thinking: {p.get('text')}")
                            continue
                        text_parts.append(p.get('text', ''))
                    else:
                        text_parts.append(str(p))
                clinical_report = "".join(text_parts).strip()
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
            logger.error(f"RAG pipeline error: {e}", exc_info=True)
            clinical_report = f"Detected {', '.join(top_findings)}. (RAG explanation temporarily unavailable)."
            cited_sources = []
    elif not RAG_ENABLED:
        clinical_report = "RAG explanation is disabled for this deployment."
    elif max_prob < RAG_MIN_TOP_PROB:
        clinical_report = (
            f"Top confidence ({max_prob*100:.1f}%) is below the RAG threshold "
            f"({RAG_MIN_TOP_PROB*100:.1f}%), so guideline generation was skipped to reduce cost."
        )
    else:
        # Fallback if RAG components failed to initialize but were expected to run
        findings_str = ", ".join([f"{k} ({v*100:.1f}%)" for k, v in results.items() if v > 0.1])
        if not findings_str:
            findings_str = "No significant findings."
        clinical_report = f"Analysis complete. Custom model breakdown: {findings_str}. (Note: Detailed AI explanation is unavailable, likely due to missing Gemini/Pinecone API keys or initialization failure)."

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
async def mcp_analyze(request: dict, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    """Execute analysis via MCP protocol"""
    try:
        if MCP_API_KEY and x_api_key != MCP_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized MCP request")

        image_b64 = request.get("image_b64")
        if not image_b64: return {"error": "No image_b64 found"}
        
        # Decode and wrap in mock UploadFile
        try:
            img_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError):
            return {"error": "Invalid base64 image payload"}

        _read_and_validate_upload(img_bytes, "image/jpeg")
        from fastapi import UploadFile
        import io
        mock_file = UploadFile(filename="mcp_input.jpg", file=io.BytesIO(img_bytes))
        
        result = await predict(mock_file)
        high_conf = [name for name, prob in result["predictions"].items() if prob > 0.5]
        return {
            "content": [
                {"type": "text", "text": f"PulmoLens Analysis Result: {result['report']}"},
                {"type": "text", "text": f"Findings summary: {', '.join(high_conf) if high_conf else 'No high-confidence findings'}"}
            ]
        }
    except Exception as e:
        logger.exception("MCP analyze failed: %s", e)
        return {"error": "Analysis failed"}

# Include router at root and /api
app.include_router(router)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
