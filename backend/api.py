import io
import numpy as np
import torch
import cv2
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
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
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv

# --- INITIALIZATION: Load environment before any cloud clients ---
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
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

API_VERSION = "1.0.0" # Sync: Gemma 4 + Reasoning Mode Stable Baseline
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").strip().lower() == "true"
RAG_MIN_TOP_PROB = float(os.getenv("RAG_MIN_TOP_PROB", "0.5"))
ALLOW_UNSAFE_MODEL_DESERIALIZATION = os.getenv("ALLOW_UNSAFE_MODEL_DESERIALIZATION", "false").strip().lower() == "true"

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
    # NOTE: system_instruction is not a valid ChatGoogleGenerativeAI constructor arg —
    # langchain silently moves it to model_kwargs and it never reaches the model.
    # The persona is delivered via SystemMessage in the request instead (see /summarize).
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        thinking_level="high",
        temperature=1.0,
    )
    logger.info("✅ RAG components initialized (Gemma 4 31B Dense + Reasoning Mode + Pinecone 3072)")
except Exception as e:
    logger.error(f"❌ RAG initialization failure: {e}")
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

# --- SCHEMAS ---
class SummarizeRequest(BaseModel):
    predictions: Dict[str, float]
    # Base64 PNG/JPEG of the X-ray with Grad-CAM heat regions baked on top
    # (a single composite, not a standalone heatmap). Produced by /predict.
    attention_overlay: str

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
        
    # Generate Grad-CAM overlay (X-ray + heat regions baked into one image)
    overlay_b64 = None
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
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        
    # Upload to Blob Storage REMOVED for privacy
    # Images are only stored if user submits feedback
    blob_url = None 
    
    return {
        "predictions": results,
        "attention_overlay": overlay_b64,
        "version": API_VERSION,
        "imageId": blob_url or file.filename,
    }

@router.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Second stage of the pipeline: Generates the RAG-grounded expert report as a stream.
    """
    if not (vector_store and llm and RAG_ENABLED):
         async def err_gen():
             yield json.dumps({"report": "AI summarization is currently unavailable."}) + "\n"
         return StreamingResponse(err_gen(), media_type="application/x-ndjson")

    results = request.predictions
    overlay_b64 = request.attention_overlay
    
    # Trigger RAG logic
    top_findings = [cls_name for cls_name, prob in results.items() if prob > 0.4]
    sorted_all_findings = sorted(results.items(), key=lambda x: x[1], reverse=True)
    max_prob = sorted_all_findings[0][1] if sorted_all_findings else 0
    
    if max_prob < RAG_MIN_TOP_PROB:
        async def threshold_gen():
            yield json.dumps({"report": f"Top confidence ({max_prob*100:.1f}%) is below the RAG threshold. Detailed report skipped."}) + "\n"
        return StreamingResponse(threshold_gen(), media_type="application/x-ndjson")

    primary = sorted_all_findings[0][0]
    primary_pct = max_prob * 100

    async def report_generator():
        try:
            # 1. Retrieval — expand bare class name into a clinical sentence so the
            # 3072-d embedding has enough lexical signal to discriminate between
            # guideline documents. Bare tokens like "Hernia" collapse to the
            # dominant doc in the index regardless of relevance.
            primary_phrase = primary.replace("_", " ")
            search_query = (
                f"radiographic findings, signs, and clinical management of "
                f"{primary_phrase} on chest x-ray"
            )
            co_findings = [t.replace("_", " ") for t in top_findings if t != primary]
            if co_findings:
                search_query += f" with co-existing {', '.join(co_findings)}"

            retrieved_docs = vector_store.similarity_search(search_query, k=3)

            # Only retain docs whose text actually mentions the primary finding.
            # Prevents the LLM from being handed irrelevant context (e.g. the
            # Fleischner nodule paper for a Pneumothorax query) which it would
            # then either ignore or, worse, get falsely cited as a source.
            primary_terms = {primary.lower(), primary_phrase.lower()}
            relevant_docs = [
                d for d in retrieved_docs
                if any(t in d.page_content.lower() for t in primary_terms)
            ]
            context = "\n\n".join(d.page_content for d in relevant_docs) if relevant_docs else ""

            formatted_vision_data = "\n".join(
                f"- {k}: {v*100:.1f}%" for k, v in sorted_all_findings if v > 0.05
            )

            # 2. Generation prompt
            system_text = (
                "You are a Senior Radiographic Consultant AI. You provide expert "
                "radiographic signatures and clinical management plans grounded in "
                "guidelines. You verify every claim against the specific vision data "
                "and any retrieved guideline context provided. You never invent "
                "drug doses, brand names, or dosing schedules. If retrieved context "
                "does not cover the finding, you fall back to standard radiology "
                "without naming the gap."
            )

            user_prompt = f"""
[INTERNAL DATA - DO NOT REFERENCE BY NAME IN OUTPUT]
FINDINGS: {formatted_vision_data}
RELEVANT_GUIDELINES: {context if context else "(no relevant guideline excerpts retrieved)"}

### TASK
Provide a clinical interpretation of this chest X-ray.

1. PRIMARY FINDING: Focus on {primary_phrase} ({primary_pct:.1f}% confidence).
2. GUIDELINE ADHERENCE: If RELEVANT_GUIDELINES contains specific management for {primary_phrase}, prioritise it. Otherwise rely on standard radiology — do not pretend guidelines covered something they did not.
3. VISUAL GROUNDING: First, in one short sentence, state which anatomical region the attached heatmap highlights (e.g. "right mid-zone", "cardiac silhouette", "left costophrenic angle"). Then state whether that region matches where {primary_phrase} typically appears. If they disagree, say so explicitly and lower diagnostic confidence in the Patient Summary.
4. NO INVENTED DOSES: Do not include specific drug doses, brand names, or dosing schedules unless they appear verbatim in RELEVANT_GUIDELINES. Use general categories ("appropriate antimicrobial therapy", "loop diuretics") otherwise.
5. SCOPE: You recommend, you do not commit to treatment. Use "recommended", "consider", "indicated" — never "we will initiate", "I will treat", or "the patient will receive". Treatment decisions belong to the clinical team.
6. URGENCY: For findings where delay causes harm — Pneumothorax, acute Pulmonary Edema, large Mass, large Effusion, suspected Tension Pneumothorax — use directive language ("immediate", "urgent", "requires prompt evaluation") rather than educational language ("typically involves", "is generally managed with").
7. TONE: Formal, consultant-level. Clear medical language. No mention of "raw data", "guidelines", "context", or "internal data".

STRUCTURE YOUR RESPONSE AS:
**Radiographic Signature**: [Visual cues for {primary_phrase}, including the heatmap-region statement from rule 3]
**Clinical Management**: [Expert next steps or diagnostic follow-up]
**Patient Summary**: [Clear, empathetic explanation of the {primary_phrase} finding]
"""

            # 3. Stream from LLM
            logger.info(f"📤 Starting multimodal stream for {llm.model}")
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": overlay_b64}},
                ]),
            ]

            async for chunk in llm.astream(messages):
                content = chunk.content
                if isinstance(content, list):
                    text = "".join(p.get('text', '') for p in content if not p.get('thought'))
                else:
                    text = content

                if text:
                    yield json.dumps({"report": text}, ensure_ascii=False) + "\n"

            # 4. Yield only sources whose content actually informed the report
            cited_sources = []
            for doc in relevant_docs:
                src_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', '?')
                cited_sources.append(f"{src_filename} (Page {page})")
            cited_sources = list(dict.fromkeys(cited_sources))

            yield json.dumps({"sources": cited_sources}, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield json.dumps({"report": f"\nError during generation: {str(e)}"}) + "\n"

    return StreamingResponse(report_generator(), media_type="application/x-ndjson")

# Include router at root and /api
app.include_router(router)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
