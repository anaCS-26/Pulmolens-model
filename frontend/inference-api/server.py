import io, os, uuid, threading, pathlib, logging
from typing import Dict, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# -----------------------------------------------------------
# App & CORS
# -----------------------------------------------------------
app = FastAPI()
SWA_ORIGIN = os.getenv("SWA_ORIGIN")
ALLOWED = [SWA_ORIGIN] if SWA_ORIGIN else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_cpu.pt")
MODEL_URL = os.getenv("MODEL_URL", "")  # optional one-time downloader
MODEL_ARCH = os.getenv("MODEL_ARCH", "densenet121").lower()  # <- set to what you trained
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "14"))
CLASS_NAMES = [c.strip() for c in os.getenv("CLASS_NAMES", "").split(",") if c.strip()]
if not CLASS_NAMES:
    # Provide meaningful defaults if not set (typical 14-class CXR set)
    CLASS_NAMES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    ][:NUM_CLASSES]

# -----------------------------------------------------------
# Optional model download
# -----------------------------------------------------------
if MODEL_URL and not os.path.exists(MODEL_PATH):
    import requests
    pathlib.Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)
    r = requests.get(MODEL_URL, timeout=120)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference-api")

# -----------------------------------------------------------
# Model utils
# -----------------------------------------------------------
device = "cpu"
_model = None
_lock = threading.Lock()


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "densenet121":
        from torchvision.models import densenet121
        m = densenet121(weights=None)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, num_classes)
        return m
    elif arch == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported MODEL_ARCH={arch}. Add it in build_model().")


def _is_torchscript(path: str) -> bool:
    # Heuristic: allow explicit hint via env; otherwise try jit-load safely
    if os.getenv("MODEL_IS_JIT", "0") == "1":
        return True
    # If it's not a file yet, just say False (download may be pending)
    if not os.path.exists(path):
        return False
    try:
        torch.jit.load(path, map_location=device)
        return True
    except Exception:
        return False


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats, or raise."""
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()):
        sd = obj  # looks like a raw state_dict
    else:
        # try nested under common keys
        for key in ("model", "net", "module"):
            if key in obj:
                inner = obj[key]
                if hasattr(inner, "state_dict"):
                    sd = inner.state_dict()
                elif isinstance(inner, dict):
                    sd = inner
                else:
                    continue
                break
        else:
            raise ValueError("Unrecognized checkpoint format; expected TorchScript or state_dict-like dict.")

    # strip DistributedDataParallel prefixes
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    return sd


def _load_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model

    if _is_torchscript(MODEL_PATH):
        log.info("Loading TorchScript model from %s", MODEL_PATH)
        m = torch.jit.load(MODEL_PATH, map_location=device)
        m.eval()
        _model = m
        return _model

    # Otherwise, assume a (PyTorch) checkpoint with state_dict
    log.info("Loading state_dict checkpoint from %s using arch=%s", MODEL_PATH, MODEL_ARCH)
    obj = torch.load(MODEL_PATH, map_location=device)
    sd = _extract_state_dict(obj)

    m = build_model(MODEL_ARCH, NUM_CLASSES)

    # Load with STRICT behavior; if keys mismatch, fail loudly.
    msg = m.load_state_dict(sd, strict=False)
    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}; "
            f"set MODEL_ARCH correctly or export a TorchScript that matches your training architecture."
        )

    m.eval()
    _model = m
    return _model

# -----------------------------------------------------------
# Preprocess
# -----------------------------------------------------------
_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Lambda(lambda img: img.convert("RGB")),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------
# In-memory result store
# -----------------------------------------------------------
RESULTS: Dict[str, Dict] = {}

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/debug/origin")
def debug_origin():
    return {"allowed": ALLOWED}


@app.get("/classes")
def classes():
    return {"numClasses": NUM_CLASSES, "classNames": CLASS_NAMES}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Please upload a PNG or JPEG image.")
    img_bytes = await file.read()  # read ONCE
    inference_id = str(uuid.uuid4())
    with _lock:
        RESULTS[inference_id] = {"status": "running"}
    # Save a temp copy (useful for debugging)
    tmp_dir = pathlib.Path("/tmp/uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{inference_id}.img"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    log.info("Saved upload %s -> %s (%d bytes)", inference_id, tmp_path, tmp_path.stat().st_size)

    threading.Thread(target=_run_inference, args=(inference_id, img_bytes), daemon=True).start()
    return {"inferenceId": inference_id}


@app.get("/result/{inference_id}")
def get_result(inference_id: str):
    with _lock:
        if inference_id not in RESULTS:
            raise HTTPException(status_code=404, detail="Unknown inference id")
        return RESULTS[inference_id]


# -----------------------------------------------------------
# Inference worker
# -----------------------------------------------------------
def _run_inference(inference_id: str, img_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes))
        x = _preprocess(image).unsqueeze(0)
        model = _load_model()
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits).squeeze(0)
            if probs.ndim != 1 or probs.numel() != NUM_CLASSES:
                raise RuntimeError(
                    f"Model output shape {tuple(probs.shape)} != NUM_CLASSES ({NUM_CLASSES}). Check MODEL_ARCH/weights."
                )
            probs = probs.tolist()

        preds = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
        top3 = sorted(preds.items(), key=lambda kv: -kv[1])[:3]
        log.info("Inference %s top3: %s", inference_id, top3)

        with _lock:
            RESULTS[inference_id] = {"status": "succeeded", "predictions": preds}

    except Exception as e:
        log.exception("Inference %s failed", inference_id)
        with _lock:
            RESULTS[inference_id] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
