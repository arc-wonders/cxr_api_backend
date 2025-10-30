import os, io, time, torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torchvision.transforms as T
import torchvision as tv

APP_NAME = "Pneumonia CXR API (Research Demo)"
VERSION = "1.1.0"

MODEL_PATH = os.getenv("MODEL_PATH", "/app/runs_resnet50/resnet50_pneumonia_traced.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.20"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

app = FastAPI(title=APP_NAME, version=VERSION, description="Not for clinical use")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class PredictResponse(BaseModel):
    probability: float
    label: int
    threshold: float
    latency_ms: float
    device: str
    load_mode: str  # "torchscript" or "state_dict"

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_resnet50_binary():
    # ImageNet weights not required for serving; keep light
    model = tv.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model

def _try_load_torchscript(path, device):
    # Returns (model, "torchscript") or raises
    m = torch.jit.load(path, map_location=device)
    m.eval().to(device)
    return m, "torchscript"

def _try_load_state_dict(path, device):
    # Accepts either a raw state_dict, or a checkpoint dict with 'model'
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], (dict, torch.nn.modules.module.Module)):
        sd = ckpt["model"] if isinstance(ckpt["model"], dict) else ckpt["model"].state_dict()
    elif isinstance(ckpt, dict):
        # Could already be a state_dict
        sd = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format for state_dict loading.")

    model = _build_resnet50_binary()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    model.eval().to(device)
    return model, "state_dict"

def load_model():
    device = _device()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # First try TorchScript
    try:
        return _try_load_torchscript(MODEL_PATH, device)
    except Exception as e_ts:
        print(f"[info] TorchScript load failed: {e_ts}. Trying state_dict...")
        # Fallback: state_dict / checkpoint
        try:
            return _try_load_state_dict(MODEL_PATH, device)
        except Exception as e_sd:
            raise RuntimeError(f"Failed to load model as TorchScript or state_dict. "
                               f"Errors:\nTS: {e_ts}\nSD: {e_sd}")

MODEL, DEVICE_MODE = load_model()
DEVICE = _device()

TFM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@app.get("/healthz")
def health():
    return {"status": "ok", "device": DEVICE.type, "load_mode": DEVICE_MODE}

@app.get("/version")
def version():
    return {
        "app": APP_NAME, "version": VERSION,
        "model_path": MODEL_PATH, "img_size": IMG_SIZE,
        "default_threshold": THRESHOLD,
        "device": DEVICE.type, "torch": torch.__version__,
        "load_mode": DEVICE_MODE,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), threshold: float | None = None):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    th = float(threshold) if threshold is not None else THRESHOLD
    x = TFM(img).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
        logits = MODEL(x).squeeze(1)
        prob = torch.sigmoid(logits).item()
    dt = (time.perf_counter() - t0) * 1000.0
    label = int(prob >= th)

    return PredictResponse(
        probability=prob, label=label, threshold=th,
        latency_ms=dt, device=DEVICE.type, load_mode=DEVICE_MODE
    )
