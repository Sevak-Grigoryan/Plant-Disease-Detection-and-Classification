import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModel

HF_REPO_ID = os.getenv("HF_REPO_ID", "SevakGrigoryan/dinov2-large-plant-disease")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"

class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Human-readable disease class name")
    class_index: int = Field(..., description="Integer class index used by the model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")

    model_config = {
        "json_schema_extra": {
            "example": {"class_name": "early_blight", "class_index": 12, "confidence": 0.93}
        }
    }

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Top-1 predicted class name")
    predicted_index: int = Field(..., description="Top-1 predicted class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 confidence")
    top_k: list[PredictionItem] = Field(..., description="Top-k ranked predictions")

class HealthResponse(BaseModel):
    status: str = Field(..., description='"ok" if model is loaded, "degraded" otherwise')
    model_loaded: bool
    device: str = Field(..., description='"cuda" or "cpu"')
    repo_id: str = Field(..., description="Hugging Face repo the model was loaded from")
    classes_count: int

tags_metadata = [
    {"name": "ui", "description": "Browser UI."},
    {"name": "system", "description": "Health and status endpoints."},
    {"name": "inference", "description": "Image classification."},
]

app = FastAPI(
    title="Plant Disease Classifier API",
    description=(
        "DINOv2-Large + LoRA fine-tuned to classify 40 plant diseases.\n\n"
        "Model is loaded at startup from the Hugging Face Hub repo "
        f"`{HF_REPO_ID}`.\n\n"
        "Use **POST /api/predict** with a multipart image to get top-k predictions."
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
    contact={"name": "Sevak Grigoryan"},
    license_info={"name": "Apache-2.0"},
)

model = None
processor = None
class_names: list[str] = []

def load_model_from_hub() -> None:
    global model, processor, class_names

    print(f"Loading model from HF Hub: {HF_REPO_ID}")
    processor_local = AutoImageProcessor.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
    model_local = AutoModel.from_pretrained(
        HF_REPO_ID, trust_remote_code=True, token=HF_TOKEN
    ).to(DEVICE).eval()

    id2label: dict[int, str] = model_local.config.id2label
    classes = [id2label[i] for i in sorted(id2label.keys(), key=lambda x: int(x))]

    processor = processor_local
    model = model_local
    class_names = classes
    print(f"Model loaded on {DEVICE} with {len(class_names)} classes")

@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model_from_hub()
    except Exception as exc:
        print(f"Startup model load failed: {exc}")

def predict_image(image: Image.Image, top_k: int) -> PredictionResponse:
    if model is None or processor is None:
        raise RuntimeError("Model is not loaded")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        output: Any = model(pixel_values=pixel_values)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        probs = torch.softmax(logits, dim=1)[0]

    k = min(top_k, len(class_names))
    confs, indices = torch.topk(probs, k=k)

    top_items = [
        PredictionItem(
            class_name=class_names[int(idx)],
            class_index=int(idx),
            confidence=float(conf),
        )
        for conf, idx in zip(confs.tolist(), indices.tolist())
    ]

    best = top_items[0]
    return PredictionResponse(
        predicted_class=best.class_name,
        predicted_index=best.class_index,
        confidence=best.confidence,
        top_k=top_items,
    )

@app.get("/", response_class=HTMLResponse, tags=["ui"], include_in_schema=False)
def home() -> HTMLResponse:
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Plant Disease API</h1><p>See <a href='/docs'>/docs</a>.</p>")

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Service health and model status",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        device=str(DEVICE),
        repo_id=HF_REPO_ID,
        classes_count=len(class_names),
    )

@app.get(
    "/api/classes",
    tags=["system"],
    summary="List all class names",
    response_model=dict,
)
def classes_endpoint() -> dict:
    return {"count": len(class_names), "classes": class_names}

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Classify a leaf image",
    description=(
        "Upload a single image (JPEG/PNG) and receive the top-k predicted "
        "plant-disease classes with confidence scores."
    ),
    responses={
        200: {"description": "Predictions returned successfully"},
        400: {"description": "Uploaded file is not a valid image"},
        503: {"description": "Model is not loaded — check /health"},
    },
)
async def api_predict(
    file: UploadFile = File(..., description="Leaf image, JPEG or PNG"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions to return"),
) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    return predict_image(image=image, top_k=top_k)

@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
