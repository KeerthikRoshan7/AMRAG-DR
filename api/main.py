"""
FastAPI backend for AM-RAG.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /analyze       -- upload a fundus image (+ optional patient metadata),
                            returns the full explainable diagnostic report.
    GET  /health         -- basic health check.
"""

import os
import io
import base64
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

from agents.orchestrator import AMRAGOrchestrator
from utils.gradcam import overlay_gradcam

load_dotenv()

CHECKPOINT_PATH = os.environ.get("LESION_CHECKPOINT_PATH", "checkpoints/lesion_detector.pt")
DEVICE = os.environ.get("DEVICE", "cpu")

app = FastAPI(title="AM-RAG: Diabetic Retinopathy Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup -- FAISS index, embedder, and lesion detector all
# stay resident in memory across requests.
orchestrator: AMRAGOrchestrator | None = None


@app.on_event("startup")
def load_pipeline():
    global orchestrator
    checkpoint = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
    orchestrator = AMRAGOrchestrator(checkpoint_path=checkpoint, device=DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_loaded": orchestrator is not None}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    patient_metadata: str | None = Form(default=None),
):
    """
    patient_metadata: optional JSON string, e.g.
        '{"age": 58, "diabetes_type": "Type 2", "diabetes_duration_years": 12, "hba1c": 8.1}'
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    metadata = None
    if patient_metadata:
        try:
            metadata = json.loads(patient_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="patient_metadata must be valid JSON")

    report, gradcam_map = orchestrator.run(pil_image, patient_metadata=metadata)

    gradcam_b64 = None
    if gradcam_map is not None:
        overlay_img = overlay_gradcam(pil_image, gradcam_map)
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    report["gradcam_overlay_png_base64"] = gradcam_b64
    return report
