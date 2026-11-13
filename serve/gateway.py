import io
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO


# Initialize FastAPI
app = FastAPI(title="VisionPort Gateway (Session 2 prototype)")


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load YOLO model
model = YOLO("yolov8n.pt")


# Home endpoint
@app.get("/")
def home():
    return {
        "project": "VisionPort",
        "message": "VisionPort FastAPI Gateway is running",
        "status": "success",
        "device": device,
        "available_endpoints": [
            "/detect",
            "/docs"
        ]
    }


# Object detection endpoint
@app.post("/detect")
def detect(file: UploadFile = File(...), conf: float = 0.35):

    # Validate image format
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image."
        )

    # Read image
    try:
        image = Image.open(
            io.BytesIO(file.file.read())
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image."
        )

    # Run YOLO inference
    r = model.predict(
        image,
        conf=conf,
        device=device,
        verbose=False
    )[0]

    # Extract detections
    detections = []

    if r.boxes is not None:
        for box in r.boxes:

            c = int(box.cls[0])
            s = float(box.conf[0])
            b = box.xyxy[0].tolist()

            detections.append({
                "class": r.names[c],
                "confidence": round(s, 3),
                "box_xyxy": [
                    round(float(v), 1) for v in b
                ]
            })

    # Return response
    return {
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "num_detections": len(detections),
        "detections": detections,
        "timing_ms": {
            k: round(v, 2) for k, v in r.speed.items()
        }
    }