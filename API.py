
# from fastapi import FastAPI, UploadFile, File, HTTPException
# import tensorflow as tf
# import numpy as np
# import cv2

# from PREPOCESSING import preprocess_rgb

# app = FastAPI(
#     title="RetinoGrade AI API",
#     description="Diabetic Retinopathy Grading API",
#     version="1.0"
# )

# # DR class labels
# CLASS_NAMES = [
#     "No DR",
#     "Mild",
#     "Moderate",
#     "Severe",
#     "Proliferative DR"
# ]

# # Load the existing trained model (unchanged)
# model = tf.keras.models.load_model(
#     "models/B_full_s42_384_mix_model.keras",
#     compile=False
# )


# @app.get("/")
# def home():
#     return {
#         "message": "RetinoGrade AI API is running!"
#     }


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):

#     # Read uploaded image
#     image_bytes = await file.read()

#     # Decode image
#     img = cv2.imdecode(
#         np.frombuffer(image_bytes, np.uint8),
#         cv2.IMREAD_COLOR
#     )

#     if img is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid image file"
#         )

#     # Convert BGR to RGB
#     img = cv2.cvtColor(
#         img,
#         cv2.COLOR_BGR2RGB
#     )

#     # Preprocess image (384x384)
#     processed = preprocess_rgb(
#         img,
#         target_size=384
#     )

#     # Convert pixel range from 0-1 to 0-255
#     processed = processed * 255.0

#     # Add batch dimension
#     processed = np.expand_dims(
#         processed,
#         axis=0
#     )

#     # Model prediction
#     predictions = model.predict(
#         processed,
#         verbose=0
#     )[0]

#     # Get predicted class
#     predicted_class = int(np.argmax(predictions))

#     # Get confidence
#     confidence = float(predictions[predicted_class])

#     # Return results
#     return {
#         "filename": file.filename,
#         "predicted_class": predicted_class,
#         "predicted_grade": CLASS_NAMES[predicted_class],
#         "confidence": round(confidence * 100, 2),
#         "probabilities": {
#             CLASS_NAMES[i]: round(float(prob) * 100, 2)
#             for i, prob in enumerate(predictions)
#         }
#     }

import json
import base64
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from PREPOCESSING import preprocess_rgb
from src.explain.gradcam import gradcam, overlay


app = FastAPI(
    title="RetinoGrade AI API",
    description="Diabetic Retinopathy Grading API",
    version="2.0"
)


# -----------------------------
# Configuration
# -----------------------------

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

QUALITY_CLASS_NAMES = [
    "Good",
    "Usable",
    "Reject"
]

QUALITY_MESSAGES = {
    "Good": "Image quality is good.",
    "Usable": "Image quality is limited. The result needs a human review.",
    "Reject": "Image quality is too low to grade. Please take a new photo."
}

MIN_SIDE = 64

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Load DR model
# -----------------------------

model = tf.keras.models.load_model(
    BASE_DIR / "models" / "B_full_s42_384_mix_model.keras",
    compile=False
)


# -----------------------------
# Load Quality model
# -----------------------------

quality_path = BASE_DIR / "models" / "quality_b0_224.keras"

quality_model = keras.models.load_model(
    quality_path,
    compile=False
)




# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "RetinoGrade AI API is running!"
    }


# -----------------------------
# Prediction
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Convert BGR to RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    height, width = img.shape[:2]

    # -----------------------------
    # Minimum image size check
    # -----------------------------

    if min(height, width) < MIN_SIDE:
        return {
            "filename": file.filename,
            "quality": {
                "class": "Reject",
                "message": "Image is too small. Please take a new photo."
            },
            "dr": None,
            "gradcam": None
        }

    # -----------------------------
    # Quality prediction
    # -----------------------------

    quality_img = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    quality_input = np.expand_dims(
        quality_img.astype(np.float32),
        axis=0
    )

    quality_predictions = quality_model.predict(
        quality_input,
        verbose=0
    )[0]

    quality_class = int(
        np.argmax(quality_predictions)
    )

    quality_label = QUALITY_CLASS_NAMES[quality_class]

    quality_confidence = float(
        quality_predictions[quality_class]
    )

    quality_result = {
        "class": quality_label,
        "confidence": round(quality_confidence * 100, 2),
        "message": QUALITY_MESSAGES[quality_label],
        "probabilities": {
            QUALITY_CLASS_NAMES[i]: round(
                float(prob) * 100, 2
            )
            for i, prob in enumerate(quality_predictions)
        }
    }

    # -----------------------------
    # Reject poor-quality images
    # -----------------------------

    if quality_label == "Reject":
        return {
            "filename": file.filename,
            "quality": quality_result,
            "dr": None,
            "gradcam": None
        }

    # -----------------------------
    # DR preprocessing
    # -----------------------------

    processed = preprocess_rgb(
        img,
        target_size=384
    )

    processed = processed * 255.0

    # -----------------------------
    # DR prediction
    # -----------------------------

    predictions = model.predict(
        np.expand_dims(processed, axis=0),
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_class]
    )

    dr_result = {
        "predicted_class": predicted_class,
        "predicted_grade": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(
                float(prob) * 100, 2
            )
            for i, prob in enumerate(predictions)
        }
    }

    # -----------------------------
    # Grad-CAM generation
    # -----------------------------

    heatmap, gradcam_probs, target = gradcam(
        model,
        processed,
        target=predicted_class
    )

    gradcam_image = overlay(
        processed,
        heatmap
    )

    # Convert RGB overlay to PNG
    success, encoded_image = cv2.imencode(
        ".png",
        cv2.cvtColor(
            gradcam_image,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not generate Grad-CAM image."
        )

    gradcam_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    gradcam_result = {
        "target_class": int(target),
        "target_grade": CLASS_NAMES[int(target)],
        "image": "data:image/png;base64," + gradcam_base64
    }

    # -----------------------------
    # Final response
    # -----------------------------

    return {
        "filename": file.filename,
        "quality": quality_result,
        "dr": dr_result,
        "gradcam": gradcam_result
    }


# """
# RetinoGrade API: the ONLY place where the models are loaded and run.

#     uvicorn API:app --host 127.0.0.1 --port 8000

# Endpoints
#     GET  /health                       -> {"status": "ok", ...}
#     POST /analyse   (file)             -> quality gate + grade + probabilities + flags + review notice
#     POST /heatmap   (file, ?target=)   -> Grad-CAM arrays (base64 .npy); target = referable | pred
# Decision support for screening research, not a diagnostic device.
# """
# import base64
# import io
# import os
# import sys

# import numpy as np
# from fastapi import FastAPI, File, HTTPException, Query, UploadFile
# from PIL import Image

# ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(ROOT, "src"))

# import retinograde_inference as R  # noqa: E402
# from explain import gradcam as G  # noqa: E402
# from pipeline import RetinoPipeline  # noqa: E402

# MAX_MB = 25
# MAX_MEGAPIXELS = 60

# app = FastAPI(title="RetinoGrade API", version="2.0")
# pipe = RetinoPipeline()          # models are loaded once, when the server starts


# def clean(o):
#     """Make numpy values JSON-safe."""
#     if isinstance(o, dict):
#         return {k: clean(v) for k, v in o.items()}
#     if isinstance(o, (list, tuple)):
#         return [clean(v) for v in o]
#     if isinstance(o, np.generic):
#         return o.item()
#     if isinstance(o, np.ndarray):
#         return o.tolist()
#     return o


# def read_image(file: UploadFile) -> np.ndarray:
#     data = file.file.read()
#     if not data:
#         raise HTTPException(400, "The file is empty.")
#     if len(data) > MAX_MB * 1024 * 1024:
#         raise HTTPException(413, f"The file is larger than {MAX_MB} MB.")
#     try:
#         w, h = Image.open(io.BytesIO(data)).size          # reads only the header
#     except Image.DecompressionBombError:
#         raise HTTPException(400, f"It has too many pixels; the limit is {MAX_MEGAPIXELS} megapixels.")
#     except Exception:
#         raise HTTPException(400, "The file is not a valid image.")
#     if w * h > MAX_MEGAPIXELS * 1e6:
#         raise HTTPException(400, f"It has too many pixels ({w}x{h}); the limit is {MAX_MEGAPIXELS} megapixels.")
#     try:
#         return R.load_rgb(data)
#     except Exception:
#         raise HTTPException(400, "The file could not be read as an image.")


# def npy_b64(a: np.ndarray) -> str:
#     buf = io.BytesIO()
#     np.save(buf, np.asarray(a, np.float32), allow_pickle=False)
#     return base64.b64encode(buf.getvalue()).decode()


# @app.get("/health")
# def health():
#     return {"status": "ok", "model": R.MODEL_NAME, "quality_gate": pipe.gate is not None}


# @app.post("/analyse")
# def analyse(file: UploadFile = File(...)):
#     img = read_image(file)
#     return clean(pipe.predict(img))


# @app.post("/heatmap")
# def heatmap(file: UploadFile = File(...), target: str = Query("referable")):
#     if target not in ("referable", "pred"):
#         raise HTTPException(400, "target must be 'referable' or 'pred'.")
#     img = read_image(file)
#     x = pipe.grader.prepare(img)
#     h, _, used = G.gradcam(pipe.grader.model, x, target)
#     return {
#         "target": used,
#         "prepared_npy": npy_b64(x),      # the image the model sees (0..255)
#         "heatmap_npy": npy_b64(h),       # Grad-CAM map in [0, 1]
#     }