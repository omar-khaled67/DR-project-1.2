# """
# RetinoGrade API: the ONLY place where the models are loaded and run.

#     uvicorn api_server:app --host 127.0.0.1 --port 8000

# Endpoints
#     GET  /health                       -> {"status": "ok", ...}
#     POST /analyse   (file)             -> quality gate + grade + probabilities + flags + review notice
#     POST /heatmap   (file, ?target=)   -> Grad-CAM images (base64 PNG); target = referable | pred
# Decision support for screening research, not a diagnostic device.
# """
# import base64
# import io
# import os
# import sys

# import cv2
# import numpy as np
# from fastapi import FastAPI, File, HTTPException, Query, UploadFile
# from PIL import Image

# ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(ROOT, "src"))

# import retinograde_inference as R  # noqa: E402
# import review  # noqa: E402
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


# def png_b64(rgb: np.ndarray) -> str:
#     ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
#     return base64.b64encode(buf.tobytes()).decode()


# @app.get("/health")
# def health():
#     return {"status": "ok", "model": R.MODEL_NAME, "quality_gate": pipe.gate is not None}


# @app.post("/analyse")
# def analyse(file: UploadFile = File(...)):
#     img = read_image(file)
#     result = pipe.predict(img)
#     q = result["quality"]
#     result["review"] = review.assess(result["probabilities"], q["label"] if q else None)
#     result["image_size"] = [int(img.shape[1]), int(img.shape[0])]
#     return clean(result)


# @app.post("/heatmap")
# def heatmap(file: UploadFile = File(...), target: str = Query("referable")):
#     if target not in ("referable", "pred"):
#         raise HTTPException(400, "target must be 'referable' or 'pred'.")
#     img = read_image(file)
#     x = pipe.grader.prepare(img)
#     h, _, used = G.gradcam(pipe.grader.model, x, target)
#     return {
#         "target": used,
#         "empty": bool(h.max() == 0),
#         "prepared_png": png_b64(np.clip(x, 0, 255).astype(np.uint8)),
#         "overlay_png": png_b64(G.overlay(x, h)),
#     }
