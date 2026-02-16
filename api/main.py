# api/main.py
import os
import io
from pathlib import Path

import numpy as np
import cv2
import tifffile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from preprocessing.resize_pad import resize_and_pad
from localization.localize import localize_box
from segmentation.sam_wrapper import SAMSegmenter
from measurement.geometry import measure



def decode_upload(filename: str, file_bytes: bytes) -> np.ndarray:
    ext = os.path.splitext(filename.lower())[1]

    # TIFF / TIF
    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(io.BytesIO(file_bytes))
        # ensure 2D grayscale
        if img.ndim > 2:
            img = img[..., 0]
        # normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = img - img.min()
            img = (img / (img.max() + 1e-8) * 255.0).astype(np.uint8)
        return img

    # PNG/JPG fallback via OpenCV
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image: {filename}")
    if img.ndim == 3:
        img = img.mean(axis=2).astype(np.uint8)
    return img


app = FastAPI()

# ---- Serve frontend (optional) ----
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def ui_root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---- Load model once ----
# in cloud, encoder and decoder are included via download and special push, just to ensure they exist.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (adjust if needed)

ENC = ROOT / "sam" / "sam_vit_b_01ec64.pth"
DEC = ROOT / "model_registry" / "livecell_sam_vit_b_boxprompt" / "20260201_173652" / "mask_decoder.pt"

if not ENC.exists():
    raise RuntimeError(f"Missing encoder_ckpt: {ENC}")
if not DEC.exists():
    raise RuntimeError(f"Missing decoder_ckpt: {DEC}")

segmenter = SAMSegmenter(encoder_ckpt=str(ENC), decoder_ckpt=str(DEC))

# segmenter = SAMSegmenter(
#     encoder_ckpt="sam/sam_vit_b_01ec64.pth",
#     decoder_ckpt="model_registry/livecell_sam_vit_b_boxprompt/20260201_173652/mask_decoder.pt",
# )


@app.post("/measure")
async def measure_sem_image(file: UploadFile = File(...)):
    img_bytes = await file.read()

    try:
        image = decode_upload(file.filename, img_bytes)  # uint8 2D
    except Exception as e:
        return {"error": f"Decode failed: {type(e).__name__}: {e}"}

    # Preprocess expects an image array (not raw bytes)
    image_1024, meta = resize_and_pad(image)

    # --- debug / guards ---
    if not isinstance(image_1024, np.ndarray):
        return {
            "error": "resize_and_pad did not return np.ndarray",
            "type": str(type(image_1024)),
        }

    if image_1024.dtype == object:
        return {
            "error": "image_1024 has dtype=object (not numeric)",
            "shape": getattr(image_1024, "shape", None),
        }

    # If image is 3-channel, convert to grayscale
    if image_1024.ndim == 3:
        image_1024 = cv2.cvtColor(image_1024, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 (Canny expects 8-bit single-channel in the common case)
    if image_1024.dtype != np.uint8:
        image_1024 = cv2.normalize(image_1024, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ensure contiguous memory
    image_1024 = np.ascontiguousarray(image_1024)
    # Localize box on the preprocessed image
    box_1024 = localize_box(image_1024)

    # SAM segmentation using box prompt
    pred_mask = segmenter.segment_with_box(image_1024, box_1024)

    # Geometry measurement
    measurements = measure(pred_mask, pixel_nm=1.0)
    return to_jsonable({
        "box_xyxy": box_1024,
        "measurements": measurements,
        "preprocess_meta": meta,
    })

# JSON serialization. FastAPI can’t serialize NumPy scalar types like np.int64, np.float32, and also struggles with arrays unless they’re converted to native Python types.
def to_jsonable(x):
    # numpy scalar -> python scalar
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # dict -> recurse
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}

    # list/tuple -> recurse
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # fallback (already jsonable: str/int/float/bool/None)
    return x