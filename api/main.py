# api/main.py

# FastAPI is the web framework that exposes your functions as HTTP APIs
from fastapi import FastAPI, UploadFile, File

# NumPy for array handling
import numpy as np

# OpenCV for image decoding
import cv2

import tifffile

def decode_upload(filename: str, file_bytes: bytes) -> np.ndarray:
    ext = os.path.splitext(filename.lower())[1]
    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(file_bytes)  # robust TIFF decode
        # ensure 2D
        if img.ndim > 2:
            img = img[..., 0]
        # normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = img - img.min()
            img = (img / (img.max() + 1e-8) * 255.0).astype(np.uint8)
        return img

    # fallback for png/jpg
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image: {filename}")
    if img.ndim == 3:
        img = img.mean(axis=2).astype(np.uint8)
    return img

# Your own modules (already implemented elsewhere)
from preprocessing.resize_pad import resize_and_pad
from localization.localize import localize_box
from segmentation.sam_wrapper import SAMSegmenter
from measurement.geometry import measure


# -------------------------
# Create API app instance
# -------------------------
app = FastAPI()


# -------------------------
# Load model ONCE at startup
# -------------------------
# This avoids reloading the model for every request
segmenter = SAMSegmenter(
    encoder_ckpt="sam/sam_vit_b_01ec64.pth",
    decoder_ckpt="model_registry/livecell_sam_vit_b_boxprompt/20260123_220829/mask_decoder.pt"
)

# -------------------------
# Define inference endpoint
# -------------------------
@app.post("/measure")
async def measure_sem_image(file: UploadFile = File(...)):
    """
    API endpoint:
    - Accepts an uploaded SEM image
    - Runs preprocess → localization → SAM → measurement
    - Returns geometry results as JSON
    """

    # Read raw file bytes from the request
    img_bytes = await file.read()
    image = decode_upload(file.filename, img_bytes)

    # Convert bytes to NumPy buffer
    np_buf = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode image into grayscale array
    gray = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)

    # Safety check: ensure decoding worked
    if gray is None:
        return {"error": "Invalid image file or unsupported format."}

    # Preprocess image (your implementation)
    image_1024 = resize_and_pad(np_buf)

    # Localize bounding box (your implementation)
    box_1024 = localize_box(image_1024)

    # Run SAM with box prompt
    pred_mask = segmenter.segment_with_box(image_1024, box_1024)

    # Compute measurements from predicted mask
    measurements = measure(pred_mask, pixel_nm=1.0)

    # Return structured JSON response
    return {
        "box_xyxy": box_1024,
        "measurements": measurements
    }
