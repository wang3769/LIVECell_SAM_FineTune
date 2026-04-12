# LIVECell_SAM_FineTune Architecture

## Overview

This project implements a **SAM (Segment Anything Model) based measurement system** for SEM (Scanning Electron Microscopy) images. It fine-tunes the SAM mask decoder on the LIVECell dataset to perform precise cell/object segmentation and measurement in electron microscopy images.

The system receives SEM images with bounding box prompts and returns precise measurements (area, width, height) in physical units (nanometers).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LIVECell_SAM_FineTune System                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Layer (FastAPI)                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────┐   │
│  │  /measure POST  │  │  / (static UI)   │  │   decode_upload()              │   │
│  │  - file upload  │  │  - Web frontend  │  │   - TIFF/PNG/JPG decoding      │   │
│  │  - pixel_nm     │  │  - Box drawing   │  │   - Grayscale normalization    │   │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Preprocessing Module                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │                         resize_and_pad()                                   │    │
│  │  - Input: Variable resolution SEM image                                    │    │
│  │  - Output: 1024×1024 padded image + transformation metadata               │    │
│  │  - Maintains scale factor and padding info for inverse transform          │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │                         localize_box()                                     │    │
│  │  - Canny edge detection to find prominent features                         │    │
│  │  - Returns box coordinates [x1, y1, x2, y2] in image space                 │    │
│  │  - Used as SAM prompt when no user box provided                           │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SAM Segmentation Module                             │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │                       SAMSegmenter Class                                   │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐    │    │
│  │  │  Image Encoder  │  │ Prompt Encoder │  │  Mask Decoder (Fine-     │    │    │
│  │  │  (Frozen/Fixed) │  │  (Box → Embed)  │  │  Tuned with LoRA)       │    │    │
│  │  │  ViT-Base       │  │                 │  │                         │    │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘    │    │
│  │         │                    │                        │                  │    │
│  │         └────────────────────┼────────────────────────┘                  │    │
│  │                              ▼                                           │    │
│  │              ┌───────────────────────────────┐                           │    │
│  │              │    Prediction: Binary Mask   │                           │    │
│  │              └───────────────────────────────┘                           │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  Model Checkpoints:                                                               │
│  - Encoder: sam/sam_vit_b_01ec64.pth                                             │
│  - Decoder: model_registry/livecell_sam_vit_b_boxprompt/*/mask_decoder.pt        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Measurement Module                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │                          measure()                                         │    │
│  │  Input: Binary mask, pixel_nm (nm per pixel)                              │    │
│  │                                                                             │    │
│  │  Computes:                                                                  │    │
│  │  - area_nm2 = mask.sum() × pixel_nm²                                      │    │
│  │  - width_nm = (x_max - x_min) × pixel_nm                                  │    │
│  │  - height_nm = (y_max - y_min) × pixel_nm                                 │    │
│  │                                                                             │    │
│  │  Output: {"area_nm2": float, "width_nm": float, "height_nm": float}       │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Response Layer                                      │
│  ┌───────────────────────────────────────────────────────────────────────────┐    │
│  │                         to_jsonable()                                     │    │
│  │  - Converts numpy types to Python native types                            │    │
│  │  - Ensures JSON serializable output                                       │    │
│  └───────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  Response JSON:                                                                  │
│  {                                                                                │
│    "box_xyxy": [x1, y1, x2, y2],      // Box prompt used                         │
│    "measurements": {                    // Physical measurements                │
│      "area_nm2": float,                                                            │
│      "width_nm": float,                                                           │
│      "height_nm": float                                                           │
│    },                                                                             │
│    "preprocess_meta": {                  // For coordinate transformation         │
│      "scale": float,                                                               │
│      "pad_h": int,                                                                  │
│      "pad_w": int,                                                                  │
│      "orig_shape": [H, W]                                                         │
│    }                                                                              │
│  }                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
LIVECell_SAM_FineTune/
├── ARCHITECTURE.md              # This file
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
├── Jenkinsfile                  # CI/CD configuration
│
├── api/                         # FastAPI application
│   ├── main.py                  # API endpoints, model loading
│   └── io_utils.py              # (embedded in main.py)
│
├── preprocessing/               # Image preprocessing
│   └── resize_pad.py            # resize_and_pad() function
│
├── localization/                # Box detection
│   └── localize.py              # localize_box() using Canny
│
├── segmentation/                # SAM segmentation
│   └── sam_wrapper.py           # SAMSegmenter class
│
├── measurement/                 # Post-segmentation analysis
│   └── geometry.py              # measure() function
│
├── sam/                         # SAM model weights
│   └── sam_vit_b_01ec64.pth     # Base encoder checkpoint
│
├── model_registry/               # Trained model artifacts
│   └── livecell_sam_vit_b_boxprompt/
│       └── 20260201_173652/
│           └── mask_decoder.pt  # Fine-tuned decoder weights
│
├── training_WIP/                 # Training scripts (Work In Progress)
│   ├── train_sam.py              # Main training script
│   ├── prepare_data.py           # LIVECell dataset preparation
│   ├── eval_iou.py               # IoU evaluation with MLflow
│   └── mlflow_util.py            # MLflow logging utilities
│
├── frontend/                     # Web UI
│   └── index.html                # Interactive measurement interface
│
├── POC/                          # Proof of Concept notebooks
│   └── LoRA_encoder_fine_tune.ipynb
│
└── docker/                       # Docker configuration
```

## Data Flow

### Inference Pipeline

```
1. Image Upload
   └─> File bytes (TIFF/PNG/JPG)
       │
       ▼
2. Decode
   └─> numpy array (H×W grayscale uint8)
       │
       ▼
3. Preprocess
   └─> 1024×1024 padded image + metadata
       │
       ▼
4. Localize Box (or use provided)
   └─> Box coordinates [x1,y1,x2,y2]
       │
       ▼
5. SAM Segmentation
   └─> Binary mask (1024×1024)
       │
       ▼
6. Measurement
   └─> {area_nm2, width_nm, height_nm}
       │
       ▼
7. Response
   └─> JSON with measurements + metadata
```

### Training Pipeline

```
1. Dataset Preparation (LIVECell COCO format)
   └─> Images + Annotations
       │
       ▼
2. Model Initialization
   ├─> Base SAM encoder (frozen)
   └─> Base SAM decoder (LoRA fine-tuned)
       │
       ▼
3. Training Loop
   ├─> Forward pass with box prompts
   ├─> Compute IoU loss vs ground truth
   ├─> Backprop through LoRA adapters only
   └─> Log metrics to MLflow
       │
       ▼
4. Save Artifacts
   └─> mask_decoder.pt (LoRA weights only)
```

## Key Components

### 1. SAMSegmenter (`segmentation/sam_wrapper.py`)

```python
class SAMSegmenter:
    def __init__(self, encoder_ckpt, decoder_ckpt, model_type="vit_b", device=None):
        # Load base SAM with frozen encoder
        # Load fine-tuned decoder weights (LoRA)
        # Initialize SamPredictor
    
    def segment_with_box(self, image, box_xyxy):
        # Input: H×W×3 RGB image, box [x1,y1,x2,y2]
        # Output: Binary mask (H×W)
```

**Key Features:**
- Uses `segment_anything` library from Meta
- ViT-Base architecture for balance of speed/accuracy
- Only decoder is fine-tuned (encoder remains frozen)
- Supports LoRA adapter injection

### 2. resize_and_pad (`preprocessing/resize_pad.py`)

```python
def resize_and_pad(image, target=1024):
    # Scales image to fit within target×target
    # Pads with zeros to reach exact target size
    # Returns padded image + metadata dict
```

**Transformation Metadata:**
```python
{
    "scale": float,           # Scaling factor applied
    "pad_h": int,             # Padding added to height
    "pad_w": int,             # Padding added to width
    "orig_shape": (H, W)     # Original image dimensions
}
```

### 3. localize_box (`localization/localize.py`)

```python
def localize_box(image, box_size=100):
    # Canny edge detection
    # Find brightest edge pixel
    # Return centered bounding box
```

### 4. measure (`measurement/geometry.py`)

```python
def measure(mask, pixel_nm=1.0):
    # Find bounding box of mask pixels
    # Compute area and dimensions in physical units
    # Returns dict with nanometer measurements
```

## API Specification

### POST /measure

**Request:**
```
Content-Type: multipart/form-data

file: <image file> (.tif, .tiff, .png, .jpg, .jpeg)
box: "x1,y1,x2,y2" (optional, in original image coordinates)
pixel_nm: float (optional, default 1.0)
```

**Response:**
```json
{
  "box_xyxy": [100, 150, 300, 400],
  "measurements": {
    "area_nm2": 45000.0,
    "width_nm": 200.0,
    "height_nm": 250.0
  },
  "preprocess_meta": {
    "scale": 0.5,
    "pad_h": 0,
    "pad_w": 100,
    "orig_shape": [1000, 1200]
  }
}
```

### GET /

Serves the interactive web UI for manual measurements.

## Dependencies

```
fastapi==0.128.0           # Web framework
uvicorn==0.30.6           # ASGI server
python-multipart==0.0.9   # File upload support

numpy==2.2.6              # Numerical operations
opencv-python==4.10.0.84  # Image processing
pillow==10.4.0             # Image I/O
tifffile==2024.8.10        # TIFF file support

matplotlib==3.9.2         # Visualization

segment-anything          # Meta's SAM library (git+https)
```

## Model Weights

### Base Model (SAM ViT-B)
- **Source:** HuggingFace (Gourieff/ReActor models)
- **URL:** `sam_vit_b_01ec64.pth`
- **Purpose:** Pre-trained image encoder + default mask decoder

### Fine-tuned Decoder
- **Location:** `model_registry/livecell_sam_vit_b_boxprompt/*/mask_decoder.pt`
- **Training:** Fine-tuned on LIVECell dataset
- **Format:** PyTorch state dict (LoRA weights)

## Dataset

### LIVECell Dataset
- **Source:** https://sartorius-research.github.io/LIVECell/
- **Format:** COCO annotations
- **Content:** Fluorescence microscopy cell images
- **Adaptation:** Applied to SEM images for this project

## Coordinate Systems

The system operates in multiple coordinate spaces:

```
┌──────────────────────────────────────────────────────────────┐
│  Original Image Space                                         │
│  - User-provided box coordinates                              │
│  - Final measurement units                                   │
│  - Shape: variable (e.g., 1024×768)                          │
└──────────────────────────────────────────────────────────────┘
                            │
                      resize_and_pad
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Preprocessed Space (1024)                                    │
│  - SAM input coordinates                                     │
│  - Fixed 1024×1024 with padding                              │
│  - All internal computations                                 │
└──────────────────────────────────────────────────────────────┘
```

## Extensions & Future Work

### Training Improvements
- [ ] Experiment with ViT-Large encoder
- [ ] Add data augmentation during training
- [ ] Implement mixed precision training
- [ ] Add cross-validation for model selection

### Inference Optimizations
- [ ] ONNX export for faster inference
- [ ] TensorRT deployment
- [ ] Batch processing support

### Additional Features
- [ ] Multi-object segmentation
- [ ] Instance segmentation with tracking
- [ ] 3D reconstruction from multiple angles
- [ ] Uncertainty quantification

## Deployment

### Local Development
```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

### Docker
```bash
docker build -t sam-measurement .
docker run -p 8000:8000 sam-measurement
```

### Cloud Deployment
The system is designed for containerized deployment with:
- Azure Blob Storage for model persistence
- MongoDB for result logging
- MLflow for experiment tracking
