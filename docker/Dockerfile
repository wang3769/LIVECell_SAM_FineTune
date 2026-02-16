# CUDA + PyTorch runtime image (keeps “torch with CUDA” consistent)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# If you use opencv + tifffile
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# We added git because segment anything get installed from git repo and Docker by default does not have git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Put your 45MB decoder/optimizer/etc inside the image (simple)
# e.g. your repo already has: model_registry/ mask_decoder.pt optimizer.pt ...
# and SAM encoder checkpoint in models/
# download SAM encoder
RUN mkdir -p /app/sam
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O /app/sam/sam_vit_b_01ec64.pth

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
