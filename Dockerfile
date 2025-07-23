FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    SHELL=/bin/bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) System deps
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-dev python3-pip python3.10-venv \
      fonts-dejavu-core git git-lfs jq wget curl \
      libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
      ffmpeg procps && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /workspace

# 2) Copy requirements
COPY requirements.txt .

# 3) Install torch + torchvision + torchaudio + xformers (all pinned) with retries
RUN pip3 install --upgrade pip && \
    pip3 install \
      --no-cache-dir \
      --timeout=120 \
      --retries=5 \
      torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install \
      --no-cache-dir \
      --timeout=120 \
      --retries=5 \
      xformers==0.0.28.post1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir -r requirements.txt

# 4) Copy rest of the code
COPY . .

# 5) Prepare dirs & checkpoints
RUN mkdir -p loras checkpoints && \
    python3 download_checkpoints.py

# 6) Entry point
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]
