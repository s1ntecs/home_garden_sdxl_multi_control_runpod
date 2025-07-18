FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    SHELL=/bin/bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) Системные зависимости
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-dev python3-pip python3.10-venv \
      fonts-dejavu-core git git-lfs jq wget curl \
      libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
      ffmpeg procps && \
    rm -rf /var/lib/apt/lists/*
RUN git lfs install

WORKDIR /workspace

# 2) Копируем файл зависимостей
COPY requirements.txt .

# 3) Устанавливаем PyTorch, torchvision, torchaudio и xFormers
RUN pip3 install --upgrade pip && \
    pip3 install \
      torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124
    # pip3 install \
    #   xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt

# 4) Копируем остальной код
COPY . .

# 5) Создаём каталоги и загружаем чекпоинты
RUN mkdir -p loras checkpoints && \
    python3 download_checkpoints.py

# 6) Точка входа
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]