FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Clone MimicMotion repository
RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

# 2. Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

WORKDIR /app/MimicMotion

# 3. Download DWPose weights (~300MB total)
RUN mkdir -p models/DWPose && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx -O models/DWPose/yolox_l.onnx && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx -O models/DWPose/dw-ll_ucoco_384.onnx

# 4. Download MimicMotion checkpoint (~3.05GB)
RUN wget -q https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth -O models/MimicMotion_1-1.pth

# 5. Download SVD XT 1.1 base unet checkpoint directly (~3.0GB)
RUN mkdir -p models/SVD && \
    wget -q https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -O models/SVD/diffusion_pytorch_model.fp16.safetensors

# 6. Copy serverless handler
COPY handler.py /app/MimicMotion/handler.py

ENV PYTHONPATH=/app/MimicMotion

CMD ["python", "-u", "/app/MimicMotion/handler.py"]
