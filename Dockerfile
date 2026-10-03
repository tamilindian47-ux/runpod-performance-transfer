FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Clone MimicMotion repository
RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

# 2. Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

WORKDIR /app/MimicMotion

# 3. Download DWPose models (~300MB total)
RUN mkdir -p models/DWPose && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx -O models/DWPose/yolox_l.onnx && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx -O models/DWPose/dw-ll_ucoco_384.onnx

# 4. Download MimicMotion checkpoint (~3.05GB)
RUN wget -q https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth -O models/MimicMotion_1-1.pth

# 5. Pre-cache SVD base weights into the image (~3.1GB) so worker starts instantly
RUN python -c "from diffusers import StableVideoDiffusionPipeline; StableVideoDiffusionPipeline.from_pretrained('vdo/stable-video-diffusion-img2vid-xt-1-1', torch_dtype=None)"



# 6. Copy serverless handler
COPY handler.py /app/MimicMotion/handler.py

ENV PYTHONPATH=/app/MimicMotion

CMD ["python", "-u", "/app/MimicMotion/handler.py"]
