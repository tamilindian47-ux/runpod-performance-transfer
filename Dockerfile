FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

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

WORKDIR /app/MimicMotion

# 2. Update conda base environment and install all required runtime dependencies
RUN conda env update -n base -f environment.yaml && \
    pip install --no-cache-dir \
        runpod \
        requests \
        matplotlib \
        opencv-python-headless \
        onnxruntime-gpu \
        accelerate

# 3. Download DWPose weights (~300MB total)
RUN mkdir -p models/DWPose && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx -O models/DWPose/yolox_l.onnx && \
    wget -q https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx -O models/DWPose/dw-ll_ucoco_384.onnx

# 4. Download MimicMotion checkpoint (~3.05GB)
RUN wget -q https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth -O models/MimicMotion_1-1.pth

# 5. Download SVD XT 1.1 base unet checkpoint directly (~3.0GB)
RUN mkdir -p models/SVD && \
    wget -q https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -O models/SVD/diffusion_pytorch_model.fp16.safetensors

# 6. HARD CHECK: Import EVERY submodule used by the pipeline to guarantee a clean runtime
RUN python -c "\
import numpy; \
import torch; \
import cv2; \
import matplotlib; \
import onnxruntime; \
import accelerate; \
import runpod; \
from mimicmotion.utils.loader import create_pipeline; \
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose; \
from mimicmotion.dwpose.util import draw_pose; \
print('>>> ALL DEPENDENCIES & SUBMODULES VERIFIED SUCCESSFULLY <<<')"

# 7. Copy handler
COPY handler.py /app/MimicMotion/handler.py

ENV PYTHONPATH=/app/MimicMotion

CMD ["python", "-u", "/app/MimicMotion/handler.py"]
