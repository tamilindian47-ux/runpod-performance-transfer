import os
import sys
import uuid
import base64
import requests
import torch
import runpod
from omegaconf import OmegaConf
from PIL import Image

from mimicmotion.utils.loader import create_pipeline
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
from mimicmotion.utils.utils import save_to_mp4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = "/app/MimicMotion"
SVD_CKPT = os.path.join(BASE_DIR, "models/SVD/diffusion_pytorch_model.fp16.safetensors")
CKPT_PATH = os.path.join(BASE_DIR, "models/MimicMotion_1-1.pth")

print("Initializing MimicMotion Pipeline...", flush=True)

# Build configuration structure expected by MimicMotion
infer_config = OmegaConf.create({
    "base_model_path": SVD_CKPT,
    "ckpt_path": CKPT_PATH
})

pipeline = create_pipeline(infer_config, device=DEVICE)
print("Pipeline loaded successfully.", flush=True)

def download_file(url_or_b64, target_path):
    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
        res = requests.get(url_or_b64, timeout=120)
        res.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(res.content)
    else:
        # Base64 string
        data = base64.b64decode(url_or_b64.split(",")[-1])
        with open(target_path, "wb") as f:
            f.write(data)

def handler(job):
    job_input = job.get("input", {})
    
    # 1. Inputs: reference image and motion video
    ref_image_src = job_input.get("ref_image")
    ref_video_src = job_input.get("ref_video")
    
    if not ref_image_src or not ref_video_src:
        return {"error": "Both 'ref_image' and 'ref_video' (URLs or base64) are required."}
    
    resolution = int(job_input.get("resolution", 576))
    num_inference_steps = int(job_input.get("num_inference_steps", 25))
    guidance_scale = float(job_input.get("guidance_scale", 2.0))
    sample_stride = int(job_input.get("sample_stride", 2))
    fps = int(job_input.get("fps", 15))
    
    task_id = str(uuid.uuid4())[:8]
    work_dir = f"/tmp/{task_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    img_path = os.path.join(work_dir, "ref.jpg")
    vid_path = os.path.join(work_dir, "ref.mp4")
    out_path = os.path.join(work_dir, "output.mp4")
    
    try:
        # 2. Save incoming inputs
        download_file(ref_image_src, img_path)
        download_file(ref_video_src, vid_path)
        
        # 3. Process image and video motion skeletons
        image_pixels = Image.open(img_path).convert("RGB")
        w, h = image_pixels.size
        scale = resolution / min(h, w)
        image_pixels = image_pixels.resize((int(w * scale), int(h * scale)))
        
        image_pose = get_image_pose(image_pixels)
        video_pose = get_video_pose(vid_path, image_pixels, sample_stride=sample_stride)
        pose_pixels = torch.from_numpy(video_pose).to(DEVICE)
        
        # 4. Generate animated frames
        with torch.no_grad():
            frames = pipeline(
                image=image_pixels,
                image_pose=image_pose,
                video_pose=pose_pixels,
                height=image_pixels.size[1],
                width=image_pixels.size[0],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=DEVICE)
            ).frames[0]
            
        save_to_mp4(frames, out_path, fps=fps)
        
        with open(out_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")
            
        return {
            "status": "success",
            "video_base64": encoded_video,
            "fps": fps
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        # 5. Clean up temporary working directory
        if os.path.exists(work_dir):
            os.system(f"rm -rf {work_dir}")

runpod.serverless.start({"handler": handler})
