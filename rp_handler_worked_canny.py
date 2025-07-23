import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
)
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from controlnet_aux import MidasDetector

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

from colors import ade_palette

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250
TARGET_RES = 1024  # SDXL рекомендует 1024×1024

logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-depth-sdxl-1.0",
#     torch_dtype=DTYPE,
#     use_safetensors=True
# )

controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=DTYPE
            )

# cn_seg = ControlNetModel.from_pretrained(
#     "SargeZT/sdxl-controlnet-seg",
#     torch_dtype=DTYPE)


PIPELINE = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    # controlnet=[cn_depth, cn_seg],
    controlnet=controlnet,
    torch_dtype=DTYPE,
    # variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
    requires_safety_checker=False,
    add_watermarker=False,
    use_safetensors=True,
    resume_download=True,
)
PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
    PIPELINE.scheduler.config)
PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.to(DEVICE)

REFINER = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=DTYPE,
    variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
)
REFINER.scheduler = DDIMScheduler.from_config(REFINER.scheduler.config)
REFINER.to(DEVICE)

CURRENT_LORA = "None"

# midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")

# seg_image_processor = AutoImageProcessor.from_pretrained(
#     "nvidia/segformer-b5-finetuned-ade-640-640"
# )
# image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b5-finetuned-ade-640-640"
# )


def resize_dimensions(dimensions, target_size):
    """
    Resize PIL to target size while maintaining aspect ratio
    If smaller than target size leave it as is
    """
    width, height = dimensions

    # Check if both dimensions are smaller than the target size
    if width < target_size and height < target_size:
        return dimensions

    # Determine the larger side
    if width > height:
        # Calculate the aspect ratio
        aspect_ratio = height / width
        # Resize dimensions
        return (target_size, int(target_size * aspect_ratio))
    else:
        # Calculate the aspect ratio
        aspect_ratio = width / height
        # Resize dimensions
        return (int(target_size * aspect_ratio), target_size)


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        negative_prompt = payload.get(
            "negative_prompt", "")
        img_strength = payload.get(
            "img_strength", 4.0)
        guidance_scale = float(payload.get(
            "guidance_scale", 7.5))
        steps = min(int(payload.get(
            "steps", MAX_STEPS)),
                    MAX_STEPS)

        seed = int(payload.get(
            "seed",
            random.randint(0, MAX_SEED)))
        generator = torch.Generator(
            device=DEVICE).manual_seed(seed)

        # refiner
        refiner_strength = float(payload.get(
            "refiner_strength", 0.2))
        refiner_steps = int(payload.get(
            "refiner_steps", 15))
        refiner_scale = float(payload.get(
            "refiner_scale", 7.5))

        # control scales
        depth_scale = float(payload.get(
            "depth_conditioning_scale", 0.9))
        # ---------- препроцессинг входа ------------

        image_pil = url_to_pil(image_url)

        # ---- canny --------------------------------------------------------------
        control_image = make_canny_condition(image_pil)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)

        # resize *both* the init image and the control image to the same, /8-aligned size
        image_pil = image_pil.resize((work_w, work_h), Image.Resampling.LANCZOS)
        canny_cond = control_image.resize((work_w, work_h), Image.Resampling.LANCZOS)
        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            control_image=canny_cond,
            controlnet_conditioning_scale=depth_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            strength=img_strength,
        ).images

        final = []
        for im in images:
            im = im.resize((orig_w, orig_h),
                           Image.Resampling.LANCZOS).convert("RGB")
            ref = REFINER(
                prompt=prompt, image=im, strength=refiner_strength,
                num_inference_steps=refiner_steps, guidance_scale=refiner_scale
            ).images[0]
            final.append(ref)
        torch.cuda.empty_cache()

        return {
            "images_base64": [pil_to_b64(i) for i in final],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps, "seed": seed,
            "lora": CURRENT_LORA if CURRENT_LORA != "None" else None,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
