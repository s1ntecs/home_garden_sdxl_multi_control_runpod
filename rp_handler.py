import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetPipeline,
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


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
cn_depth = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
)

cn_seg = ControlNetModel.from_pretrained(
    "SargeZT/sdxl-controlnet-seg",
    torch_dtype=DTYPE)

# lineart_cn = ControlNetModel.from_pretrained(
#     "ShermanG/ControlNet-Standard-Lineart-for-SDXL",
#     torch_dtype=torch.float16)


PIPELINE = StableDiffusionXLControlNetPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    controlnet=[cn_depth, cn_seg],
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

midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
# line_det = LineartDetector.from_pretrained("lllyasviel/Annotators")

seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)

image_segmentor.to(DEVICE)


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
        segm_scale = float(payload.get(
            "segm_conditioning_scale", 0.45))

        # ---------- препроцессинг входа ------------
        image_pil = url_to_pil(image_url)
        orig_w, orig_h = image_pil.size

        pixel = seg_image_processor(image_pil,
                                    return_tensors="pt"
                                    ).pixel_values
        pixel = pixel.to(DEVICE)
        outputs = image_segmentor(pixel)

        # ---- segmentation -------------------------------------------------------
        seg = seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_pil.size[::-1]]
        )[0]
        palette = np.array(ade_palette())
        color_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label] = color
        seg_pil = Image.fromarray(color_seg).convert("RGB")

        # ---- depth --------------------------------------------------------------
        depth_cond = midas(image_pil)

        # ---- 5. Lineart карта (LineartStandardDetector / тот же что в обучении модели) ----
        # line_np = line_det(image_pil)
        # line_img = Image.fromarray(line_np)

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[depth_cond, seg_pil],
            control_image=[depth_cond, seg_pil],
            # image=[depth_cond, line_img],
            # control_image=[depth_cond, line_img],
            controlnet_conditioning_scale=[depth_scale, segm_scale],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        final = []
        for im in images:
            im = im.resize((orig_w, orig_h), Image.Resampling.LANCZOS).convert("RGB")
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
