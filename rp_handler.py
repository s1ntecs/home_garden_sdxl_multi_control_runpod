import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image

from diffusers import (
    # StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL, DDIMScheduler
)
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from controlnet_aux import PidiNetDetector, HEDdetector

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
def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


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

# controlnet = ControlNetModel.from_pretrained(
#                 "diffusers/controlnet-canny-sdxl-1.0",
#                 torch_dtype=DTYPE
#             )

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-scribble-sdxl-1.0",
    torch_dtype=torch.float16
)

# cn_seg = ControlNetModel.from_pretrained(
#     "SargeZT/sdxl-controlnet-seg",
#     torch_dtype=DTYPE)
eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler")

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


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
    scheduler=eulera_scheduler,
    vae=vae,
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
processor = HEDdetector.from_pretrained('lllyasviel/Annotators')


def make_scribble_condition(image: Image.Image) -> Image.Image:
    controlnet_img = processor(image, scribble=False)
    # controlnet_img.save("a hed detect path for an image")

    # following is some processing to simulate human sketch draw, different threshold can generate different width of lines
    controlnet_img = np.array(controlnet_img)
    controlnet_img = nms(controlnet_img, 127, 3)
    controlnet_img = cv2.GaussianBlur(controlnet_img, (0, 0), 3)

    # higher threshold, thiner line
    random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
    controlnet_img[controlnet_img > random_val] = 255
    controlnet_img[controlnet_img < 255] = 0
    controlnet_img = Image.fromarray(controlnet_img)
    return controlnet_img


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
            "img_strength", 0.5)
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
        scribble_scale = float(payload.get(
            "scribble_scale",
            0.9)
        )
        scribble_guidance_start = float(payload.get(
            "scribble_guidance_start",
            0.0)
         )
        scribble_guidance_end = float(payload.get(
            "scribble_guidance_end",
            1.0)
        )
        # ---------- препроцессинг входа ------------

        image_pil = url_to_pil(image_url)

        # ---- canny --------------------------------------------------------------
        # control_image = make_canny_condition(image_pil)
        control_image = make_scribble_condition(image_pil)

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
            controlnet_conditioning_scale=scribble_scale,
            control_guidance_start=scribble_guidance_start,
            control_guidance_end=scribble_guidance_end,
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
