import os
import torch

from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline
)

# from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LORA_NAMES = [
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""

    # for fname in LORA_NAMES:
    #     hf_hub_download(
    #         repo_id="sintecs/interior",
    #         filename=fname,
    #         local_dir="loras",
    #         local_dir_use_symlinks=False,
    #     )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=DTYPE,
        )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16,
                                        use_safetensors=True)

    PIPELINE = StableDiffusionXLControlNetPipeline.from_pretrained(
        # "RunDiffusion/Juggernaut-XL-v9",
        # "SG161222/RealVisXL_V5.0",
        # "misri/cyberrealisticPony_v90Alt1",
        "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
        torch_dtype=torch.float16,
        add_watermarker=False,
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        resume_download=True,
    ).to(DEVICE)
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
        PIPELINE.scheduler.config)

    StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        safety_checker=None,
    )
    return


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
