import os
import torch

from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline
)
from controlnet_aux import ZoeDetector, LineartDetector
from transformers import (AutoImageProcessor,
                          SegformerForSemanticSegmentation)

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
    cn_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-zoe-depth-sdxl-1.0",
        torch_dtype=DTYPE
        )

    cn_seg = ControlNetModel.from_pretrained(
        "SargeZT/sdxl-controlnet-seg",
        torch_dtype=DTYPE)

    # lineart_cn = ControlNetModel.from_pretrained(
    #     "ShermanG/ControlNet-Standard-Lineart-for-SDXL",
    #     torch_dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16,
                                        use_safetensors=True)

    PIPELINE = StableDiffusionXLControlNetPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        # "SG161222/RealVisXL_V5.0",
        # "misri/cyberrealisticPony_v90Alt1",
        # "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
        torch_dtype=DTYPE,
        add_watermarker=False,
        controlnet=[cn_depth, cn_seg],
        vae=vae,
        # variant="fp16",
        use_safetensors=True,
        resume_download=True,
    )

    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
        PIPELINE.scheduler.config)

    StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        safety_checker=None,
    )
    zoe = ZoeDetector.from_pretrained(
        "lllyasviel/Annotators")

    seg_image_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    # line_det = LineartDetector.from_pretrained("lllyasviel/Annotators")


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
