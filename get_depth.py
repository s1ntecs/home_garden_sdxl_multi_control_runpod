import torch, numpy as np
from PIL import Image
from transformers import pipeline

DEVICE = 0            # cuda:id  или  -1 для CPU
IMG_PATH = "input.jpg"   # ваша исходная картинка

# ---------- depth ----------------------------------------------------------
depth_pipe = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu",      # любая Zoe / DPT / DepthAnything
    device=DEVICE
)

img = Image.open(IMG_PATH).convert("RGB")
w, h = img.size

depth = depth_pipe(img)["predicted_depth"][0]        # → torch 1×H×W
depth = torch.nn.functional.interpolate(             # растягиваем до full-res
           depth.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False
         ).squeeze().cpu().numpy()

# нормализуем и переводим в 3-канальный вид
d_min, d_max = np.percentile(depth, (5, 95))
depth_norm = (np.clip((depth - d_min) / (d_max - d_min), 0, 1) * 255).astype(np.uint8)
depth_map = Image.fromarray(depth_norm).convert("RGB")
depth_map.save("depth_map.png")

# ---------- semantic segmentation -----------------------------------------
seg_pipe = pipeline(
    task="image-segmentation",
    model="nvidia/segformer-b0-finetuned-ade-512-512",   # ADE20K, 150 классов
    device=DEVICE
)                                                       # :contentReference[oaicite:1]{index=1}

segments = seg_pipe(img)

# создаём карту с id-классами
label_map = np.zeros((h, w), dtype=np.uint8)
for part in segments:
    mask = np.array(part["mask"])           # 255 = входит, 0 = нет
    class_id = part["label_id"]             # >= 1
    label_map[mask == 255] = class_id

# ADE20K-палитра (150 цветов). Можно импортировать готовый список,
# например из  controlnet_aux.oneformer. Здесь — короткий вариант:
ADE20K_COLORS = np.loadtxt(
    "https://raw.githubusercontent.com/CSAILVision/ADE20K/master/ade20k_colors.txt",
    dtype=np.uint8
)[:150]                                      # (150,3)  R,G,B

seg_rgb = ADE20K_COLORS[label_map]           # → H×W×3
seg_map = Image.fromarray(seg_rgb)
seg_map.save("seg_map.png")
