import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# === Paths ===
lr_folder = 'train/LR'
output_folder = 'train/teacher_outputs'
os.makedirs(output_folder, exist_ok=True)

# === Model Config ===
model_path = 'RealESRGAN_x4plus.pth'
model_scale = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# === Load RealESRGAN Teacher ===
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=model_scale
)

state_dict = torch.load(model_path, map_location='cuda')['params_ema']
model.load_state_dict(state_dict, strict=True)

upsampler = RealESRGANer(
    scale=model_scale,
    model_path=model_path,
    model=model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device='cuda'
)

# === Inference for All Images ===
# === Inference for Remaining Images Only ===
# Start from 146th image → index 145
lr_files = sorted([f for f in os.listdir(lr_folder) if f.endswith('.png')])
remaining_files = lr_files[2909:]  # Skip the first 145

print(f"Generating teacher outputs for {len(remaining_files)} remaining images...")

for fname in tqdm(remaining_files):
    img_path = os.path.join(lr_folder, fname)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)

    try:
        output, _ = upsampler.enhance(img, outscale=model_scale)
        output_img = Image.fromarray(output)
        output_img.save(os.path.join(output_folder, fname))
    except Exception as e:
        print(f"Error processing {fname}: {e}")


print("✅ All teacher outputs generated and saved.")
torch.cuda.empty_cache()
