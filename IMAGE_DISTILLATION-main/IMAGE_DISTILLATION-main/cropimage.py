import os
import random
from PIL import Image
from pathlib import Path

def random_crop_patches(hr_dir, lr_dir, out_hr_dir, out_lr_dir, patch_size=256, patches_per_image=4, seed=42):
    os.makedirs(out_hr_dir, exist_ok=True)
    os.makedirs(out_lr_dir, exist_ok=True)

    random.seed(seed)
    patch_id = 0

    hr_files = sorted(Path(hr_dir).glob("*.png"))

    for hr_path in hr_files:
        lr_path = Path(lr_dir) / hr_path.name

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        W, H = hr_img.size

        for _ in range(patches_per_image):
            left = random.randint(0, W - patch_size)
            top = random.randint(0, H - patch_size)

            hr_patch = hr_img.crop((left, top, left + patch_size, top + patch_size))
            lr_patch = lr_img.crop((left, top, left + patch_size, top + patch_size))

            hr_patch.save(f"{out_hr_dir}/{patch_id:05d}.png")
            lr_patch.save(f"{out_lr_dir}/{patch_id:05d}.png")
            patch_id += 1

    print(f"✅ Done: {patch_id} patches saved.")

# Example: 800 images × 4 patches = 3200 total
random_crop_patches(
    hr_dir="test/HR",
    lr_dir="test/LR",
    out_hr_dir="patches/HR",
    out_lr_dir="patches/LR",
    patch_size=256,
    patches_per_image=4
)
