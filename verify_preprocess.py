"""Verify that preprocessed image_512.png + mask_512.png pairs are valid."""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from data.class_mapping import CLASSES, COLOR_MAP

# Find first house with both files
base = Path("data/cubicasa5k")
found = False

for split in ["train", "val", "test"]:
    split_file = base / f"{split}.txt"
    if not split_file.exists():
        continue
    
    with open(split_file) as f:
        for line in f:
            rel = line.strip().strip('/')
            house_dir = base / rel
            img_path = house_dir / "image_512.png"
            mask_path = house_dir / "mask_512.png"
            
            if img_path.exists() and mask_path.exists():
                found = True
                break
        if found:
            break

if not found:
    print("❌ No preprocessed files found yet. Wait for preprocessing to finish.")
    exit(1)

print(f"Checking: {house_dir.name}")
print(f"  Image: {img_path}")
print(f"  Mask:  {mask_path}")

# Load
img = np.array(Image.open(img_path).convert('RGB'))
mask = np.array(Image.open(mask_path).convert('L'))

print(f"\n📐 Image shape: {img.shape}, dtype: {img.dtype}")
print(f"📐 Mask shape:  {mask.shape}, dtype: {mask.dtype}")

# Check mask values
unique = sorted(np.unique(mask))
print(f"\n🔢 Unique values in mask: {unique}")
print(f"   Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (or subset)")

# Check each class
print(f"\n📊 Class distribution:")
for cid in unique:
    count = np.sum(mask == cid)
    pct = count / mask.size * 100
    name = CLASSES[cid] if cid < len(CLASSES) else "UNKNOWN"
    print(f"   Class {cid:2d} ({name:15s}): {count:6,} px ({pct:5.2f}%)")

# Verify values are in valid range
if max(unique) > 9:
    print(f"\n⚠️  WARNING: Max value {max(unique)} > 9 — mask may be corrupted!")
else:
    print(f"\n✅ Mask values are in valid range 0–9")

# Check image/mask alignment
if img.shape[:2] == mask.shape:
    print(f"✅ Image and mask dimensions match: {mask.shape}")
else:
    print(f"❌ Dimension mismatch: img {img.shape[:2]} vs mask {mask.shape}")

# Create a human-visible colorized mask for confirmation
print(f"\n🎨 Saving colorized mask for visual confirmation...")
colorized = np.zeros((*mask.shape, 3), dtype=np.uint8)
for cid, color in COLOR_MAP.items():
    colorized[mask == cid] = color

# Blend with image
alpha = 0.5
overlay = (img * (1 - alpha) + colorized * alpha).astype(np.uint8)

# Save verification images
os.makedirs("verify_outputs", exist_ok=True)
Image.fromarray(img).save("verify_outputs/01_image.png")
Image.fromarray(colorized).save("verify_outputs/02_mask_colorized.png")
Image.fromarray(overlay).save("verify_outputs/03_overlay.png")

print(f"   Saved: verify_outputs/01_image.png")
print(f"   Saved: verify_outputs/02_mask_colorized.png")
print(f"   Saved: verify_outputs/03_overlay.png")
print(f"\n👀 Open verify_outputs/03_overlay.png to confirm alignment.")