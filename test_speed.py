"""Test training speed on preprocessed samples only."""

import os
import glob
import time
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from data.dataset import FloorPlanDataset
from data.class_mapping import NUM_CLASSES

# Find all houses with image_512.png + mask_512.png
base = "data/cubicasa5k"
samples = []

for split in ["train", "val"]:
    split_file = os.path.join(base, f"{split}.txt")
    if not os.path.exists(split_file):
        continue
    with open(split_file) as f:
        for line in f:
            rel = line.strip().strip('/')
            house_dir = os.path.join(base, rel)
            img_512 = os.path.join(house_dir, "image_512.png")
            mask_512 = os.path.join(house_dir, "mask_512.png")
            if os.path.exists(img_512) and os.path.exists(mask_512):
                samples.append({
                    'image_path': img_512,
                    'mask_path': mask_512,
                    'house_id': os.path.basename(house_dir),
                    'source': 'cubicasa_preprocessed',
                })

print(f"Found {len(samples)} preprocessed samples")

# Create dataset
ds = FloorPlanDataset(samples, target_size=(512, 512), augment=False)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Warm-up (first batch always slow)
_ = next(iter(loader))

# Time 10 batches
print("\nTiming 10 batches...")
start = time.time()
for i, (images, masks) in enumerate(loader):
    if i >= 10:
        break
    images = images.to(device)
    masks = masks.to(device).long()
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    print(f"  Batch {i+1}/10 — loss: {loss.item():.4f}")

elapsed = time.time() - start
per_batch = elapsed / 10
per_epoch_est = (len(ds) / 8) * per_batch / 60  # minutes

print(f"\n{'='*50}")
print(f"Speed test complete!")
print(f"  Time for 10 batches: {elapsed:.1f}s")
print(f"  Per batch: {per_batch:.2f}s")
print(f"  Estimated epoch time: {per_epoch_est:.1f} minutes")
print(f"  Estimated 50 epochs: {per_epoch_est * 50 / 60:.1f} hours")
print(f"{'='*50}")