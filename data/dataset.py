"""
Plan2Scene — PyTorch Dataset (Ultra-Fast: preprocessed 512x512 images + masks)
"""

import os
import glob
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

from data.class_mapping import NUM_CLASSES
from data.cubicasa_parser import parse_cubicasa_sample

logger = logging.getLogger(__name__)


class FloorPlanDataset(Dataset):
    def __init__(self, samples: List[dict], target_size: Tuple[int, int] = (512, 512), augment: bool = False):
        self.samples = samples
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        source = sample['source']

        if source == 'cubicasa_preprocessed':
            # ULTRA-FAST: load pre-resized 512x512 PNGs directly
            image = np.array(Image.open(sample['image_path']).convert('RGB'))
            mask = np.array(Image.open(sample['mask_path']).convert('L'))
        elif source == 'cubicasa':
            image, mask, _ = parse_cubicasa_sample(
                image_path=sample['image_path'],
                svg_path=sample['svg_path'],
                target_size=self.target_size
            )
        else:
            image = np.array(Image.open(sample['image_path']).convert('RGB'))
            mask = np.array(Image.open(sample['mask_path']).convert('L'))

        # Convert to tensors (no resize needed — already 512x512)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask

    @classmethod
    def from_cubicasa5k(
        cls,
        root_dir: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        prefer_preprocessed: bool = True,
    ) -> "FloorPlanDataset":
        split_file = os.path.join(root_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        samples = []
        skipped = 0

        for line in lines:
            rel_path = line.strip('/')
            house_dir = os.path.join(root_dir, rel_path)

            if not os.path.isdir(house_dir):
                skipped += 1
                continue

            house_id = os.path.basename(house_dir)

            # Check for preprocessed files first
            img_512 = os.path.join(house_dir, "image_512.png")
            mask_512 = os.path.join(house_dir, "mask_512.png")
            mask_png = os.path.join(house_dir, "mask.png")  # old preprocessed mask
            svg_path = os.path.join(house_dir, "model.svg")

            # Find original image
            orig_image = None
            for cand_name in ['F1_scaled.png', 'F1.png']:
                cand = os.path.join(house_dir, cand_name)
                if os.path.exists(cand):
                    orig_image = cand
                    break
            if orig_image is None:
                pngs = glob.glob(os.path.join(house_dir, "*.png"))
                if pngs:
                    orig_image = pngs[0]

            if orig_image is None:
                skipped += 1
                continue

            # Priority: image_512 + mask_512 > image + mask.png > image + svg
            if prefer_preprocessed and os.path.exists(img_512) and os.path.exists(mask_512):
                samples.append({
                    'image_path': img_512,
                    'mask_path': mask_512,
                    'house_id': house_id,
                    'source': 'cubicasa_preprocessed',
                })
            elif os.path.exists(mask_png):
                samples.append({
                    'image_path': orig_image,
                    'mask_path': mask_png,
                    'house_id': house_id,
                    'source': 'cubicasa_preprocessed',
                })
            elif os.path.exists(svg_path):
                samples.append({
                    'image_path': orig_image,
                    'svg_path': svg_path,
                    'house_id': house_id,
                    'source': 'cubicasa',
                })
            else:
                skipped += 1

        mode = "preprocessed" if prefer_preprocessed else "raw SVG"
        logger.info(f"CubiCasa5k [{split}] ({mode}): {len(samples)} samples, {skipped} skipped")
        return cls(samples, target_size=target_size, augment=augment)

    @classmethod
    def from_synthetic(cls, root_dir: str, target_size: Tuple[int, int] = (512, 512), augment: bool = False):
        images_dir = os.path.join(root_dir, "images")
        masks_dir = os.path.join(root_dir, "masks")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        samples = []
        for img_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
            basename = os.path.basename(img_path)
            mask_path = os.path.join(masks_dir, basename)
            if os.path.exists(mask_path):
                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'house_id': basename.replace('.png', ''),
                    'source': 'synthetic',
                })

        logger.info(f"Synthetic: {len(samples)} samples loaded")
        return cls(samples, target_size=target_size, augment=augment)