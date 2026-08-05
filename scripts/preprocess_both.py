"""
Pre-process CubiCasa5k: resize BOTH image and mask to 512x512 once.
Saves as image_512.png and mask_512.png in each house folder.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from data.dataset import FloorPlanDataset
from data.cubicasa_parser import parse_cubicasa_sample


def preprocess_split(root_dir: str, split: str, target_size=(512, 512)):
    print(f"\nPre-processing {split} split...")
    
    # Load with SVG parsing to get mask
    ds = FloorPlanDataset.from_cubicasa5k(
        root_dir, split=split, target_size=target_size, augment=False, prefer_preprocessed=False
    )
    
    processed = 0
    skipped = 0
    
    for sample in tqdm(ds.samples, desc=f"Processing {split}"):
        house_dir = Path(sample['image_path']).parent
        img_512_path = house_dir / "image_512.png"
        mask_512_path = house_dir / "mask_512.png"
        
        if img_512_path.exists() and mask_512_path.exists():
            skipped += 1
            continue
        
        try:
            image, mask, _ = parse_cubicasa_sample(
                image_path=sample['image_path'],
                svg_path=sample['svg_path'],
                target_size=target_size
            )
            
            Image.fromarray(image).save(img_512_path)
            Image.fromarray(mask).save(mask_512_path)
            processed += 1
            
        except Exception as e:
            print(f"  Error: {house_dir.name}: {e}")
    
    print(f"  Done: {processed} processed, {skipped} already existed")
    return processed, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cubicasa5k")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    
    total_p = 0
    total_s = 0
    
    for split in ["train", "val", "test"]:
        split_file = Path(args.data) / f"{split}.txt"
        if not split_file.exists():
            continue
        p, s = preprocess_split(args.data, split, (args.size, args.size))
        total_p += p
        total_s += s
    
    print(f"\n{'='*50}")
    print(f"Done! Processed: {total_p}, Skipped: {total_s}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()