"""
Pre-process CubiCasa5k: parse all SVGs into PNG masks once.
After this, training loads PNG masks directly (no SVG parsing).
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from data.dataset import FloorPlanDataset


def preprocess_split(root_dir: str, split: str, target_size=(512, 512)):
    """Parse all SVGs in a split and save masks as PNGs."""
    print(f"\nPre-processing {split} split...")
    
    ds = FloorPlanDataset.from_cubicasa5k(
        root_dir, split=split, target_size=target_size, augment=False
    )
    
    processed = 0
    skipped = 0
    
    for sample in tqdm(ds.samples, desc=f"Processing {split}"):
        house_dir = Path(sample['image_path']).parent
        mask_path = house_dir / "mask.png"
        
        # Skip if already processed
        if mask_path.exists():
            skipped += 1
            continue
        
        try:
            # Parse SVG to get mask
            from data.cubicasa_parser import parse_cubicasa_sample
            _, mask, _ = parse_cubicasa_sample(
                image_path=sample['image_path'],
                svg_path=sample['svg_path'],
                target_size=target_size
            )
            
            # Save as PNG
            Image.fromarray(mask).save(mask_path)
            processed += 1
            
        except Exception as e:
            print(f"  Error processing {house_dir.name}: {e}")
    
    print(f"  Done: {processed} processed, {skipped} already existed")
    return processed, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cubicasa5k")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    total_processed = 0
    total_skipped = 0
    
    for split in ["train", "val", "test"]:
        split_file = Path(args.data) / f"{split}.txt"
        if not split_file.exists():
            print(f"Skipping {split} (no {split}.txt found)")
            continue
        
        p, s = preprocess_split(args.data, split, target_size)
        total_processed += p
        total_skipped += s
    
    print(f"\n{'='*50}")
    print(f"Pre-processing complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print(f"Masks saved as mask.png in each house folder")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()