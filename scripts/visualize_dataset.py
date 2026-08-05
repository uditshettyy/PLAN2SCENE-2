#!/usr/bin/env python3
"""
Plan2Scene — Dataset Visualization Tool

Visualizes floor plan images alongside their segmentation masks.

Usage:
    python scripts/visualize_dataset.py --source cubicasa --root data/cubicasa5k --split train --num 5 --save_dir viz_outputs
    python scripts/visualize_dataset.py --source synthetic --root data/synthetic --num 10 --save_dir viz_outputs
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.class_mapping import CLASSES, COLOR_MAP
from data.dataset import FloorPlanDataset


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class ID mask to RGB visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    return rgb


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend image and colored mask."""
    colored_mask = colorize_mask(mask)
    overlay = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlay


def visualize_sample(image: np.ndarray, mask: np.ndarray, title: str = "", figsize: tuple = (16, 5)):
    """Create a 4-panel visualization: image, mask, overlay, legend."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Floor Plan Image")
    axes[0].axis('off')

    # Panel 2: Segmentation mask
    colored_mask = colorize_mask(mask)
    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    # Panel 3: Overlay
    overlay = create_overlay(image, mask)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (α=0.4)")
    axes[2].axis('off')

    # Panel 4: Class legend + statistics
    axes[3].axis('off')
    unique_classes = sorted(np.unique(mask))
    patches = []
    stats_text = []
    total_pixels = mask.size

    for cid in unique_classes:
        if cid < len(CLASSES):
            name = CLASSES[cid]
            color = np.array(COLOR_MAP[cid]) / 255.0
            count = np.sum(mask == cid)
            pct = count / total_pixels * 100
            patches.append(mpatches.Patch(color=color, label=f"{name}"))
            stats_text.append(f"{name}: {count:,} px ({pct:.1f}%)")

    axes[3].legend(handles=patches, loc='upper left', fontsize=10, frameon=True, title="Classes", title_fontsize=11)
    stats_str = "\n".join(stats_text)
    axes[3].text(0.05, 0.55, stats_str, transform=axes[3].transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize floor plan dataset samples")
    parser.add_argument("--source", choices=["synthetic", "cubicasa"], required=True)
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="Split (for CubiCasa5k)")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default="viz_outputs", help="Directory to save images")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Target size (W H)")
    args = parser.parse_args()

    target_size = tuple(args.size)

    if args.source == "synthetic":
        dataset = FloorPlanDataset.from_synthetic(args.root, target_size=target_size)
    else:
        dataset = FloorPlanDataset.from_cubicasa5k(args.root, split=args.split, target_size=target_size)

    print(f"Loaded dataset with {len(dataset)} samples")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving visualizations to: {args.save_dir}")

    num_samples = min(args.num, len(dataset))

    for i in range(num_samples):
        image_tensor, mask_tensor = dataset[i]
        image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask_tensor.numpy()

        sample_info = dataset.samples[i]
        title = f"Sample {i} — {sample_info.get('house_id', 'unknown')}"

        fig = visualize_sample(image, mask, title=title)
        save_path = os.path.join(args.save_dir, f"viz_{i:04d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)

    print(f"\nDone! Visualized {num_samples} samples.")


if __name__ == "__main__":
    main()