"""
Plan2Scene — Phase 2: Fast Training (256x256, EfficientNet-B0, AMP)
~1–2 hours for 10 epochs. Good enough to validate the pipeline.
"""

import os
import argparse
import json
import platform
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from tqdm import tqdm

from data.dataset import FloorPlanDataset
from data.class_mapping import NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def compute_iou(pred, target, num_classes=NUM_CLASSES, ignore_index=0):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            continue
        ious.append(float(intersection) / float(union))
    return np.mean(ious) if ious else 0.0


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=num_classes, from_logits=True)

    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(outputs, dim=1)
        iou = compute_iou(preds, masks)

        total_loss += loss.item()
        total_iou += iou
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{iou:.4f}"})

    return total_loss / num_batches, total_iou / num_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device).long()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks)

            total_loss += loss.item()
            total_iou += iou
            num_batches += 1

    return total_loss / num_batches, total_iou / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cubicasa5k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds = FloorPlanDataset.from_cubicasa5k(
        args.data, split="train", target_size=(args.img_size, args.img_size), augment=True,
    )
    val_ds = FloorPlanDataset.from_cubicasa5k(
        args.data, split="val", target_size=(args.img_size, args.img_size), augment=False,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    num_workers = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model: U-Net + EfficientNet-B0 (much faster than ResNet-34) ──
    print("Building U-Net (EfficientNet-B0 encoder)...")
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = GradScaler()

    best_iou = 0.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_iou = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_iou)

        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (args.epochs - epoch)

        print(f"Train — Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}")
        print(f"Val   — Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        print(f"Time  — Elapsed: {elapsed/60:.0f}min, ETA: {eta/60:.0f}min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "args": vars(args),
            }, os.path.join(args.save_dir, "segmentation_best.pth"))
            print(f"💾 New best model! mIoU: {best_iou:.4f}")

    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Best mIoU: {best_iou:.4f}")
    print(f"Total time: {total/60:.0f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()