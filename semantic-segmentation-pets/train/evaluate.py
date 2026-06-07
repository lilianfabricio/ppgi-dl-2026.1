#!/usr/bin/env python3
"""Avalia o checkpoint salvo no conjunto de validação."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import create_dataloaders
from src.metrics import build_metrics, update_metrics
from src.model import UNet
from train.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    SUBSET_TRAIN,
    SUBSET_VAL,
)


@torch.no_grad()
def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {CHECKPOINT_PATH}. Rode train.py primeiro.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    _, val_loader = create_dataloaders(
        root=str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        subset_train=SUBSET_TRAIN,
        subset_val=SUBSET_VAL,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    iou, accuracy = build_metrics(NUM_CLASSES, device)
    total_loss = 0.0

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        update_metrics(iou, accuracy, logits, masks)

    print(f"Val loss: {total_loss / len(val_loader):.4f}")
    print(f"IoU:      {iou.compute().item():.4f}")
    print(f"Acurácia: {accuracy.compute().item():.4f}")


if __name__ == "__main__":
    main()
