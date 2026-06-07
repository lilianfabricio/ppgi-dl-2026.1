#!/usr/bin/env python3
"""Treina UNet no Oxford-IIIT Pet (subset) e salva o melhor checkpoint."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import create_dataloaders
from src.metrics import build_metrics, update_metrics
from src.model import UNet
from train.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATA_DIR,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_WORKERS,
    SUBSET_TRAIN,
    SUBSET_VAL,
)


def train_one_epoch(model, loader, criterion, optimizer, device, iou, accuracy):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        update_metrics(iou, accuracy, logits.detach(), masks)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, iou, accuracy):
    model.eval()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        update_metrics(iou, accuracy, logits, masks)

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        root=str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        subset_train=SUBSET_TRAIN,
        subset_val=SUBSET_VAL,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_iou, train_acc = build_metrics(NUM_CLASSES, device)
        val_iou, val_acc = build_metrics(NUM_CLASSES, device)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_iou, train_acc
        )
        val_loss = validate(model, val_loader, criterion, device, val_iou, val_acc)

        epoch_train_iou = train_iou.compute().item()
        epoch_train_acc = train_acc.compute().item()
        epoch_val_iou = val_iou.compute().item()
        epoch_val_acc = val_acc.compute().item()

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={train_loss:.4f} IoU={epoch_train_iou:.4f} acc={epoch_train_acc:.4f} | "
            f"val loss={val_loss:.4f} IoU={epoch_val_iou:.4f} acc={epoch_val_acc:.4f}"
        )

        if epoch_val_iou > best_iou:
            best_iou = epoch_val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "val_iou": epoch_val_iou,
                    "val_acc": epoch_val_acc,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> checkpoint salvo em {CHECKPOINT_PATH}")

    print(f"Treino finalizado. Melhor IoU (val): {best_iou:.4f}")


if __name__ == "__main__":
    main()
