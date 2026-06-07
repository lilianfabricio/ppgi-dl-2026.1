#!/usr/bin/env python3
"""Exporta o modelo treinado para TorchScript (FP32) para uso no Android."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import UNet
from train.config import ANDROID_ASSETS, CHECKPOINT_PATH, EXPORT_PATH, IMAGE_SIZE, NUM_CLASSES


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {CHECKPOINT_PATH}. Rode train.py primeiro.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.float()

    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced = traced.float()

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)

    traced.save(str(EXPORT_PATH))
    android_path = ANDROID_ASSETS / "model.pt"
    traced.save(str(android_path))

    print(f"Modelo exportado (FP32):")
    print(f"  - {EXPORT_PATH}")
    print(f"  - {android_path}")
    print(f"Input shape: [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")


if __name__ == "__main__":
    main()
