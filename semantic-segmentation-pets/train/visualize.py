#!/usr/bin/env python3
"""Gera visualização com máscara sobreposta (Etapa 3)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, PetSegmentationDataset
from src.model import UNet
from train.config import CHECKPOINT_PATH, DATA_DIR, IMAGE_SIZE, NUM_CLASSES


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def overlay_mask(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    overlay = image.clone()
    color = torch.tensor([1.0, 0.2, 0.2]).view(3, 1, 1)
    pet = mask.bool()
    overlay[:, pet] = overlay[:, pet] * (1 - alpha) + color * alpha
    return overlay


@torch.no_grad()
def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {CHECKPOINT_PATH}. Rode train.py primeiro.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    dataset = PetSegmentationDataset(root=str(DATA_DIR), split="test", image_size=IMAGE_SIZE)
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    for index in range(min(3, len(dataset))):
        image, target = dataset[index]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu()

        image_rgb = denormalize(image)
        result = overlay_mask(image_rgb, pred)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb.permute(1, 2, 0))
        axes[0].set_title("Original")
        axes[1].imshow(target, cmap="gray")
        axes[1].set_title("Máscara GT")
        axes[2].imshow(result.permute(1, 2, 0))
        axes[2].set_title("Predição sobreposta")

        for ax in axes:
            ax.axis("off")

        out_path = output_dir / f"sample_{index}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
