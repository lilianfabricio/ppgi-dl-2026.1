from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mask_to_binary(mask: torch.Tensor) -> torch.Tensor:
    """Oxford trimap: 1=pet, 2=background, 3=border -> 1=pet, 0=background."""
    return (mask == 1).long()


class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
            transform=transform,
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base[index]
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        mask = torch.as_tensor(mask, dtype=torch.long)
        mask = mask_to_binary(mask)

        return image, mask


def create_dataloaders(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    subset_train: int,
    subset_val: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PetSegmentationDataset(root=root, split="trainval", image_size=image_size)
    val_dataset = PetSegmentationDataset(root=root, split="test", image_size=image_size)

    train_indices = list(range(min(subset_train, len(train_dataset))))
    val_indices = list(range(min(subset_val, len(val_dataset))))

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
