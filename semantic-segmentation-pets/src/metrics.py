import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex


def build_metrics(num_classes: int, device: torch.device):
    iou = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    return iou, accuracy


def update_metrics(
    iou: MulticlassJaccardIndex,
    accuracy: MulticlassAccuracy,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    preds = logits.argmax(dim=1)
    iou.update(preds, targets)
    accuracy.update(preds, targets)
