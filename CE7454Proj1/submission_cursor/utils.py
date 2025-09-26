import torch


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    valid = target != ignore_index
    if valid.sum() == 0:
        return 0.0
    correct = (pred[valid] == target[valid]).sum().item()
    total = valid.sum().item()
    return correct / total


def compute_fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    beta: float = 1.0,
    ignore_index: int = 255,
) -> float:
    scores = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_mask = pred == class_id
        target_mask = target == class_id

        tp = (pred_mask & target_mask).sum().item()
        fp = (pred_mask & (~target_mask)).sum().item()
        fn = ((~pred_mask) & target_mask).sum().item()

        denom = (1 + beta**2) * tp + beta**2 * fn + fp
        if denom == 0:
            continue
        fscore = (1 + beta**2) * tp / denom
        scores.append(fscore)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def poly_lr_scheduler(optimizer: torch.optim.Optimizer, base_lr: float, iter: int, max_iter: int, power: float = 0.9) -> None:
    lr = base_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

