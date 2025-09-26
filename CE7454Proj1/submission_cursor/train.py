import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import FaceSegmentationDataset
from model import build_model
from model_utils import count_parameters, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face segmentation network")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing train/val datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training data used for validation if explicit val set missing")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of classes")
    parser.add_argument("--model_type", type=str, default=None, choices=["v1", "v2", "v3"], help="Choose model architecture")
    return parser.parse_args()


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def compute_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    return criterion(outputs, targets)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = compute_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = compute_loss(logits, masks)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = compute_loss(logits, masks)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def load_config(config_path: str | None) -> Dict:
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def has_explicit_val_split(data_root: Path) -> bool:
    return (data_root / "val" / "images").is_dir()


def split_train_val(data_root: Path, val_split: float, seed: int) -> Tuple[List[Path], List[Path]]:
    train_image_dir = data_root / "train" / "images"
    image_paths = sorted(list(train_image_dir.glob("*.jpg")) + list(train_image_dir.glob("*.png")) + list(train_image_dir.glob("*.jpeg")))
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {train_image_dir}")
    val_count = max(1, int(len(image_paths) * val_split))
    random.Random(seed).shuffle(image_paths)
    val_paths = image_paths[:val_count]
    train_paths = image_paths[val_count:]
    if not train_paths:
        raise ValueError("Validation split too large; no training samples left")
    return train_paths, val_paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    image_transform = build_transforms()

    data_root = Path(args.data_root)
    if has_explicit_val_split(data_root):
        train_dataset = FaceSegmentationDataset(
            root_dir=args.data_root,
            split="train",
            transform=image_transform,
            augment=True,
        )
        val_dataset = FaceSegmentationDataset(
            root_dir=args.data_root,
            split="val",
            transform=image_transform,
            augment=False,
        )
    else:
        train_paths, val_paths = split_train_val(data_root, args.val_split, args.seed)
        train_dataset = FaceSegmentationDataset(
            root_dir=args.data_root,
            split="train",
            transform=image_transform,
            file_paths=train_paths,
            augment=True,
        )
        val_dataset = FaceSegmentationDataset(
            root_dir=args.data_root,
            split="train",
            transform=image_transform,
            file_paths=val_paths,
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = args.num_classes if args.num_classes is not None else config.get("num_classes", 16)
    model_type = args.model_type if args.model_type is not None else config.get("model_type", "v1")
    model = build_model({"num_classes": num_classes, "model_type": model_type})
    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")
    if param_count >= 1_800_000:
        raise RuntimeError(f"Model has {param_count} parameters which exceeds the limit")

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        ckpt_model_type = ckpt.get("model_type")
        if ckpt_model_type and ckpt_model_type != model_type:
            model_type = ckpt_model_type
            model = build_model({"num_classes": num_classes, "model_type": model_type}).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(f"Epoch {epoch + 1}/{args.num_epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "num_classes": num_classes,
            "model_type": model_type,
        }
        if is_best:
            best_state = copy.deepcopy(state)
            best_path = os.path.join(args.output_dir, "ckpt_best.pth")
            save_checkpoint(best_state, best_path)

    final_state = {
        "epoch": args.num_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "num_classes": num_classes,
        "model_type": model_type,
    }
    final_path = os.path.join(args.output_dir, "ckpt_last.pth")
    if best_state is None:
        best_state = copy.deepcopy(final_state)
        best_path = os.path.join(args.output_dir, "ckpt_best.pth")
        save_checkpoint(best_state, best_path)
    save_checkpoint(final_state, final_path)


if __name__ == "__main__":
    main()

