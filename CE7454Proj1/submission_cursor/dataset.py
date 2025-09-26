import random
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

import torch
from torch.utils.data import Dataset


class FaceSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        file_paths: Optional[Sequence[Path]] = None,
        augment: bool = False,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
        self.ignore_index = ignore_index

        image_dir = self.root_dir / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Could not find image directory at {image_dir}")

        if file_paths is not None:
            self.image_paths = [Path(path) for path in file_paths]
        else:
            exts: Iterable[str] = ("*.jpg", "*.jpeg", "*.png")
            image_files = []
            for ext in exts:
                image_files.extend(image_dir.glob(ext))
            self.image_paths = sorted(image_files)

        if split != "test":
            mask_dir = self.root_dir / split / "masks"
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Could not find mask directory at {mask_dir}")
            self.mask_paths = [mask_dir / (path.stem + ".png") for path in self.image_paths]
        else:
            self.mask_paths = [None] * len(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(
        self, image: Image.Image, mask: Optional[Image.Image]
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        mask_path = self.mask_paths[idx]
        mask: Optional[Image.Image]
        if mask_path is not None:
            if not mask_path.is_file():
                raise FileNotFoundError(f"Could not find mask file at {mask_path}")
            mask = Image.open(mask_path).convert("P")
        else:
            mask = None

        image, mask = self._apply_augmentations(image, mask)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        if mask is not None:
            mask_array = np.array(mask, dtype=np.int64)
            mask_array = np.where(mask_array > 15, 0, mask_array)
            if self.target_transform:
                mask_tensor = self.target_transform(mask_array)
            else:
                mask_tensor = torch.from_numpy(mask_array)
        else:
            mask_tensor = torch.full((1,), self.ignore_index, dtype=torch.long)

        return image_tensor, mask_tensor

