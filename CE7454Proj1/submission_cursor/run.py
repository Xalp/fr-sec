import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_model
from model_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for face segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save predicted mask")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    return parser.parse_args()


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def create_palette() -> np.ndarray:
    palette = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
    palette[:16] = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [191, 0, 0],
            [64, 128, 0],
            [191, 128, 0],
            [64, 0, 128],
            [191, 0, 128],
            [64, 128, 128],
            [191, 128, 128],
        ],
        dtype=np.uint8,
    )
    return palette.reshape(-1)


def save_mask(mask: torch.Tensor, output_path: Path) -> None:
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask_np, mode="P")
    img.putpalette(create_palette().tolist())
    img.save(output_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found at {input_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(str(weights_path), map_location="cpu")
    num_classes = checkpoint.get("num_classes", 19)
    model = build_model({"num_classes": num_classes})
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image_tensor = load_image(input_path).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        mask = torch.argmax(logits, dim=1)

    save_mask(mask, output_path)


if __name__ == "__main__":
    main()

