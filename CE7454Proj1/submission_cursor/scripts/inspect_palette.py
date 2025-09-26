import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = np.array(
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

LABEL_NAMES = {
    4: "left_eye",
    5: "right_eye",
    6: "left_brow",
    7: "right_brow",
    8: "left_ear",
    9: "right_ear",
}


def get_palette_color(label_id: int) -> tuple[int, int, int]:
    if not 0 <= label_id < len(PALETTE):
        raise ValueError(f"Label ID {label_id} outside palette range")
    return tuple(int(x) for x in PALETTE[label_id])


def inspect_mask(mask_path: Path) -> None:
    mask = Image.open(mask_path).convert("P")
    mask_np = np.array(mask, dtype=np.uint8)

    unique_values = np.unique(mask_np)
    print(f"Mask: {mask_path}")
    for label_id, name in LABEL_NAMES.items():
        color = get_palette_color(label_id)
        present = label_id in unique_values
        print(f"  {name:<12} (id={label_id:2d}) color={color} present={present}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect palette colors for key facial labels")
    parser.add_argument("mask", type=str, help="Path to mask image")
    args = parser.parse_args()

    inspect_mask(Path(args.mask))


if __name__ == "__main__":
    main()

