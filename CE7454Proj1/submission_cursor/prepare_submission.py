import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate masks and package submission for dev/test modes")
    parser.add_argument("mode", choices=["dev", "test"], help="Submission mode")
    parser.add_argument("--images", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=str, required=True, help="Directory to build submission structure")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--run-script", type=str, default="run.py", help="Inference script to execute")
    parser.add_argument(
        "--solution",
        type=str,
        default=None,
        help="Directory hosting solution files (used only in test mode)",
    )
    parser.add_argument("--zip-name", type=str, default=None, help="Optional custom name for submission zip")
    parser.add_argument("--image-ext", type=str, default=".jpg", help="Input image extension (default: .jpg)")
    return parser.parse_args()


def run_inference(image_paths: Iterable[Path], masks_dir: Path, weights: Path, run_script: Path) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        cmd = [
            "python",
            str(run_script),
            "--input",
            str(image_path),
            "--output",
            str(mask_path),
            "--weights",
            str(weights),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Inference failed for {image_path.name}: {result.stderr or result.stdout}"
            )


def copy_masks(masks_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in masks_dir.glob("*.png"):
        shutil.copy2(mask_path, target_dir / mask_path.name)


def copy_solution_artifacts(solution_dir: Path, target_dir: Path) -> None:
    required_files = ["ckpt.pth", "requirements.txt", "run.py"]
    for name in required_files:
        src = solution_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing required solution file: {src}")
        shutil.copy2(src, target_dir / name)


def create_zip(source_dir: Path, zip_name: str) -> Path:
    zip_path = source_dir.parent / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
    return zip_path


def main() -> None:
    args = parse_args()
    mode = args.mode
    images_dir = Path(args.images)
    output_dir = Path(args.output)
    weights_path = Path(args.weights)
    run_script = Path(args.run_script)
    solution_dir = Path(args.solution) if args.solution else None
    image_ext = args.image_ext

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not run_script.is_file():
        raise FileNotFoundError(f"Run script not found: {run_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    masks_output = output_dir / "masks"
    temp_masks_dir = output_dir / "_generated_masks"
    temp_masks_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob(f"*{image_ext}"))
    if not image_paths:
        raise FileNotFoundError(f"No images with extension {image_ext} found in {images_dir}")

    run_inference(image_paths, temp_masks_dir, weights_path, run_script)
    copy_masks(temp_masks_dir, masks_output)
    shutil.rmtree(temp_masks_dir)

    if mode == "test":
        if solution_dir is None:
            raise ValueError("--solution must be provided in test mode")
        if not solution_dir.is_dir():
            raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

        solution_output = output_dir / "solution"
        solution_output.mkdir(parents=True, exist_ok=True)
        copy_solution_artifacts(solution_dir, solution_output)

    if args.zip_name:
        zip_name = args.zip_name
    else:
        zip_name = f"submission_{mode}.zip"
    zip_path = create_zip(output_dir, zip_name)
    print(f"Submission prepared at {output_dir}")
    print(f"Zip archive created: {zip_path}")


if __name__ == "__main__":
    main()

