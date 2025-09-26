import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare submission artifacts for dev or test mode")
    parser.add_argument("mode", choices=["dev", "test"], help="Submission mode")
    parser.add_argument("--images", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--masks", type=str, required=True, help="Directory containing generated masks")
    parser.add_argument("--output", type=str, required=True, help="Directory to build submission structure")
    parser.add_argument(
        "--solution",
        type=str,
        default=None,
        help="Directory hosting solution files (used only in test mode)",
    )
    parser.add_argument("--zip-name", type=str, default=None, help="Optional custom name for submission zip")
    return parser.parse_args()


def validate_masks(images_dir: Path, masks_dir: Path) -> None:
    image_files = sorted([path.stem for path in images_dir.glob("*.jpg")])
    mask_files = sorted([path.stem for path in masks_dir.glob("*.png")])
    missing = sorted(set(image_files) - set(mask_files))
    if missing:
        raise FileNotFoundError(
            "Missing mask files for images: " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
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
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    solution_dir = Path(args.solution) if args.solution else None

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    validate_masks(images_dir, masks_dir)

    masks_output = output_dir / "masks"
    copy_masks(masks_dir, masks_output)

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

